import logging
import os
import time
import json
import hashlib
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import requests
from huggingface_hub import HfApi, login, get_hf_file_metadata, hf_hub_url
import psutil

# Set up logging with thread-safe handler
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
)

# Thread-safe lock for logging
log_lock = threading.Lock()

def safe_log(level, message):
    """Thread-safe logging function."""
    with log_lock:
        if level == 'info':
            logging.info(message)
        elif level == 'error':
            logging.error(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'debug':
            logging.debug(message)

# Configuration
MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100MB - files larger than this use multipart
MULTIPART_CHUNKSIZE = 100 * 1024 * 1024  # 100MB chunks for multipart upload
STREAM_CHUNK_SIZE = 10 * 1024 * 1024     # 10MB chunks for streaming from HF
MAX_RETRIES = 3
RETRY_DELAY = 5
VERIFY_SIZE = True  # Verify file sizes match
FORCE_REDOWNLOAD = False  # Set to True to redownload all files

# Environment variables
model_id = os.getenv("MODEL_ID")
huggingface_token = os.getenv("HF_TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_HOST = os.getenv("AWS_HOST")  # Your Ceph endpoint
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = "-".join((model_id.split("/")[1]).split("-")[:3])
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")  # May not matter for Ceph

# Cache files for tracking state
UPLOAD_CACHE_FILE = ".upload_cache.json"
MULTIPART_CACHE_FILE = ".multipart_cache.json"

# File patterns to skip (optional)
SKIP_PATTERNS = [
    # "*.gguf",  # Uncomment to skip GGUF files
    # "*.onnx",  # Uncomment to skip ONNX files
]

@dataclass
class FileInfo:
    """Information about a file to transfer."""
    path: str
    size: int
    url: str
    etag: Optional[str] = None
    sha256: Optional[str] = None

@dataclass
class S3FileInfo:
    """Information about a file in S3."""
    key: str
    size: int
    etag: Optional[str] = None
    last_modified: Optional[str] = None

@dataclass
class MultipartUploadState:
    """State of an incomplete multipart upload."""
    upload_id: str
    key: str
    parts_completed: List[int]
    total_parts: int

class StreamingUploader:
    """Handles streaming uploads from HuggingFace to S3-compatible storage with resume capability."""
    
    def __init__(self, s3_client, bucket: str, token: str):
        self.s3_client = s3_client
        self.bucket = bucket
        self.token = token
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.multipart_cache = self.load_multipart_cache()
    
    def load_multipart_cache(self) -> Dict[str, MultipartUploadState]:
        """Load cache of incomplete multipart uploads."""
        if os.path.exists(MULTIPART_CACHE_FILE):
            try:
                with open(MULTIPART_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    return {
                        k: MultipartUploadState(**v) 
                        for k, v in data.items()
                    }
            except:
                return {}
        return {}
    
    def save_multipart_cache(self):
        """Save multipart upload cache."""
        try:
            cache_data = {
                k: {
                    'upload_id': v.upload_id,
                    'key': v.key,
                    'parts_completed': v.parts_completed,
                    'total_parts': v.total_parts
                }
                for k, v in self.multipart_cache.items()
            }
            with open(MULTIPART_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            safe_log('warning', f"Could not save multipart cache: {e}")
    
    def check_file_exists(self, s3_key: str) -> Optional[S3FileInfo]:
        """Check if a file already exists in S3 and get its info."""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return S3FileInfo(
                key=s3_key,
                size=response.get('ContentLength', 0),
                etag=response.get('ETag', '').strip('"'),
                last_modified=str(response.get('LastModified', ''))
            )
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise
    
    def list_incomplete_multipart_uploads(self) -> Dict[str, str]:
        """List all incomplete multipart uploads for the bucket."""
        incomplete = {}
        try:
            response = self.s3_client.list_multipart_uploads(Bucket=self.bucket)
            for upload in response.get('Uploads', []):
                incomplete[upload['Key']] = upload['UploadId']
        except Exception as e:
            safe_log('debug', f"Could not list multipart uploads: {e}")
        return incomplete
    
    def resume_or_abort_multipart(self, s3_key: str, file_info: FileInfo) -> Optional[MultipartUploadState]:
        """Check for incomplete multipart upload and decide whether to resume or abort."""
        # Check cache first
        if s3_key in self.multipart_cache:
            state = self.multipart_cache[s3_key]
            safe_log('info', f"Found cached multipart upload for {s3_key}")
            
            # Verify the upload still exists
            try:
                self.s3_client.list_parts(
                    Bucket=self.bucket,
                    Key=s3_key,
                    UploadId=state.upload_id
                )
                safe_log('info', f"Resuming multipart upload with {len(state.parts_completed)} parts completed")
                return state
            except:
                safe_log('info', f"Cached upload no longer exists, starting fresh")
                del self.multipart_cache[s3_key]
        
        # Check for any incomplete uploads for this key
        incomplete = self.list_incomplete_multipart_uploads()
        if s3_key in incomplete:
            upload_id = incomplete[s3_key]
            safe_log('info', f"Found incomplete multipart upload for {s3_key}")
            
            # Get list of completed parts
            try:
                response = self.s3_client.list_parts(
                    Bucket=self.bucket,
                    Key=s3_key,
                    UploadId=upload_id
                )
                parts_completed = [part['PartNumber'] for part in response.get('Parts', [])]
                total_parts = (file_info.size + MULTIPART_CHUNKSIZE - 1) // MULTIPART_CHUNKSIZE
                
                state = MultipartUploadState(
                    upload_id=upload_id,
                    key=s3_key,
                    parts_completed=parts_completed,
                    total_parts=total_parts
                )
                
                if parts_completed:
                    safe_log('info', f"Found {len(parts_completed)} completed parts, resuming upload")
                    self.multipart_cache[s3_key] = state
                    self.save_multipart_cache()
                    return state
                else:
                    # No parts completed, abort and start fresh
                    safe_log('info', f"No parts completed, aborting old upload and starting fresh")
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=s3_key,
                        UploadId=upload_id
                    )
            except Exception as e:
                safe_log('warning', f"Could not list parts for upload: {e}")
                try:
                    self.s3_client.abort_multipart_upload(
                        Bucket=self.bucket,
                        Key=s3_key,
                        UploadId=upload_id
                    )
                except:
                    pass
        
        return None
    
    def should_download_file(self, file_info: FileInfo, s3_info: Optional[S3FileInfo]) -> Tuple[bool, str]:
        """Determine if a file should be downloaded."""
        if FORCE_REDOWNLOAD:
            return True, "force redownload enabled"
        
        if s3_info is None:
            return True, "file not found in S3"
        
        # Check size match
        if VERIFY_SIZE and s3_info.size != file_info.size:
            safe_log('warning', f"Size mismatch for {file_info.path}: "
                              f"S3={s3_info.size} bytes, HF={file_info.size} bytes")
            return True, f"size mismatch (S3={s3_info.size}, HF={file_info.size})"
        
        # If sizes match, consider it complete
        if s3_info.size == file_info.size:
            return False, f"already exists with correct size ({file_info.size} bytes)"
        
        return True, "verification failed"
    
    def stream_to_s3(self, file_info: FileInfo, s3_key: str) -> Tuple[bool, str]:
        """Stream a file directly from HuggingFace to S3 without local storage."""
        try:
            # Check if file already exists
            s3_info = self.check_file_exists(s3_key)
            should_download, reason = self.should_download_file(file_info, s3_info)
            
            if not should_download:
                safe_log('info', f"✓ Skipping {file_info.path}: {reason}")
                return True, f"skipped - {reason}"
            
            safe_log('info', f"Downloading {file_info.path}: {reason}")
            
            if file_info.size < MULTIPART_THRESHOLD:
                return self._simple_upload(file_info, s3_key)
            else:
                return self._multipart_upload_with_resume(file_info, s3_key)
        except Exception as e:
            safe_log('error', f"Failed to stream {file_info.path}: {e}")
            return False, str(e)
    
    def _simple_upload(self, file_info: FileInfo, s3_key: str) -> Tuple[bool, str]:
        """Simple upload for small files (< 100MB)."""
        safe_log('info', f"Simple upload: {file_info.path} ({file_info.size:,} bytes)")
        
        try:
            # Stream the entire file into memory (safe for small files)
            response = requests.get(file_info.url, headers=self.headers, stream=False)
            response.raise_for_status()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=response.content,
                ContentLength=file_info.size
            )
            
            safe_log('info', f"✓ Uploaded: {s3_key}")
            return True, "uploaded successfully"
            
        except Exception as e:
            safe_log('error', f"✗ Simple upload failed for {file_info.path}: {e}")
            return False, str(e)
    
    def _multipart_upload_with_resume(self, file_info: FileInfo, s3_key: str) -> Tuple[bool, str]:
        """Multipart upload with resume capability."""
        safe_log('info', f"Multipart upload: {file_info.path} ({file_info.size:,} bytes)")
        
        # Check for existing incomplete upload
        existing_state = self.resume_or_abort_multipart(s3_key, file_info)
        
        # Calculate total parts
        num_parts = (file_info.size + MULTIPART_CHUNKSIZE - 1) // MULTIPART_CHUNKSIZE
        safe_log('info', f"Will upload in {num_parts} parts")
        
        if existing_state:
            upload_id = existing_state.upload_id
            completed_parts = set(existing_state.parts_completed)
            safe_log('info', f"Resuming upload: {len(completed_parts)}/{num_parts} parts already completed")
            
            # Get existing parts info
            response = self.s3_client.list_parts(
                Bucket=self.bucket,
                Key=s3_key,
                UploadId=upload_id
            )
            parts = [{'ETag': part['ETag'], 'PartNumber': part['PartNumber']} 
                    for part in response.get('Parts', [])]
        else:
            # Initiate new multipart upload
            try:
                mpu = self.s3_client.create_multipart_upload(
                    Bucket=self.bucket,
                    Key=s3_key
                )
                upload_id = mpu['UploadId']
                completed_parts = set()
                parts = []
                safe_log('debug', f"Initiated new multipart upload: {upload_id}")
                
                # Save to cache
                self.multipart_cache[s3_key] = MultipartUploadState(
                    upload_id=upload_id,
                    key=s3_key,
                    parts_completed=[],
                    total_parts=num_parts
                )
                self.save_multipart_cache()
                
            except Exception as e:
                safe_log('error', f"Failed to initiate multipart upload: {e}")
                return False, str(e)
        
        bytes_uploaded = len(completed_parts) * MULTIPART_CHUNKSIZE
        
        try:
            # Stream file in chunks
            response = requests.get(file_info.url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            buffer = BytesIO()
            part_number = 1
            start_time = time.time()
            bytes_read = 0
            
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                if not chunk:
                    continue
                
                buffer.write(chunk)
                bytes_read += len(chunk)
                
                # When buffer reaches MULTIPART_CHUNKSIZE or we're at the end
                should_upload_part = (buffer.tell() >= MULTIPART_CHUNKSIZE or 
                                    bytes_read >= file_info.size)
                
                if should_upload_part and buffer.tell() > 0:
                    # Check if this part was already uploaded
                    if part_number in completed_parts:
                        safe_log('info', f"Part {part_number}/{num_parts} already uploaded, skipping")
                        
                        # Skip to next part
                        buffer = BytesIO()
                        part_number += 1
                        continue
                    
                    buffer.seek(0)
                    part_data = buffer.read(MULTIPART_CHUNKSIZE)
                    
                    # Upload part with retries
                    for attempt in range(MAX_RETRIES):
                        try:
                            part_response = self.s3_client.upload_part(
                                Body=part_data,
                                Bucket=self.bucket,
                                Key=s3_key,
                                PartNumber=part_number,
                                UploadId=upload_id,
                                ContentLength=len(part_data)
                            )
                            
                            parts.append({
                                'ETag': part_response['ETag'],
                                'PartNumber': part_number
                            })
                            
                            completed_parts.add(part_number)
                            bytes_uploaded += len(part_data)
                            
                            # Update cache
                            if s3_key in self.multipart_cache:
                                self.multipart_cache[s3_key].parts_completed = list(completed_parts)
                                self.save_multipart_cache()
                            
                            elapsed = time.time() - start_time
                            speed = (bytes_uploaded - len(completed_parts) * MULTIPART_CHUNKSIZE) / elapsed if elapsed > 0 else 0
                            progress = (bytes_uploaded / file_info.size) * 100
                            
                            safe_log('info', f"Part {part_number}/{num_parts} uploaded "
                                           f"({progress:.1f}%, {speed/1024/1024:.1f} MB/s)")
                            break
                            
                        except Exception as e:
                            if attempt == MAX_RETRIES - 1:
                                raise
                            safe_log('warning', f"Retry {attempt + 1}/{MAX_RETRIES} for part {part_number}: {e}")
                            time.sleep(RETRY_DELAY)
                    
                    # Reset buffer for next part
                    remaining = buffer.read()
                    buffer = BytesIO()
                    if remaining:
                        buffer.write(remaining)
                    
                    part_number += 1
            
            # Handle any remaining data in buffer
            if buffer.tell() > 0 and part_number <= num_parts:
                if part_number not in completed_parts:
                    buffer.seek(0)
                    part_data = buffer.read()
                    
                    for attempt in range(MAX_RETRIES):
                        try:
                            part_response = self.s3_client.upload_part(
                                Body=part_data,
                                Bucket=self.bucket,
                                Key=s3_key,
                                PartNumber=part_number,
                                UploadId=upload_id,
                                ContentLength=len(part_data)
                            )
                            
                            parts.append({
                                'ETag': part_response['ETag'],
                                'PartNumber': part_number
                            })
                            
                            bytes_uploaded += len(part_data)
                            safe_log('info', f"Final part {part_number}/{num_parts} uploaded")
                            break
                            
                        except Exception as e:
                            if attempt == MAX_RETRIES - 1:
                                raise
                            safe_log('warning', f"Retry {attempt + 1}/{MAX_RETRIES} for final part: {e}")
                            time.sleep(RETRY_DELAY)
            
            # Sort parts by part number for completion
            parts.sort(key=lambda x: x['PartNumber'])
            
            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=s3_key,
                UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            
            # Remove from cache after successful completion
            if s3_key in self.multipart_cache:
                del self.multipart_cache[s3_key]
                self.save_multipart_cache()
            
            elapsed_total = time.time() - start_time
            avg_speed = file_info.size / elapsed_total if elapsed_total > 0 else 0
            safe_log('info', f"✓ Multipart upload complete: {s3_key} "
                           f"({elapsed_total:.1f}s, {avg_speed/1024/1024:.1f} MB/s avg)")
            return True, "uploaded successfully"
            
        except Exception as e:
            safe_log('error', f"✗ Multipart upload failed: {e}")
            # Don't abort on failure - we can resume later
            return False, str(e)

def get_s3_client():
    """Create S3 client for Ceph or S3-compatible storage."""
    # Configure for S3-compatible storage (Ceph)
    config = Config(
        region_name=AWS_REGION,
        signature_version='s3v4',
        retries={
            'max_attempts': 3,
            'mode': 'standard'
        },
        max_pool_connections=50
    )
    
    # Use custom endpoint for Ceph
    endpoint_url = f"https://{AWS_HOST}"
    if not AWS_HOST.startswith('http'):
        endpoint_url = f"https://{AWS_HOST}"
    elif AWS_HOST.startswith('http://'):
        endpoint_url = AWS_HOST  # Allow HTTP for internal Ceph
    else:
        endpoint_url = AWS_HOST
    
    safe_log('info', f"Connecting to S3-compatible storage at: {endpoint_url}")
    
    return boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=config,
        use_ssl=not endpoint_url.startswith('http://'),  # Disable SSL for HTTP endpoints
        verify=False  # You might need this for self-signed certificates
    )

def detect_resources():
    """Detect available system resources."""
    cpu_count = os.cpu_count() or 1
    memory_gb = psutil.virtual_memory().total / (1024**3)
    disk_gb = psutil.disk_usage('/').free / (1024**3)
    
    # Check if running in Kubernetes
    in_kubernetes = os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount')
    
    if in_kubernetes:
        safe_log('info', "Running in Kubernetes")
        # Try to read cgroup limits
        try:
            # Try cgroup v2 first
            if os.path.exists('/sys/fs/cgroup/cpu.max'):
                with open('/sys/fs/cgroup/cpu.max', 'r') as f:
                    cpu_max = f.read().strip().split()
                    if cpu_max[0] != 'max':
                        cpu_quota = int(cpu_max[0])
                        cpu_period = int(cpu_max[1])
                        cpu_count = max(1, cpu_quota // cpu_period)
            # Fall back to cgroup v1
            elif os.path.exists('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'):
                with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us', 'r') as f:
                    cpu_quota = int(f.read().strip())
                with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us', 'r') as f:
                    cpu_period = int(f.read().strip())
                if cpu_quota > 0 and cpu_period > 0:
                    cpu_count = max(1, cpu_quota // cpu_period)
        except Exception as e:
            safe_log('warning', f"Could not read CPU limits: {e}")
        
        # Memory limits
        try:
            if os.path.exists('/sys/fs/cgroup/memory.max'):
                with open('/sys/fs/cgroup/memory.max', 'r') as f:
                    mem_max = f.read().strip()
                    if mem_max != 'max':
                        memory_gb = int(mem_max) / (1024**3)
            elif os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
                with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                    mem_limit = int(f.read().strip())
                if mem_limit < psutil.virtual_memory().total:
                    memory_gb = mem_limit / (1024**3)
        except Exception as e:
            safe_log('warning', f"Could not read memory limits: {e}")
    
    # For streaming operations, we can use more workers since we don't store locally
    # Network I/O bound, not disk I/O bound
    max_workers = max(1, min(cpu_count * 6, 32))  # 6x CPUs for network streaming
    
    safe_log('info', f"Detected resources - CPUs: {cpu_count}, Memory: {memory_gb:.2f}GB, Free disk: {disk_gb:.2f}GB")
    safe_log('info', f"Using {max_workers} parallel workers for streaming")
    
    return max_workers, memory_gb, disk_gb

def load_upload_cache():
    """Load cache of already uploaded files."""
    if os.path.exists(UPLOAD_CACHE_FILE):
        try:
            with open(UPLOAD_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_upload_cache(cache):
    """Save upload cache."""
    try:
        with open(UPLOAD_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        safe_log('warning', f"Could not save cache: {e}")

def should_skip_file(file_path):
    """Check if file should be skipped based on patterns."""
    from fnmatch import fnmatch
    for pattern in SKIP_PATTERNS:
        if fnmatch(file_path, pattern):
            return True
    return False

def get_file_info_list(api: HfApi, model_id: str, token: str) -> List[FileInfo]:
    """Get list of files with their metadata."""
    safe_log('info', f"Fetching file list for {model_id}...")
    
    try:
        # Get repository info
        repo_info = api.repo_info(repo_id=model_id, token=token, files_metadata=True)
        
        file_infos = []
        for file in repo_info.siblings:
            # Get the download URL
            url = hf_hub_url(
                repo_id=model_id,
                filename=file.rfilename,
                repo_type="model"
            )
            
            file_infos.append(FileInfo(
                path=file.rfilename,
                size=file.size if hasattr(file, 'size') else 0,
                url=url,
                etag=file.lfs.sha256 if hasattr(file, 'lfs') and file.lfs else None,
                sha256=file.lfs.sha256 if hasattr(file, 'lfs') and file.lfs else None
            ))
        
        # Sort by size (smallest first for quick wins)
        file_infos.sort(key=lambda x: x.size)
        
        total_size = sum(f.size for f in file_infos)
        safe_log('info', f"Found {len(file_infos)} files, total size: {total_size / (1024**3):.2f} GB")
        
        return file_infos
        
    except Exception as e:
        safe_log('error', f"Failed to get file list: {e}")
        raise

def process_single_file(args) -> Tuple[str, bool, str, Optional[str]]:
    """Process a single file by streaming it directly to S3."""
    file_info, uploader, upload_cache, file_index, total_files = args
    
    # Check skip patterns
    if should_skip_file(file_info.path):
        safe_log('info', f"[{file_index}/{total_files}] Skipping by pattern: {file_info.path}")
        return (file_info.path, True, "skipped by pattern", None)
    
    safe_log('info', f"[{file_index}/{total_files}] Processing: {file_info.path} ({file_info.size:,} bytes)")
    
    # Stream directly to S3
    s3_key = f"{S3_PREFIX}/{file_info.path}"
    success, message = uploader.stream_to_s3(file_info, s3_key)
    
    # Create cache key
    cache_key = f"{file_info.path}_{file_info.size}_{file_info.sha256}"
    
    if success:
        return (file_info.path, True, message, cache_key)
    else:
        return (file_info.path, False, message, None)

def cleanup_incomplete_uploads(s3_client, bucket: str):
    """Clean up old incomplete multipart uploads."""
    try:
        safe_log('info', "Checking for old incomplete uploads...")
        response = s3_client.list_multipart_uploads(Bucket=bucket)
        uploads = response.get('Uploads', [])
        
        if uploads:
            safe_log('info', f"Found {len(uploads)} incomplete uploads")
            for upload in uploads:
                # Check age of upload
                upload_time = upload['Initiated']
                age_hours = (time.time() - upload_time.timestamp()) / 3600
                
                # Clean up uploads older than 24 hours
                if age_hours > 24:
                    try:
                        s3_client.abort_multipart_upload(
                            Bucket=bucket,
                            Key=upload['Key'],
                            UploadId=upload['UploadId']
                        )
                        safe_log('info', f"Cleaned up old upload for {upload['Key']} (age: {age_hours:.1f} hours)")
                    except:
                        pass
        else:
            safe_log('info', "No incomplete uploads found")
    except Exception as e:
        safe_log('warning', f"Could not check for incomplete uploads: {e}")

def stream_all_files(model_id: str, token: str, max_workers: int):
    """Main function to stream all files from HuggingFace to S3."""
    try:
        # Initialize
        api = HfApi()
        s3_client = get_s3_client()
        uploader = StreamingUploader(s3_client, S3_BUCKET, token)
        
        # Clean up old incomplete uploads (optional)
        cleanup_incomplete_uploads(s3_client, S3_BUCKET.replace("s3://", ""))
        
        # Get file list
        file_infos = get_file_info_list(api, model_id, token)
        
        if not file_infos:
            safe_log('error', "No files found in repository")
            return
        
        # Load cache
        upload_cache = load_upload_cache()
        
        # Statistics
        successful_uploads = 0
        skipped_files = 0
        failed_uploads = 0
        failed_files = []
        total_bytes = sum(f.size for f in file_infos)
        bytes_transferred = 0
        
        # Prepare arguments for parallel processing
        file_args = [
            (file_info, uploader, upload_cache, i, len(file_infos))
            for i, file_info in enumerate(file_infos, 1)
        ]
        
        safe_log('info', f"Starting parallel streaming with {max_workers} workers...")
        safe_log('info', "Files will be checked and resumed if already partially uploaded")
        start_time = time.time()
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(process_single_file, args): args[0]
                for args in file_args
            }
            
            for future in as_completed(future_to_file):
                file_info = future_to_file[future]
                try:
                    file_path, success, message, cache_key = future.result()
                    
                    if success:
                        if "skipped" in message.lower():
                            skipped_files += 1
                        else:
                            successful_uploads += 1
                            
                        if cache_key:
                            upload_cache[cache_key] = True
                            save_upload_cache(upload_cache)
                        
                        # Update bytes transferred
                        bytes_transferred += file_info.size
                    else:
                        failed_uploads += 1
                        failed_files.append(f"{file_path}: {message}")
                    
                    # Progress report
                    progress = successful_uploads + skipped_files + failed_uploads
                    elapsed = time.time() - start_time
                    rate = bytes_transferred / elapsed if elapsed > 0 else 0
                    eta = (total_bytes - bytes_transferred) / rate if rate > 0 else 0
                    
                    safe_log('info', 
                            f"Progress: {progress}/{len(file_infos)} files | "
                            f"Uploaded: {successful_uploads} | Skipped: {skipped_files} | Failed: {failed_uploads} | "
                            f"{bytes_transferred/(1024**3):.1f}/{total_bytes/(1024**3):.1f} GB | "
                            f"Speed: {rate/(1024**2):.1f} MB/s | "
                            f"ETA: {eta/60:.1f} min")
                    
                except Exception as e:
                    safe_log('error', f"Exception processing file: {e}")
                    failed_uploads += 1
                    if hasattr(file_info, 'path'):
                        failed_files.append(f"{file_info.path}: {str(e)}")
        
        # Final summary
        elapsed_total = time.time() - start_time
        avg_speed = bytes_transferred / elapsed_total if elapsed_total > 0 else 0
        
        safe_log('info', "\n" + "="*50)
        safe_log('info', "FINAL SUMMARY")
        safe_log('info', "="*50)
        safe_log('info', f"Total files: {len(file_infos)}")
        safe_log('info', f"Newly uploaded: {successful_uploads}")
        safe_log('info', f"Skipped (already exist): {skipped_files}")
        safe_log('info', f"Failed: {failed_uploads}")
        safe_log('info', f"Success rate: {((successful_uploads + skipped_files)/len(file_infos)*100):.1f}%")
        safe_log('info', f"Total time: {elapsed_total/60:.1f} minutes")
        safe_log('info', f"Average speed: {avg_speed/(1024**2):.1f} MB/s")
        safe_log('info', f"Data transferred: {bytes_transferred/(1024**3):.2f} GB")
        
        if failed_files:
            safe_log('error', "Failed files:")
            for file in failed_files[:10]:  # Show first 10 failures
                safe_log('error', f"  - {file}")
            if len(failed_files) > 10:
                safe_log('error', f"  ... and {len(failed_files) - 10} more")
            
            with open('failed_files.txt', 'w') as f:
                for file in failed_files:
                    f.write(f"{file}\n")
            safe_log('info', "Failed files saved to failed_files.txt")
        
    except Exception as e:
        safe_log('error', f"Fatal error: {e}")
        raise

def monitor_resources():
    """Monitor system resources during transfer."""
    def monitor():
        while not stop_monitoring.is_set():
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            net = psutil.net_io_counters()
            
            safe_log('info', 
                    f"System: CPU {cpu:.1f}% | "
                    f"RAM {mem.percent:.1f}% ({mem.used/(1024**3):.1f}GB) | "
                    f"Disk {disk.percent:.1f}% ({disk.free/(1024**3):.1f}GB free) | "
                    f"Net ↓{net.bytes_recv/(1024**3):.1f}GB ↑{net.bytes_sent/(1024**3):.1f}GB")
            
            time.sleep(30)
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    return thread

# Global monitoring flag
stop_monitoring = threading.Event()

def login_to_huggingface(token):
    """Login to Hugging Face."""
    try:
        login(token=token)
        safe_log('info', "Logged in to Hugging Face")
    except Exception as e:
        safe_log('error', f"Failed to login to Hugging Face: {e}")
        raise

if __name__ == "__main__":
    # Validate environment
    if not all([model_id, huggingface_token, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_HOST, S3_BUCKET]):
        logging.error("Missing required environment variables")
        logging.error("Required: MODEL_ID, HF_TOKEN, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_HOST, S3_BUCKET")
        exit(1)
    
    # Detect resources
    max_workers, memory_gb, disk_gb = detect_resources()
    
    if disk_gb < 10:
        safe_log('warning', f"Low disk space: {disk_gb:.1f}GB free. Streaming mode active (no local storage needed).")
    
    # Start monitoring
    monitor_thread = monitor_resources()
    
    try:
        # Login to HuggingFace
        login_to_huggingface(huggingface_token)
        
        # Stream all files
        stream_all_files(model_id, huggingface_token, max_workers)
        
    finally:
        # Stop monitoring
        stop_monitoring.set()
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=2)
    
    safe_log('info', "Transfer complete!")
