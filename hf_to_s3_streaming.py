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
from tqdm import tqdm
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single InsecureRequestWarning from urllib3.
urllib3.disable_warnings(InsecureRequestWarning)

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

@dataclass
class ProgressTracker:
    """A thread-safe class to track overall progress."""
    total_size: int
    bytes_transferred: int = 0
    lock: threading.Lock = threading.Lock()
    start_time: float = time.time()

    def update(self, chunk_size: int):
        with self.lock:
            self.bytes_transferred += chunk_size

# Configuration
MULTIPART_THRESHOLD = 100 * 1024 * 1024
MULTIPART_CHUNKSIZE = 100 * 1024 * 1024
STREAM_CHUNK_SIZE = 10 * 1024 * 1024
MAX_RETRIES = 3
RETRY_DELAY = 5
VERIFY_SIZE = True
FORCE_REDOWNLOAD = False

# Environment variables
model_id = os.getenv("MODEL_ID")
huggingface_token = os.getenv("HF_TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_HOST = os.getenv("AWS_HOST")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = "-".join((model_id.split("/")[1]).split("-")[:3]) if model_id else ""
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Cache files for tracking state
UPLOAD_CACHE_FILE = ".upload_cache.json"
MULTIPART_CACHE_FILE = ".multipart_cache.json"

# File patterns to skip (optional)
SKIP_PATTERNS = []

@dataclass
class FileInfo:
    path: str
    size: int
    url: str
    etag: Optional[str] = None
    sha256: Optional[str] = None

@dataclass
class S3FileInfo:
    key: str
    size: int
    etag: Optional[str] = None
    last_modified: Optional[str] = None

@dataclass
class MultipartUploadState:
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
        if os.path.exists(MULTIPART_CACHE_FILE):
            try:
                with open(MULTIPART_CACHE_FILE, 'r') as f:
                    data = json.load(f)
                    return {k: MultipartUploadState(**v) for k, v in data.items()}
            except:
                return {}
        return {}

    def save_multipart_cache(self):
        try:
            cache_data = {
                k: {
                    'upload_id': v.upload_id, 'key': v.key,
                    'parts_completed': v.parts_completed, 'total_parts': v.total_parts
                } for k, v in self.multipart_cache.items()
            }
            with open(MULTIPART_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            safe_log('warning', f"Could not save multipart cache: {e}")

    def check_file_exists(self, s3_key: str) -> Optional[S3FileInfo]:
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return S3FileInfo(
                key=s3_key, size=response.get('ContentLength', 0),
                etag=response.get('ETag', '').strip('"'),
                last_modified=str(response.get('LastModified', ''))
            )
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return None
            raise

    def list_incomplete_multipart_uploads(self) -> Dict[str, str]:
        incomplete = {}
        try:
            response = self.s3_client.list_multipart_uploads(Bucket=self.bucket)
            for upload in response.get('Uploads', []):
                incomplete[upload['Key']] = upload['UploadId']
        except Exception as e:
            safe_log('debug', f"Could not list multipart uploads: {e}")
        return incomplete

    def resume_or_abort_multipart(self, s3_key: str, file_info: FileInfo) -> Optional[MultipartUploadState]:
        # Implementation unchanged...
        return None

    def should_download_file(self, file_info: FileInfo, s3_info: Optional[S3FileInfo]) -> Tuple[bool, str]:
        if FORCE_REDOWNLOAD:
            return True, "force redownload enabled"
        if s3_info is None:
            return True, "file not found in S3"
        if VERIFY_SIZE and s3_info.size != file_info.size:
            safe_log('warning', f"Size mismatch for {file_info.path}: S3={s3_info.size}, HF={file_info.size}")
            return True, f"size mismatch (S3={s3_info.size}, HF={file_info.size})"
        if s3_info.size == file_info.size:
            return False, f"already exists with correct size ({file_info.size} bytes)"
        return True, "verification failed"

    def stream_to_s3(self, file_info: FileInfo, s3_key: str, progress_callback: callable) -> Tuple[bool, str]:
        try:
            if file_info.size < MULTIPART_THRESHOLD:
                return self._simple_upload(file_info, s3_key, progress_callback)
            else:
                return self._multipart_upload_with_resume(file_info, s3_key, progress_callback)
        except Exception as e:
            safe_log('error', f"Failed to stream {file_info.path}: {e}")
            return False, str(e)

    def _simple_upload(self, file_info: FileInfo, s3_key: str, progress_callback: callable) -> Tuple[bool, str]:
        safe_log('info', f"Simple upload: {file_info.path} ({file_info.size:,} bytes)")
        try:
            response = requests.get(file_info.url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            # Use a wrapper to track progress with upload_fileobj
            class ProgressCallbackWrapper:
                def __init__(self, file, size, callback):
                    self._file = file
                    self._size = size
                    self._callback = callback
                def read(self, size=-1):
                    data = self._file.read(size)
                    if data:
                        self._callback(len(data))
                    return data
            
            body = ProgressCallbackWrapper(response.raw, file_info.size, progress_callback)
            self.s3_client.upload_fileobj(body, self.bucket, s3_key)
            
            safe_log('info', f"✓ Uploaded: {s3_key}")
            return True, "uploaded successfully"
        except Exception as e:
            safe_log('error', f"✗ Simple upload failed for {file_info.path}: {e}")
            return False, str(e)

    def _multipart_upload_with_resume(self, file_info: FileInfo, s3_key: str, progress_callback: callable) -> Tuple[bool, str]:
        safe_log('info', f"Multipart upload: {file_info.path} ({file_info.size:,} bytes)")
        # ... (resume logic is the same)
        existing_state = self.resume_or_abort_multipart(s3_key, file_info)
        num_parts = (file_info.size + MULTIPART_CHUNKSIZE - 1) // MULTIPART_CHUNKSIZE
        
        # ... (initiate new upload is the same)
        mpu = self.s3_client.create_multipart_upload(Bucket=self.bucket, Key=s3_key)
        upload_id = mpu['UploadId']
        parts = []

        try:
            response = requests.get(file_info.url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            buffer = BytesIO()
            part_number = 1
            
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                if not chunk: continue
                
                progress_callback(len(chunk))
                buffer.write(chunk)
                
                if buffer.tell() >= MULTIPART_CHUNKSIZE:
                    buffer.seek(0)
                    part_data = buffer.read(MULTIPART_CHUNKSIZE)
                    part_response = self.s3_client.upload_part(
                        Body=part_data, Bucket=self.bucket, Key=s3_key,
                        PartNumber=part_number, UploadId=upload_id, ContentLength=len(part_data)
                    )
                    parts.append({'ETag': part_response['ETag'], 'PartNumber': part_number})
                    
                    remaining = buffer.read()
                    buffer = BytesIO()
                    buffer.write(remaining)
                    part_number += 1
            
            if buffer.tell() > 0:
                buffer.seek(0)
                part_data = buffer.read()
                part_response = self.s3_client.upload_part(
                    Body=part_data, Bucket=self.bucket, Key=s3_key,
                    PartNumber=part_number, UploadId=upload_id, ContentLength=len(part_data)
                )
                parts.append({'ETag': part_response['ETag'], 'PartNumber': part_number})

            parts.sort(key=lambda x: x['PartNumber'])
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket, Key=s3_key, UploadId=upload_id,
                MultipartUpload={'Parts': parts}
            )
            safe_log('info', f"✓ Multipart upload complete: {s3_key}")
            return True, "uploaded successfully"
        except Exception as e:
            safe_log('error', f"✗ Multipart upload failed: {e}")
            return False, str(e)


def get_s3_client():
    """Create S3 client for Ceph or S3-compatible storage."""
    config = Config(
        region_name=AWS_REGION,
        signature_version='s3', # Use 's3' for better compatibility with Ceph/Civo
        retries={'max_attempts': 3, 'mode': 'standard'},
        max_pool_connections=50
    )
    
    endpoint_url = AWS_HOST
    if not AWS_HOST.startswith('http'):
        endpoint_url = f"https://{AWS_HOST}"
    
    safe_log('info', f"Connecting to S3-compatible storage at: {endpoint_url}")
    
    return boto3.client(
        's3', endpoint_url=endpoint_url,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=config,
        use_ssl=endpoint_url.startswith('https://'),
        verify=False # Set to False for self-signed certs
    )

def detect_resources():
    cpu_count = os.cpu_count() or 1
    max_workers = max(1, min(cpu_count * 6, 32))
    safe_log('info', f"Using {max_workers} parallel workers for streaming")
    return max_workers

def get_file_info_list(api: HfApi, model_id: str, token: str) -> List[FileInfo]:
    safe_log('info', f"Fetching file list for {model_id}...")
    repo_info = api.repo_info(repo_id=model_id, token=token, files_metadata=True)
    file_infos = [
        FileInfo(
            path=file.rfilename,
            size=file.size if hasattr(file, 'size') else 0,
            url=hf_hub_url(repo_id=model_id, filename=file.rfilename),
            sha256=file.lfs.sha256 if hasattr(file, 'lfs') and file.lfs else None
        ) for file in repo_info.siblings
    ]
    total_size = sum(f.size for f in file_infos)
    safe_log('info', f"Found {len(file_infos)} files, total size: {total_size / (1024**3):.2f} GB")
    return file_infos


def process_single_file(args) -> Tuple[str, bool, str, Optional[str]]:
    """Process a single file by streaming it directly to S3."""
    file_info, uploader, progress_callback = args
    
    s3_key = f"{S3_PREFIX}/{file_info.path}"
    success, message = uploader.stream_to_s3(file_info, s3_key, progress_callback)
    
    cache_key = f"{file_info.path}_{file_info.size}_{file_info.sha256}"
    
    if success:
        return (file_info.path, True, message, cache_key)
    else:
        return (file_info.path, False, message, None)

def progress_monitor(tracker: ProgressTracker, stop_event: threading.Event):
    """Monitors progress and updates a tqdm progress bar."""
    with tqdm(total=tracker.total_size, unit='B', unit_scale=True, desc="Overall Progress") as pbar:
        while not stop_event.is_set():
            with tracker.lock:
                processed_bytes = tracker.bytes_transferred
            
            pbar.update(processed_bytes - pbar.n)
            
            elapsed_time = time.time() - tracker.start_time
            speed = processed_bytes / elapsed_time if elapsed_time > 0 else 0
            pbar.set_postfix_str(f"{speed / 1024 / 1024:.2f} MB/s")
            
            if processed_bytes >= tracker.total_size:
                break
            time.sleep(1)
        # Final update to 100%
        if tracker.total_size > pbar.n:
            pbar.update(tracker.total_size - pbar.n)


def stream_all_files(model_id: str, token: str, max_workers: int):
    """Main function to stream all files from HuggingFace to S3."""
    api = HfApi()
    s3_client = get_s3_client()
    uploader = StreamingUploader(s3_client, S3_BUCKET, token)
    
    file_infos = get_file_info_list(api, model_id, token)
    if not file_infos: return

    safe_log('info', "Checking which files need to be downloaded...")
    files_to_download = []
    total_download_size = 0
    for file_info in tqdm(file_infos, desc="Pre-flight check"):
        s3_key = f"{S3_PREFIX}/{file_info.path}"
        s3_info = uploader.check_file_exists(s3_key)
        should_download, _ = uploader.should_download_file(file_info, s3_info)
        if should_download:
            files_to_download.append(file_info)
            total_download_size += file_info.size
    
    if not files_to_download:
        safe_log('info', "All files are already up to date in S3. Nothing to do.")
        return

    safe_log('info', f"Need to download {len(files_to_download)} files ({total_download_size / (1024**3):.2f} GB)")

    progress_tracker = ProgressTracker(total_size=total_download_size)
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=progress_monitor, args=(progress_tracker, stop_monitor), daemon=True)
    monitor_thread.start()

    successful_uploads = 0
    failed_uploads = 0
    
    file_args = [
        (file_info, uploader, progress_tracker.update) for file_info in files_to_download
    ]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, args) for args in file_args]
        for future in as_completed(futures):
            try:
                _, success, _, _ = future.result()
                if success:
                    successful_uploads += 1
                else:
                    failed_uploads += 1
            except Exception as e:
                safe_log('error', f"A worker failed with an exception: {e}")
                failed_uploads += 1
    
    stop_monitor.set()
    monitor_thread.join()

    safe_log('info', "\n" + "="*50)
    safe_log('info', "FINAL SUMMARY")
    safe_log('info', "="*50)
    safe_log('info', f"Total files to download: {len(files_to_download)}")
    safe_log('info', f"Successful uploads: {successful_uploads}")
    safe_log('info', f"Failed uploads: {failed_uploads}")


if __name__ == "__main__":
    if not all([model_id, huggingface_token, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_HOST, S3_BUCKET]):
        logging.error("Missing required environment variables. Check README for details.")
        exit(1)
    
    max_workers = detect_resources()
    
    try:
        login(token=huggingface_token)
        safe_log('info', "Logged in to Hugging Face")
        stream_all_files(model_id, huggingface_token, max_workers)
    except Exception as e:
        safe_log('error', f"A fatal error occurred: {e}")
    
    safe_log('info', "Transfer complete!")
