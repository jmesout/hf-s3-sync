import logging
import os
import time
import json
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError
import requests
from requests.exceptions import RequestException
from huggingface_hub import HfApi, login, hf_hub_url
import psutil
from tqdm import tqdm
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single InsecureRequestWarning from urllib3.
urllib3.disable_warnings(InsecureRequestWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
)
log_lock = threading.Lock()

def safe_log(level, message):
    with log_lock:
        getattr(logging, level)(message)

@dataclass
class ProgressTracker:
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
# This will be set from environment variable below
FORCE_REDOWNLOAD = None

# Environment variables
model_id = os.getenv("MODEL_ID")
huggingface_token = os.getenv("HF_TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_HOST = os.getenv("AWS_HOST")
S3_BUCKET = os.getenv("S3_BUCKET")
# Extract prefix from model_id - handle both org/model and simple model formats
if model_id and "/" in model_id:
    # Format: org/model-name -> extract model-name and take first 3 parts
    model_name = model_id.split("/")[1]
    S3_PREFIX = "-".join(model_name.split("-")[:3])
elif model_id:
    # Simple format without org
    S3_PREFIX = "-".join(model_id.split("-")[:3])
else:
    S3_PREFIX = ""
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# --- NEW FEATURE: Add environment variable to skip the existence check ---
SKIP_EXISTENCE_CHECK = os.getenv("SKIP_EXISTENCE_CHECK", "false").lower() == "true"
FORCE_REDOWNLOAD = os.getenv("FORCE_REDOWNLOAD", "false").lower() == "true"

# File patterns to skip for vLLM
SKIP_PATTERNS = [
    "*.h5", "*.msgpack", "flax_model*", "tf_model*",
    "*.onnx*", "*.gguf", "optimizer.pt", "scheduler.pt",
    "trainer_state.json", "training_args.bin", "README.md", ".gitattributes"
]

@dataclass
class FileInfo:
    path: str; size: int; url: str; sha256: Optional[str] = None

class StreamingUploader:
    def __init__(self, s3_client, bucket: str, token: str):
        self.s3_client = s3_client
        self.bucket = bucket
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def check_file_exists(self, s3_key: str):
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return response
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404' or error_code == 'NoSuchKey':
                return None
            # Log the actual error for debugging
            safe_log('debug', f"HeadObject error for {s3_key}: {error_code} - {e}")
            raise

    def should_download_file(self, file_info: FileInfo, s3_info) -> Tuple[bool, str]:
        if FORCE_REDOWNLOAD: return True, "force redownload enabled"
        if s3_info is None: return True, "file not found in S3"
        s3_size = s3_info.get('ContentLength', 0)
        if VERIFY_SIZE and s3_size != file_info.size:
            safe_log('warning', f"Size mismatch for {file_info.path}: S3={s3_size}, HF={file_info.size}")
            return True, f"size mismatch"
        if s3_size == file_info.size: return False, f"already exists with correct size"
        return True, "verification failed"

    def stream_to_s3(self, file_info: FileInfo, s3_key: str, progress_callback: callable) -> Tuple[bool, str]:
        for attempt in range(MAX_RETRIES):
            try:
                if file_info.size < MULTIPART_THRESHOLD:
                    return self._simple_upload(file_info, s3_key, progress_callback)
                else:
                    return self._multipart_upload_with_resume(file_info, s3_key, progress_callback)
            except (RequestException, BotoCoreError, ClientError) as e:
                safe_log('warning', f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {file_info.path}: {e}")
                if attempt + 1 == MAX_RETRIES:
                    return False, str(e)
                time.sleep(RETRY_DELAY * (attempt + 1))
        return False, "Exhausted all retries."

    def _simple_upload(self, file_info: FileInfo, s3_key: str, progress_callback: callable) -> Tuple[bool, str]:
        safe_log('info', f"Simple upload: {file_info.path}")
        response = requests.get(file_info.url, headers=self.headers, stream=True)
        response.raise_for_status()
        
        class ProgressCallbackWrapper:
            def __init__(self, file, callback): self._file = file; self._callback = callback
            def read(self, size=-1):
                data = self._file.read(size)
                if data: self._callback(len(data))
                return data
        
        body = ProgressCallbackWrapper(response.raw, progress_callback)
        self.s3_client.upload_fileobj(body, self.bucket, s3_key)
        safe_log('info', f"✓ Uploaded: {s3_key}")
        return True, "uploaded successfully"

    def _multipart_upload_with_resume(self, file_info: FileInfo, s3_key: str, progress_callback: callable) -> Tuple[bool, str]:
        safe_log('info', f"Multipart upload: {file_info.path}")
        mpu = self.s3_client.create_multipart_upload(Bucket=self.bucket, Key=s3_key)
        upload_id = mpu['UploadId']
        parts = []

        try:
            response = requests.get(file_info.url, headers=self.headers, stream=True)
            response.raise_for_status()
            
            buffer, part_number = BytesIO(), 1
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                if not chunk: continue
                progress_callback(len(chunk))
                buffer.write(chunk)
                if buffer.tell() >= MULTIPART_CHUNKSIZE:
                    buffer.seek(0)
                    part_data = buffer.read(MULTIPART_CHUNKSIZE)
                    part_resp = self.s3_client.upload_part(
                        Body=part_data, Bucket=self.bucket, Key=s3_key,
                        PartNumber=part_number, UploadId=upload_id)
                    parts.append({'ETag': part_resp['ETag'], 'PartNumber': part_number})
                    buffer = BytesIO(buffer.read())
                    part_number += 1
            
            if buffer.tell() > 0:
                buffer.seek(0)
                part_resp = self.s3_client.upload_part(
                    Body=buffer.read(), Bucket=self.bucket, Key=s3_key,
                    PartNumber=part_number, UploadId=upload_id)
                parts.append({'ETag': part_resp['ETag'], 'PartNumber': part_number})

            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket, Key=s3_key, UploadId=upload_id,
                MultipartUpload={'Parts': sorted(parts, key=lambda x: x['PartNumber'])}
            )
            safe_log('info', f"✓ Multipart upload complete: {s3_key}")
            return True, "uploaded successfully"
        except Exception as e:
            safe_log('warning', f"Aborting multipart upload for {s3_key} due to error: {e}")
            self.s3_client.abort_multipart_upload(Bucket=self.bucket, Key=s3_key, UploadId=upload_id)
            raise

def get_s3_client():
    # Get configuration from environment
    signature_version = os.getenv("S3_SIGNATURE_VERSION", "s3v4")
    verify_ssl = os.getenv("VERIFY_SSL", "true").lower() == "true"
    use_path_style = os.getenv("USE_PATH_STYLE", "true").lower() == "true"

    # Build endpoint URL
    endpoint_url = AWS_HOST
    if not AWS_HOST.startswith('http'):
        endpoint_url = f"https://{AWS_HOST}"

    safe_log('info', f"Connecting to S3-compatible storage at: {endpoint_url}")
    safe_log('info', f"Configuration: signature={signature_version}, path_style={use_path_style}, verify_ssl={verify_ssl}")

    # Configure boto3 for S3-compatible storage (Civo)
    # Path-style addressing is required for most S3-compatible services
    config = Config(
        region_name=AWS_REGION,
        signature_version=signature_version,
        retries={'max_attempts': 5, 'mode': 'adaptive'},
        max_pool_connections=50,
        s3={
            'addressing_style': 'path',  # Force path style addressing
            'payload_signing_enabled': True
        }
    )

    # Create S3 client
    client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=config,
        verify=verify_ssl,
        region_name=AWS_REGION
    )

    # Test connection - but don't fail if list_buckets doesn't work
    # Some S3-compatible services don't support list_buckets
    try:
        response = client.list_buckets()
        safe_log('info', f"Successfully connected to S3-compatible storage. Found {len(response.get('Buckets', []))} buckets")
        # Verify our target bucket exists
        bucket_names = [b['Name'] for b in response.get('Buckets', [])]
        if S3_BUCKET not in bucket_names:
            safe_log('warning', f"Target bucket '{S3_BUCKET}' not found in available buckets: {bucket_names}")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        if error_code == 'AccessDenied':
            safe_log('warning', f"Cannot list buckets (AccessDenied) - will attempt to use bucket '{S3_BUCKET}' directly")
        else:
            safe_log('warning', f"Connection test warning: {error_code} - {e}")
            # Don't raise - try to continue anyway
    except Exception as e:
        safe_log('warning', f"Connection test warning: {e} - continuing anyway")

    return client

def detect_resources():
    cpu_count = os.cpu_count() or 1
    max_workers = max(1, min(cpu_count * 6, 32))
    safe_log('info', f"Using {max_workers} parallel workers for streaming")
    return max_workers

def get_file_info_list(api: HfApi, model_id: str, token: str) -> List[FileInfo]:
    from fnmatch import fnmatch
    safe_log('info', f"Fetching file list for {model_id}...")
    repo_info = api.repo_info(repo_id=model_id, token=token, files_metadata=True)
    all_files = [
        FileInfo(
            path=file.rfilename, size=file.size or 0,
            url=hf_hub_url(repo_id=model_id, filename=file.rfilename),
            sha256=file.lfs.sha256 if file.lfs else None
        ) for file in repo_info.siblings
    ]
    files_to_process = [
        f for f in all_files if not any(fnmatch(f.path, pattern) for pattern in SKIP_PATTERNS)
    ]
    skipped_count = len(all_files) - len(files_to_process)
    safe_log('info', f"Found {len(all_files)} total files. Skipping {skipped_count} based on patterns.")
    total_size = sum(f.size for f in files_to_process)
    safe_log('info', f"Processing {len(files_to_process)} files, total size: {total_size / (1024**3):.2f} GB")
    return files_to_process

def process_single_file(args):
    file_info, uploader, progress_callback = args
    s3_key = f"{S3_PREFIX}/{file_info.path}"
    success, message = uploader.stream_to_s3(file_info, s3_key, progress_callback)
    return (file_info.path, success, message)

def progress_monitor(tracker: ProgressTracker, stop_event: threading.Event, files_to_process: int, completed_files: List):
    with tqdm(total=tracker.total_size, unit='B', unit_scale=True, desc="Overall Progress") as pbar:
        while not stop_event.is_set():
            with tracker.lock:
                processed_bytes = tracker.bytes_transferred
            
            pbar.update(processed_bytes - pbar.n)
            
            elapsed_time = time.time() - tracker.start_time
            speed = processed_bytes / elapsed_time if elapsed_time > 0 else 0
            pbar.set_description(f"Files: {len(completed_files)}/{files_to_process}")
            pbar.set_postfix_str(f"{speed / 1024 / 1024:.2f} MB/s")
            
            if processed_bytes >= tracker.total_size: break
            time.sleep(1)
        if tracker.total_size > pbar.n:
            pbar.update(tracker.total_size - pbar.n)

def stream_all_files(model_id: str, token: str, max_workers: int):
    api = HfApi()
    s3_client = get_s3_client()
    uploader = StreamingUploader(s3_client, S3_BUCKET, token)

    file_infos = get_file_info_list(api, model_id, token)
    if not file_infos:
        safe_log('warning', "No files found to process - exiting")
        return
    safe_log('info', f"Found {len(file_infos)} files to process after filtering")

    files_to_download = []
    total_download_size = 0
    
    # --- NEW FEATURE: Conditionally skip the pre-flight check ---
    if not SKIP_EXISTENCE_CHECK:
        safe_log('info', "Checking which files need to be downloaded...")
        for file_info in tqdm(file_infos, desc="Pre-flight check"):
            try:
                s3_key = f"{S3_PREFIX}/{file_info.path}"
                s3_info = uploader.check_file_exists(s3_key)
                should_download, reason = uploader.should_download_file(file_info, s3_info)
                if should_download:
                    files_to_download.append(file_info)
                    total_download_size += file_info.size
                    safe_log('debug', f"Will download {file_info.path}: {reason}")
            except (BotoCoreError, ClientError) as e:
                safe_log('warning', f"Pre-flight check failed for {file_info.path}: {e}. Assuming it needs download.")
                files_to_download.append(file_info)
                total_download_size += file_info.size
    else:
        safe_log('info', f"Skipping existence check (SKIP_EXISTENCE_CHECK={SKIP_EXISTENCE_CHECK}), assuming all files need to be downloaded...")
        files_to_download = file_infos
        total_download_size = sum(f.size for f in file_infos)
        safe_log('info', f"Will download all {len(files_to_download)} files")
    
    if not files_to_download:
        safe_log('info', "All files are already up to date in S3. Nothing to do.")
        return

    safe_log('info', f"Need to download {len(files_to_download)} files ({total_download_size / (1024**3):.2f} GB)")
    safe_log('info', f"Starting parallel download with {max_workers} workers...")

    progress_tracker = ProgressTracker(total_size=total_download_size)
    stop_monitor, completed_files, failed_files = threading.Event(), [], []
    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(progress_tracker, stop_monitor, len(files_to_download), completed_files),
        daemon=True
    )
    monitor_thread.start()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, (f, uploader, progress_tracker.update)): f
            for f in files_to_download
        }
        try:
            for future in as_completed(futures):
                file_path, success, message = future.result()
                if success:
                    completed_files.append(file_path)
                else:
                    failed_files.append((file_path, message))
        except KeyboardInterrupt:
            safe_log('warning', "Cancellation requested. Shutting down workers...")
            executor.shutdown(wait=False, cancel_futures=True)
    
    stop_monitor.set()
    monitor_thread.join()

    safe_log('info', "\n" + "="*50)
    safe_log('info', "FINAL SUMMARY")
    safe_log('info', "="*50)
    safe_log('info', f"Total files to download: {len(files_to_download)}")
    safe_log('info', f"Successful uploads: {len(completed_files)}")
    safe_log('info', f"Failed uploads: {len(failed_files)}")
    if failed_files:
        safe_log('error', "--- FAILED FILES ---")
        for path, msg in failed_files:
            safe_log('error', f"  - {path}: {msg}")

if __name__ == "__main__":
    # Log environment configuration
    safe_log('info', f"Environment flags: SKIP_EXISTENCE_CHECK={SKIP_EXISTENCE_CHECK}, FORCE_REDOWNLOAD={FORCE_REDOWNLOAD}")
    safe_log('info', f"Model: {model_id}, Bucket: {S3_BUCKET}, Prefix: {S3_PREFIX}")

    if not all([model_id, huggingface_token, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_HOST, S3_BUCKET]):
        logging.error("Missing required environment variables.")
        logging.error(f"model_id={model_id}, HF_TOKEN={'set' if huggingface_token else 'missing'}")
        logging.error(f"AWS_ACCESS_KEY_ID={'set' if AWS_ACCESS_KEY_ID else 'missing'}")
        logging.error(f"AWS_SECRET_ACCESS_KEY={'set' if AWS_SECRET_ACCESS_KEY else 'missing'}")
        logging.error(f"AWS_HOST={AWS_HOST}, S3_BUCKET={S3_BUCKET}")
        exit(1)

    max_workers = detect_resources()
    
    try:
        login(token=huggingface_token)
        safe_log('info', "Logged in to Hugging Face")
        stream_all_files(model_id, huggingface_token, max_workers)
    except KeyboardInterrupt:
        safe_log('info', "Process interrupted by user.")
    except Exception as e:
        safe_log('error', f"A fatal error occurred: {e}")
        import traceback
        safe_log('error', f"Traceback: {traceback.format_exc()}")
        exit(1)  # Exit with error code

    safe_log('info', "Transfer complete!")
