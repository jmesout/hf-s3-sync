#!/usr/bin/env python3
"""
HuggingFace to S3 Streaming v2
Optimized for Ceph/Rados Gateway with techniques from s3-benchmark
Auto-scales based on available memory and CPU
Uses temporary storage instead of persistent volumes
"""

import logging
import os
import sys
import time
import tempfile
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError
import requests
from requests.exceptions import RequestException
from huggingface_hub import HfApi, login, hf_hub_url
import psutil
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
log_lock = threading.Lock()

def safe_log(level: str, message: str):
    """Thread-safe logging"""
    with log_lock:
        getattr(logger, level)(message)

@dataclass
class SystemResources:
    """System resource information for auto-scaling"""
    cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    recommended_workers: int
    chunk_size_mb: int
    max_concurrency: int
    multipart_threshold_mb: int

@dataclass
class ProgressTracker:
    """Track download/upload progress"""
    total_size: int
    bytes_transferred: int = 0
    lock: threading.Lock = None
    start_time: float = None

    def __post_init__(self):
        if self.lock is None:
            self.lock = threading.Lock()
        if self.start_time is None:
            self.start_time = time.time()

    def update(self, chunk_size: int):
        with self.lock:
            self.bytes_transferred += chunk_size

    def get_speed(self) -> float:
        with self.lock:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                return self.bytes_transferred / elapsed
            return 0

@dataclass
class FileInfo:
    """Information about a file to transfer"""
    path: str
    size: int
    url: str
    sha256: Optional[str] = None

# Hardcoded optimizations for Ceph/Rados Gateway
CEPH_CONFIG = {
    'signature_version': 's3v4',
    'addressing_style': 'path',
    'payload_signing_enabled': True,
    'max_pool_connections': 100,
    'retries': {'max_attempts': 3, 'mode': 'adaptive'},
}

# File patterns to skip for vLLM
SKIP_PATTERNS = [
    "*.h5", "*.msgpack", "flax_model*", "tf_model*",
    "*.onnx*", "*.gguf", "optimizer.pt", "scheduler.pt",
    "trainer_state.json", "training_args.bin", "README.md", ".gitattributes"
]

def detect_system_resources() -> SystemResources:
    """
    Detect system resources and calculate optimal settings.
    Uses techniques from s3-benchmark for auto-scaling.
    """
    cpu_cores = os.cpu_count() or 1
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024**3)
    available_memory_gb = memory.available / (1024**3)

    # Calculate workers based on CPU and memory
    # Reserve 20% memory for system
    usable_memory_gb = available_memory_gb * 0.8

    # Each worker should have at least 512MB
    max_workers_by_memory = int(usable_memory_gb * 1024 / 512)

    # Use 2-4 workers per CPU core for I/O bound tasks
    max_workers_by_cpu = cpu_cores * 3

    # Take the minimum and cap at 32
    recommended_workers = min(max_workers_by_memory, max_workers_by_cpu, 32)
    recommended_workers = max(1, recommended_workers)

    # Calculate chunk size based on available memory per worker
    # Each worker gets equal share of usable memory
    memory_per_worker_mb = int((usable_memory_gb * 1024) / recommended_workers)

    # Chunk size should be between 8MB and 128MB
    chunk_size_mb = min(128, max(8, memory_per_worker_mb // 4))

    # Multipart threshold at 2x chunk size
    multipart_threshold_mb = chunk_size_mb * 2

    # Max concurrency for transfers (similar to workers but can be higher for small files)
    max_concurrency = min(recommended_workers * 2, 50)

    safe_log('info', f"System Resources Detected:")
    safe_log('info', f"  CPU Cores: {cpu_cores}")
    safe_log('info', f"  Total Memory: {total_memory_gb:.2f} GB")
    safe_log('info', f"  Available Memory: {available_memory_gb:.2f} GB")
    safe_log('info', f"  Recommended Workers: {recommended_workers}")
    safe_log('info', f"  Chunk Size: {chunk_size_mb} MB")
    safe_log('info', f"  Multipart Threshold: {multipart_threshold_mb} MB")
    safe_log('info', f"  Max Concurrency: {max_concurrency}")

    return SystemResources(
        cpu_cores=cpu_cores,
        total_memory_gb=total_memory_gb,
        available_memory_gb=available_memory_gb,
        recommended_workers=recommended_workers,
        chunk_size_mb=chunk_size_mb,
        max_concurrency=max_concurrency,
        multipart_threshold_mb=multipart_threshold_mb
    )

def get_transfer_config(resources: SystemResources) -> TransferConfig:
    """
    Create optimized TransferConfig based on system resources.
    Uses techniques from s3-benchmark.
    """
    return TransferConfig(
        multipart_threshold=resources.multipart_threshold_mb * 1024 * 1024,
        multipart_chunksize=resources.chunk_size_mb * 1024 * 1024,
        max_concurrency=resources.max_concurrency,
        use_threads=True,
        max_io_queue=resources.max_concurrency * 2,
    )

def get_s3_client():
    """
    Create S3 client optimized for Ceph/Rados Gateway.
    Configuration hardcoded for Ceph compatibility.
    """
    # Mandatory environment variables
    aws_access_key = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
    aws_host = os.environ['AWS_HOST']

    # Build endpoint URL
    endpoint_url = aws_host
    if not aws_host.startswith('http'):
        endpoint_url = f"https://{aws_host}"

    safe_log('info', f"Connecting to S3-compatible storage at: {endpoint_url}")

    # Get region from environment or default to us-east-1
    region = os.getenv('AWS_REGION', 'us-east-1')

    # Hardcoded Ceph-optimized configuration
    config = Config(
        region_name=region,
        signature_version=CEPH_CONFIG['signature_version'],
        retries=CEPH_CONFIG['retries'],
        max_pool_connections=CEPH_CONFIG['max_pool_connections'],
        s3={
            'addressing_style': CEPH_CONFIG['addressing_style'],
            'payload_signing_enabled': CEPH_CONFIG['payload_signing_enabled']
        }
    )

    # Create S3 client
    client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        config=config,
        verify=False,  # Ceph often uses self-signed certs
        region_name=region
    )

    # Quick connectivity test
    bucket = os.environ['S3_BUCKET']
    try:
        client.head_bucket(Bucket=bucket)
        safe_log('info', f"Successfully connected to bucket: {bucket}")
    except ClientError as e:
        # Ceph might not allow HeadBucket, try a simple list instead
        try:
            client.list_objects_v2(Bucket=bucket, MaxKeys=1)
            safe_log('info', f"Successfully connected to bucket: {bucket}")
        except:
            safe_log('warning', f"Could not verify bucket access: {e}")

    return client

class OptimizedUploader:
    """
    Optimized uploader using techniques from s3-benchmark.
    Uses TransferConfig and temporary storage.
    """

    def __init__(self, s3_client, bucket: str, prefix: str,
                 transfer_config: TransferConfig, hf_token: str):
        self.s3_client = s3_client
        self.bucket = bucket
        self.prefix = prefix
        self.transfer_config = transfer_config
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

    def check_file_exists(self, s3_key: str) -> Optional[Dict]:
        """Check if file exists in S3"""
        try:
            response = self.s3_client.head_object(Bucket=self.bucket, Key=s3_key)
            return response
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            # Civo returns 403 for non-existent objects
            if error_code in ['404', 'NoSuchKey', 'AccessDenied', 'Forbidden', '403']:
                return None
            safe_log('warning', f"Unexpected error checking {s3_key}: {error_code}")
            raise

    def upload_file_streaming(self, file_info: FileInfo, s3_key: str,
                             progress_callback: callable) -> Tuple[bool, str]:
        """
        Stream upload with temporary storage and memory optimization.
        Uses techniques from s3-benchmark for efficient transfers.
        """
        # Check if file already exists with correct size
        s3_info = self.check_file_exists(s3_key)
        if s3_info and s3_info.get('ContentLength') == file_info.size:
            safe_log('info', f"Skipping {file_info.path}: already exists with correct size")
            return True, "already exists"

        safe_log('info', f"Uploading {file_info.path} ({file_info.size / (1024**2):.2f} MB)")

        # Use temporary file for large files to avoid memory issues
        use_temp_file = file_info.size > (self.transfer_config.multipart_threshold * 2)

        try:
            # Stream download from HuggingFace
            response = requests.get(file_info.url, headers=self.headers, stream=True)
            response.raise_for_status()

            if use_temp_file:
                # Use temporary file for large files
                with tempfile.NamedTemporaryFile(delete=True) as temp_file:
                    # Stream to temporary file
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            temp_file.write(chunk)
                            progress_callback(len(chunk))

                    temp_file.flush()
                    temp_file.seek(0)

                    # Upload from temporary file using upload_file
                    self.s3_client.upload_file(
                        temp_file.name,
                        self.bucket,
                        s3_key,
                        Config=self.transfer_config,
                        Callback=progress_callback
                    )
            else:
                # For smaller files, use memory buffer
                buffer = BytesIO()
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        buffer.write(chunk)
                        progress_callback(len(chunk))

                buffer.seek(0)

                # Upload from memory using put_object for small files
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Body=buffer.getvalue()
                )

            safe_log('info', f"✓ Uploaded: {s3_key}")
            return True, "uploaded successfully"

        except Exception as e:
            safe_log('error', f"Failed to upload {file_info.path}: {e}")
            return False, str(e)

def get_file_list(model_id: str, token: str) -> List[FileInfo]:
    """Get list of files to transfer from HuggingFace"""
    from fnmatch import fnmatch

    safe_log('info', f"Fetching file list for {model_id}...")
    api = HfApi()

    try:
        repo_info = api.repo_info(repo_id=model_id, token=token, files_metadata=True)
    except Exception as e:
        safe_log('error', f"Failed to get repo info: {e}")
        raise

    all_files = [
        FileInfo(
            path=file.rfilename,
            size=file.size or 0,
            url=hf_hub_url(repo_id=model_id, filename=file.rfilename),
            sha256=file.lfs.sha256 if file.lfs else None
        ) for file in repo_info.siblings
    ]

    # Filter out unwanted files
    files_to_process = [
        f for f in all_files
        if not any(fnmatch(f.path, pattern) for pattern in SKIP_PATTERNS)
    ]

    skipped = len(all_files) - len(files_to_process)
    total_size = sum(f.size for f in files_to_process)

    safe_log('info', f"Found {len(all_files)} files, skipping {skipped}")
    safe_log('info', f"Will process {len(files_to_process)} files ({total_size / (1024**3):.2f} GB)")

    return files_to_process

def process_file(args) -> Tuple[str, bool, str]:
    """Process a single file upload"""
    file_info, uploader, progress_callback = args
    model_id = os.environ['MODEL_ID']

    # Extract prefix from model_id
    if "/" in model_id:
        model_name = model_id.split("/")[1]
        prefix = "-".join(model_name.split("-")[:3])
    else:
        prefix = "-".join(model_id.split("-")[:3])

    s3_key = f"{prefix}/{file_info.path}"
    success, message = uploader.upload_file_streaming(file_info, s3_key, progress_callback)
    return file_info.path, success, message

def progress_monitor(tracker: ProgressTracker, stop_event: threading.Event,
                    total_files: int, completed_files: List):
    """Monitor and display progress"""
    with tqdm(total=tracker.total_size, unit='B', unit_scale=True,
              desc="Overall Progress") as pbar:
        last_update = 0
        while not stop_event.is_set():
            with tracker.lock:
                current = tracker.bytes_transferred

            if current > last_update:
                pbar.update(current - last_update)
                last_update = current

            speed = tracker.get_speed()
            completed = len(completed_files)
            pbar.set_description(f"Files: {completed}/{total_files}")
            pbar.set_postfix_str(f"{speed / (1024**2):.2f} MB/s")

            if current >= tracker.total_size:
                break
            time.sleep(0.5)

        # Final update
        if tracker.total_size > last_update:
            pbar.update(tracker.total_size - last_update)

def main():
    """Main execution function"""
    # Check mandatory environment variables
    required_vars = ['MODEL_ID', 'HF_TOKEN', 'AWS_ACCESS_KEY_ID',
                     'AWS_SECRET_ACCESS_KEY', 'AWS_HOST', 'S3_BUCKET']

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        safe_log('error', f"Missing required environment variables: {missing}")
        sys.exit(1)

    model_id = os.environ['MODEL_ID']
    hf_token = os.environ['HF_TOKEN']
    bucket = os.environ['S3_BUCKET']

    # Extract prefix from model_id
    if "/" in model_id:
        model_name = model_id.split("/")[1]
        prefix = "-".join(model_name.split("-")[:3])
    else:
        prefix = "-".join(model_id.split("-")[:3])

    safe_log('info', f"Model: {model_id}")
    safe_log('info', f"Bucket: {bucket}")
    safe_log('info', f"Prefix: {prefix}")

    # Detect system resources
    resources = detect_system_resources()
    transfer_config = get_transfer_config(resources)

    # Login to HuggingFace
    try:
        login(token=hf_token)
        safe_log('info', "Logged in to HuggingFace")
    except Exception as e:
        safe_log('error', f"Failed to login to HuggingFace: {e}")
        sys.exit(1)

    # Get S3 client
    try:
        s3_client = get_s3_client()
    except Exception as e:
        safe_log('error', f"Failed to create S3 client: {e}")
        sys.exit(1)

    # Get file list
    try:
        files = get_file_list(model_id, hf_token)
    except Exception as e:
        safe_log('error', f"Failed to get file list: {e}")
        sys.exit(1)

    if not files:
        safe_log('info', "No files to process")
        return

    # Create uploader
    uploader = OptimizedUploader(s3_client, bucket, prefix, transfer_config, hf_token)

    # Calculate total size
    total_size = sum(f.size for f in files)
    tracker = ProgressTracker(total_size=total_size)

    # Start progress monitor
    stop_event = threading.Event()
    completed_files = []
    failed_files = []

    monitor_thread = threading.Thread(
        target=progress_monitor,
        args=(tracker, stop_event, len(files), completed_files),
        daemon=True
    )
    monitor_thread.start()

    # Process files in parallel
    with ThreadPoolExecutor(max_workers=resources.recommended_workers) as executor:
        futures = {
            executor.submit(process_file, (f, uploader, tracker.update)): f
            for f in files
        }

        try:
            for future in as_completed(futures):
                file_path, success, message = future.result()
                if success:
                    completed_files.append(file_path)
                else:
                    failed_files.append((file_path, message))
        except KeyboardInterrupt:
            safe_log('warning', "Interrupted by user, shutting down...")
            executor.shutdown(wait=False, cancel_futures=True)
            stop_event.set()
            sys.exit(1)

    # Stop progress monitor
    stop_event.set()
    monitor_thread.join(timeout=2)

    # Print summary
    safe_log('info', "\n" + "="*50)
    safe_log('info', "TRANSFER SUMMARY")
    safe_log('info', "="*50)
    safe_log('info', f"Total files: {len(files)}")
    safe_log('info', f"Successful: {len(completed_files)}")
    safe_log('info', f"Failed: {len(failed_files)}")

    if failed_files:
        safe_log('error', "Failed files:")
        for path, error in failed_files[:10]:  # Show first 10 failures
            safe_log('error', f"  - {path}: {error}")
        if len(failed_files) > 10:
            safe_log('error', f"  ... and {len(failed_files) - 10} more")

    # Calculate final speed
    elapsed = time.time() - tracker.start_time
    if elapsed > 0:
        avg_speed = tracker.bytes_transferred / elapsed
        safe_log('info', f"Average speed: {avg_speed / (1024**2):.2f} MB/s")
        safe_log('info', f"Total time: {elapsed:.2f} seconds")

    safe_log('info', "Transfer complete!")

if __name__ == "__main__":
    main()