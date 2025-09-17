# HuggingFace to S3 Sync - Optimized for Civo Object Store

A high-performance Go application for synchronizing HuggingFace model repositories to S3-compatible storage, specifically optimized for **Civo Object Store** and Ceph/RadosGW backends. Features concurrent downloads, intelligent file management, and seamless integration with Kubernetes workloads.

## Features

- **Civo Object Store Optimized**: Built and tested specifically for Civo's S3-compatible object storage
- **Go-powered Performance**: Native Go implementation for high performance and low resource usage
- **Dual Storage Modes**: Support for both local storage and direct S3 streaming
- **Concurrent Processing**: Multi-threaded downloads with configurable concurrency
- **Smart File Management**: Skip existing files, verify checksums, and organize with intelligent folder structures
- **Kubernetes Ready**: Built-in support for containerized deployments and K8s jobs
- **Progress Tracking**: Real-time progress monitoring with detailed logging
- **Resilient**: Automatic retries with exponential backoff and comprehensive error handling
- **Memory Efficient**: Configurable multipart uploads and smart memory management

## Quick Start

### Prerequisites

- Go 1.25+ (for building from source)
- HuggingFace account with access token  
- Civo Object Store credentials (or any S3-compatible storage)
- Docker (for containerized deployment)

### Installation

#### Option 1: Build from Source
```bash
# Clone the repository
git clone https://github.com/jmesout/hf-s3-sync.git
cd hf-s3-sync

# Build the binary
go build -o hfsyncs3 main.go
```

#### Option 2: Using Docker
```bash
# Build the Docker image
docker build -t hf-s3-sync .
```

### Basic Usage with Civo Object Store

#### Local Binary
```bash
# Set required environment variables for Civo
export MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export AWS_ACCESS_KEY_ID="your_civo_access_key"
export AWS_SECRET_ACCESS_KEY="your_civo_secret_key"
export AWS_HOST="objectstore.lon1.civo.com"  # Civo London region
export S3_BUCKET="your-bucket-name"
export USE_LOCAL_STORAGE="false"  # Set to "true" for local storage mode

# Run the sync
./hfsyncs3
```

## Docker Usage

```bash
# Build the image
docker build -t hf-s3-sync .

# Run with Civo Object Store
docker run --rm \
  -e MODEL_ID="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="hf_your_token" \
  -e AWS_ACCESS_KEY_ID="your_civo_key" \
  -e AWS_SECRET_ACCESS_KEY="your_civo_secret" \
  -e AWS_HOST="objectstore.lon1.civo.com" \
  -e S3_BUCKET="your-bucket" \
  -e USE_LOCAL_STORAGE="false" \
  hf-s3-sync
```

## Kubernetes Deployment

The application is designed to run efficiently as Kubernetes Jobs. Here's a complete example using the provided job configuration:

### Kubernetes Job Example

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: hf-s3-sync-openai-gpt-oss-120b
  namespace: default
spec:
  backoffLimit: 2
  completionMode: NonIndexed
  completions: 1
  parallelism: 1
  template:
    spec:
      containers:
      - name: hf-s3-streamer
        image: ttl.sh/hftos3:1h  # Replace with your image
        imagePullPolicy: Always
        env:
        - name: MODEL_ID
          value: "openai/gpt-oss-120b"
        - name: S3_BUCKET
          value: "model-cache"
        - name: AWS_HOST
          value: "objectstore.lon1.civo.com"
        - name: USE_LOCAL_STORAGE
          value: "false"
        envFrom:
        - secretRef:
            name: hf-token-secret
        - secretRef:
            name: s3-credentials
        resources:
          limits:
            cpu: "4"
            memory: 8Gi
          requests:
            cpu: "2"
            memory: 4Gi
      restartPolicy: OnFailure
```

### Required Kubernetes Secrets

Create the necessary secrets for HuggingFace and S3 credentials:

```bash
# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="hf_your_token_here"

# Create S3 credentials secret  
kubectl create secret generic s3-credentials \
  --from-literal=AWS_ACCESS_KEY_ID="your_civo_access_key" \
  --from-literal=AWS_SECRET_ACCESS_KEY="your_civo_secret_key"
```

### Deploy the Job

```bash
# Apply the job configuration
kubectl apply -f k8s/job.yaml

# Monitor the job progress
kubectl logs -f job/hf-s3-sync-openai-gpt-oss-120b

# Check job status
kubectl get jobs
```

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `MODEL_ID` | HuggingFace model identifier (e.g., `meta-llama/Llama-3.1-8B-Instruct`) | Yes | - |
| `HF_TOKEN` | HuggingFace access token | Yes | - |
| `AWS_ACCESS_KEY_ID` | S3 access key | Yes | - |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key | Yes | - |
| `AWS_HOST` | S3 endpoint (e.g., `objectstore.lon1.civo.com`) | Yes | - |
| `S3_BUCKET` | Target S3 bucket name | Yes | - |
| `USE_LOCAL_STORAGE` | Storage mode: `"true"` for local then upload, `"false"` for direct streaming | No | `"false"` |

### Application Settings

The application includes several built-in optimizations:

- **Concurrency**: 8 concurrent downloads
- **Max Active Downloads**: 3 simultaneous file transfers  
- **Multipart Threshold**: 256MiB for large file uploads
- **Verification**: SHA256 checksums for data integrity
- **Retries**: 4 attempts with exponential backoff (400ms to 10s)
- **File Organization**: Automatic organization by model organization name

## Storage Modes

### Direct Streaming Mode (`USE_LOCAL_STORAGE="false"`)
- **Default mode**: Files are downloaded and immediately uploaded to S3
- **Memory efficient**: No local storage required
- **Ideal for**: Kubernetes jobs, limited storage environments
- **Process**: Download → Upload → Delete local copy

### Local Storage Mode (`USE_LOCAL_STORAGE="true"`)  
- **Batch mode**: All files downloaded locally first, then batch uploaded
- **Higher storage requirements**: Needs space for complete model
- **Ideal for**: Development, backup scenarios
- **Process**: Download all → Upload all → Keep local copies

## Key Features Explained

### Intelligent File Organization
Files are automatically organized in S3 using the model's organization structure:
```
bucket/
├── meta-llama/           # Organization name extracted from model ID
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── tokenizer_config.json
├── openai/
│   ├── config.json
│   └── pytorch_model.bin
```

### Smart Upload Management
- **Duplicate Detection**: Skips files that already exist with the same size
- **Resume Capability**: Can resume interrupted transfers
- **Size Verification**: Compares local and remote file sizes before upload
- **Concurrent Uploads**: Uses goroutines for non-blocking uploads in streaming mode

### Error Handling & Resilience
- **Retry Logic**: Exponential backoff for failed downloads (400ms to 10s)
- **Comprehensive Logging**: Detailed progress and error reporting
- **Graceful Failures**: Continues processing other files when individual files fail
- **Memory Safety**: Automatic cleanup of temporary files

## Why Civo Object Store?

This tool was specifically built and optimized for Civo Object Store because:
- **Cost-effective**: Competitive pricing for ML model storage
- **S3-compatible**: Full S3 API compatibility with MinIO client
- **Performance**: Optimized for large file transfers
- **Regional presence**: Multiple regions for low-latency access
- **Ceph backend**: Enterprise-grade reliability with Ceph/RadosGW

## Performance

- **Concurrent Downloads**: 8 parallel downloads with 3 max active transfers
- **Streaming Mode**: Direct S3 upload without intermediate storage
- **Multipart Uploads**: 256MiB threshold for large files
- **Memory Efficient**: Go's native memory management and garbage collection
- **Skip Existing**: Intelligent file existence and size checking
- **Verification**: SHA256 checksum validation for data integrity

## System Requirements

### Minimum (Container)
- 1 CPU core
- 2GB RAM  
- Network bandwidth: 50 Mbps

### Recommended (Kubernetes Job)
- 2-4 CPU cores
- 4-8GB RAM
- Network bandwidth: 500 Mbps+

### High-Performance Setup
- 4+ CPU cores
- 8GB+ RAM
- Network bandwidth: 1 Gbps+

## Supported Storage Backends

Primary support:
- **Civo Object Store** (primary target)
- Ceph/RadosGW

Also compatible with:
- AWS S3
- MinIO
- DigitalOcean Spaces
- Any S3-compatible storage

## Civo Object Store Regions

Currently tested with:
- `objectstore.lon1.civo.com` (London)
- `objectstore.nyc1.civo.com` (New York)  
- `objectstore.fra1.civo.com` (Frankfurt)

## Example Use Cases

### Development & Testing
```bash
# Quick model sync for development
export MODEL_ID="microsoft/DialoGPT-medium"
export USE_LOCAL_STORAGE="true"
./hfsyncs3
```

### Production Kubernetes Job
```bash
# Deploy large model sync job
export MODEL_ID="meta-llama/Llama-3.1-70B-Instruct"
kubectl apply -f k8s/job.yaml
```

### Batch Processing Multiple Models
```bash
# Process multiple models with different jobs
for model in "microsoft/DialoGPT-small" "microsoft/DialoGPT-medium" "microsoft/DialoGPT-large"; do
  sed "s/MODEL_ID_PLACEHOLDER/$model/g" k8s/job-template.yaml | kubectl apply -f -
done
```

## Troubleshooting

### Common Issues

**Permission Denied**
```bash
# Ensure S3 credentials have proper permissions
aws s3 ls s3://your-bucket --endpoint-url=https://objectstore.lon1.civo.com
```

**Out of Memory**  
```bash
# Reduce concurrency or use local storage mode
export USE_LOCAL_STORAGE="true"
```

**Network Timeouts**
```bash
# Check connectivity to HuggingFace and S3 endpoint
curl -I https://huggingface.co
curl -I https://objectstore.lon1.civo.com
```

### Monitoring Kubernetes Jobs

```bash
# Watch job progress
kubectl get jobs -w

# View detailed logs
kubectl logs -f job/hf-s3-sync-model-name

# Check resource usage
kubectl top pods -l job-name=hf-s3-sync-model-name
```

## Architecture

The application is built with:
- **Go 1.25**: High-performance, concurrent runtime
- **HuggingFace Downloader**: Efficient model downloading with retries
- **MinIO Client**: Robust S3-compatible storage interface
- **Docker**: Containerized deployment
- **Kubernetes**: Scalable job orchestration

## Contributing

Contributions are welcome! Please ensure:
1. Go code follows `gofmt` standards
2. Tests pass with `go test ./...`
3. Docker builds successfully
4. Kubernetes manifests are valid

## License

MIT License - See LICENSE file for details

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/jmesout/hf-s3-sync/issues) page.

