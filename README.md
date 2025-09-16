# HuggingFace to S3 Sync - Optimized for Civo Object Store

A high-performance streaming tool for synchronizing HuggingFace model repositories to S3-compatible storage, specifically optimized for **Civo Object Store** and Ceph/RadosGW backends. Features auto-scaling based on system resources, intelligent file filtering for vLLM compatibility, and parallel transfer optimization.

## Features

- **Civo Object Store Optimized**: Built and tested specifically for Civo's S3-compatible object storage
- **Streaming Transfer**: Direct streaming from HuggingFace to S3 without intermediate storage
- **Auto-scaling**: Automatically detects and optimizes for available CPU and memory resources
- **Parallel Processing**: Multi-threaded uploads with configurable concurrency
- **vLLM Optimized**: Automatically filters unnecessary files for vLLM deployments
- **Ceph/RadosGW Compatible**: Hardcoded optimizations for Ceph S3 compatibility
- **Progress Tracking**: Real-time progress monitoring with speed metrics
- **Resilient**: Automatic retries, skip existing files, and detailed error reporting
- **Memory Efficient**: Uses temporary files for large uploads to prevent memory exhaustion

## Quick Start

### Prerequisites

- Python 3.11+
- HuggingFace account with access token
- Civo Object Store credentials (or any S3-compatible storage)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hf-s3-sync.git
cd hf-s3-sync

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage with Civo Object Store

```bash
# Set required environment variables for Civo
export MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_your_token_here"
export AWS_ACCESS_KEY_ID="your_civo_access_key"
export AWS_SECRET_ACCESS_KEY="your_civo_secret_key"
export AWS_HOST="objectstore.lon1.civo.com"  # Civo London region
export S3_BUCKET="your-bucket-name"
export AWS_REGION="lon1"  # Optional, defaults to us-east-1

# Run the sync
python hf_to_s3_streaming.py
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
  -e AWS_REGION="lon1" \
  hf-s3-sync
```

## Documentation

- [Configuration Guide](docs/CONFIGURATION.md) - Detailed environment variables and options
- [Usage Examples](docs/USAGE.md) - Common use cases and examples
- [Architecture](docs/ARCHITECTURE.md) - Technical details and design decisions
- [Docker Guide](docs/DOCKER.md) - Containerization and deployment
- [Troubleshooting](docs/TROUBLESHOOTING.md) - Common issues and solutions
- [Civo Setup Guide](docs/CIVO_SETUP.md) - Specific instructions for Civo Object Store

## Why Civo Object Store?

This tool was specifically built and optimized for Civo Object Store because:
- **Cost-effective**: Competitive pricing for ML model storage
- **S3-compatible**: Full S3 API compatibility
- **Performance**: Optimized for large file transfers
- **Regional presence**: Multiple regions for low-latency access
- **Ceph backend**: Enterprise-grade reliability with Ceph/RadosGW

## Key Features Explained

### Auto-scaling
The tool automatically detects system resources and adjusts:
- Worker threads based on CPU cores and available memory
- Chunk sizes for optimal memory usage
- Multipart upload thresholds
- Maximum concurrent connections

### File Filtering for vLLM
Automatically skips unnecessary files for vLLM deployments:
- TensorFlow checkpoints (`*.h5`, `tf_model*`)
- Flax models (`flax_model*`)
- ONNX models (`*.onnx*`)
- GGUF quantized models (`*.gguf`)
- Training artifacts (`optimizer.pt`, `scheduler.pt`)
- Documentation files

### S3 Key Structure
Files are organized with intelligent prefixing:
```
bucket/
├── model-name-prefix/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model.safetensors
│   └── ...
```

## Performance

- **Parallel transfers**: Up to 32 concurrent workers
- **Streaming**: No intermediate storage required
- **Memory optimization**: Automatic switching between memory and temp file storage
- **Bandwidth efficient**: Chunk sizes optimized for available memory
- **Skip existing**: Checks file existence and size before transfer

## System Requirements

### Minimum
- 2 CPU cores
- 4GB RAM
- Network bandwidth: 100 Mbps

### Recommended
- 8+ CPU cores
- 16GB+ RAM
- Network bandwidth: 1 Gbps+

## Supported Storage Backends

Primary support:
- **Civo Object Store** (primary target)
- Ceph/RadosGW

Also compatible with:
- AWS S3
- MinIO
- Any S3-compatible storage

## Civo Object Store Regions

Currently tested with:
- `objectstore.lon1.civo.com` (London)
- `objectstore.nyc1.civo.com` (New York)
- `objectstore.fra1.civo.com` (Frankfurt)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please read our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## Support

For issues and feature requests, please use the [GitHub Issues](https://github.com/yourusername/hf-s3-sync/issues) page.

