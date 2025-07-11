# WildTrain Data Pipeline

A comprehensive data pipeline for managing computer vision datasets with support for multiple annotation formats, data transformations, and DVC integration for data versioning.

## Features

### 🎯 Core Features
- **Multi-format Support**: Import from COCO and YOLO formats
- **Master Format**: Unified internal representation for all datasets
- **Data Validation**: Comprehensive validation for each format
- **Framework Export**: Export to COCO and YOLO formats for training
- **Data Transformations**: Augmentation and tiling capabilities

### 📦 DVC Integration
- **Data Versioning**: Git-like versioning for datasets
- **Remote Storage**: Support for S3, GCS, Azure, SSH, and HDFS
- **Pipeline Management**: Automated workflows with `dvc.yaml`
- **Experiment Tracking**: Track dataset versions and transformations
- **Collaboration**: Share datasets across teams

### 🔧 Data Transformations
- **Augmentation**: Rotation, brightness, contrast, noise
- **Tiling**: Image tiling with configurable parameters
- **Pipeline Support**: Chain multiple transformations

## Installation

### Basic Installation
```bash
# Install with pip
pip install wildtrain

# Or install with uv
uv add wildtrain
```

### With DVC Support
```bash
# Install with DVC support
pip install "wildtrain[dvc]"

# For cloud storage support
pip install "wildtrain[dvc]" "dvc[s3]"    # AWS S3
pip install "wildtrain[dvc]" "dvc[gcs]"   # Google Cloud Storage
pip install "wildtrain[dvc]" "dvc[azure]" # Azure Blob Storage
```

## Quick Start

### 1. Basic Usage

```bash
# Import a COCO dataset
wildtrain dataset import /path/to/annotations.json coco my_dataset

# Import a YOLO dataset
wildtrain dataset import /path/to/data.yaml yolo my_dataset

# List all datasets
wildtrain dataset list

# Export to framework format
wildtrain dataset export my_dataset coco
```

### 2. With Data Transformations

```bash
# Import with augmentation
wildtrain dataset import /path/to/data coco my_dataset --augment

# Import with tiling
wildtrain dataset import /path/to/data yolo my_dataset --tile

# Import with both transformations
wildtrain dataset import /path/to/data coco my_dataset --augment --tile
```

### 3. With DVC Integration

```bash
# Setup DVC remote storage
wildtrain dvc setup --storage-type local --storage-path ./dvc_storage

# Import with DVC tracking
wildtrain dataset import /path/to/data coco my_dataset --track-with-dvc

# Check DVC status
wildtrain dvc status

# Pull data from remote
wildtrain dvc pull

# Push data to remote
wildtrain dvc push
```

## CLI Commands

### Dataset Management
```bash
# Import dataset
wildtrain dataset import <source_path> <format> <dataset_name> [options]

# List datasets
wildtrain dataset list

# Get dataset info
wildtrain dataset info <dataset_name>

# Export dataset
wildtrain dataset export <dataset_name> <format>

# Delete dataset
wildtrain dataset delete <dataset_name>
```

### DVC Operations
```bash
# Setup DVC
wildtrain dvc setup [options]

# Check status
wildtrain dvc status

# Pull data
wildtrain dvc pull [dataset_name]

# Push data
wildtrain dvc push

# Create pipeline
wildtrain dvc pipeline <pipeline_name> [options]

# Run pipeline
wildtrain dvc run <pipeline_name>
```

### Data Validation
```bash
# Validate dataset
wildtrain validate <source_path> <format>
```

## Python API

### Basic Usage
```python
from wildtrain.pipeline.data_pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline("data")

# Import dataset
result = pipeline.import_dataset(
    source_path="/path/to/data",
    source_format="coco",
    dataset_name="my_dataset"
)

# List datasets
datasets = pipeline.list_datasets()
```

### With DVC Integration
```python
from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.pipeline.dvc_manager import DVCConfig, DVCStorageType

# Configure DVC
config = DVCConfig(
    storage_type=DVCStorageType.S3,
    storage_path="s3://my-bucket/datasets"
)

# Initialize pipeline with DVC
pipeline = DataPipeline("data", enable_dvc=True, dvc_config=config)

# Import with DVC tracking
result = pipeline.import_dataset(
    source_path="/path/to/data",
    source_format="coco",
    dataset_name="my_dataset",
    track_with_dvc=True
)
```

## Configuration

### DVC Storage Types

#### Local Storage
```bash
wildtrain dvc setup --storage-type local --storage-path ./dvc_storage
```

#### AWS S3
```bash
wildtrain dvc setup --storage-type s3 --storage-path s3://my-bucket/datasets
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

#### Google Cloud Storage
```bash
wildtrain dvc setup --storage-type gcs --storage-path gs://my-bucket/datasets
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

#### Azure Blob Storage
```bash
wildtrain dvc setup --storage-type azure --storage-path azure://my-container/datasets
export AZURE_STORAGE_CONNECTION_STRING=your_connection_string
```

## Data Transformations

### Augmentation
```bash
# Basic augmentation
wildtrain dataset import data coco dataset --augment

# Custom augmentation parameters
wildtrain dataset import data coco dataset --augment \
  --rotation -15 15 \
  --probability 0.7 \
  --brightness 0.8 1.2 \
  --contrast 0.9 1.1 \
  --noise 0.02
```

### Tiling
```bash
# Basic tiling
wildtrain dataset import data coco dataset --tile

# Custom tiling parameters
wildtrain dataset import data coco dataset --tile \
  --tile-size 512 \
  --stride 256 \
  --min-visibility 0.1 \
  --max-negative-tiles 3
```

## Project Structure

```
project/
├── data/                    # Master data storage
│   ├── images/             # Image files
│   └── annotations/        # Master annotations
├── .dvc/                   # DVC configuration
├── dvc.yaml               # Pipeline definitions
└── .gitignore             # Git ignore rules
```

## Examples

### Complete Workflow Example
```bash
# 1. Setup DVC
wildtrain dvc setup --storage-type local

# 2. Import dataset with transformations
wildtrain dataset import /path/to/raw_data coco my_dataset \
  --augment \
  --tile \
  --track-with-dvc

# 3. Export for training
wildtrain dataset export my_dataset yolo

# 4. Check status
wildtrain dvc status
wildtrain dataset list
```

### Pipeline Example
```yaml
# dvc.yaml
stages:
  import:
    cmd: wildtrain dataset import data/raw coco raw_dataset --track-with-dvc
    deps:
      - data/raw
    outs:
      - data/processed

  augment:
    cmd: wildtrain dataset transform raw_dataset --augment --output-name augmented_dataset
    deps:
      - data/processed
    outs:
      - data/augmented

  export:
    cmd: wildtrain dataset export augmented_dataset yolo
    deps:
      - data/augmented
    outs:
      - data/exports
```

## Testing

```bash
# Run all tests
uv run python -m pytest -v

# Run specific test file
uv run python -m pytest tests/test_dvc_integration.py -v

# Run with coverage
uv run python -m pytest --cov=wildtrain tests/
```

## Documentation

- [DVC Integration Guide](docs/DVC_INTEGRATION.md)
- [Transformation Documentation](docs/TRANSFORMATIONS.md)
- [API Reference](docs/API.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DVC](https://dvc.org/) for data versioning capabilities
- [COCO](https://cocodataset.org/) for the annotation format
- [YOLO](https://github.com/ultralytics/yolov5) for the annotation format
- [Albumentations](https://albumentations.ai/) for data augmentation
