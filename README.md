# WildTrain Data Pipeline

A robust data pipeline for managing deep learning datasets in a unified master format and creating framework-specific formats using symlinks.

## 🎯 Overview

WildTrain provides a unified data management system that:

1. **Validates** input data formats (COCO, YOLO)
2. **Converts** to a master annotation format
3. **Stores** data efficiently with symlinks to avoid duplication
4. **Generates** framework-specific formats on-demand

## 📁 Project Structure

```
project_root/
├── data/
│   ├── images/                    # Master storage (real files)
│   │   ├── train/
│   │   │   ├── image001.jpg
│   │   │   └── image002.jpg
│   │   └── val/
│   │       ├── image003.jpg
│   │       └── image004.jpg
│   └── annotations/
│       └── master/
│           └── annotations.json
├── framework_configs/
│   ├── coco/
│   │   ├── data/
│   │   │   ├── train/            # Symlinks to master images
│   │   │   └── val/              # Symlinks to master images
│   │   └── annotations/
│   │       └── instances_train.json
│   └── yolo/
│       ├── images/
│       │   ├── train/            # Symlinks to master images
│       │   └── val/              # Symlinks to master images
│       └── labels/
│           ├── train/
│           └── val/
└── src/wildtrain/
    ├── adapters/                  # Format converters
    ├── converters/                # Input format converters
    ├── validators/                # Format validators
    └── pipeline/                  # Pipeline components
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd wildtrain

# Install dependencies
uv sync
```

### Basic Usage

```bash
# Import a COCO dataset
python main.py import /path/to/coco/dataset coco my_dataset

# Import a YOLO dataset
python main.py import /path/to/yolo/dataset yolo my_dataset

# List all datasets
python main.py list

# Get dataset information
python main.py info my_dataset

# Export to framework format
python main.py export my_dataset coco
python main.py export my_dataset yolo
```

## 📋 CLI Commands

### Import Dataset
```bash
python main.py import <source_path> <format_type> <dataset_name> [--no-hints]
```

- `source_path`: Path to the source dataset
- `format_type`: Either 'coco' or 'yolo'
- `dataset_name`: Name for the dataset in master storage
- `--no-hints`: Disable validation hints

### List Datasets
```bash
python main.py list
```

Lists all available datasets in master storage with statistics.

### Dataset Information
```bash
python main.py info <dataset_name>
```

Shows detailed information about a specific dataset.

### Export Framework Format
```bash
python main.py export <dataset_name> <framework>
```

- `dataset_name`: Name of the dataset
- `framework`: Target framework ('coco' or 'yolo')

### Delete Dataset
```bash
python main.py delete <dataset_name> [--force]
```

- `dataset_name`: Name of the dataset to delete
- `--force`: Force deletion without confirmation

## 🔧 Workflow

### 1. Data Validation
The pipeline first validates your input data:

- **COCO**: Validates JSON structure, required fields, and image file existence
- **YOLO**: Validates data.yaml structure, label files, and image paths

### 2. Master Format Conversion
Validated data is converted to the master annotation format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image001.jpg",
      "file_path": "/path/to/image001.jpg",
      "width": 640,
      "height": 480,
      "split": "train"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [100, 100, 200, 150],
      "type": "detection"
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person"
    }
  ]
}
```

### 3. Framework-Specific Generation
Using adapters, the pipeline generates framework-specific formats:

- **COCO**: Creates COCO JSON annotations and symlinks to images
- **YOLO**: Creates label files, data.yaml, and symlinks to images

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
uv run python -m pytest -v

# Run specific test files
uv run python -m pytest tests/test_validators.py -v
uv run python -m pytest tests/test_pipeline.py -v
```

## 📊 Supported Formats

### Input Formats
- **COCO**: Standard COCO JSON format with images and annotations
- **YOLO**: YOLO format with data.yaml and label files

### Output Formats
- **COCO**: Compatible with OpenMMLab and other COCO-based frameworks
- **YOLO**: Compatible with Ultralytics YOLO and other YOLO-based frameworks

## 🔍 Validation Features

### COCO Validation
- ✅ Required fields (images, annotations, categories)
- ✅ Image file existence
- ✅ Annotation consistency
- ✅ Category mapping

### YOLO Validation
- ✅ data.yaml structure
- ✅ Required fields (path, train, names)
- ✅ Label file existence
- ✅ Image file existence
- ✅ Path resolution

## 🛠️ Architecture

### Core Components

1. **DataPipeline**: Main orchestrator coordinating the workflow
2. **MasterDataManager**: Manages master data storage and operations
3. **FrameworkDataManager**: Creates framework-specific formats using symlinks
4. **Validators**: Validate input data formats
5. **Converters**: Convert input formats to master format
6. **Adapters**: Convert master format to framework-specific formats

### Key Features

- **Symlink-based storage**: Efficient storage using symlinks to avoid duplication
- **Validation with hints**: Helpful error messages and suggestions
- **Framework agnostic**: Master format works with any framework
- **Extensible**: Easy to add new formats and frameworks

## 📝 Examples

### Import COCO Dataset
```bash
python main.py import /path/to/coco_dataset coco my_coco_dataset
```

### Import YOLO Dataset
```bash
python main.py import /path/to/yolo_dataset yolo my_yolo_dataset
```

### Export for Training
```bash
# Export for OpenMMLab training
python main.py export my_dataset coco

# Export for YOLO training
python main.py export my_dataset yolo
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions, please open an issue on GitHub.
