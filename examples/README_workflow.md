# WildTrain Data Pipeline Workflow Example

This example demonstrates how to use the WildTrain data pipeline with both synthetic and real data.

## Features

- **Synthetic Data Processing**: Creates and processes synthetic COCO format data for testing
- **Real Data Processing**: Processes real COCO and YOLO datasets
- **Multiple Format Support**: Handles both COCO and YOLO input/output formats
- **Dataset Management**: Lists and provides information about processed datasets
- **Error Handling**: Robust error handling with detailed error messages
- **Flexible Configuration**: Command-line arguments for different use cases

## Usage

### Basic Usage (Synthetic Data)

```bash
# Run with synthetic data (default)
uv run python examples/workflow_example.py
```

### Real Data Usage

```bash
# Run with real data
uv run python examples/workflow_example.py --use-real-data
```

### Advanced Options

```bash
# Use a specific data directory
uv run python examples/workflow_example.py --data-dir /path/to/data

# Keep temporary files
uv run python examples/workflow_example.py --keep-temp

# Combine options
uv run python examples/workflow_example.py --use-real-data --data-dir /path/to/data --keep-temp
```

## Configuration

### Real Data Paths

To use real data, update the paths in the `setup_real_data_paths()` function:

```python
real_data_paths = {
    'coco': {
        'annotation_path': Path(r"path/to/your/coco/annotations.json"),
        'description': "COCO format annotation file"
    },
    'yolo': {
        'data_yaml_path': Path(r"path/to/your/yolo/data.yaml"),
        'description': "YOLO format data.yaml file"
    }
}
```

### Supported Data Formats

- **COCO**: JSON annotation files with images, annotations, and categories
- **YOLO**: data.yaml files with image paths and label files

## Output

The workflow produces:

1. **Master Format**: Unified dataset format stored in the master data directory
2. **Framework Formats**: COCO and YOLO format exports
3. **Dataset Information**: Statistics about images, annotations, and splits
4. **Pipeline Status**: Information about supported formats and available datasets

## Example Output

```
--- WildTrain Data Pipeline Workflow Example ---
[INFO] Created temp directory: /tmp/tmp12345
[INFO] DataPipeline initialized.
[INFO] Pipeline status:
  - Master data directory: /tmp/tmp12345/data
  - Supported formats: ['coco', 'yolo']
  - Available datasets: []

--- Processing COCO Dataset ---
Source: /path/to/coco/annotations.json
Description: COCO format annotation file
[SUCCESS] Imported dataset 'real_coco_dataset'
Master annotations: /tmp/tmp12345/data/real_coco_dataset/annotations.json
Dataset info:
  - Total images: 1000
  - Total annotations: 2500
  - Images by split: {'train': 800, 'val': 200}
  - Annotations by type: {'detection': 2500}
[SUCCESS] Exported to COCO: /tmp/tmp12345/data/framework_formats/coco/real_coco_dataset
[SUCCESS] Exported to YOLO: /tmp/tmp12345/data/framework_formats/yolo/real_coco_dataset

--- Available Datasets ---
Dataset: real_coco_dataset
  - Total images: 1000
  - Total annotations: 2500
  - Images by split: {'train': 800, 'val': 200}
  - Annotations by type: {'detection': 2500}

--- Workflow Complete ---
[INFO] Cleaning up temp directory: /tmp/tmp12345
```

## Error Handling

The workflow includes comprehensive error handling:

- **Missing Data**: Graceful fallback to synthetic data if real data is not found
- **Validation Errors**: Detailed error messages for data validation issues
- **Import Failures**: Clear error reporting with hints for resolution
- **Export Failures**: Individual format export failures don't stop the entire workflow

## Integration with Tests

This workflow example is designed to work with the test suite in `tests/test_complete_workflow.py`. The test file includes:

- Synthetic data tests
- Real data tests (with actual data paths)
- Component-level tests
- Error handling tests

## Dependencies

- `wildtrain.pipeline.data_pipeline`: Main pipeline orchestrator
- `wildtrain.transformations`: Data transformation components
- `wildtrain.config`: Configuration classes
- Standard Python libraries: `pathlib`, `json`, `argparse`, etc.

## Troubleshooting

1. **Real data not found**: Update paths in `setup_real_data_paths()`
2. **Import errors**: Check data format and validation requirements
3. **Export failures**: Ensure master data is properly formatted
4. **Permission errors**: Check file/directory permissions for data access 