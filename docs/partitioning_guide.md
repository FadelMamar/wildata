# WildTrain Partitioning System

## Overview

The WildTrain partitioning system provides robust train-val-test splitting strategies specifically designed for aerial imagery data with high spatial autocorrelation. This system addresses the unique challenges of wildlife monitoring datasets by offering multiple partitioning strategies that respect spatial relationships and prevent data leakage.

## Key Features

### 🗺️ Spatial Autocorrelation Handling
- **GPS Coordinate Analysis**: Extracts and uses GPS coordinates from image metadata
- **Spatial Clustering**: Groups spatially close images together using DBSCAN or grid-based methods
- **Distance Thresholds**: Configurable spatial thresholds for grouping (in degrees)

### 🏕️ Camp-Based Grouping
- **Wildlife Camp Areas**: Groups images by camp areas to ensure spatial integrity
- **Camp Metadata**: Uses camp identifiers from image metadata
- **Fallback Keys**: Multiple metadata keys for camp identification

### 📊 Metadata-Based Partitioning
- **Dataset Tags**: Groups by dataset identifiers and tags
- **Acquisition Dates**: Temporal grouping for time-series data
- **Flexible Keys**: Configurable metadata keys for grouping

### 🔄 Hybrid Strategy
- **Multi-Strategy Fallback**: Combines spatial, camp-based, and metadata strategies
- **Automatic Fallback**: Tries strategies in order until one succeeds
- **Robust Handling**: Graceful handling of missing metadata

## Installation

The partitioning system is included with WildTrain and requires scikit-learn:

```bash
# Already included in pyproject.toml
scikit-learn>=1.7.0
```

## Quick Start

### Basic Usage

```python
from wildtrain.partitioning.partitioning_pipeline import PartitioningPipeline, PartitioningStrategy

# Create partitioning pipeline
pipeline = PartitioningPipeline(
    strategy=PartitioningStrategy.SPATIAL,
    test_size=0.2,
    val_size=0.2,
    random_state=42,
    spatial_threshold=0.01,  # degrees for GPS coordinates
)

# Apply partitioning to COCO data
split_data = pipeline.apply_partitioning_to_coco_data(coco_data)

# Access splits
train_data = split_data["train"]
val_data = split_data["val"]
test_data = split_data["test"]
```

### Camp-Based Partitioning

```python
# For wildlife camp areas
pipeline = PartitioningPipeline(
    strategy=PartitioningStrategy.CAMP_BASED,
    test_size=0.25,
    val_size=0.25,
    camp_metadata_key="camp_id",
)

split_data = pipeline.apply_partitioning_to_coco_data(coco_data)
```

### Hybrid Strategy

```python
# Combines multiple strategies with fallback
pipeline = PartitioningPipeline(
    strategy=PartitioningStrategy.HYBRID,
    test_size=0.2,
    val_size=0.2,
    spatial_threshold=0.01,
    camp_metadata_key="camp_id",
    metadata_keys=["dataset_id", "camp_id", "acquisition_date"],
)

split_data = pipeline.apply_partitioning_to_coco_data(coco_data)
```

## Partitioning Strategies

### 1. Spatial Partitioning

**Use Case**: Datasets with GPS coordinates and high spatial autocorrelation

**Features**:
- Extracts GPS coordinates from image metadata
- Uses DBSCAN clustering for spatial grouping
- Configurable distance thresholds
- Grid-based fallback option

**Configuration**:
```python
pipeline = PartitioningPipeline(
    strategy=PartitioningStrategy.SPATIAL,
    spatial_threshold=0.01,  # degrees
    clustering_method="dbscan",  # or "grid"
    gps_keys=["gps_lat", "gps_lon", "latitude", "longitude"],
)
```

### 2. Camp-Based Partitioning

**Use Case**: Wildlife monitoring with defined camp areas

**Features**:
- Groups images by camp identifiers
- Respects camp boundaries
- Fallback keys for different metadata structures
- Individual group creation for orphaned images

**Configuration**:
```python
pipeline = PartitioningPipeline(
    strategy=PartitioningStrategy.CAMP_BASED,
    camp_metadata_key="camp_id",
    fallback_keys=["camp", "area", "region"],
    create_individual_groups=True,
)
```

### 3. Metadata-Based Partitioning

**Use Case**: Datasets with rich metadata (tags, dates, sources)

**Features**:
- Groups by dataset identifiers
- Temporal grouping by acquisition dates
- Tag-based grouping
- Composite key creation

**Configuration**:
```python
pipeline = PartitioningPipeline(
    strategy=PartitioningStrategy.METADATA_BASED,
    metadata_keys=["dataset_id", "source", "acquisition_date"],
    dataset_id_key="dataset_id",
    date_keys=["date", "acquisition_date", "timestamp"],
    tag_keys=["tags", "labels", "categories"],
)
```

### 4. Hybrid Strategy

**Use Case**: Complex datasets with multiple metadata types

**Features**:
- Tries multiple strategies in sequence
- Automatic fallback mechanisms
- Robust handling of missing data
- Best-effort partitioning

**Configuration**:
```python
pipeline = PartitioningPipeline(
    strategy=PartitioningStrategy.HYBRID,
    # All parameters from individual strategies
    spatial_threshold=0.01,
    camp_metadata_key="camp_id",
    metadata_keys=["dataset_id", "camp_id"],
)
```

## Integration with Data Pipeline

### Basic Integration

```python
from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.partitioning.partitioning_pipeline import PartitioningPipeline

# Create data pipeline
data_pipeline = DataPipeline(
    root="data/",
    split_name="train",  # Will be overridden by partitioning
    enable_dvc=True,
)

# Create partitioning pipeline
partitioning_pipeline = PartitioningPipeline(
    strategy=PartitioningStrategy.SPATIAL,
    test_size=0.2,
    val_size=0.2,
)

# Load dataset
loader = Loader()
dataset_info, split_data = loader.load(
    source_path="path/to/dataset",
    source_format="coco",
    dataset_name="my_dataset",
)

# Apply partitioning to create train/val/test splits
partitioned_data = partitioning_pipeline.apply_partitioning_to_coco_data(
    split_data["train"]  # Assuming original data is in train split
)

# Store each split
for split_name, split_coco in partitioned_data.items():
    data_pipeline.split_name = split_name
    data_pipeline.import_dataset(
        source_path=None,  # Use in-memory data
        source_format="coco",
        dataset_name=f"my_dataset_{split_name}",
        # ... other parameters
    )
```

### Advanced Integration

```python
# Custom partitioning workflow
def partition_and_store_dataset(
    source_path: str,
    dataset_name: str,
    partitioning_config: dict,
    output_root: str,
):
    """Complete workflow for partitioning and storing datasets."""
    
    # Load original dataset
    loader = Loader()
    dataset_info, split_data = loader.load(
        source_path=source_path,
        source_format="coco",
        dataset_name=dataset_name,
    )
    
    # Create partitioning pipeline
    pipeline = PartitioningPipeline(**partitioning_config)
    
    # Apply partitioning
    partitioned_data = pipeline.apply_partitioning_to_coco_data(
        split_data["train"]
    )
    
    # Store each split
    for split_name, split_coco in partitioned_data.items():
        data_pipeline = DataPipeline(
            root=output_root,
            split_name=split_name,
            enable_dvc=True,
        )
        
        # Store split
        data_pipeline.import_dataset(
            source_path=None,
            source_format="coco",
            dataset_name=f"{dataset_name}_{split_name}",
            # Use in-memory data
            dataset_info=dataset_info,
            split_data={split_name: split_coco},
        )
    
    # Save partitioning configuration
    pipeline.save_partitioning_config(
        f"{output_root}/partitioning_config.json",
        additional_info={
            "source_dataset": dataset_name,
            "partitioning_date": "2024-01-15",
        }
    )
```

## Configuration

### Partitioning Configuration File

```json
{
  "strategy": "spatial",
  "test_size": 0.2,
  "val_size": 0.2,
  "random_state": 42,
  "spatial_threshold": 0.01,
  "clustering_method": "dbscan",
  "camp_metadata_key": "camp_id",
  "metadata_keys": ["dataset_id", "camp_id", "acquisition_date"],
  "create_individual_groups": true
}
```

### Loading from Configuration

```python
# Load from file
pipeline = PartitioningPipeline.from_config("partitioning_config.json")

# Or create with parameters
pipeline = PartitioningPipeline(
    strategy="spatial",
    test_size=0.2,
    val_size=0.2,
    spatial_threshold=0.01,
)
```

## Statistics and Analysis

### Get Partitioning Statistics

```python
# Get comprehensive statistics
stats = pipeline.get_statistics(images, metadata)

print(f"Strategy: {stats['strategy']}")
print(f"Total images: {stats['total_images']}")
print(f"Coverage: {stats['spatial']['coverage_percentage']:.1f}%")

# Spatial statistics
if 'spatial' in stats:
    spatial = stats['spatial']
    print(f"Spatial bounds: {spatial['spatial_bounds']}")
    print(f"Unique spatial groups: {len(spatial.get('unique_groups', []))}")

# Camp statistics
if 'camp_based' in stats:
    camp = stats['camp_based']
    print(f"Unique camps: {camp['unique_camps']}")
    print(f"Avg images per camp: {camp['avg_images_per_camp']:.1f}")

# Metadata statistics
if 'metadata_based' in stats:
    meta = stats['metadata_based']
    print(f"Unique datasets: {meta['unique_datasets']}")
    print(f"Dataset distribution: {meta['dataset_distribution']}")
```

## Best Practices

### 1. Data Preparation

- **GPS Coordinates**: Ensure GPS coordinates are in decimal degrees
- **Metadata Consistency**: Use consistent metadata keys across datasets
- **Camp Identifiers**: Use unique, consistent camp identifiers
- **Date Formats**: Use ISO format (YYYY-MM-DD) for dates

### 2. Strategy Selection

- **Spatial**: Use when GPS coordinates are available and spatial autocorrelation is high
- **Camp-Based**: Use when camp areas are well-defined and important
- **Metadata-Based**: Use when dataset tags or temporal grouping is important
- **Hybrid**: Use for complex datasets with multiple metadata types

### 3. Parameter Tuning

- **Spatial Threshold**: Start with 0.01 degrees, adjust based on your data density
- **Test/Val Sizes**: Consider your dataset size and evaluation needs
- **Random State**: Use fixed random state for reproducible results

### 4. Validation

- **Check Splits**: Verify that splits are reasonable and balanced
- **Spatial Integrity**: Ensure spatial groups are meaningful
- **Metadata Coverage**: Check that metadata extraction is working correctly

## Troubleshooting

### Common Issues

1. **No GPS Coordinates Found**
   - Check metadata keys in `gps_keys` parameter
   - Verify coordinate format (decimal degrees)
   - Use `get_statistics()` to check coverage

2. **No Camp Information**
   - Check `camp_metadata_key` parameter
   - Verify camp identifiers in metadata
   - Enable `create_individual_groups` for orphaned images

3. **Poor Split Balance**
   - Adjust `test_size` and `val_size` parameters
   - Check group distribution with statistics
   - Consider using hybrid strategy

4. **Memory Issues**
   - Use smaller spatial thresholds
   - Process datasets in chunks
   - Use grid-based clustering instead of DBSCAN

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Get detailed statistics
stats = pipeline.get_statistics(images, metadata)
print(json.dumps(stats, indent=2))
```

## Examples

See `examples/partitioning_example.py` for complete working examples demonstrating all partitioning strategies and integration patterns.

## API Reference

### PartitioningPipeline

Main class for orchestrating partitioning strategies.

**Methods**:
- `partition_dataset()`: Partition images using selected strategy
- `apply_partitioning_to_coco_data()`: Apply partitioning to COCO format data
- `get_statistics()`: Get comprehensive dataset statistics
- `save_partitioning_config()`: Save configuration to file
- `from_config()`: Create pipeline from configuration file

### PartitioningStrategy

Enumeration of available partitioning strategies:
- `SPATIAL`: Spatial autocorrelation handling
- `CAMP_BASED`: Camp-based grouping
- `METADATA_BASED`: Metadata-based partitioning
- `HYBRID`: Multi-strategy with fallback

### Individual Partitioners

- `SpatialPartitioner`: GPS coordinate analysis
- `CampPartitioner`: Camp area grouping
- `MetadataPartitioner`: Metadata-based grouping

## Contributing

The partitioning system is designed to be extensible. To add new partitioning strategies:

1. Create a new partitioner class inheriting from base strategies
2. Implement required methods (`partition_dataset`, `get_statistics`)
3. Add strategy to `PartitioningStrategy` enum
4. Update `PartitioningPipeline` to handle new strategy
5. Add tests and documentation

## License

This partitioning system is part of the WildTrain project and follows the same license terms. 