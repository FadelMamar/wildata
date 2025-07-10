# Data Transformation System

The transformation system provides a flexible and extensible framework for applying data transformations to images and their annotations. This system is integrated into the main data pipeline and supports common operations like data augmentation and image tiling.

## Overview

The transformation system consists of:

1. **Base Transformer Classes** - Abstract interfaces for all transformers
2. **Concrete Transformers** - Implementations for specific transformations
3. **Transformation Pipeline** - Orchestrates multiple transformers
4. **Integration with Data Pipeline** - Seamless integration with the main pipeline

## Architecture

### Base Classes

#### `BaseTransformer`
The abstract base class that all transformers must inherit from. It defines the interface:

```python
class BaseTransformer(ABC):
    def transform_image(self, image, image_info) -> Tuple[np.ndarray, Dict]:
        """Transform an image."""
        pass
    
    def transform_annotations(self, annotations, image_info) -> List[Dict]:
        """Transform annotations."""
        pass
    
    def transform(self, image, annotations, image_info) -> Tuple[np.ndarray, List[Dict], Dict]:
        """Transform both image and annotations together."""
        pass
```


### Concrete Transformers

#### `AugmentationTransformer`
Applies data augmentation techniques to images and their annotations.

**Supported augmentations:**
- Random rotation
- Random horizontal/vertical flip
- Brightness/contrast adjustment
- Gaussian noise
- Random crop
- Color jittering (HSV adjustments)

**Configuration:**
```python
config = {
    'rotation_range': (-15, 15),      # Rotation angle range
    'flip_probability': 0.5,          # Probability of horizontal flip
    'brightness_range': (0.8, 1.2),   # Brightness adjustment range
    'contrast_range': (0.8, 1.2),     # Contrast adjustment range
    'noise_std': 0.01,               # Gaussian noise standard deviation
    'crop_size': (512, 512),         # Random crop size
    'color_jitter': {                 # Color jittering parameters
        'hue_shift': 0.1,
        'saturation_shift': 0.1,
        'value_shift': 0.1
    }
}
```

#### `TilingTransformer`
Extracts tiles/patches from large images and their annotations.

**Features:**
- Regular grid tiling
- Overlapping tiles
- Random tile extraction
- Annotation-aware tiling
- Minimum annotation area filtering

**Configuration:**
```python
config = {
    'tile_size': (512, 512),         # Tile dimensions
    'overlap': 50,                   # Overlap between tiles
    'stride': None,                  # Stride (auto-calculated from overlap)
    'random_tiles': False,           # Use random tile extraction
    'min_annotation_area': 0.01,     # Minimum annotation area ratio
    'max_tiles_per_image': 10,       # Maximum tiles per image
    'padding': 0                     # Padding around tiles
}
```

### Transformation Pipeline

The `TransformationPipeline` class orchestrates multiple transformers:

```python
pipeline = TransformationPipeline()

# Add transformers
pipeline.add_transformer(augmentation_transformer)
pipeline.add_transformer(tiling_transformer)

# Apply transformations
transformed_image, transformed_annotations, updated_info = pipeline.transform(
    image, annotations, image_info
)
```

## Integration with Data Pipeline

### Basic Integration

The transformation system is integrated into the main data pipeline:

```python
from wildtrain.transformations import TransformationPipeline, AugmentationTransformer
from wildtrain.pipeline.data_pipeline import DataPipeline

# Create transformation pipeline
transformation_pipeline = TransformationPipeline()
transformation_pipeline.add_transformer(AugmentationTransformer())

# Create data pipeline with transformations
data_pipeline = DataPipeline(
    master_data_dir="./master_data",
    transformation_pipeline=transformation_pipeline
)

# Import dataset with transformations
success = data_pipeline.import_dataset(
    source_path="./dataset",
    source_format="coco",
    dataset_name="my_dataset",
    apply_transformations=True
)
```

### Advanced Usage

#### Custom Transformers

You can create custom transformers by inheriting from `BaseTransformer`:

```python
from wildtrain.transformations.base_transformer import BaseTransformer

class CustomResizeTransformer(BaseTransformer):
    def __init__(self, target_size=(640, 640)):
        super().__init__({'target_size': target_size})
    
    def transform_image(self, image, image_info):
        target_w, target_h = self.config['target_size']
        resized_image = cv2.resize(image, (target_w, target_h))
        
        updated_info = image_info.copy()
        updated_info['resized'] = True
        updated_info['original_size'] = image.shape[:2]
        updated_info['new_size'] = (target_h, target_w)
        
        return resized_image, updated_info
    
    def transform_annotations(self, annotations, image_info):
        # Scale annotation coordinates
        original_h, original_w = image_info['original_size']
        new_h, new_w = image_info['new_size']
        
        scale_x = new_w / original_w
        scale_y = new_h / original_h
        
        transformed_annotations = []
        for ann in annotations:
            transformed_ann = ann.copy()
            
            # Scale bounding box
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                transformed_ann['bbox'] = [
                    x * scale_x, y * scale_y,
                    w * scale_x, h * scale_y
                ]
            
            # Scale segmentation
            if 'segmentation' in ann:
                scaled_segmentation = []
                for polygon in ann['segmentation']:
                    scaled_polygon = []
                    for i in range(0, len(polygon), 2):
                        x, y = polygon[i], polygon[i+1]
                        scaled_polygon.extend([x * scale_x, y * scale_y])
                    scaled_segmentation.append(scaled_polygon)
                transformed_ann['segmentation'] = scaled_segmentation
            
            transformed_annotations.append(transformed_ann)
        
        return transformed_annotations
```

#### Chaining Transformations

You can chain multiple transformations together:

```python
# Create pipeline with multiple transformers
pipeline = TransformationPipeline()

# Add resize transformer
pipeline.add_transformer(CustomResizeTransformer((640, 640)))

# Add augmentation transformer
pipeline.add_transformer(AugmentationTransformer({
    'rotation_range': (-10, 10),
    'flip_probability': 0.3
}))

# Add tiling transformer
pipeline.add_transformer(TilingTransformer({
    'tile_size': (256, 256),
    'overlap': 32
}))
```

## Usage Examples

### Data Augmentation

```python
# Create augmentation transformer
aug_transformer = AugmentationTransformer({
    'rotation_range': (-15, 15),
    'flip_probability': 0.5,
    'brightness_range': (0.8, 1.2),
    'contrast_range': (0.8, 1.2),
    'noise_std': 0.01
})

# Apply to single image
transformed_image, transformed_annotations, updated_info = aug_transformer.transform(
    image, annotations, image_info
)
```

### Image Tiling

```python
# Create tiling transformer
tiling_transformer = TilingTransformer({
    'tile_size': (512, 512),
    'overlap': 50,
    'max_tiles_per_image': 10,
    'min_annotation_area': 0.01
})

# Extract tiles
tiles = tiling_transformer.extract_tiles(image, annotations, image_info)

for tile_image, tile_annotations, tile_info in tiles:
    print(f"Tile shape: {tile_image.shape}")
    print(f"Tile annotations: {len(tile_annotations)}")
    print(f"Tile coordinates: {tile_info['tile_coords']}")
```

### Pipeline Integration

```python
# Create data pipeline with transformations
transformation_pipeline = TransformationPipeline()
transformation_pipeline.add_transformer(AugmentationTransformer())
transformation_pipeline.add_transformer(TilingTransformer())

data_pipeline = DataPipeline(
    master_data_dir="./master_data",
    transformation_pipeline=transformation_pipeline
)

# Import dataset with transformations
success = data_pipeline.import_dataset(
    source_path="./coco_dataset",
    source_format="coco",
    dataset_name="augmented_dataset",
    apply_transformations=True
)
```

## Configuration Management

### Saving Pipeline Configuration

```python
# Save pipeline configuration
pipeline.save_pipeline_config("transformation_config.json")
```

### Loading Pipeline Configuration

```python
# Load pipeline configuration
pipeline.load_pipeline_config("transformation_config.json")
```

## Best Practices

### 1. Transformer Order
- Apply geometric transformations (resize, crop) first
- Apply augmentation transformations second
- Apply tiling transformations last

### 2. Memory Management
- For large datasets, process images in batches
- Use generators for memory-efficient processing
- Consider using lazy loading for images

### 3. Validation
- Always validate transformer configurations
- Test transformations on small datasets first
- Verify annotation consistency after transformations

### 4. Performance
- Use vectorized operations when possible
- Cache transformation results for repeated operations
- Consider using multiprocessing for batch operations

## Error Handling

The transformation system includes comprehensive error handling:

```python
try:
    transformed_image, transformed_annotations, updated_info = pipeline.transform(
        image, annotations, image_info
    )
except Exception as e:
    logger.error(f"Transformation failed: {str(e)}")
    # Handle error appropriately
```

## Future Extensions

The transformation system is designed to be extensible. Future additions could include:

1. **Advanced Augmentations**
   - Mixup/CutMix techniques
   - AutoAugment policies
   - RandAugment

2. **Specialized Transformers**
   - Medical image transformations
   - Satellite image processing
   - Video frame transformations

3. **Performance Optimizations**
   - GPU acceleration
   - Parallel processing
   - Caching mechanisms

4. **Integration with External Libraries**
   - Albumentations
   - Torchvision transforms
   - OpenCV-based transformations

## Dependencies

The transformation system requires:
- `numpy` - For array operations
- `opencv-python` - For image processing
- `PIL` - For additional image operations (optional)

Install dependencies:
```bash
pip install numpy opencv-python pillow
```

## Conclusion

The transformation system provides a robust, flexible, and extensible framework for data transformations in the wildtrain pipeline. It seamlessly integrates with the existing data pipeline and supports both common and custom transformation needs. 