"""
Example demonstrating how to use the transformation system in the data pipeline.
"""

import numpy as np
from pathlib import Path
import cv2

from wildtrain.transformations import (
    TransformationPipeline, 
    AugmentationTransformer, 
    TilingTransformer
)
from wildtrain.pipeline.data_pipeline import DataPipeline


def create_augmentation_transformer():
    """Create an augmentation transformer with common settings."""
    config = {
        'rotation_range': (-15, 15),  # Random rotation between -15 and 15 degrees
        'flip_probability': 0.5,      # 50% chance of horizontal flip
        'brightness_range': (0.8, 1.2),  # Brightness adjustment
        'contrast_range': (0.8, 1.2),    # Contrast adjustment
        'noise_std': 0.01,           # Small amount of Gaussian noise
        'color_jitter': {
            'hue_shift': 0.1,
            'saturation_shift': 0.1,
            'value_shift': 0.1
        }
    }
    
    return AugmentationTransformer(config)


def create_tiling_transformer():
    """Create a tiling transformer for extracting patches."""
    config = {
        'tile_size': (512, 512),     # 512x512 pixel tiles
        'overlap': 50,               # 50 pixel overlap between tiles
        'max_tiles_per_image': 10,   # Maximum 10 tiles per image
        'min_annotation_area': 0.01, # At least 1% of tile must have annotations
        'random_tiles': False        # Use regular grid tiling
    }
    
    return TilingTransformer(config)


def example_basic_transformation():
    """Example of basic transformation usage."""
    print("=== Basic Transformation Example ===")
    
    # Create a simple test image
    image = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
    annotations = [
        {
            'id': 1,
            'category_id': 1,
            'bbox': [100, 100, 200, 150],  # [x, y, width, height]
            'segmentation': [[100, 100, 300, 100, 300, 250, 100, 250]],
            'keypoints': [150, 125, 2, 250, 125, 2, 200, 175, 2]  # [x, y, visibility, ...]
        }
    ]
    image_info = {
        'id': 1,
        'file_name': 'test_image.jpg',
        'width': 600,
        'height': 800
    }
    
    # Create transformation pipeline
    pipeline = TransformationPipeline()
    
    # Add augmentation transformer
    aug_transformer = create_augmentation_transformer()
    pipeline.add_transformer(aug_transformer)
    
    # Apply transformation
    transformed_image, transformed_annotations, updated_image_info = pipeline.transform(
        image, annotations, image_info
    )
    
    print(f"Original image shape: {image.shape}")
    print(f"Transformed image shape: {transformed_image.shape}")
    print(f"Number of annotations: {len(transformed_annotations)}")
    print(f"Transformation history: {updated_image_info.get('transformation_history', [])}")
    
    return transformed_image, transformed_annotations, updated_image_info


def example_tiling_transformation():
    """Example of tiling transformation usage."""
    print("\n=== Tiling Transformation Example ===")
    
    # Create a test image
    image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    annotations = [
        {
            'id': 1,
            'category_id': 1,
            'bbox': [200, 200, 300, 250],
            'segmentation': [[200, 200, 500, 200, 500, 450, 200, 450]]
        }
    ]
    image_info = {
        'id': 1,
        'file_name': 'large_image.jpg',
        'width': 1024,
        'height': 1024
    }
    
    # Create tiling transformer
    tiling_transformer = create_tiling_transformer()
    
    # Extract tiles
    tiles = tiling_transformer.transform(image, annotations, image_info)
    
    print(f"Original image shape: {image.shape}")
    print(f"Number of tiles extracted: {len(tiles)}")
    
    for i, (tile_image, tile_annotations, tile_info) in enumerate(tiles):
        print(f"Tile {i+1}: shape={tile_image.shape}, annotations={len(tile_annotations)}")
        print(f"  Tile coords: {tile_info['tile_coords']}")
    
    return tiles


def example_pipeline_integration():
    """Example of integrating transformations into the data pipeline."""
    print("\n=== Pipeline Integration Example ===")
    
    # Create transformation pipeline
    transformation_pipeline = TransformationPipeline()
    
    # Add augmentation transformer
    aug_transformer = create_augmentation_transformer()
    transformation_pipeline.add_transformer(aug_transformer)
    
    # Add tiling transformer
    tiling_transformer = create_tiling_transformer()
    transformation_pipeline.add_transformer(tiling_transformer)
    
    # Create data pipeline with transformations
    data_pipeline = DataPipeline(
        master_data_dir="./master_data",
        transformation_pipeline=transformation_pipeline
    )
    
    # Show pipeline status
    status = data_pipeline.get_pipeline_status()
    print(f"Pipeline status: {status}")
    
    return data_pipeline


def example_custom_transformer():
    """Example of creating a custom transformer."""
    print("\n=== Custom Transformer Example ===")
    
    from wildtrain.transformations.base_transformer import BaseTransformer
    
    class CustomBrightnessTransformer(BaseTransformer):
        """Custom transformer that only adjusts brightness."""
        
        def __init__(self, brightness_factor=1.2):
            super().__init__({'brightness_factor': brightness_factor})
        
        def transform_image(self, image, image_info):
            """Apply brightness adjustment."""
            factor = self.config['brightness_factor']
            transformed_image = np.clip(image * factor, 0, 255).astype(np.uint8)
            
            updated_info = image_info.copy()
            updated_info['brightness_adjusted'] = factor
            
            return transformed_image, updated_info
        
        def transform_annotations(self, annotations, image_info):
            """Annotations remain unchanged for brightness adjustment."""
            return annotations
    
    # Use custom transformer
    custom_transformer = CustomBrightnessTransformer(brightness_factor=1.5)
    
    # Test with sample data
    image = np.random.randint(0, 255, (400, 300, 3), dtype=np.uint8)
    annotations = []
    image_info = {'id': 1, 'file_name': 'test.jpg'}
    
    transformed_image, transformed_annotations, updated_info = custom_transformer.transform(
        image, annotations, image_info
    )
    
    print(f"Original image mean: {np.mean(image):.2f}")
    print(f"Transformed image mean: {np.mean(transformed_image):.2f}")
    print(f"Brightness factor applied: {updated_info.get('brightness_adjusted')}")


if __name__ == "__main__":
    # Run examples
    example_basic_transformation()
    example_tiling_transformation()
    example_pipeline_integration()
    example_custom_transformer()
    
    print("\n=== Transformation System Ready ===")
    print("The transformation system is now integrated into the data pipeline!")
    print("You can use it to:")
    print("1. Apply data augmentation during dataset import")
    print("2. Extract tiles/patches from large images")
    print("3. Create custom transformations for specific needs")
    print("4. Chain multiple transformations together") 