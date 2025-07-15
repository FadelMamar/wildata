"""
Example script demonstrating the partitioning system for aerial imagery data.

This example shows how to use the partitioning pipeline to create train-val-test
splits that respect spatial autocorrelation in aerial imagery data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# Add the src directory to the path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from wildtrain.partitioning.partitioning_pipeline import PartitioningPipeline, PartitioningStrategy


def create_sample_coco_data() -> Dict[str, Any]:
    """Create sample COCO format data with spatial metadata."""
    
    # Create sample images with GPS coordinates
    images = []
    annotations = []
    categories = [
        {"id": 1, "name": "wildlife", "supercategory": "animal"},
        {"id": 2, "name": "vehicle", "supercategory": "transport"},
    ]
    
    # Create images from different camps with GPS coordinates
    camps = ["camp_alpha", "camp_beta", "camp_gamma", "camp_delta"]
    camp_coordinates = {
        "camp_alpha": [(1.0, 1.0), (1.01, 1.0), (1.0, 1.01), (1.01, 1.01)],
        "camp_beta": [(2.0, 2.0), (2.01, 2.0), (2.0, 2.01), (2.01, 2.01)],
        "camp_gamma": [(3.0, 3.0), (3.01, 3.0), (3.0, 3.01), (3.01, 3.01)],
        "camp_delta": [(4.0, 4.0), (4.01, 4.0), (4.0, 4.01), (4.01, 4.01)],
    }
    
    image_id = 1
    annotation_id = 1
    
    for camp_name, coordinates in camp_coordinates.items():
        for i, (lat, lon) in enumerate(coordinates):
            # Create image with GPS coordinates and camp metadata
            image = {
                "id": image_id,
                "file_name": f"{camp_name}_image_{i+1}.jpg",
                "width": 640,
                "height": 480,
                "gps_lat": lat,
                "gps_lon": lon,
                "camp_id": camp_name,
                "dataset_id": f"dataset_{camp_name}",
                "acquisition_date": "2024-01-15",
            }
            images.append(image)
            
            # Create some annotations
            for cat_id in [1, 2]:
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": [100, 100, 200, 150],
                    "area": 30000,
                    "iscrowd": 0,
                }
                annotations.append(annotation)
                annotation_id += 1
            
            image_id += 1
    
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def demonstrate_spatial_partitioning():
    """Demonstrate spatial partitioning strategy."""
    print("=== Spatial Partitioning Demo ===")
    
    # Create sample data
    coco_data = create_sample_coco_data()
    
    # Initialize spatial partitioning pipeline
    pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.SPATIAL,
        test_size=0.25,
        val_size=0.25,
        random_state=42,
        spatial_threshold=0.02,  # 0.02 degrees for GPS coordinates
        clustering_method="dbscan",
    )
    
    # Get statistics
    stats = pipeline.get_statistics(coco_data["images"])
    print(f"Spatial statistics: {json.dumps(stats, indent=2)}")
    
    # Apply partitioning
    split_data = pipeline.apply_partitioning_to_coco_data(coco_data)
    
    # Print results
    for split_name, split_coco in split_data.items():
        print(f"{split_name}: {len(split_coco['images'])} images, "
              f"{len(split_coco['annotations'])} annotations")
    
    return split_data


def demonstrate_camp_based_partitioning():
    """Demonstrate camp-based partitioning strategy."""
    print("\n=== Camp-Based Partitioning Demo ===")
    
    # Create sample data
    coco_data = create_sample_coco_data()
    
    # Initialize camp-based partitioning pipeline
    pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.CAMP_BASED,
        test_size=0.25,
        val_size=0.25,
        random_state=42,
        camp_metadata_key="camp_id",
    )
    
    # Get statistics
    stats = pipeline.get_statistics(coco_data["images"])
    print(f"Camp-based statistics: {json.dumps(stats, indent=2)}")
    
    # Apply partitioning
    split_data = pipeline.apply_partitioning_to_coco_data(coco_data)
    
    # Print results
    for split_name, split_coco in split_data.items():
        print(f"{split_name}: {len(split_coco['images'])} images, "
              f"{len(split_coco['annotations'])} annotations")
    
    return split_data


def demonstrate_hybrid_partitioning():
    """Demonstrate hybrid partitioning strategy."""
    print("\n=== Hybrid Partitioning Demo ===")
    
    # Create sample data
    coco_data = create_sample_coco_data()
    
    # Initialize hybrid partitioning pipeline
    pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.HYBRID,
        test_size=0.25,
        val_size=0.25,
        random_state=42,
        spatial_threshold=0.02,
        camp_metadata_key="camp_id",
        metadata_keys=["dataset_id", "camp_id", "acquisition_date"],
    )
    
    # Get statistics
    stats = pipeline.get_statistics(coco_data["images"])
    print(f"Hybrid statistics: {json.dumps(stats, indent=2)}")
    
    # Apply partitioning
    split_data = pipeline.apply_partitioning_to_coco_data(coco_data)
    
    # Print results
    for split_name, split_coco in split_data.items():
        print(f"{split_name}: {len(split_coco['images'])} images, "
              f"{len(split_coco['annotations'])} annotations")
    
    return split_data


def demonstrate_integration_with_data_pipeline():
    """Demonstrate integration with existing data pipeline."""
    print("\n=== Integration with Data Pipeline Demo ===")
    
    # This would typically be used with the existing DataPipeline class
    # Here we show how the partitioning could be integrated
    
    coco_data = create_sample_coco_data()
    
    # Create partitioning pipeline
    partitioning_pipeline = PartitioningPipeline(
        strategy=PartitioningStrategy.SPATIAL,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )
    
    # Apply partitioning to get split data
    split_data = partitioning_pipeline.apply_partitioning_to_coco_data(coco_data)
    
    # Save partitioning configuration
    config_path = Path("partitioning_config.json")
    partitioning_pipeline.save_partitioning_config(
        config_path,
        additional_info={
            "description": "Spatial partitioning for aerial imagery",
            "dataset_name": "sample_aerial_dataset",
        }
    )
    
    print(f"Partitioning configuration saved to {config_path}")
    
    # Load partitioning pipeline from config
    loaded_pipeline = PartitioningPipeline.from_config(config_path)
    print(f"Loaded pipeline with strategy: {loaded_pipeline.strategy.value}")
    
    return split_data


def main():
    """Run all partitioning demonstrations."""
    print("WildTrain Partitioning System Demo")
    print("=" * 50)
    
    try:
        # Demonstrate different partitioning strategies
        spatial_splits = demonstrate_spatial_partitioning()
        camp_splits = demonstrate_camp_based_partitioning()
        hybrid_splits = demonstrate_hybrid_partitioning()
        
        # Demonstrate integration
        integration_splits = demonstrate_integration_with_data_pipeline()
        
        print("\n" + "=" * 50)
        print("All demonstrations completed successfully!")
        print("\nKey Features Demonstrated:")
        print("- Spatial autocorrelation handling with GPS coordinates")
        print("- Camp-based grouping for wildlife areas")
        print("- Metadata-based partitioning for dataset tags")
        print("- Hybrid strategy with fallback mechanisms")
        print("- Integration with existing data pipeline")
        print("- Configuration persistence and loading")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 