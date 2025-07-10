#!/usr/bin/env python3
"""
Demo script showing how to use the WildTrain data pipeline.
"""

import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildtrain.pipeline.data_pipeline import DataPipeline


def demo_pipeline():
    """Demonstrate the data pipeline functionality."""
    print("🚀 WildTrain Data Pipeline Demo")
    print("=" * 50)
    
    # Initialize pipeline
    project_root = Path.cwd()
    pipeline = DataPipeline(str(project_root))
    
    print(f"Project root: {project_root}")
    print()
    
    # List existing datasets
    print("📋 Existing datasets:")
    datasets = pipeline.list_datasets()
    if datasets:
        for dataset in datasets:
            print(f"  - {dataset['dataset_name']}: {dataset['total_images']} images, {dataset['total_annotations']} annotations")
    else:
        print("  No datasets found.")
    print()
    
    # Example: Import a COCO dataset (if you have one)
    print("💡 To import a dataset, you would run:")
    print("   python main.py import /path/to/coco/dataset coco my_dataset")
    print("   python main.py import /path/to/yolo/dataset yolo my_dataset")
    print()
    
    # Example: Get dataset info
    print("💡 To get dataset info:")
    print("   python main.py info my_dataset")
    print()
    
    # Example: Export to framework format
    print("💡 To export to framework format:")
    print("   python main.py export my_dataset coco")
    print("   python main.py export my_dataset yolo")
    print()
    
    # Example: List all commands
    print("💡 Available commands:")
    print("   python main.py --help")
    print()
    
    print("✅ Pipeline demo completed!")


if __name__ == "__main__":
    demo_pipeline() 