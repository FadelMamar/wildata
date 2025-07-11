import tempfile
import shutil
import os
import json
from pathlib import Path
import numpy as np
import argparse
import sys

from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.transformations import AugmentationTransformer
from wildtrain.config import AugmentationConfig


def create_synthetic_coco_data(images_dir: Path, annotation_file: Path):
    """Create synthetic COCO data for testing purposes."""
    # Create synthetic images
    images = []
    for i in range(2):
        img_name = f"test_image_{i + 1}.jpg"
        img_path = images_dir / img_name
        # Create a random image
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            import cv2

            cv2.imwrite(str(img_path), arr)
        except ImportError:
            from PIL import Image

            Image.fromarray(arr).save(str(img_path))
        images.append(
            {
                "id": i + 1,
                "file_name": img_name,
                "width": 640,
                "height": 480,
                "split": "train",
            }
        )
    # Create annotations
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 200, 150],
            "area": 30000,
            "iscrowd": 0,
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 1,
            "bbox": [150, 150, 250, 200],
            "area": 25000,
            "iscrowd": 0,
        },
    ]
    categories = [{"id": 1, "name": "test_category", "supercategory": "test"}]
    coco_data = {"images": images, "annotations": annotations, "categories": categories}
    with open(annotation_file, "w") as f:
        json.dump(coco_data, f)


def setup_real_data_paths():
    """Setup paths for real data - modify these paths as needed."""
    # Define paths to your real data
    # These should be updated to point to your actual data locations
    real_data_paths = {
        "coco": {
            "annotation_path": Path(r"D:/workspace/savmap/coco/annotations.json"),
            "description": "COCO format annotation file",
        },
        "yolo": {
            "data_yaml_path": Path(r"D:/workspace/savmap/yolo/data.yaml"),
            "description": "YOLO format data.yaml file",
        },
    }
    return real_data_paths


def check_data_availability(real_data_paths):
    """Check which real data is available."""
    available_data = {}

    for format_name, data_info in real_data_paths.items():
        if format_name == "coco":
            path = data_info["annotation_path"]
            if path.exists():
                available_data[format_name] = {
                    "path": path,
                    "description": data_info["description"],
                }
                print(f"[INFO] Found COCO data: {path}")
            else:
                print(f"[WARNING] COCO data not found: {path}")

        elif format_name == "yolo":
            path = data_info["data_yaml_path"]
            if path.exists():
                available_data[format_name] = {
                    "path": path,
                    "description": data_info["description"],
                }
                print(f"[INFO] Found YOLO data: {path}")
            else:
                print(f"[WARNING] YOLO data not found: {path}")

    return available_data


def process_real_dataset(pipeline, data_info, dataset_name, format_type):
    """Process a real dataset with the pipeline."""
    print(f"\n--- Processing {format_type.upper()} Dataset ---")
    print(f"Source: {data_info['path']}")
    print(f"Description: {data_info['description']}")

    # Import dataset
    result = pipeline.import_dataset(
        source_path=str(data_info["path"]),
        source_format=format_type,
        dataset_name=dataset_name,
    )

    if result["success"]:
        print(f"[SUCCESS] Imported dataset '{dataset_name}'")
        print(f"Master annotations: {result['master_path']}")

        # Get dataset info
        try:
            info = pipeline.get_dataset_info(dataset_name)
            print(f"Dataset info:")
            print(f"  - Total images: {info['total_images']}")
            print(f"  - Total annotations: {info['total_annotations']}")
            print(f"  - Images by split: {info['images_by_split']}")
            print(f"  - Annotations by type: {info['annotations_by_type']}")
        except Exception as e:
            print(f"[WARNING] Could not get dataset info: {e}")

        # Export to framework formats
        for export_format in ["coco", "yolo"]:
            try:
                export_result = pipeline.export_framework_format(
                    dataset_name, export_format
                )
                print(
                    f"[SUCCESS] Exported to {export_format.upper()}: {export_result['output_path']}"
                )
            except Exception as e:
                print(f"[WARNING] Failed to export to {export_format.upper()}: {e}")

        return True
    else:
        print(f"[ERROR] Failed to import dataset '{dataset_name}'")
        print(f"Error: {result.get('error', 'Unknown error')}")
        if result.get("validation_errors"):
            print("Validation errors:")
            for error in result["validation_errors"]:
                print(f"  - {error}")
        if result.get("hints"):
            print("Hints:")
            for hint in result["hints"]:
                print(f"  - {hint}")
        return False


def process_synthetic_dataset(pipeline, temp_dir):
    """Process synthetic dataset for demonstration."""
    print("\n--- Processing Synthetic Dataset ---")

    project_root = Path(temp_dir)
    data_dir = project_root / "data"
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True)
    annotation_file = data_dir / "annotations_train.json"

    # Create synthetic COCO data
    create_synthetic_coco_data(images_dir, annotation_file)
    print(f"[INFO] Synthetic COCO data created at {annotation_file}")

    # Import dataset
    result = pipeline.import_dataset(
        source_path=str(annotation_file),
        source_format="coco",
        dataset_name="synthetic_demo_dataset",
    )

    if result["success"]:
        print(f"[SUCCESS] Imported synthetic dataset")
        print(f"Master annotations: {result['master_path']}")

        # Add transformation (augmentation)
        try:
            config = AugmentationConfig(
                rotation_range=(-10, 10), probability=0.5, brightness_range=(0.9, 1.1)
            )
            transformer = AugmentationTransformer(config)
            pipeline.add_transformation(transformer)
            print("[INFO] AugmentationTransformer added to pipeline.")
        except Exception as e:
            print(f"[WARNING] Could not add augmentation: {e}")

        # Export to framework formats
        for export_format in ["coco", "yolo"]:
            try:
                export_result = pipeline.export_framework_format(
                    "synthetic_demo_dataset", export_format
                )
                print(
                    f"[SUCCESS] Exported to {export_format.upper()}: {export_result['output_path']}"
                )
            except Exception as e:
                print(f"[WARNING] Failed to export to {export_format.upper()}: {e}")

        return True
    else:
        print(f"[ERROR] Failed to import synthetic dataset")
        print(f"Error: {result.get('error', 'Unknown error')}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="WildTrain Data Pipeline Workflow Example"
    )
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="Use real data instead of synthetic data",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory to store master data (default: temp directory)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true", help="Keep temporary files after processing"
    )

    args = parser.parse_args()

    print("--- WildTrain Data Pipeline Workflow Example ---")

    # Setup data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
        print(f"[INFO] Using specified data directory: {data_dir}")
    else:
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)
        print(f"[INFO] Created temp directory: {temp_dir}")

    try:
        # Initialize pipeline
        pipeline = DataPipeline(str(data_dir))
        print("[INFO] DataPipeline initialized.")

        # Get pipeline status
        status = pipeline.get_pipeline_status()
        print(f"[INFO] Pipeline status:")
        print(f"  - Master data directory: {status['master_data_dir']}")
        print(f"  - Supported formats: {status['supported_formats']}")
        print(f"  - Available datasets: {status['available_datasets']}")

        if args.use_real_data:
            # Process real data
            real_data_paths = setup_real_data_paths()
            available_data = check_data_availability(real_data_paths)

            if not available_data:
                print("[WARNING] No real data found. Falling back to synthetic data.")
                process_synthetic_dataset(pipeline, temp_dir or data_dir)
            else:
                # Process each available real dataset
                for format_type, data_info in available_data.items():
                    dataset_name = f"real_{format_type}_dataset"
                    process_real_dataset(pipeline, data_info, dataset_name, format_type)
        else:
            # Process synthetic data
            process_synthetic_dataset(pipeline, temp_dir or data_dir)

        # List all datasets
        print("\n--- Available Datasets ---")
        datasets = pipeline.list_datasets()
        if datasets:
            for dataset in datasets:
                print(f"Dataset: {dataset['dataset_name']}")
                print(f"  - Total images: {dataset['total_images']}")
                print(f"  - Total annotations: {dataset['total_annotations']}")
                print(f"  - Images by split: {dataset['images_by_split']}")
                print(f"  - Annotations by type: {dataset['annotations_by_type']}")
        else:
            print("No datasets found.")

        print("\n--- Workflow Complete ---")

        if args.keep_temp and temp_dir:
            print(f"[INFO] Keeping temp directory: {temp_dir}")
        elif temp_dir:
            print(f"[INFO] Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)

    except Exception as e:
        print(f"[ERROR] Workflow failed: {e}")
        if temp_dir and not args.keep_temp:
            shutil.rmtree(temp_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
