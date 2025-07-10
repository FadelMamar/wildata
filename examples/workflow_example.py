import tempfile
import shutil
import os
import json
from pathlib import Path
import numpy as np

from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.transformations import AugmentationTransformer
from wildtrain.config import AugmentationConfig


def create_synthetic_coco_data(images_dir: Path, annotation_file: Path):
    # Create synthetic images
    images = []
    for i in range(2):
        img_name = f"test_image_{i+1}.jpg"
        img_path = images_dir / img_name
        # Create a random image
        arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        try:
            import cv2
            cv2.imwrite(str(img_path), arr)
        except ImportError:
            from PIL import Image
            Image.fromarray(arr).save(str(img_path))
        images.append({
            "id": i+1,
            "file_name": img_name,
            "width": 640,
            "height": 480,
            "split": "train"
        })
    # Create annotations
    annotations = [
        {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [100, 100, 200, 150],
            "area": 30000,
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 2,
            "category_id": 1,
            "bbox": [150, 150, 250, 200],
            "area": 25000,
            "iscrowd": 0
        }
    ]
    categories = [
        {"id": 1, "name": "test_category", "supercategory": "test"}
    ]
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(annotation_file, 'w') as f:
        json.dump(coco_data, f)


def main():
    print("--- WildTrain Data Pipeline Workflow Example ---")
    temp_dir = tempfile.mkdtemp()
    print(f"[INFO] Created temp directory: {temp_dir}")
    try:
        project_root = Path(temp_dir)
        data_dir = project_root / "data"
        images_dir = data_dir / "images"
        images_dir.mkdir(parents=True)
        annotation_file = data_dir / "annotations_train.json"
        # Step 1: Create synthetic COCO data
        create_synthetic_coco_data(images_dir, annotation_file)
        print(f"[INFO] Synthetic COCO data created at {annotation_file}")

        # Step 2: Initialize pipeline
        pipeline = DataPipeline(str(data_dir))
        print("[INFO] DataPipeline initialized.")

        # Step 3: Import dataset
        result = pipeline.import_dataset(
            source_path=str(annotation_file),
            source_format="coco",
            dataset_name="demo_dataset"
        )
        print(f"[INFO] Import result: {result}")

        # Step 4: Add transformation (augmentation)
        config = AugmentationConfig(
            rotation_range=(-10, 10),
            probability=0.5,
            brightness_range=(0.9, 1.1)
        )
        transformer = AugmentationTransformer(config)
        pipeline.add_transformation(transformer)
        print("[INFO] AugmentationTransformer added to pipeline.")

        # Step 5: Export to COCO format
        coco_export = pipeline.export_framework_format("demo_dataset", "coco")
        print(f"[INFO] Exported COCO format: {coco_export}")

        # Step 6: Export to YOLO format
        yolo_export = pipeline.export_framework_format("demo_dataset", "yolo")
        print(f"[INFO] Exported YOLO format: {yolo_export}")

        # Step 7: Print out file locations
        print("\n--- Results ---")
        print(f"Master annotations: {result['master_path']}")
        print(f"COCO format dir: {coco_export['output_path']}")
        print(f"YOLO format dir: {yolo_export['output_path']}")
        print("\nYou can inspect these files before cleanup.")

    finally:
        # Clean up temp files
        print(f"[INFO] Cleaning up temp directory: {temp_dir}")
        shutil.rmtree(temp_dir)
        print("[INFO] Done.")


if __name__ == "__main__":
    main() 