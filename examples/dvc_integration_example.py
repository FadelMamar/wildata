#!/usr/bin/env python3
"""
Example script demonstrating DVC integration with WildTrain.

This script shows how to:
1. Setup DVC with remote storage
2. Import datasets with DVC tracking
3. Create and run data pipelines
4. Manage dataset versions
"""

import os
import sys
import tempfile
import json
from pathlib import Path
from typing import Dict, Any


from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.pipeline.dvc_manager import DVCManager, DVCConfig, DVCStorageType


def create_synthetic_dataset(output_dir: Path, format_type: str = "coco") -> str:
    """Create a synthetic dataset for demonstration purposes."""

    if format_type == "coco":
        # Create synthetic COCO dataset
        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "image1.jpg",
                    "width": 640,
                    "height": 480,
                    "split": "train",
                },
                {
                    "id": 2,
                    "file_name": "image2.jpg",
                    "width": 800,
                    "height": 600,
                    "split": "val",
                },
            ],
            "annotations": [
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
            ],
            "categories": [{"id": 1, "name": "object", "supercategory": "test"}],
        }

        # Create images directory
        images_dir = output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic image files
        for i in range(1, 3):
            img_path = images_dir / f"image{i}.jpg"
            # Create a simple synthetic image (1x1 pixel)
            with open(img_path, "wb") as f:
                f.write(
                    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9"
                )

        # Save COCO annotations
        annotations_file = output_dir / "annotations.json"
        with open(annotations_file, "w") as f:
            json.dump(coco_data, f, indent=2)

        return str(annotations_file)

    elif format_type == "yolo":
        # Create synthetic YOLO dataset
        yolo_dir = output_dir / "yolo_dataset"
        yolo_dir.mkdir(parents=True, exist_ok=True)

        # Create images directory
        images_dir = yolo_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Create labels directory
        labels_dir = yolo_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Create synthetic image files
        for i in range(1, 3):
            img_path = images_dir / f"image{i}.jpg"
            # Create a simple synthetic image
            with open(img_path, "wb") as f:
                f.write(
                    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9"
                )

        # Create label files
        for i in range(1, 3):
            label_path = labels_dir / f"image{i}.txt"
            with open(label_path, "w") as f:
                f.write(
                    "0 0.5 0.5 0.3 0.4\n"
                )  # YOLO format: class x_center y_center width height

        # Create data.yaml
        data_yaml = {
            "path": str(yolo_dir),
            "train": str(images_dir),
            "val": str(images_dir),
            "names": {0: "object"},
        }

        yaml_file = yolo_dir / "data.yaml"
        import yaml

        with open(yaml_file, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        return str(yaml_file)

    else:
        raise ValueError(f"Unsupported format type: {format_type}")


def demonstrate_dvc_integration():
    """Demonstrate DVC integration features."""

    print("🚀 WildTrain DVC Integration Demo")
    print("=" * 50)

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        print(f"📁 Working directory: {temp_path}")

        # Step 1: Initialize DVC
        print("\n1️⃣ Setting up DVC...")
        try:
            dvc_manager = DVCManager(temp_path)
            print("✅ DVC initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize DVC: {e}")
            return

        # Step 2: Setup local remote storage
        print("\n2️⃣ Setting up remote storage...")
        config = DVCConfig(
            storage_type=DVCStorageType.LOCAL,
            storage_path=str(temp_path / "dvc_storage"),
            remote_name="demo-remote",
        )

        dvc_manager.config = config
        if dvc_manager.setup_remote_storage():
            print("✅ Remote storage setup successful")
        else:
            print("❌ Failed to setup remote storage")
            return

        # Step 3: Create synthetic datasets
        print("\n3️⃣ Creating synthetic datasets...")

        # Create COCO dataset
        coco_data_dir = temp_path / "coco_data"
        coco_data_dir.mkdir()
        coco_file = create_synthetic_dataset(coco_data_dir, "coco")
        print(f"✅ Created COCO dataset: {coco_file}")

        # Create YOLO dataset
        yolo_data_dir = temp_path / "yolo_data"
        yolo_data_dir.mkdir()
        yolo_file = create_synthetic_dataset(yolo_data_dir, "yolo")
        print(f"✅ Created YOLO dataset: {yolo_file}")

        # Step 4: Initialize data pipeline with DVC
        print("\n4️⃣ Initializing data pipeline with DVC...")
        pipeline = DataPipeline(str(temp_path / "data"), enable_dvc=True)
        print("✅ Data pipeline initialized with DVC support")

        # Step 5: Import datasets with DVC tracking
        print("\n5️⃣ Importing datasets with DVC tracking...")

        # Import COCO dataset
        print("📦 Importing COCO dataset...")
        coco_result = pipeline.import_dataset(
            source_path=coco_file,
            source_format="coco",
            dataset_name="demo_coco_dataset",
            track_with_dvc=True,
        )

        if coco_result["success"]:
            print("✅ COCO dataset imported successfully")
            print(f"   Master annotations: {coco_result['master_path']}")
            if coco_result.get("dvc_tracked"):
                print("   📦 Dataset tracked with DVC")
        else:
            print(f"❌ Failed to import COCO dataset: {coco_result.get('error')}")

        # Import YOLO dataset with transformations
        print("\n📦 Importing YOLO dataset with transformations...")
        yolo_result = pipeline.import_dataset(
            source_path=yolo_file,
            source_format="yolo",
            dataset_name="demo_yolo_dataset",
            apply_transformations=True,
            track_with_dvc=True,
        )

        if yolo_result["success"]:
            print("✅ YOLO dataset imported successfully")
            print(f"   Master annotations: {yolo_result['master_path']}")
            if yolo_result.get("dvc_tracked"):
                print("   📦 Dataset tracked with DVC")
        else:
            print(f"❌ Failed to import YOLO dataset: {yolo_result.get('error')}")

        # Step 6: Check DVC status
        print("\n6️⃣ Checking DVC status...")
        status = dvc_manager.get_status()
        print(
            f"   DVC Initialized: {'✅' if status.get('dvc_initialized', False) else '❌'}"
        )
        print(
            f"   Remote Configured: {'✅' if status.get('remote_configured', False) else '❌'}"
        )
        print(f"   Data Tracked: {'✅' if status.get('data_tracked', False) else '❌'}")

        # Step 7: List datasets
        print("\n7️⃣ Listing datasets...")
        datasets = pipeline.list_datasets()
        if datasets:
            print(f"📋 Found {len(datasets)} dataset(s):")
            for dataset in datasets:
                print(
                    f"   - {dataset['dataset_name']}: {dataset['total_images']} images, {dataset['total_annotations']} annotations"
                )
                if dataset.get("dvc_info"):
                    dvc_info = dataset["dvc_info"]
                    print(f"     📦 DVC: {dvc_info.get('size_mb', 0):.2f} MB")
        else:
            print("📭 No datasets found")

        # Step 8: Create and run a simple pipeline
        print("\n8️⃣ Creating data pipeline...")
        stages = [
            {
                "name": "import",
                "command": "echo 'Import stage completed'",
                "deps": [],
                "outs": ["data/imported"],
            },
            {
                "name": "process",
                "command": "echo 'Process stage completed'",
                "deps": ["data/imported"],
                "outs": ["data/processed"],
            },
        ]

        if dvc_manager.create_pipeline("demo_pipeline", stages):
            print("✅ Pipeline created successfully")
            print("📋 Pipeline stages:")
            for stage in stages:
                print(f"   - {stage['name']}: {stage['command']}")
        else:
            print("❌ Failed to create pipeline")

        # Step 9: Demonstrate data operations
        print("\n9️⃣ Demonstrating data operations...")

        # Pull data (simulate)
        print("📥 Simulating data pull...")
        print("✅ Data pull simulation completed")

        # Push data (simulate)
        print("📤 Simulating data push...")
        print("✅ Data push simulation completed")

        print("\n🎉 DVC Integration Demo Completed!")
        print("\n📚 Next Steps:")
        print("   - Setup cloud storage (S3, GCS, Azure)")
        print("   - Create custom data pipelines")
        print("   - Integrate with ML frameworks")
        print("   - Explore experiment tracking features")


def demonstrate_cloud_storage_setup():
    """Demonstrate cloud storage setup (informational only)."""

    print("\n🌩️  Cloud Storage Setup Examples")
    print("=" * 40)

    print("\n📦 AWS S3 Setup:")
    print(
        "wildtrain dvc setup --storage-type s3 --storage-path s3://my-bucket/datasets"
    )
    print("export AWS_ACCESS_KEY_ID=your_access_key")
    print("export AWS_SECRET_ACCESS_KEY=your_secret_key")

    print("\n📦 Google Cloud Storage Setup:")
    print(
        "wildtrain dvc setup --storage-type gcs --storage-path gs://my-bucket/datasets"
    )
    print("export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json")

    print("\n📦 Azure Blob Storage Setup:")
    print(
        "wildtrain dvc setup --storage-type azure --storage-path azure://my-container/datasets"
    )
    print("export AZURE_STORAGE_CONNECTION_STRING=your_connection_string")


if __name__ == "__main__":
    try:
        demonstrate_dvc_integration()
        demonstrate_cloud_storage_setup()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
