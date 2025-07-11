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
        # for export_format in ["coco", "yolo"]:
        #     try:
        #         export_result = pipeline.export_framework_format(
        #             dataset_name, export_format
        #         )
        #         print(
        #             f"[SUCCESS] Exported to {export_format.upper()}: {export_result['output_path']}"
        #         )
        #     except Exception as e:
        #         print(f"[WARNING] Failed to export to {export_format.upper()}: {e}")

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


if __name__ == "__main__":
    data_dir = r"D:\workspace\data\MyNewData"
    pipeline = DataPipeline(data_dir)

    data_info = {
        "path": Path(r"D:/workspace/savmap/coco/annotations.json"),
        "description": "COCO format annotation file",
    }
    dataset_name = "savmap"
    format_type = "coco"

    process_real_dataset(pipeline, data_info, dataset_name, format_type)
