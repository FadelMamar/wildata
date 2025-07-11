import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..validators.coco_validator import COCOValidator
from .base_converter import BaseConverter


class COCOToMasterConverter(BaseConverter):
    """
    Converter from COCO format to master annotation format.
    """

    def __init__(self, coco_annotation_path: str):
        """
        Initialize the converter with the path to the COCO annotation file.
        """
        self.coco_annotation_path = coco_annotation_path
        self.coco_images_path = None
        self.coco_data: Dict[str, Any] = {}

    def load_coco_annotation(self, filter_invalid_annotations: bool = False) -> None:
        """
        Load the COCO annotation JSON file into memory.
        Args:
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        """
        # Validate before loading
        validator = COCOValidator(
            self.coco_annotation_path,
            filter_invalid_annotations=filter_invalid_annotations,
        )
        is_valid, errors, warnings = validator.validate()

        if not is_valid:
            error_msg = f"COCO validation failed for {self.coco_annotation_path}:\n"
            error_msg += "\n".join(errors)
            if warnings:
                error_msg += f"\nWarnings:\n" + "\n".join(warnings)
            raise ValueError(error_msg)

        # Use the filtered data from validator if filtering was enabled
        if filter_invalid_annotations:
            self.coco_data = validator.coco_data
            skipped_count = validator.get_skipped_count()
            if skipped_count > 0:
                print(
                    f"Warning: Skipped {skipped_count} invalid annotations during COCO validation"
                )
        else:
            # Load the data normally
            with open(self.coco_annotation_path, "r", encoding="utf-8") as f:
                self.coco_data = json.load(f)

        self.coco_images_path = Path(self.coco_annotation_path).parent / "images"

    def convert_to_master(
        self,
        dataset_name: str,
        version: str = "1.0",
        task_type: str = "detection",
        validate_output: bool = True,
        filter_invalid_annotations: bool = False,
    ) -> Dict[str, Any]:
        """
        Convert the loaded COCO annotation to master format.
        Args:
            dataset_name (str): Name of the dataset.
            version (str): Version of the dataset.
            task_type (str): Type of task (detection, segmentation, keypoints).
            validate_output (bool): Whether to validate the output master annotation.
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        Returns:
            Dict[str, Any]: The master annotation dictionary.
        """
        # Extract categories
        categories = self.coco_data.get("categories", [])
        classes = [
            {
                "id": cat["id"],
                "name": cat["name"],
                "supercategory": cat.get("supercategory", ""),
            }
            for cat in categories
        ]

        # Extract images
        images = []
        for img in self.coco_data.get("images", []):
            # Determine split based on file path or other logic
            # For now, assume all images are in 'train' split
            split = self._determine_split(img)
            file_name = os.path.basename(img["file_name"])
            master_image = {
                "id": img["id"],
                "file_name": file_name,
                "width": img["width"],
                "height": img["height"],
                "split": split,
                "path": os.path.join(self.coco_images_path, file_name),
            }
            images.append(master_image)

        # Extract annotations
        annotations = []
        for ann in self.coco_data.get("annotations", []):
            master_annotation = {
                "id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "bbox": ann.get("bbox", []),
                "area": ann.get("area", 0),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
                "keypoints": ann.get("keypoints", []),
                "attributes": ann.get("attributes", {}),
            }
            annotations.append(master_annotation)

        # Create master annotation structure
        master_annotation = {
            "dataset_info": {
                "name": dataset_name,
                "version": version,
                "schema_version": "1.0",
                "task_type": task_type,
                "classes": classes,
            },
            "images": images,
            "annotations": annotations,
        }

        # Validate the output if requested
        if validate_output:
            self._validate_master_annotation(
                master_annotation, filter_invalid_annotations
            )

        return master_annotation

    def _determine_split(self, image: Dict[str, Any]) -> str:
        """
        Determine the split for an image based on file path or other logic.
        Args:
            image (Dict[str, Any]): COCO image object.
        Returns:
            str: Split name (train, val, test).
        """
        # Simple logic: check if file path contains split information
        file_name = image.get("file_name", "")
        if "val" in file_name.lower() or "validation" in file_name.lower():
            return "val"
        elif "test" in file_name.lower():
            return "test"
        else:
            return "train"  # Default to train
