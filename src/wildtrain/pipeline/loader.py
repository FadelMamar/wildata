import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..converters.yolo_to_master import YOLOToMasterConverter
from ..validators.coco_validator import COCOValidator
from ..validators.yolo_validator import YOLOValidator


class Loader:
    def __init__(self):
        self.split_name = "NOT_SET"

    def _load_json(self, annotation_path: str) -> Dict[str, Any]:
        with open(annotation_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        bbox_tolerance: int,
        split_name: str,
    ):
        if split_name not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split name: {split_name}")

        self.split_name = split_name

        dataset_info, split_data = self._load_and_validate_dataset(
            source_path=source_path,
            source_format=source_format,
            dataset_name=dataset_name,
            bbox_tolerance=bbox_tolerance,
        )

        return dataset_info, split_data

    def _load_coco_to_split_format(
        self,
        coco_data: Dict[str, Any],
        dataset_name: str,
        image_dir: Path,
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Load COCO annotation file and convert to split-based format.

        Args:
            coco_annotation_path: Path to COCO annotation file
            dataset_name: Name of the dataset

        Returns:
            Tuple of (dataset_info, split_data)
        """
        # Extract dataset info
        dataset_info = {
            "name": dataset_name,
            "version": "1.0",
            "schema_version": "1.0",
            "task_type": "detection",
            "classes": coco_data.get("categories", []),
        }

        # Group images and annotations by split
        split_data = {}
        images_by_split = {}
        annotations_by_split = {}

        # Determine split for each image (simple logic - can be improved)
        for image in coco_data.get("images", []):
            path = image_dir / self.split_name / Path(image["file_name"]).name
            image["file_name"] = str(Path(path).resolve())
            if self.split_name not in images_by_split:
                images_by_split[self.split_name] = []
                annotations_by_split[self.split_name] = []
            images_by_split[self.split_name].append(image)

        # Group annotations by split
        for annotation in coco_data.get("annotations", []):
            image_id = annotation["image_id"]
            # Find which split this annotation belongs to
            for split, images in images_by_split.items():
                if any(img["id"] == image_id for img in images):
                    annotations_by_split[split].append(annotation)
                    break

        # Create split data
        for split in images_by_split.keys():
            split_data[split] = {
                "images": images_by_split[split],
                "annotations": annotations_by_split[split],
                "categories": coco_data.get("categories", []),
            }

        return dataset_info, split_data

    def _load_and_validate_dataset(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        bbox_tolerance: int,
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Dict[str, Any]]]]:
        """
        Load and validate dataset based on source format.

        Returns:
            Tuple of (dataset_info, split_data) or (None, None) if validation fails
        """
        if source_format == "coco":
            # For COCO, source_path should be the annotation file path
            validator = COCOValidator(source_path)
            is_valid, errors, warnings = validator.validate(
                bbox_tolerance=bbox_tolerance
            )
            if not is_valid:
                print("[DEBUG] COCO validation failed")
                return None, None

            # Load COCO data and convert to split-based structure
            coco_data = self._load_json(source_path)
            image_dir = Path(source_path).parents[1] / "images"

            if not image_dir.exists():
                raise FileNotFoundError(f"The expected format {image_dir}")

            dataset_info, split_data = self._load_coco_to_split_format(
                coco_data=coco_data,
                dataset_name=dataset_name,
                image_dir=image_dir,
            )

        elif source_format == "yolo":
            # For YOLO, source_path should be the data.yaml file path
            validator = YOLOValidator(source_path)
            is_valid, errors, warnings = validator.validate()
            if not is_valid:
                print("[DEBUG] YOLO validation failed")
                return None, None

            # Create converter and convert YOLO to COCO format
            converter = YOLOToMasterConverter(source_path)
            converter.load_yolo_data()
            dataset_info, split_data = converter.convert(dataset_name)
        else:
            raise ValueError(f"Unsupported source format: {source_format}")

        return dataset_info, split_data
