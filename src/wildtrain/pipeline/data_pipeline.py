"""
Data pipeline for managing deep learning datasets with transformations.
"""

import json
import logging
import os
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yaml

from ..adapters.coco_adapter import COCOAdapter
from ..adapters.yolo_adapter import YOLOAdapter
from ..converters.coco_to_master import COCOToMasterConverter
from ..converters.yolo_to_master import YOLOToMasterConverter
from ..transformations import TransformationPipeline
from ..validators.coco_validator import COCOValidator
from ..validators.yolo_validator import YOLOValidator
from .framework_data_manager import FrameworkDataManager
from .master_data_manager import MasterDataManager
from .path_manager import PathManager


class DataPipeline:
    """
    Main data pipeline for managing deep learning datasets.

    This pipeline integrates:
    - Data validation
    - Format conversion
    - Data transformations (augmentation, tiling)
    - Framework-specific format generation
    - DVC data versioning

    Responsibilities:
    - High-level orchestration of dataset operations
    - Integration with MasterDataManager and FrameworkDataManager
    - Transformation pipeline management
    - Dataset import/export workflow coordination
    """

    def __init__(
        self,
        root: str,
        transformation_pipeline: Optional[TransformationPipeline] = None,
        enable_dvc: bool = True,
    ):
        """
        Initialize the data pipeline.

        Args:
            root_data_directory: Root directory of the project
            transformation_pipeline: Optional transformation pipeline
            enable_dvc: Whether to enable DVC integration
        """
        self.root = Path(root)

        # Initialize path manager for consistent path resolution
        self.path_manager = PathManager(self.root)

        # Initialize transformation pipeline
        self.transformation_pipeline = (
            transformation_pipeline or TransformationPipeline()
        )
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize master data manager with DVC support
        self.master_data_manager = MasterDataManager(
            self.path_manager, enable_dvc=enable_dvc
        )

        # Initialize framework data manager
        self.framework_data_manager = FrameworkDataManager(self.path_manager)

        # Don't pre-instantiate converters and validators - create on demand
        self.adapters = {}

    def import_dataset(
        self,
        source_path: str,
        source_format: str,
        dataset_name: str,
        apply_transformations: bool = False,
        track_with_dvc: bool = True,
    ) -> Dict[str, Any]:
        """
        Import a dataset from source format to master format.

        Args:
            source_path: Path to source dataset
            source_format: Format of source dataset ('coco' or 'yolo')
            dataset_name: Name for the dataset in master format
            apply_transformations: Whether to apply transformations during import
            track_with_dvc: Whether to track the dataset with DVC

        Returns:
            Dictionary with import result information
        """
        print(
            f"[DEBUG] Starting import_dataset: {source_path}, {source_format}, {dataset_name}"
        )
        try:
            self.logger.info(
                f"Importing dataset from {source_path} ({source_format} format)"
            )

            # Validate source format
            if source_format not in ["coco", "yolo"]:
                print("[DEBUG] Unsupported source format")
                return {
                    "success": False,
                    "error": f"Unsupported source format: {source_format}",
                    "validation_errors": [],
                    "hints": ["Supported formats: coco, yolo"],
                }

            # Create validator and validate dataset
            if source_format == "coco":
                # For COCO, source_path should be the annotation file path
                validator = COCOValidator(source_path)
                is_valid, errors, warnings = validator.validate()
                if not is_valid:
                    print("[DEBUG] COCO validation failed")
                    return {
                        "success": False,
                        "error": "Validation failed",
                        "validation_errors": errors,
                        "hints": warnings,
                    }

                # Create converter and convert
                converter = COCOToMasterConverter(source_path)
                converter.load_coco_annotation()
                master_data = converter.convert_to_master(dataset_name)

            elif source_format == "yolo":
                # For YOLO, source_path should be the data.yaml file path
                validator = YOLOValidator(source_path)
                is_valid, errors, warnings = validator.validate()
                if not is_valid:
                    print("[DEBUG] YOLO validation failed")
                    return {
                        "success": False,
                        "error": "Validation failed",
                        "validation_errors": errors,
                        "hints": warnings,
                    }

                # Create converter and convert
                converter = YOLOToMasterConverter(source_path)
                converter.load_yolo_data()
                master_data = converter.convert_to_master(dataset_name)

            # Apply transformations if requested
            if apply_transformations and self.transformation_pipeline:
                print("[DEBUG] Applying transformations")
                master_data = self._apply_transformations_to_dataset(master_data)

            # Store dataset using master data manager
            print("[DEBUG] Storing dataset with master data manager")
            try:
                master_annotations_path = self.master_data_manager.store_dataset(
                    dataset_name, master_data, track_with_dvc=track_with_dvc
                )
            except Exception as e:
                self.logger.error(f"Error storing dataset: {traceback.format_exc()}")
                return {
                    "success": False,
                    "error": str(e),
                    "validation_errors": [],
                    "hints": [],
                }

            # Create framework formats using framework data manager
            print("[DEBUG] Creating framework formats")
            framework_paths = self.framework_data_manager.create_framework_formats(
                dataset_name
            )

            self.logger.info(f"Successfully imported dataset '{dataset_name}'")
            print(f"[DEBUG] import_dataset completed successfully for {dataset_name}")
            return {
                "success": True,
                "dataset_name": dataset_name,
                "master_path": master_annotations_path,
                "framework_paths": framework_paths,
                "dvc_tracked": track_with_dvc
                and self.master_data_manager.dvc_manager is not None,
            }

        except Exception as e:
            self.logger.error(f"Error importing dataset: {str(e)}")
            print(f"[DEBUG] Exception in import_dataset: {e}")
            return {
                "success": False,
                "error": str(e),
                "validation_errors": [],
                "hints": [],
            }

    def _apply_transformations_to_dataset(
        self, master_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply transformations to all images and annotations in the dataset.

        Args:
            master_data: Master format dataset data

        Returns:
            Transformed master data
        """
        transformed_images = []
        transformed_annotations = []
        transformed_image_info = []

        for image_info in master_data["images"]:
            # Load image
            image_path = Path(image_info["path"])
            if not image_path.is_absolute():
                image_path = self.path_manager.master_data_dir / image_path

            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                continue

            # Get annotations for this image
            image_annotations = [
                ann
                for ann in master_data["annotations"]
                if ann["image_id"] == image_info["id"]
            ]

            # Apply transformations with correct interface
            try:
                inputs = {
                    "image": image,
                    "annotations": image_annotations,
                    "info": image_info,
                }

                transformed_data = self.transformation_pipeline.transform(inputs)

                for data in transformed_data:
                    transformed_images.extend(data["image"])
                    transformed_annotations.extend(data.get("annotations", []))
                    transformed_image_info.extend(data["info"])

            except Exception as e:
                self.logger.error(
                    f"Error transforming image {image_info['path']}: {str(e)}"
                )
                continue

        return {
            "images": transformed_images,
            "annotations": transformed_annotations,
            "categories": master_data.get("categories", []),
            "info": transformed_image_info,
        }

    def export_dataset(
        self, dataset_name: str, target_format: str, target_path: str
    ) -> bool:
        """
        Export a dataset from master format to target format.

        Args:
            dataset_name: Name of the dataset in master format
            target_format: Target format ('coco' or 'yolo')
            target_path: Path to save exported dataset

        Returns:
            True if export successful, False otherwise
        """
        try:
            self.logger.info(
                f"Exporting dataset '{dataset_name}' to {target_format} format"
            )

            # Check if dataset exists
            if not self.path_manager.dataset_exists(dataset_name):
                self.logger.error(f"Dataset '{dataset_name}' not found")
                return False

            # Get or create adapter
            if target_format not in self.adapters:
                # Create adapter based on target format with proper master annotation path
                master_annotation_path = str(
                    self.path_manager.get_dataset_annotations_file(dataset_name)
                )
                if target_format == "coco":
                    self.adapters[target_format] = COCOAdapter(master_annotation_path)
                elif target_format == "yolo":
                    self.adapters[target_format] = YOLOAdapter(master_annotation_path)
                else:
                    self.logger.error(f"Unsupported target format: {target_format}")
                    return False

            # Convert to target format
            adapter = self.adapters[target_format]
            adapter.load_master_annotation()

            # Convert for each split
            for split in ["train", "val", "test"]:
                try:
                    converted_data = adapter.convert(split)
                    adapter.save(converted_data, target_path)
                except Exception as e:
                    self.logger.warning(f"Could not convert split '{split}': {str(e)}")
                    continue

            self.logger.info(f"Successfully exported dataset to {target_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            return False

    def add_transformation(self, transformer) -> None:
        """
        Add a transformation to the pipeline.

        Args:
            transformer: Transformer to add
        """
        self.transformation_pipeline.add_transformer(transformer)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get status information about the pipeline.

        Returns:
            Dictionary with pipeline status
        """
        return {
            "root": str(self.root),
            "master_data_dir": str(self.path_manager.master_data_dir),
            "transformation_pipeline": self.transformation_pipeline.get_pipeline_info(),
            "supported_formats": ["coco", "yolo"],
            "available_datasets": self.path_manager.list_datasets(),
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets in master storage.

        Returns:
            List of dataset information dictionaries
        """
        datasets = []

        for dataset_name in self.path_manager.list_datasets():
            try:
                dataset_info = self.get_dataset_info(dataset_name)
                datasets.append(dataset_info)
            except Exception as e:
                self.logger.warning(f"Error reading dataset {dataset_name}: {str(e)}")

        return datasets

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        annotations_file = self.path_manager.get_dataset_annotations_file(dataset_name)

        if not annotations_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")

        with open(annotations_file, "r") as f:
            master_data = json.load(f)

        # Count images and annotations
        total_images = len(master_data.get("images", []))
        total_annotations = len(master_data.get("annotations", []))

        # Count images by split
        images_by_split = {}
        for image in master_data.get("images", []):
            split = image.get("split", "unknown")
            images_by_split[split] = images_by_split.get(split, 0) + 1

        # Count annotations by type
        annotations_by_type = {}
        for ann in master_data.get("annotations", []):
            ann_type = "detection"  # Default type
            if "segmentation" in ann and ann["segmentation"]:
                ann_type = "segmentation"
            elif "keypoints" in ann and ann["keypoints"]:
                ann_type = "keypoints"

            annotations_by_type[ann_type] = annotations_by_type.get(ann_type, 0) + 1

        # Get categories
        categories = master_data.get("dataset_info", {}).get("classes", [])

        # Get framework format availability
        framework_formats = self.path_manager.list_framework_formats(dataset_name)

        return {
            "dataset_name": dataset_name,
            "master_annotations_file": str(annotations_file),
            "total_images": total_images,
            "total_annotations": total_annotations,
            "images_by_split": images_by_split,
            "annotations_by_type": annotations_by_type,
            "categories": categories,
            "framework_formats": framework_formats,
        }

    def export_framework_format(
        self, dataset_name: str, framework: str
    ) -> Dict[str, Any]:
        """
        Export a dataset to a specific framework format.

        Args:
            dataset_name: Name of the dataset
            framework: Framework name ('coco' or 'yolo')

        Returns:
            Dictionary with export information
        """
        return self.framework_data_manager.export_framework_format(
            dataset_name, framework
        )

    def list_framework_formats(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        List available framework formats for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of framework format information
        """
        return self.framework_data_manager.list_framework_formats(dataset_name)
