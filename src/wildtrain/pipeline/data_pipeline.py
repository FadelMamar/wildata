"""
Data pipeline for importing, transforming, and exporting datasets.
"""

import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from tqdm import tqdm

from ..converters.yolo_to_master import YOLOToMasterConverter
from ..logging_config import get_logger
from ..transformations.transformation_pipeline import TransformationPipeline
from ..validators.coco_validator import COCOValidator
from ..validators.yolo_validator import YOLOValidator
from .data_manager import DataManager
from .framework_data_manager import FrameworkDataManager
from .path_manager import PathManager


class DataPipeline:
    """
    High-level data pipeline for managing dataset operations.

    This class orchestrates the complete data workflow:
    - Import datasets from various formats (COCO, YOLO)
    - Apply transformations and augmentations
    - Store data in COCO format with split-based organization
    - Export to framework-specific formats
    - Manage DVC integration for version control
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
            root: Root directory for data storage
            transformation_pipeline: Optional transformation pipeline
            enable_dvc: Whether to enable DVC integration
        """
        self.root = Path(root)
        self.logger = get_logger(self.__class__.__name__)

        # Initialize path manager for consistent path resolution
        self.path_manager = PathManager(self.root)

        # Initialize transformation pipeline
        self.transformation_pipeline = (
            transformation_pipeline or TransformationPipeline()
        )

        # Initialize data manager with DVC support
        self.data_manager = DataManager(self.path_manager, enable_dvc=enable_dvc)

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
        track_with_dvc: bool = False,
        bbox_tolerance: int = 5,
    ) -> Dict[str, Any]:
        """
        Import a dataset from source format to COCO format.

        Args:
            source_path: Path to source dataset
            source_format: Format of source dataset ('coco' or 'yolo')
            dataset_name: Name for the dataset in COCO format
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
                is_valid, errors, warnings = validator.validate(
                    bbox_tolerance=bbox_tolerance
                )
                if not is_valid:
                    print("[DEBUG] COCO validation failed")
                    return {
                        "success": False,
                        "error": "Validation failed",
                        "validation_errors": errors,
                        "hints": warnings,
                    }

                # For COCO, we can store directly in COCO format
                # Load COCO data and convert to split-based structure
                dataset_info, split_data = self._load_coco_to_split_format(
                    source_path, dataset_name
                )

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

                # Create converter and convert YOLO to COCO format
                converter = YOLOToMasterConverter(source_path)
                converter.load_yolo_data()
                dataset_info, split_data = converter.convert(dataset_name)
            else:
                raise ValueError(f"Unsupported source format: {source_format}")

            # Apply transformations if requested
            if apply_transformations:
                print("[DEBUG] Applying transformations")
                split_data = self._apply_transformations_to_dataset(split_data)

            # Store dataset using data manager
            print("[DEBUG] Storing dataset with data manager")
            dataset_info_path = self.data_manager.store_dataset(
                dataset_name, dataset_info, split_data, track_with_dvc=track_with_dvc
            )

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
                "dataset_info_path": dataset_info_path,
                "framework_paths": framework_paths,
                "dvc_tracked": track_with_dvc
                and self.data_manager.dvc_manager is not None,
            }

        except Exception as e:
            self.logger.error(f"Error importing dataset: {str(e)}")
            print(f"[DEBUG] Exception in import_dataset: {traceback.format_exc()}")
            return {
                "success": False,
                "error": str(e),
                "validation_errors": [],
                "hints": [],
            }

    def _load_coco_to_split_format(
        self, coco_annotation_path: str, dataset_name: str
    ) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Load COCO annotation file and convert to split-based format.

        Args:
            coco_annotation_path: Path to COCO annotation file
            dataset_name: Name of the dataset

        Returns:
            Tuple of (dataset_info, split_data)
        """

        with open(coco_annotation_path, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        image_dir = Path(coco_annotation_path).parents[1] / "images"

        assert image_dir.exists(), f"The expected format "

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
            split = self._determine_split_from_image(image, coco_annotation_path)
            path = image_dir / split / Path(image["file_name"]).name
            image["file_name"] = str(Path(path).resolve())
            if split not in images_by_split:
                images_by_split[split] = []
                annotations_by_split[split] = []
            images_by_split[split].append(image)

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

    def _determine_split_from_image(
        self, image: Dict[str, Any], annotation_path: str
    ) -> str:
        """
        Determine split for an image based on file path or annotation path.

        Args:
            image: COCO image object
            annotation_path: Path to annotation file

        Returns:
            Split name (train, val, test)
        """
        file_name = image.get("file_name", "").lower()
        annotation_path_lower = annotation_path.lower()

        if (
            "val" in file_name
            or "validation" in file_name
            or "val" in annotation_path_lower
        ):
            return "val"
        elif "test" in file_name or "test" in annotation_path_lower:
            return "test"
        else:
            return "train"

    def _apply_transformations_to_dataset(
        self, split_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply transformations to all images and annotations in the dataset.

        Args:
            split_data: Dictionary mapping split names to COCO format data

        Returns:
            Transformed split data
        """
        transformed_split_data = {}

        for split_name, split_coco_data in tqdm(
            split_data.items(), desc="Applying transformations to dataset"
        ):
            transformed_images = []
            transformed_annotations = []

            for image_info in split_coco_data["images"]:
                # Load image
                image_path = Path(image_info["file_name"])
                if not Path(image_path).exists():
                    self.logger.warning(f"Could not load image: {image_path}")
                    continue
                # Get annotations for this image
                image_annotations = [
                    ann
                    for ann in split_coco_data["annotations"]
                    if ann["image_id"] == image_info["id"]
                ]
                image = cv2.imread(str(image_path))

                # Apply transformations
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

                except Exception as e:
                    print(
                        f"Error transforming image {image_info['file_name']}: {str(e)}"
                    )
                    continue

            # Create transformed split data
            transformed_split_data[split_name] = {
                "images": transformed_images,
                "annotations": transformed_annotations,
                "categories": split_coco_data.get("categories", []),
            }

        return transformed_split_data

    def export_dataset(
        self, dataset_name: str, target_format: str, target_path: str
    ) -> bool:
        """
        Export a dataset from COCO format to target format.

        Args:
            dataset_name: Name of the dataset in COCO format
            target_format: Target format ('coco' or 'yolo')
            target_path: Path where to export the dataset

        Returns:
            True if export successful, False otherwise
        """
        try:
            # Load dataset data
            dataset_data = self.data_manager.load_dataset_data(dataset_name)

            if target_format == "coco":
                # Export COCO format
                return self._export_coco_format(dataset_data, target_path)
            elif target_format == "yolo":
                # Export YOLO format
                return self._export_yolo_format(dataset_data, target_path)
            else:
                self.logger.error(f"Unsupported target format: {target_format}")
                return False

        except Exception as e:
            self.logger.error(f"Error exporting dataset: {str(e)}")
            return False

    def _export_coco_format(
        self, dataset_data: Dict[str, Any], target_path: str
    ) -> bool:
        """Export dataset to COCO format."""
        try:
            import json

            target_path_obj = Path(target_path)
            target_path_obj.mkdir(parents=True, exist_ok=True)

            # Export each split
            for split_name, split_data in dataset_data["split_data"].items():
                split_file = target_path_obj / f"{split_name}.json"
                with open(split_file, "w") as f:
                    json.dump(split_data, f, indent=2)

            # Export dataset info
            info_file = target_path_obj / "dataset_info.json"
            with open(info_file, "w") as f:
                json.dump(dataset_data["dataset_info"], f, indent=2)

            return True
        except Exception as e:
            self.logger.error(f"Error exporting COCO format: {str(e)}")
            return False

    def _export_yolo_format(
        self, dataset_data: Dict[str, Any], target_path: str
    ) -> bool:
        """Export dataset to YOLO format."""
        try:
            # Use framework data manager to convert COCO to YOLO
            result = self.framework_data_manager.export_framework_format(
                dataset_data["dataset_info"]["name"], "yolo"
            )
            return "path" in result
        except Exception as e:
            self.logger.error(f"Error exporting YOLO format: {str(e)}")
            return False

    def add_transformation(self, transformer) -> None:
        """
        Add a transformation to the pipeline.

        Args:
            transformer: Transformation to add
        """
        if self.transformation_pipeline:
            self.transformation_pipeline.add_transformer(transformer)

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the current status of the data pipeline.

        Returns:
            Dictionary with pipeline status information
        """
        return {
            "root_directory": str(self.root),
            "dvc_enabled": self.data_manager.dvc_manager is not None,
            "transformation_pipeline": self.transformation_pipeline.get_pipeline_info()
            if self.transformation_pipeline
            else None,
            "datasets": self.list_datasets(),
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.

        Returns:
            List of dataset information dictionaries
        """
        return self.data_manager.list_datasets()

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        return self.data_manager.get_dataset_info(dataset_name)

    def delete_dataset(self, dataset_name: str, remove_from_dvc: bool = True) -> bool:
        """
        Delete a dataset.

        Args:
            dataset_name: Name of the dataset to delete
            remove_from_dvc: Whether to remove from DVC tracking

        Returns:
            True if deletion successful, False otherwise
        """
        return self.data_manager.delete_dataset(dataset_name, remove_from_dvc)

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
