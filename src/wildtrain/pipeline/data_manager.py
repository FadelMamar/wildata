"""
Data manager for storing and managing datasets in COCO format.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..logging_config import get_logger
from .dvc_manager import DVCConfig, DVCManager, DVCStorageType
from .path_manager import PathManager


class DataManager:
    """
    Manages dataset storage in COCO format with split-based organization.

    This class is responsible for:
    - Storing datasets in COCO format with split-based organization
    - Managing DVC integration for version control
    - Providing dataset information and statistics
    - Supporting dataset operations (list, delete, pull)
    """

    def __init__(
        self,
        path_manager: PathManager,
        enable_dvc: bool = True,
        dvc_config: Optional[DVCConfig] = None,
    ):
        """
        Initialize the data manager.

        Args:
            path_manager: PathManager instance for consistent path resolution
            enable_dvc: Whether to enable DVC integration
            dvc_config: DVC configuration (optional)
        """
        self.path_manager = path_manager
        self.data_dir = path_manager.data_dir

        # Setup logging
        self.logger = get_logger(__name__)

        # Initialize DVC integration
        self.dvc_manager = None
        if enable_dvc:
            try:
                # Use project root from path manager
                project_root = path_manager.project_root
                self.dvc_manager = DVCManager(project_root, dvc_config)
                self.logger.info("DVC integration enabled")
            except Exception as e:
                self.logger.warning(
                    f"DVC integration failed: {e}. Continuing without DVC."
                )

    def store_dataset(
        self,
        dataset_name: str,
        dataset_info: Dict[str, Any],
        split_data: Dict[str, Dict[str, Any]],
        track_with_dvc: bool = False,
    ) -> str:
        """
        Store a dataset in COCO format with split-based organization.

        Args:
            dataset_name: Name of the dataset
            dataset_info: Common dataset metadata (classes, version, etc.)
            split_data: Dictionary mapping split names to COCO format data
            track_with_dvc: Whether to track the dataset with DVC

        Returns:
            Path to the stored dataset info file
        """

        self.path_manager.ensure_directories(dataset_name)

        # Store dataset info
        dataset_info_file = self.path_manager.get_dataset_info_file(dataset_name)
        with open(dataset_info_file, "w") as f:
            json.dump(dataset_info, f, indent=2)

        # Store each split
        for split_name, split_coco_data in split_data.items():
            # Store split annotations
            split_annotations_file = (
                self.path_manager.get_dataset_split_annotations_file(
                    dataset_name, split_name
                )
            )

            # Copy images for this split
            split_coco_data = self._copy_images_for_split(
                dataset_name, split_name, split_coco_data
            )

            with open(split_annotations_file, "w") as f:
                json.dump(split_coco_data, f, indent=2)

        self.logger.info(f"Stored dataset '{dataset_name}' in COCO format")

        # Track with DVC if enabled
        if track_with_dvc and self.dvc_manager:
            try:
                dataset_path = self.path_manager.get_dataset_dir(dataset_name)
                if self.dvc_manager.add_data_to_dvc(dataset_path, dataset_name):
                    self.logger.info(f"Dataset '{dataset_name}' tracked with DVC")
                else:
                    self.logger.warning(
                        f"Failed to track dataset '{dataset_name}' with DVC"
                    )
            except Exception as e:
                self.logger.error(f"Error tracking dataset with DVC: {e}")

        return str(dataset_info_file)

    def _copy_images_for_split(
        self, dataset_name: str, split_name: str, split_coco_data: Dict[str, Any]
    ):
        """Copy images for a specific split to the dataset directory."""
        images = split_coco_data.get("images", [])

        # Create split images directory
        split_images_dir = self.path_manager.get_dataset_split_images_dir(
            dataset_name, split_name
        )
        split_images_dir.mkdir(parents=True, exist_ok=True)

        for image_info in images:
            # Extract original image path
            original_path = image_info.get("file_name", "")
            if not original_path:
                self.logger.warning(
                    f"Image file_name is not set for image {image_info}"
                )
                continue

            # Try to find the image file
            original_file = Path(original_path)
            if not original_file.exists():
                # Try relative to the annotation file directory
                annotation_dir = self.path_manager.get_dataset_annotations_dir(
                    dataset_name
                )
                original_file = annotation_dir.parent / original_path
                if not original_file.exists():
                    self.logger.warning(f"Image file not found: {original_path}")
                    continue

            # Copy image to split directory
            filename = original_file.name
            new_path = split_images_dir / filename

            if not new_path.exists():
                shutil.copy2(original_file, new_path)

            # Update the file_name in COCO data to relative to master data_dir
            image_info["file_name"] = os.path.relpath(
                new_path, start=self.path_manager.data_dir
            )

        return split_coco_data

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a specific dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset information
        """
        dataset_info_file = self.path_manager.get_dataset_info_file(dataset_name)

        if not dataset_info_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")

        with open(dataset_info_file, "r") as f:
            dataset_info = json.load(f)

        # Get split information
        existing_splits = self.path_manager.get_existing_splits(dataset_name)
        images_by_split = {}
        annotations_by_split = {}

        for split in existing_splits:
            split_annotations_file = (
                self.path_manager.get_dataset_split_annotations_file(
                    dataset_name, split
                )
            )
            if split_annotations_file.exists():
                with open(split_annotations_file, "r") as f:
                    split_data = json.load(f)
                    images_by_split[split] = len(split_data.get("images", []))
                    annotations_by_split[split] = len(split_data.get("annotations", []))

        # Get DVC information if available
        dvc_info = {}
        if self.dvc_manager:
            try:
                dvc_status = self.dvc_manager.get_status()
                dvc_info = {
                    "dvc_enabled": True,
                    "dvc_status": dvc_status,
                }
            except Exception as e:
                dvc_info = {
                    "dvc_enabled": True,
                    "dvc_error": str(e),
                }
        else:
            dvc_info = {"dvc_enabled": False}

        return {
            "dataset_name": dataset_name,
            "dataset_info": dataset_info,
            "splits": existing_splits,
            "images_by_split": images_by_split,
            "annotations_by_split": annotations_by_split,
            "total_images": sum(images_by_split.values()),
            "total_annotations": sum(annotations_by_split.values()),
            "dvc_info": dvc_info,
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets in data storage.

        Returns:
            List of dataset information dictionaries
        """
        datasets = []

        # Use PathManager to list datasets
        for dataset_name in self.path_manager.list_datasets():
            try:
                dataset_info = self.get_dataset_info(dataset_name)
                datasets.append(dataset_info)
            except Exception as e:
                self.logger.warning(f"Error reading dataset {dataset_name}: {str(e)}")

        return datasets

    def load_dataset_data(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load all data for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary with dataset info and split data
        """
        dataset_info_file = self.path_manager.get_dataset_info_file(dataset_name)

        if not dataset_info_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")

        with open(dataset_info_file, "r") as f:
            dataset_info = json.load(f)

        # Load all split data
        split_data = {}
        existing_splits = self.path_manager.get_existing_splits(dataset_name)

        for split in existing_splits:
            split_annotations_file = (
                self.path_manager.get_dataset_split_annotations_file(
                    dataset_name, split
                )
            )
            if split_annotations_file.exists():
                with open(split_annotations_file, "r") as f:
                    split_data[split] = json.load(f)

        return {"dataset_info": dataset_info, "split_data": split_data}

    def delete_dataset(self, dataset_name: str, remove_from_dvc: bool = True) -> bool:
        """
        Delete a dataset from data storage.

        Args:
            dataset_name: Name of the dataset to delete
            remove_from_dvc: Whether to remove from DVC tracking

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            dataset_dir = self.path_manager.get_dataset_dir(dataset_name)
            if not dataset_dir.exists():
                self.logger.warning(f"Dataset '{dataset_name}' not found")
                return False

            # Remove from DVC if enabled
            if remove_from_dvc and self.dvc_manager:
                try:
                    # Note: This method may not exist in DVCManager, handle gracefully
                    if hasattr(self.dvc_manager, "remove_data_from_dvc"):
                        self.dvc_manager.remove_data_from_dvc(dataset_name)
                        self.logger.info(f"Removed dataset '{dataset_name}' from DVC")
                    else:
                        self.logger.warning(
                            "DVC remove_data_from_dvc method not available"
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to remove dataset from DVC: {e}")

            # Remove dataset directory
            shutil.rmtree(dataset_dir)
            self.logger.info(f"Deleted dataset '{dataset_name}'")

            return True

        except Exception as e:
            self.logger.error(f"Error deleting dataset '{dataset_name}': {e}")
            return False

    def pull_dataset(self, dataset_name: str) -> bool:
        """
        Pull a dataset from DVC remote storage.

        Args:
            dataset_name: Name of the dataset to pull

        Returns:
            True if pull successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.pull_data(dataset_name)
        except Exception as e:
            self.logger.error(f"Error pulling dataset '{dataset_name}': {e}")
            return False

    def setup_remote_storage(
        self, storage_type: DVCStorageType, storage_path: str, force: bool = False
    ) -> bool:
        """
        Setup remote storage for DVC.

        Args:
            storage_type: Type of storage (local, s3, gcs, etc.)
            storage_path: Path to remote storage
            force: Whether to force setup even if already configured

        Returns:
            True if setup successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.setup_remote_storage(
                storage_type, storage_path, force
            )
        except Exception as e:
            self.logger.error(f"Error setting up remote storage: {e}")
            return False

    def get_dvc_status(self) -> Dict[str, Any]:
        """
        Get DVC status information.

        Returns:
            Dictionary with DVC status information
        """
        if not self.dvc_manager:
            return {"dvc_enabled": False}

        try:
            return {
                "dvc_enabled": True,
                "status": self.dvc_manager.get_status(),
                "config": self.dvc_manager.get_config(),
            }
        except Exception as e:
            return {
                "dvc_enabled": True,
                "error": str(e),
            }

    def create_data_pipeline(
        self, pipeline_name: str, stages: List[Dict[str, Any]]
    ) -> bool:
        """
        Create a data pipeline configuration.

        Args:
            pipeline_name: Name of the pipeline
            stages: List of pipeline stages

        Returns:
            True if creation successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.create_pipeline(pipeline_name, stages)
        except Exception as e:
            self.logger.error(f"Error creating data pipeline: {e}")
            return False

    def run_data_pipeline(self, pipeline_name: str) -> bool:
        """
        Run a data pipeline.

        Args:
            pipeline_name: Name of the pipeline to run

        Returns:
            True if pipeline execution successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.run_pipeline(pipeline_name)
        except Exception as e:
            self.logger.error(f"Error running data pipeline: {e}")
            return False
