"""
Master data manager for storing and managing data in the master format with DVC integration.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..adapters.coco_adapter import COCOAdapter
from ..adapters.yolo_adapter import YOLOAdapter
from .dvc_manager import DVCConfig, DVCManager, DVCStorageType
from .path_manager import PathManager


class MasterDataManager:
    """
    Manages master data storage and operations with DVC integration.

    Master data structure:
    data/
    ├── images/                    # Master storage (real files)
    │   ├── train/
    │   │   ├── image001.jpg
    │   │   └── image002.jpg
    │   └── val/
    │       ├── image003.jpg
    │       └── image004.jpg
    └── annotations/
        └── master/
            └── annotations.json
    """

    def __init__(
        self,
        path_manager: PathManager,
        enable_dvc: bool = True,
        dvc_config: Optional[DVCConfig] = None,
    ):
        """
        Initialize the master data manager.

        Args:
            path_manager: PathManager instance for consistent path resolution
            enable_dvc: Whether to enable DVC integration
            dvc_config: DVC configuration (optional)
        """
        self.path_manager = path_manager
        self.master_data_dir = path_manager.master_data_dir

        # Setup logging
        self.logger = logging.getLogger(__name__)

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
        master_data: Dict[str, Any],
        track_with_dvc: bool = False,
    ) -> str:
        """
        Store a dataset in master format with optional DVC tracking.

        Args:
            dataset_name: Name of the dataset
            master_data: Master format data dictionary
            track_with_dvc: Whether to track the dataset with DVC

        Returns:
            Path to the stored master annotations file
        """

        self.path_manager.ensure_directories(dataset_name=dataset_name)

        # Copy images to master storage
        self._copy_images_to_master(dataset_name, master_data)

        # Store master annotations using PathManager
        annotations_file = self.path_manager.get_dataset_annotations_file(dataset_name)

        with open(annotations_file, "w") as f:
            json.dump(master_data, f, indent=2)

        self.logger.info(f"Stored dataset '{dataset_name}' in master format")

        # Track with DVC if enabled
        if track_with_dvc and self.dvc_manager:
            try:
                dataset_path = self.path_manager.get_dataset_master_dir(dataset_name)
                if self.dvc_manager.add_data_to_dvc(dataset_path, dataset_name):
                    self.logger.info(f"Dataset '{dataset_name}' tracked with DVC")
                else:
                    self.logger.warning(
                        f"Failed to track dataset '{dataset_name}' with DVC"
                    )
            except Exception as e:
                self.logger.error(f"Error tracking dataset with DVC: {e}")

        return str(annotations_file)

    def _copy_images_to_master(self, dataset_name: str, master_data: Dict[str, Any]):
        """Copy images to master storage and update paths."""
        images = master_data.get("images", [])

        for image_info in images:
            # Extract original image path
            original_path = image_info.get("path", "")
            if not original_path:
                raise ValueError(f"Image path is not set for image {image_info}")

            # Determine split (train/val/test)
            split = image_info.get("split")
            if not split:
                raise ValueError(f"Split is not set for image {image_info}")

            # Use PathManager to get split directory
            split_dir = self.path_manager.get_master_split_images_dir(
                dataset_name, split
            )
            split_dir.mkdir(parents=False, exist_ok=True)

            # Copy image to master storage
            original_file = Path(original_path)
            if original_file.exists():
                filename = original_file.name
                new_path = split_dir / filename

                # Copy file if it doesn't exist or is different
                if not new_path.exists():
                    shutil.copy2(original_file, new_path)

                # get relative path to root data dir
                image_info["path"] = self.path_manager.get_relative_path(
                    new_path, start=self.path_manager.project_root
                )
            else:
                self.logger.warning(f"Image file not found: {original_path}")

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get information about a specific dataset.

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

        # Count images by split
        images_by_split = {}
        for image in master_data.get("images", []):
            split = image.get("split", "train")
            if split not in images_by_split:
                images_by_split[split] = 0
            images_by_split[split] += 1

        # Count annotations by type
        annotations_by_type = {}
        for annotation in master_data.get("annotations", []):
            annotation_type = annotation.get("type", "unknown")
            if annotation_type not in annotations_by_type:
                annotations_by_type[annotation_type] = 0
            annotations_by_type[annotation_type] += 1

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
            "total_images": len(master_data.get("images", [])),
            "total_annotations": len(master_data.get("annotations", [])),
            "images_by_split": images_by_split,
            "annotations_by_type": annotations_by_type,
            "dvc_info": dvc_info,
        }

    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets in master storage.

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

    def load_master_data(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load master data for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Master data dictionary
        """
        annotations_file = self.path_manager.get_dataset_annotations_file(dataset_name)

        if not annotations_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")

        with open(annotations_file, "r") as f:
            return json.load(f)

    def delete_dataset(self, dataset_name: str, remove_from_dvc: bool = True) -> bool:
        """
        Delete a dataset from master storage.

        Args:
            dataset_name: Name of the dataset to delete
            remove_from_dvc: Whether to remove from DVC tracking

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            dataset_dir = self.path_manager.get_dataset_master_dir(dataset_name)
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
            storage_path: Path to storage location
            force: Whether to force setup even if already configured

        Returns:
            True if setup successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            # Note: This method signature may need to be adjusted based on DVCManager implementation
            if hasattr(self.dvc_manager, "setup_remote_storage"):
                return self.dvc_manager.setup_remote_storage(
                    storage_type, storage_path, force
                )
            else:
                self.logger.warning("DVC setup_remote_storage method not available")
                return False
        except Exception as e:
            self.logger.error(f"Error setting up remote storage: {e}")
            return False

    def get_dvc_status(self) -> Dict[str, Any]:
        """
        Get DVC status information.

        Returns:
            Dictionary with DVC status
        """
        if not self.dvc_manager:
            return {"dvc_enabled": False}

        try:
            status_info = {
                "dvc_enabled": True,
                "status": self.dvc_manager.get_status(),
            }

            # Add config if available
            if hasattr(self.dvc_manager, "get_config"):
                status_info["config"] = self.dvc_manager.get_config()

            return status_info
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
            True if execution successful, False otherwise
        """
        if not self.dvc_manager:
            self.logger.error("DVC not enabled")
            return False

        try:
            return self.dvc_manager.run_pipeline(pipeline_name)
        except Exception as e:
            self.logger.error(f"Error running data pipeline: {e}")
            return False
