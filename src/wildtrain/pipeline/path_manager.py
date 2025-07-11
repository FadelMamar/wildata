"""
Centralized path management for the data pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class PathManager:
    """
    Centralized path management for the data pipeline.

    Provides consistent path resolution and eliminates hardcoded paths.
    """

    def __init__(self, root_data_directory: Path):
        """
        Initialize the path manager.
        """
        self.project_root = Path(root_data_directory)
        self.logger = logging.getLogger(__name__)

        # Define standard directory structure
        self._setup_directory_structure()

    def _setup_directory_structure(self):
        """Setup the standard directory structure."""
        # Master data storage
        self.master_data_dir = self.project_root / "master"

        # Framework formats
        self.framework_formats_dir = self.project_root / "framework_formats"
        # self.coco_formats_dir = self.framework_formats_dir / "coco"
        # self.yolo_formats_dir = self.framework_formats_dir / "yolo"

        # DVC and configuration
        self.dvc_dir = self.project_root / ".dvc"
        self.config_dir = self.project_root / "config"

    def get_dataset_master_dir(self, dataset_name: str) -> Path:
        """Get the master data directory for a specific dataset."""
        return self.master_data_dir / dataset_name

    def get_dataset_images_dir(self, dataset_name: str) -> Path:
        """Get the images directory for a specific dataset."""
        return self.get_dataset_master_dir(dataset_name) / "images"

    def get_dataset_annotations_file(self, dataset_name: str) -> Path:
        """Get the annotations file path for a specific dataset."""
        return self.get_dataset_master_dir(dataset_name) / "annotations.json"

    def get_framework_format_dir(self, dataset_name: str, framework: str) -> Path:
        """Get the framework format directory for a dataset."""
        if framework.lower() == "coco":
            return self.framework_formats_dir / dataset_name / "coco"
        elif framework.lower() == "yolo":
            return self.framework_formats_dir / dataset_name / "yolo"
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def get_framework_images_dir(self, dataset_name: str, framework: str) -> Path:
        """Get the images directory for a framework format."""
        framework_dir = self.get_framework_format_dir(dataset_name, framework)
        if framework.lower() == "coco":
            return framework_dir / "images"
        elif framework.lower() == "yolo":
            return framework_dir / "images"
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def get_framework_annotations_dir(self, dataset_name: str, framework: str) -> Path:
        """Get the annotations directory for a framework format."""
        framework_dir = self.get_framework_format_dir(dataset_name, framework)
        if framework.lower() == "coco":
            return framework_dir / "annotations"
        elif framework.lower() == "yolo":
            return framework_dir / "labels"
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def ensure_directories(self, dataset_name: str, frameworks: Optional[list] = None):
        """Ensure all necessary directories exist."""
        # Master directories
        self.get_dataset_master_dir(dataset_name).mkdir(parents=True, exist_ok=True)
        self.get_dataset_images_dir(dataset_name).mkdir(parents=True, exist_ok=True)
        self.get_dataset_master_dir(dataset_name).mkdir(parents=True, exist_ok=True)

        # Framework directories
        if frameworks:
            for framework in frameworks:
                framework_dir = self.get_framework_format_dir(dataset_name, framework)
                framework_dir.mkdir(parents=True, exist_ok=True)

                # Create split directories only for existing splits
                images_dir = self.get_framework_images_dir(dataset_name, framework)
                annotations_dir = self.get_framework_annotations_dir(
                    dataset_name, framework
                )
                images_dir.mkdir(parents=True, exist_ok=True)
                annotations_dir.mkdir(parents=True, exist_ok=True)

                # Get existing splits from master data
                existing_splits = self._get_existing_splits(dataset_name)

                for split in existing_splits:
                    (images_dir / split).mkdir(exist_ok=True)

                    if framework != "coco":  # coco image paths are in the annotations
                        (annotations_dir / split).mkdir(exist_ok=True)

    def get_existing_splits(self, dataset_name: str) -> List[str]:
        """
        Get list of splits that actually exist in the master data.
        """
        try:
            annotations_file = self.get_dataset_annotations_file(dataset_name)
            if not annotations_file.exists():
                return []

            with open(annotations_file, "r") as f:
                master_data = json.load(f)

            existing_splits = set()
            for image in master_data.get("images", []):
                split = image.get("split")
                if split:
                    existing_splits.add(split)

            return sorted(list(existing_splits))
        except Exception as e:
            self.logger.warning(
                f"Error getting existing splits for dataset '{dataset_name}': {e}"
            )
            return []

    def _get_existing_splits(self, dataset_name: str) -> List[str]:
        """Get list of splits that actually exist in the master data."""
        try:
            annotations_file = self.get_dataset_annotations_file(dataset_name)
            if not annotations_file.exists():
                return []

            with open(annotations_file, "r") as f:
                master_data = json.load(f)

            existing_splits = set()
            for image in master_data.get("images", []):
                split = image.get("split")
                if split:
                    existing_splits.add(split)

            return sorted(list(existing_splits))
        except Exception as e:
            # If we can't read the master data, return empty list
            return []

    def get_split_image_dir(
        self, dataset_name: str, framework: str, split: str
    ) -> Path:
        """Get the images directory for a specific split."""
        return self.get_framework_images_dir(dataset_name, framework) / split

    def get_split_annotations_dir(
        self, dataset_name: str, framework: str, split: str
    ) -> Path:
        """Get the annotations directory for a specific split."""
        return self.get_framework_annotations_dir(dataset_name, framework) / split

    def get_master_split_images_dir(self, dataset_name: str, split: str) -> Path:
        """Get the master images directory for a specific split."""
        return self.get_dataset_images_dir(dataset_name) / split

    def dataset_exists(self, dataset_name: str) -> bool:
        """Check if a dataset exists in master storage."""
        return self.get_dataset_annotations_file(dataset_name).exists()

    def framework_format_exists(self, dataset_name: str, framework: str) -> bool:
        """Check if a framework format exists for a dataset."""
        return self.get_framework_format_dir(dataset_name, framework).exists()

    def list_datasets(self) -> list:
        """List all available datasets."""
        datasets = []
        if self.master_data_dir.exists():
            for dataset_dir in self.master_data_dir.iterdir():
                if dataset_dir.is_dir():
                    annotations_file = dataset_dir / "annotations.json"
                    if annotations_file.exists():
                        datasets.append(dataset_dir.name)
        return datasets

    def list_framework_formats(self, dataset_name: str) -> Dict[str, bool]:
        """List available framework formats for a dataset."""
        formats = {}
        for framework in ["coco", "yolo"]:
            formats[framework] = self.framework_format_exists(dataset_name, framework)
        return formats

    def get_relative_path(self, path: Path, start: Path) -> str:
        """Get a relative of path from start."""
        try:
            return os.path.relpath(
                path, start=start
            )  # str(to_path.relative_to(from_path))
        except ValueError:
            # If paths are on different drives (Windows), use absolute path
            return str(path)
