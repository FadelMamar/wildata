"""
Framework data manager for creating framework-specific formats using symlinks.
"""

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..adapters.coco_adapter import COCOAdapter
from ..adapters.yolo_adapter import YOLOAdapter
from .path_manager import PathManager

logger = logging.getLogger(__name__)


class FrameworkDataManager:
    """
    Manages framework-specific data formats using symlinks to master data.

    This class is responsible for:
    - Creating framework-specific formats (COCO, YOLO)
    - Managing symlinks to master data
    - Coordinating with adapters for format conversion
    - Maintaining framework directory structures
    """

    def __init__(self, path_manager: PathManager):
        """
        Initialize the framework data manager.

        Args:
            path_manager: PathManager instance for consistent path resolution
        """
        self.path_manager = path_manager
        self.logger = logging.getLogger(__name__)

    def create_framework_formats(self, dataset_name: str) -> Dict[str, str]:
        """
        Create framework-specific formats for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dictionary mapping framework names to their output paths
        """
        framework_paths = {}

        # Create COCO format
        try:
            coco_path = self._create_coco_format(dataset_name)
            framework_paths["coco"] = coco_path
        except Exception as e:
            self.logger.error(f"Error creating COCO format: {e}")

        # Create YOLO format
        try:
            yolo_path = self._create_yolo_format(dataset_name)
            framework_paths["yolo"] = yolo_path
        except Exception as e:
            self.logger.error(f"Error creating YOLO format: {e}")

        return framework_paths

    def _create_coco_format(self, dataset_name: str) -> str:
        """Create COCO format for a dataset."""
        # Ensure directories exist
        self.path_manager.ensure_directories(dataset_name, ["coco"])

        # Get paths using PathManager
        coco_dir = self.path_manager.get_framework_format_dir(dataset_name, "coco")
        coco_data_dir = self.path_manager.get_framework_images_dir(dataset_name, "coco")
        coco_annotations_dir = self.path_manager.get_framework_annotations_dir(
            dataset_name, "coco"
        )

        # Create symlinks for images
        self._create_image_symlinks(dataset_name, coco_data_dir, "coco")

        # Generate COCO annotations using adapter
        master_data = self._load_master_data(dataset_name)

        # Create temporary master annotation file for adapter
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(master_data, tmp_file)
            tmp_master_path = tmp_file.name

        try:
            adapter = COCOAdapter(tmp_master_path)
            adapter.load_master_annotation()

            # Convert for each existing split using PathManager
            all_coco_data = {"images": [], "annotations": [], "categories": []}
            existing_splits = self.path_manager.get_existing_splits(dataset_name)

            for split in existing_splits:
                try:
                    split_data = adapter.convert(split)
                    all_coco_data["images"].extend(split_data.get("images", []))
                    all_coco_data["annotations"].extend(
                        split_data.get("annotations", [])
                    )
                    if not all_coco_data["categories"]:
                        all_coco_data["categories"] = split_data.get("categories", [])
                except Exception as e:
                    self.logger.warning(f"Could not convert split '{split}': {str(e)}")
                    continue

            # Save COCO annotations
            coco_annotations_file = coco_annotations_dir / "annotations.json"
            adapter.save(all_coco_data, str(coco_annotations_file))
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_master_path):
                os.unlink(tmp_master_path)

        self.logger.info(f"Created COCO format for dataset '{dataset_name}'")
        return str(coco_dir)

    def _create_yolo_format(self, dataset_name: str) -> str:
        """Create YOLO format for a dataset."""
        # Ensure directories exist
        self.path_manager.ensure_directories(dataset_name, ["yolo"])

        # Get paths using PathManager
        yolo_dir = self.path_manager.get_framework_format_dir(dataset_name, "yolo")
        yolo_images_dir = self.path_manager.get_framework_images_dir(
            dataset_name, "yolo"
        )
        yolo_labels_dir = self.path_manager.get_framework_annotations_dir(
            dataset_name, "yolo"
        )

        # Create symlinks for images
        self._create_image_symlinks(dataset_name, yolo_images_dir, "yolo")

        # Generate YOLO annotations using adapter
        master_data = self._load_master_data(dataset_name)

        # Create temporary master annotation file for adapter
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(master_data, tmp_file)
            tmp_master_path = tmp_file.name

        try:
            adapter = YOLOAdapter(tmp_master_path)
            adapter.load_master_annotation()

            # Convert for each existing split using PathManager
            all_yolo_data = {"annotations": {}, "names": {}}
            existing_splits = self.path_manager.get_existing_splits(dataset_name)

            for split in existing_splits:
                try:
                    split_data = adapter.convert(split)
                    all_yolo_data["annotations"][split] = split_data

                    # Get class names from master data
                    if not all_yolo_data["names"]:
                        classes = master_data.get("dataset_info", {}).get("classes", [])
                        all_yolo_data["names"] = {
                            cat["id"]: cat["name"] for cat in classes
                        }
                except Exception as e:
                    self.logger.warning(f"Could not convert split '{split}': {str(e)}")
                    continue

            # Save YOLO annotations and data.yaml
            self._save_yolo_annotations(yolo_labels_dir, all_yolo_data)
            self._save_yolo_data_yaml(yolo_dir, dataset_name, all_yolo_data)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_master_path):
                os.unlink(tmp_master_path)

        self.logger.info(f"Created YOLO format for dataset '{dataset_name}'")
        return str(yolo_dir)

    def _create_image_symlinks(
        self, dataset_name: str, target_dir: Path, framework: str
    ):
        """Create symlinks for images in the target directory."""
        logger.info(f"Creating symlinks for images in {target_dir} for {framework}")

        # Get existing splits from PathManager
        existing_splits = self.path_manager.get_existing_splits(dataset_name)

        if not existing_splits:
            self.logger.warning(
                f"No existing splits found for dataset '{dataset_name}'"
            )
            return

        # Create split directories only for existing splits
        for split in existing_splits:
            split_dir = target_dir / split
            split_dir.mkdir(exist_ok=True)

            # Get master images directory for this split
            master_split_images_dir = self.path_manager.get_master_split_images_dir(
                dataset_name, split
            )

            if master_split_images_dir.exists():
                # Create symlinks for each image
                for image_file in master_split_images_dir.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                    ]:
                        symlink_path = split_dir / image_file.name

                        # Create relative symlink to master image
                        relative_path = self.path_manager.get_relative_path(
                            split_dir, image_file
                        )

                        # Remove existing symlink if it exists
                        if symlink_path.exists():
                            symlink_path.unlink()

                        # Create symlink
                        try:
                            os.symlink(relative_path, symlink_path)
                        except OSError:
                            # Fallback to copying if symlink fails (e.g., on Windows without admin)
                            shutil.copy2(image_file, symlink_path)
            else:
                self.logger.warning(
                    f"Master split directory not found: {master_split_images_dir}"
                )

    def _load_master_data(self, dataset_name: str) -> Dict[str, Any]:
        """Load master data for a dataset."""
        annotations_file = self.path_manager.get_dataset_annotations_file(dataset_name)

        if not annotations_file.exists():
            raise FileNotFoundError(f"Master annotations not found: {annotations_file}")

        with open(annotations_file, "r") as f:
            return json.load(f)

    def _save_yolo_annotations(self, labels_dir: Path, yolo_data: Dict[str, Any]):
        """Save YOLO label files."""
        annotations = yolo_data.get("annotations", {})

        for split, split_annotations in annotations.items():
            split_labels_dir = labels_dir / split
            split_labels_dir.mkdir(exist_ok=True)

            for image_name, label_lines in split_annotations.items():
                label_file = split_labels_dir / f"{Path(image_name).stem}.txt"

                with open(label_file, "w") as f:
                    for line in label_lines:
                        f.write(line + "\n")

    def _save_yolo_data_yaml(
        self, yolo_dir: Path, dataset_name: str, yolo_data: Dict[str, Any]
    ):
        """Save YOLO data.yaml file."""
        data_yaml = {
            "path": str(yolo_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": yolo_data.get("names", {}),
        }

        # Remove test if it doesn't exist
        if not (yolo_dir / "images" / "test").exists():
            del data_yaml["test"]

        yaml_file = yolo_dir / "data.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

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
        if framework.lower() == "coco":
            return self._export_coco_format(dataset_name)
        elif framework.lower() == "yolo":
            return self._export_yolo_format(dataset_name)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    def _export_coco_format(self, dataset_name: str) -> Dict[str, Any]:
        """Export to COCO format."""
        coco_dir = self.path_manager.get_framework_format_dir(dataset_name, "coco")

        if not coco_dir.exists():
            # Create the format if it doesn't exist
            self._create_coco_format(dataset_name)

        return {
            "framework": "coco",
            "output_path": str(coco_dir),
            "data_dir": str(
                self.path_manager.get_framework_images_dir(dataset_name, "coco")
            ),
            "annotations_file": str(
                self.path_manager.get_framework_annotations_dir(dataset_name, "coco")
                / "annotations.json"
            ),
        }

    def _export_yolo_format(self, dataset_name: str) -> Dict[str, Any]:
        """Export to YOLO format."""
        yolo_dir = self.path_manager.get_framework_format_dir(dataset_name, "yolo")

        if not yolo_dir.exists():
            # Create the format if it doesn't exist
            self._create_yolo_format(dataset_name)

        return {
            "framework": "yolo",
            "output_path": str(yolo_dir),
            "images_dir": str(
                self.path_manager.get_framework_images_dir(dataset_name, "yolo")
            ),
            "labels_dir": str(
                self.path_manager.get_framework_annotations_dir(dataset_name, "yolo")
            ),
            "data_yaml": str(yolo_dir / "data.yaml"),
        }

    def list_framework_formats(self, dataset_name: str) -> List[Dict[str, Any]]:
        """
        List available framework formats for a dataset.

        Args:
            dataset_name: Name of the dataset

        Returns:
            List of framework format information
        """
        formats = []

        # Check COCO format
        coco_dir = self.path_manager.get_framework_format_dir(dataset_name, "coco")
        if coco_dir.exists():
            formats.append({"framework": "coco", "path": str(coco_dir), "exists": True})

        # Check YOLO format
        yolo_dir = self.path_manager.get_framework_format_dir(dataset_name, "yolo")
        if yolo_dir.exists():
            formats.append({"framework": "yolo", "path": str(yolo_dir), "exists": True})

        return formats
