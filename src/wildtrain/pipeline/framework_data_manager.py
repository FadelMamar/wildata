"""
Framework data manager for creating framework-specific formats using symlinks.
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
import tempfile

from ..adapters.coco_adapter import COCOAdapter
from ..adapters.yolo_adapter import YOLOAdapter

logger = logging.getLogger(__name__)
class FrameworkDataManager:
    """
    Manages framework-specific data formats using symlinks to master data.
    
    Framework structure:
    framework_configs/
    ├── coco/
    │   ├── data/
    │   │   ├── train/            # Symlinks to master images
    │   │   └── val/              # Symlinks to master images
    │   └── annotations/
    │       └── instances_train.json
    └── yolo/
        ├── images/
        │   ├── train/            # Symlinks to master images
        │   └── val/              # Symlinks to master images
        └── labels/
            ├── train/
            └── val/
    """
    
    def __init__(self, project_root: Path):
        """
        Initialize the framework data manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.framework_configs_dir = project_root / "framework_configs"
        
        # Setup logging
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
            framework_paths['coco'] = coco_path
        except Exception as e:
            self.logger.error(f"Error creating COCO format: {e}")
        
        # Create YOLO format
        try:
            yolo_path = self._create_yolo_format(dataset_name)
            framework_paths['yolo'] = yolo_path
        except Exception as e:
            self.logger.error(f"Error creating YOLO format: {e}")
        
        return framework_paths
    
    def _create_coco_format(self, dataset_name: str) -> str:
        """Create COCO format for a dataset."""
        # Create COCO directory structure
        coco_dir = self.framework_configs_dir / "coco" / dataset_name
        coco_data_dir = coco_dir / "data"
        coco_annotations_dir = coco_dir / "annotations"
        
        coco_data_dir.mkdir(parents=True, exist_ok=True)
        coco_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks for images
        self._create_image_symlinks(dataset_name, coco_data_dir, "coco")
        
        # Generate COCO annotations using adapter
        master_data = self._load_master_data(dataset_name)
        
        # Create temporary master annotation file for adapter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(master_data, tmp_file)
            tmp_master_path = tmp_file.name
        
        try:
            adapter = COCOAdapter(tmp_master_path)
            adapter.load_master_annotation()
            coco_data = adapter.convert('train')
            
            # Save COCO annotations
            coco_annotations_file = coco_annotations_dir / "instances_train.json"
            adapter.save(coco_data, str(coco_annotations_file))
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(tmp_master_path):
                os.unlink(tmp_master_path)
        
        self.logger.info(f"Created COCO format for dataset '{dataset_name}'")
        return str(coco_dir)
    
    def _create_yolo_format(self, dataset_name: str) -> str:
        """Create YOLO format for a dataset."""
        # Create YOLO directory structure
        yolo_dir = self.framework_configs_dir / "yolo" / dataset_name
        yolo_images_dir = yolo_dir / "images"
        yolo_labels_dir = yolo_dir / "labels"
        
        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks for images
        self._create_image_symlinks(dataset_name, yolo_images_dir, "yolo")
        
        # Generate YOLO annotations using adapter
        master_data = self._load_master_data(dataset_name)
        
        # Create temporary master annotation file for adapter
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            import json
            json.dump(master_data, tmp_file)
            tmp_master_path = tmp_file.name
        
        try:
            adapter = YOLOAdapter(tmp_master_path)
            adapter.load_master_annotation()
            yolo_data = adapter.convert('train')
            
            # Save YOLO annotations
            adapter.save(yolo_data)
            
            # Save data.yaml
            class_names = [cat['name'] for cat in master_data.get('dataset_info', {}).get('classes', [])]
            split_image_dirs = {
                'train': 'images/train',
                'val': 'images/val'
            }
            adapter.save_data_yaml(class_names, split_image_dirs, str(yolo_dir / "data.yaml"))
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_master_path):
                os.unlink(tmp_master_path)
        
        self.logger.info(f"Created YOLO format for dataset '{dataset_name}'")
        return str(yolo_dir)
    
    def _create_image_symlinks(self, dataset_name: str, target_dir: Path, framework: str):
        """Create symlinks for images in the target directory."""
        master_images_dir = self.project_root / "data" / "images" / dataset_name

        logger.info(f"Creating symlinks for images in {target_dir} for {framework}")
        
        if not master_images_dir.exists():
            raise FileNotFoundError(f"Master images directory not found: {master_images_dir}")
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            split_dir = target_dir / split
            split_dir.mkdir(exist_ok=True)
            
            master_split_dir = master_images_dir / split
            success = 0
            failed = 0
            if master_split_dir.exists():
                # Create symlinks for each image
                for image_file in master_split_dir.iterdir():
                    if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        symlink_path = split_dir / image_file.name
                        
                        # Create relative symlink to master image
                        relative_path = os.path.relpath(image_file, split_dir)
                        
                        # Remove existing symlink if it exists
                        if symlink_path.exists():
                            symlink_path.unlink()
                        
                        # Create symlink
                        try:
                            os.symlink(relative_path, symlink_path)
                        except OSError:
                            # Fallback to copying if symlink fails (e.g., on Windows without admin)
                            shutil.copy2(image_file, symlink_path)   
                         
    
    def _load_master_data(self, dataset_name: str) -> Dict[str, Any]:
        """Load master data for a dataset."""
        annotations_file = self.project_root / "data" / "annotations" / "master" / dataset_name / "annotations.json"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Master annotations not found: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            import json
            return json.load(f)
    
    def export_framework_format(self, dataset_name: str, framework: str) -> Dict[str, Any]:
        """
        Export a dataset to a specific framework format.
        
        Args:
            dataset_name: Name of the dataset
            framework: Framework name ('coco' or 'yolo')
            
        Returns:
            Dictionary with export information
        """
        if framework.lower() == 'coco':
            return self._export_coco_format(dataset_name)
        elif framework.lower() == 'yolo':
            return self._export_yolo_format(dataset_name)
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _export_coco_format(self, dataset_name: str) -> Dict[str, Any]:
        """Export to COCO format."""
        coco_dir = self.framework_configs_dir / "coco" / dataset_name
        
        if not coco_dir.exists():
            # Create the format if it doesn't exist
            self._create_coco_format(dataset_name)
        
        return {
            'framework': 'coco',
            'output_path': str(coco_dir),
            'data_dir': str(coco_dir / "data"),
            'annotations_file': str(coco_dir / "annotations" / "instances_train.json")
        }
    
    def _export_yolo_format(self, dataset_name: str) -> Dict[str, Any]:
        """Export to YOLO format."""
        yolo_dir = self.framework_configs_dir / "yolo" / dataset_name
        
        if not yolo_dir.exists():
            # Create the format if it doesn't exist
            self._create_yolo_format(dataset_name)
        
        return {
            'framework': 'yolo',
            'output_path': str(yolo_dir),
            'images_dir': str(yolo_dir / "images"),
            'labels_dir': str(yolo_dir / "labels"),
            'data_yaml': str(yolo_dir / "data.yaml")
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
        coco_dir = self.framework_configs_dir / "coco" / dataset_name
        if coco_dir.exists():
            formats.append({
                'framework': 'coco',
                'path': str(coco_dir),
                'exists': True
            })
        
        # Check YOLO format
        yolo_dir = self.framework_configs_dir / "yolo" / dataset_name
        if yolo_dir.exists():
            formats.append({
                'framework': 'yolo',
                'path': str(yolo_dir),
                'exists': True
            })
        
        return formats 