"""
Master data manager for storing and managing data in the master format.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from ..adapters.coco_adapter import COCOAdapter
from ..adapters.yolo_adapter import YOLOAdapter


class MasterDataManager:
    """
    Manages master data storage and operations.
    
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
    
    def __init__(self, project_root: Path):
        """
        Initialize the master data manager.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        self.data_dir = project_root / "data"
        self.images_dir = self.data_dir / "images"
        self.annotations_dir = self.data_dir / "annotations" / "master"
        
        # Create directories if they don't exist
        self._ensure_directories()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
    
    def store_dataset(self, dataset_name: str, master_data: Dict[str, Any]) -> str:
        """
        Store a dataset in master format.
        
        Args:
            dataset_name: Name of the dataset
            master_data: Master format data dictionary
            
        Returns:
            Path to the stored master annotations file
        """
        # Create dataset-specific directories
        dataset_images_dir = self.images_dir / dataset_name
        dataset_annotations_dir = self.annotations_dir / dataset_name
        
        dataset_images_dir.mkdir(parents=True, exist_ok=True)
        dataset_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images to master storage
        self._copy_images_to_master(dataset_name, master_data)
        
        # Store master annotations
        annotations_file = dataset_annotations_dir / "annotations.json"
        with open(annotations_file, 'w') as f:
            json.dump(master_data, f, indent=2)
        
        self.logger.info(f"Stored dataset '{dataset_name}' in master format")
        return str(annotations_file)
    
    def _copy_images_to_master(self, dataset_name: str, master_data: Dict[str, Any]):
        """Copy images to master storage and update paths."""
        images = master_data.get('images', [])
        
        for image_info in images:
            # Extract original image path
            original_path = image_info.get('file_path', '')
            if not original_path:
                continue
            
            # Determine split (train/val/test)
            split = image_info.get('split', 'train')
            split_dir = self.images_dir / dataset_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy image to master storage
            original_file = Path(original_path)
            if original_file.exists():
                filename = original_file.name
                new_path = split_dir / filename
                
                # Copy file if it doesn't exist or is different
                if not new_path.exists():
                    shutil.copy2(original_file, new_path)
                
                # Update the file path in master data
                image_info['file_path'] = str(new_path)
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
        annotations_file = self.annotations_dir / dataset_name / "annotations.json"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")
        
        with open(annotations_file, 'r') as f:
            master_data = json.load(f)
        
        # Count images by split
        images_by_split = {}
        for image in master_data.get('images', []):
            split = image.get('split', 'train')
            if split not in images_by_split:
                images_by_split[split] = 0
            images_by_split[split] += 1
        
        # Count annotations by type
        annotations_by_type = {}
        for annotation in master_data.get('annotations', []):
            annotation_type = annotation.get('type', 'unknown')
            if annotation_type not in annotations_by_type:
                annotations_by_type[annotation_type] = 0
            annotations_by_type[annotation_type] += 1
        
        return {
            'dataset_name': dataset_name,
            'master_annotations_file': str(annotations_file),
            'images_by_split': images_by_split,
            'annotations_by_type': annotations_by_type,
            'total_images': len(master_data.get('images', [])),
            'total_annotations': len(master_data.get('annotations', [])),
            'categories': master_data.get('categories', [])
        }
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.
        
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        
        if not self.annotations_dir.exists():
            return datasets
        
        for dataset_dir in self.annotations_dir.iterdir():
            if dataset_dir.is_dir():
                annotations_file = dataset_dir / "annotations.json"
                if annotations_file.exists():
                    try:
                        dataset_info = self.get_dataset_info(dataset_dir.name)
                        datasets.append(dataset_info)
                    except Exception as e:
                        self.logger.warning(f"Error reading dataset '{dataset_dir.name}': {e}")
        
        return datasets
    
    def load_master_data(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load master data for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Master data dictionary
        """
        annotations_file = self.annotations_dir / dataset_name / "annotations.json"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")
        
        with open(annotations_file, 'r') as f:
            return json.load(f)
    
    def delete_dataset(self, dataset_name: str) -> bool:
        """
        Delete a dataset from master storage.
        
        Args:
            dataset_name: Name of the dataset to delete
            
        Returns:
            True if deletion was successful
        """
        dataset_images_dir = self.images_dir / dataset_name
        dataset_annotations_dir = self.annotations_dir / dataset_name
        
        try:
            if dataset_images_dir.exists():
                shutil.rmtree(dataset_images_dir)
            
            if dataset_annotations_dir.exists():
                shutil.rmtree(dataset_annotations_dir)
            
            self.logger.info(f"Deleted dataset '{dataset_name}'")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting dataset '{dataset_name}': {e}")
            return False 