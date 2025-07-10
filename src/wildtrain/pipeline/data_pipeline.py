"""
Data pipeline for managing deep learning datasets with transformations.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import numpy as np
import cv2
import os

from ..adapters.base_adapter import BaseAdapter
from ..converters.coco_to_master import COCOToMasterConverter
from ..converters.yolo_to_master import YOLOToMasterConverter
from ..validators.coco_validator import COCOValidator
from ..validators.yolo_validator import YOLOValidator
from ..transformations import TransformationPipeline, AugmentationTransformer, TilingTransformer


class DataPipeline:
    """
    Main data pipeline for managing deep learning datasets.
    
    This pipeline integrates:
    - Data validation
    - Format conversion
    - Data transformations (augmentation, tiling)
    - Framework-specific format generation
    """
    
    def __init__(self, master_data_dir: str, transformation_pipeline: Optional[TransformationPipeline] = None):
        """
        Initialize the data pipeline.
        
        Args:
            master_data_dir: Directory for storing master format data
            transformation_pipeline: Optional transformation pipeline
        """
        self.master_data_dir = Path(master_data_dir)
        self.master_data_dir.mkdir(parents=True, exist_ok=True)
        
        self.transformation_pipeline = transformation_pipeline or TransformationPipeline()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Don't pre-instantiate converters and validators - create on demand
        self.adapters = {}
    
    def import_dataset(self, source_path: str, source_format: str, dataset_name: str, 
                      apply_transformations: bool = False) -> Dict[str, Any]:
        """
        Import a dataset from source format to master format.
        
        Args:
            source_path: Path to source dataset
            source_format: Format of source dataset ('coco' or 'yolo')
            dataset_name: Name for the dataset in master format
            apply_transformations: Whether to apply transformations during import
            
        Returns:
            Dictionary with import result information
        """
        try:
            self.logger.info(f"Importing dataset from {source_path} ({source_format} format)")
            
            # Validate source format
            if source_format not in ['coco', 'yolo']:
                return {
                    'success': False,
                    'error': f"Unsupported source format: {source_format}",
                    'validation_errors': [],
                    'hints': ['Supported formats: coco, yolo']
                }
            
            # Create validator and validate dataset
            if source_format == 'coco':
                # For COCO, source_path should be the annotation file path
                validator = COCOValidator(source_path)
                is_valid, errors, warnings = validator.validate()
                if not is_valid:
                    return {
                        'success': False,
                        'error': 'Validation failed',
                        'validation_errors': errors,
                        'hints': warnings
                    }
                
                # Create converter and convert
                converter = COCOToMasterConverter(source_path)
                converter.load_coco_annotation()
                master_data = converter.convert_to_master(dataset_name)
                
            elif source_format == 'yolo':
                # For YOLO, source_path should be the data.yaml file path
                validator = YOLOValidator(source_path)
                is_valid, errors, warnings = validator.validate()
                if not is_valid:
                    return {
                        'success': False,
                        'error': 'Validation failed',
                        'validation_errors': errors,
                        'hints': warnings
                    }
                
                # Create converter and convert
                converter = YOLOToMasterConverter(source_path)
                converter.load_yolo_data()
                master_data = converter.convert_to_master(dataset_name)
            
            # Apply transformations if requested
            if apply_transformations and self.transformation_pipeline:
                master_data = self._apply_transformations_to_dataset(master_data)
            
            # Save master data
            dataset_dir = self.master_data_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            self._save_master_data(master_data, dataset_dir)
            
            # Create framework formats
            framework_paths = self._create_framework_formats(dataset_name)
            
            self.logger.info(f"Successfully imported dataset '{dataset_name}'")
            return {
                'success': True,
                'dataset_name': dataset_name,
                'master_path': str(dataset_dir / 'annotations.json'),
                'framework_paths': framework_paths
            }
            
        except Exception as e:
            self.logger.error(f"Error importing dataset: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'validation_errors': [],
                'hints': []
            }
    
    def _apply_transformations_to_dataset(self, master_data: Dict[str, Any]) -> Dict[str, Any]:
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
        
        for image_info in master_data['images']:
            # Load image
            image_path = Path(image_info['file_name'])
            if not image_path.is_absolute():
                image_path = self.master_data_dir / image_path
            
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"Could not load image: {image_path}")
                continue
            
            # Get annotations for this image
            image_annotations = [
                ann for ann in master_data['annotations'] 
                if ann['image_id'] == image_info['id']
            ]
            
            # Apply transformations with correct interface
            try:
                inputs = {
                    'image': image,
                    'annotations': image_annotations,
                    'info': image_info
                }
                
                transformed_data = self.transformation_pipeline.transform(inputs)
                
                for data in transformed_data:
                    transformed_images.extend(data['image'])
                    transformed_annotations.extend(data.get('annotations', []))
                    transformed_image_info.extend(data['info'])
                
            except Exception as e:
                self.logger.error(f"Error transforming image {image_info['file_name']}: {str(e)}")
                continue
        
        return {
            'images': transformed_images,
            'annotations': transformed_annotations,
            'categories': master_data.get('categories', []),
            'info': transformed_image_info
        }
    
    def _save_master_data(self, master_data: Dict[str, Any], dataset_dir: Path) -> None:
        """
        Save master format data to directory.
        
        Args:
            master_data: Master format data
            dataset_dir: Directory to save data
        """
        # Save annotations
        annotations_file = dataset_dir / 'annotations.json'
        with open(annotations_file, 'w') as f:
            json.dump(master_data, f, indent=2)
        
        # Save images (copy or symlink)
        images_dir = dataset_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        
        for image_info in master_data['images']:
            source_path = Path(image_info['file_name'])
            if not source_path.is_absolute():
                source_path = self.master_data_dir / source_path
            
            target_path = images_dir / source_path.name
            
            if source_path.exists():
                # Create symlink or copy
                if not target_path.exists():
                    target_path.symlink_to(source_path)
        
        self.logger.info(f"Saved master data to {dataset_dir}")
    
    def export_dataset(self, dataset_name: str, target_format: str, target_path: str) -> bool:
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
            self.logger.info(f"Exporting dataset '{dataset_name}' to {target_format} format")
            
            # Load master data
            dataset_dir = self.master_data_dir / dataset_name
            annotations_file = dataset_dir / 'annotations.json'
            
            if not annotations_file.exists():
                self.logger.error(f"Dataset '{dataset_name}' not found")
                return False
            
            with open(annotations_file, 'r') as f:
                master_data = json.load(f)
            
            # Get or create adapter
            if target_format not in self.adapters:
                # Create adapter based on target format with proper master annotation path
                master_annotation_path = str(annotations_file)
                if target_format == 'coco':
                    from ..adapters.coco_adapter import COCOAdapter
                    self.adapters[target_format] = COCOAdapter(master_annotation_path)
                elif target_format == 'yolo':
                    from ..adapters.yolo_adapter import YOLOAdapter
                    self.adapters[target_format] = YOLOAdapter(master_annotation_path)
                else:
                    self.logger.error(f"Unsupported target format: {target_format}")
                    return False
            
            # Convert to target format
            adapter = self.adapters[target_format]
            adapter.load_master_annotation()
            
            # Convert for each split
            for split in ['train', 'val', 'test']:
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
            'master_data_dir': str(self.master_data_dir),
            'transformation_pipeline': self.transformation_pipeline.get_pipeline_info(),
            'supported_formats': ['coco', 'yolo'],
            'available_datasets': [d.name for d in self.master_data_dir.iterdir() if d.is_dir()]
        } 

    def _create_framework_formats(self, dataset_name: str) -> Dict[str, str]:
        """
        Create framework-specific formats for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary mapping framework names to output paths
        """
        framework_paths = {}
        
        try:
            # Create COCO format
            coco_path = self._create_coco_format(dataset_name)
            framework_paths['coco'] = coco_path
        except Exception as e:
            self.logger.warning(f"Failed to create COCO format: {str(e)}")
        
        try:
            # Create YOLO format
            yolo_path = self._create_yolo_format(dataset_name)
            framework_paths['yolo'] = yolo_path
        except Exception as e:
            self.logger.warning(f"Failed to create YOLO format: {str(e)}")
        
        return framework_paths
    
    def _create_coco_format(self, dataset_name: str) -> str:
        """Create COCO format for a dataset."""
        # Create COCO directory structure
        coco_dir = self.master_data_dir / "framework_formats" / "coco" / dataset_name
        coco_data_dir = coco_dir / "data"
        coco_annotations_dir = coco_dir / "annotations"
        
        coco_data_dir.mkdir(parents=True, exist_ok=True)
        coco_annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks for images
        self._create_image_symlinks(dataset_name, coco_data_dir, "coco")
        
        # Generate COCO annotations using adapter
        master_data = self._load_master_data(dataset_name)
        adapter = self._get_coco_adapter(dataset_name)
        adapter.load_master_annotation()
        
        # Convert for each split
        all_coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        for split in ['train', 'val', 'test']:
            try:
                split_data = adapter.convert(split)
                all_coco_data['images'].extend(split_data.get('images', []))
                all_coco_data['annotations'].extend(split_data.get('annotations', []))
                if not all_coco_data['categories']:
                    all_coco_data['categories'] = split_data.get('categories', [])
            except Exception as e:
                self.logger.warning(f"Could not convert split '{split}': {str(e)}")
                continue
        
        # Save COCO annotations
        coco_annotations_file = coco_annotations_dir / "instances_train.json"
        with open(coco_annotations_file, 'w') as f:
            json.dump(all_coco_data, f, indent=2)
        
        self.logger.info(f"Created COCO format for dataset '{dataset_name}'")
        return str(coco_dir)
    
    def _create_yolo_format(self, dataset_name: str) -> str:
        """Create YOLO format for a dataset."""
        # Create YOLO directory structure
        yolo_dir = self.master_data_dir / "framework_formats" / "yolo" / dataset_name
        yolo_images_dir = yolo_dir / "images"
        yolo_labels_dir = yolo_dir / "labels"
        
        yolo_images_dir.mkdir(parents=True, exist_ok=True)
        yolo_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Create symlinks for images
        self._create_image_symlinks(dataset_name, yolo_images_dir, "yolo")
        
        # Generate YOLO annotations using adapter
        master_data = self._load_master_data(dataset_name)
        adapter = self._get_yolo_adapter(dataset_name)
        adapter.load_master_annotation()
        
        # Convert for each split
        all_yolo_data = {
            'annotations': {},
            'names': {}
        }
        
        for split in ['train', 'val', 'test']:
            try:
                split_data = adapter.convert(split)
                all_yolo_data['annotations'][split] = split_data
                
                # Get class names from master data
                if not all_yolo_data['names']:
                    classes = master_data.get('dataset_info', {}).get('classes', [])
                    all_yolo_data['names'] = {cat['id']: cat['name'] for cat in classes}
            except Exception as e:
                self.logger.warning(f"Could not convert split '{split}': {str(e)}")
                continue
        
        # Save YOLO annotations and data.yaml
        self._save_yolo_annotations(yolo_labels_dir, all_yolo_data)
        self._save_yolo_data_yaml(yolo_dir, dataset_name, all_yolo_data)
        
        self.logger.info(f"Created YOLO format for dataset '{dataset_name}'")
        return str(yolo_dir)
    
    def _create_image_symlinks(self, dataset_name: str, target_dir: Path, framework: str):
        """Create symlinks for images in the target directory."""
        master_images_dir = self.master_data_dir / dataset_name / "images"
        
        if not master_images_dir.exists():
            raise FileNotFoundError(f"Master images directory not found: {master_images_dir}")
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            split_dir = target_dir / split
            split_dir.mkdir(exist_ok=True)
            
            master_split_dir = master_images_dir / split
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
                            import shutil
                            shutil.copy2(image_file, symlink_path)
    
    def _save_yolo_annotations(self, labels_dir: Path, yolo_data: Dict[str, Any]):
        """Save YOLO label files."""
        annotations = yolo_data.get('annotations', {})
        
        for split, split_annotations in annotations.items():
            split_labels_dir = labels_dir / split
            split_labels_dir.mkdir(exist_ok=True)
            
            for image_name, label_lines in split_annotations.items():
                label_file = split_labels_dir / f"{Path(image_name).stem}.txt"
                
                with open(label_file, 'w') as f:
                    for line in label_lines:
                        f.write(line + '\n')
    
    def _save_yolo_data_yaml(self, yolo_dir: Path, dataset_name: str, yolo_data: Dict[str, Any]):
        """Save YOLO data.yaml file."""
        import yaml
        
        data_yaml = {
            'path': str(yolo_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': yolo_data.get('names', {})
        }
        
        # Remove test if it doesn't exist
        if not (yolo_dir / "images" / "test").exists():
            del data_yaml['test']
        
        yaml_file = yolo_dir / "data.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
    
    def _load_master_data(self, dataset_name: str) -> Dict[str, Any]:
        """Load master data for a dataset."""
        annotations_file = self.master_data_dir / dataset_name / "annotations.json"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Master annotations not found: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            return json.load(f)
    
    def _get_coco_adapter(self, dataset_name: str):
        """Get or create COCO adapter."""
        if 'coco' not in self.adapters:
            from ..adapters.coco_adapter import COCOAdapter
            master_annotation_path = str(self.master_data_dir / dataset_name / "annotations.json")
            self.adapters['coco'] = COCOAdapter(master_annotation_path)
        return self.adapters['coco']
    
    def _get_yolo_adapter(self, dataset_name: str):
        """Get or create YOLO adapter."""
        if 'yolo' not in self.adapters:
            from ..adapters.yolo_adapter import YOLOAdapter
            master_annotation_path = str(self.master_data_dir / dataset_name / "annotations.json")
            self.adapters['yolo'] = YOLOAdapter(master_annotation_path)
        return self.adapters['yolo']
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets in master storage.
        
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        
        for dataset_dir in self.master_data_dir.iterdir():
            if dataset_dir.is_dir():
                annotations_file = dataset_dir / "annotations.json"
                if annotations_file.exists():
                    try:
                        with open(annotations_file, 'r') as f:
                            master_data = json.load(f)
                        
                        # Count images and annotations
                        total_images = len(master_data.get('images', []))
                        total_annotations = len(master_data.get('annotations', []))
                        
                        # Count images by split
                        images_by_split = {}
                        for image in master_data.get('images', []):
                            split = image.get('split', 'unknown')
                            images_by_split[split] = images_by_split.get(split, 0) + 1
                        
                        # Count annotations by type
                        annotations_by_type = {}
                        for ann in master_data.get('annotations', []):
                            ann_type = 'detection'  # Default type
                            if 'segmentation' in ann and ann['segmentation']:
                                ann_type = 'segmentation'
                            elif 'keypoints' in ann and ann['keypoints']:
                                ann_type = 'keypoints'
                            
                            annotations_by_type[ann_type] = annotations_by_type.get(ann_type, 0) + 1
                        
                        datasets.append({
                            'dataset_name': dataset_dir.name,
                            'total_images': total_images,
                            'total_annotations': total_annotations,
                            'images_by_split': images_by_split,
                            'annotations_by_type': annotations_by_type
                        })
                    except Exception as e:
                        self.logger.warning(f"Error reading dataset {dataset_dir.name}: {str(e)}")
        
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dictionary with dataset information
        """
        dataset_dir = self.master_data_dir / dataset_name
        annotations_file = dataset_dir / "annotations.json"
        
        if not annotations_file.exists():
            raise FileNotFoundError(f"Dataset '{dataset_name}' not found")
        
        with open(annotations_file, 'r') as f:
            master_data = json.load(f)
        
        # Count images and annotations
        total_images = len(master_data.get('images', []))
        total_annotations = len(master_data.get('annotations', []))
        
        # Count images by split
        images_by_split = {}
        for image in master_data.get('images', []):
            split = image.get('split', 'unknown')
            images_by_split[split] = images_by_split.get(split, 0) + 1
        
        # Count annotations by type
        annotations_by_type = {}
        for ann in master_data.get('annotations', []):
            ann_type = 'detection'  # Default type
            if 'segmentation' in ann and ann['segmentation']:
                ann_type = 'segmentation'
            elif 'keypoints' in ann and ann['keypoints']:
                ann_type = 'keypoints'
            
            annotations_by_type[ann_type] = annotations_by_type.get(ann_type, 0) + 1
        
        # Get categories
        categories = master_data.get('dataset_info', {}).get('classes', [])
        
        return {
            'dataset_name': dataset_name,
            'master_annotations_file': str(annotations_file),
            'total_images': total_images,
            'total_annotations': total_annotations,
            'images_by_split': images_by_split,
            'annotations_by_type': annotations_by_type,
            'categories': categories
        }
    
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
        coco_dir = self.master_data_dir / "framework_formats" / "coco" / dataset_name
        
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
        yolo_dir = self.master_data_dir / "framework_formats" / "yolo" / dataset_name
        
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