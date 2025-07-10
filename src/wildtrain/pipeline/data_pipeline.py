"""
Data pipeline for managing deep learning datasets with transformations.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import numpy as np
import cv2

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
        
        # Initialize converters and validators
        self.converters = {
            'coco': COCOToMasterConverter(),
            'yolo': YOLOToMasterConverter()
        }
        
        self.validators = {
            'coco': COCOValidator(),
            'yolo': YOLOValidator()
        }
        
        self.adapters = {}
    
    def import_dataset(self, source_path: str, source_format: str, dataset_name: str, 
                      apply_transformations: bool = False) -> bool:
        """
        Import a dataset from source format to master format.
        
        Args:
            source_path: Path to source dataset
            source_format: Format of source dataset ('coco' or 'yolo')
            dataset_name: Name for the dataset in master format
            apply_transformations: Whether to apply transformations during import
            
        Returns:
            True if import successful, False otherwise
        """
        try:
            self.logger.info(f"Importing dataset from {source_path} ({source_format} format)")
            
            # Validate source format
            if source_format not in self.converters:
                self.logger.error(f"Unsupported source format: {source_format}")
                return False
            
            # Validate dataset
            validator = self.validators[source_format]
            if not validator.validate(source_path):
                self.logger.error(f"Validation failed for {source_format} dataset")
                return False
            
            # Convert to master format
            converter = self.converters[source_format]
            master_data = converter.convert(source_path)
            
            # Apply transformations if requested
            if apply_transformations and self.transformation_pipeline:
                master_data = self._apply_transformations_to_dataset(master_data)
            
            # Save master data
            dataset_dir = self.master_data_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            self._save_master_data(master_data, dataset_dir)
            
            self.logger.info(f"Successfully imported dataset '{dataset_name}'")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing dataset: {str(e)}")
            return False
    
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
            
            # Apply transformations
            try:
                transformed_data = self.transformation_pipeline.transform(
                    image, image_annotations, image_info
                )
                
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
                # Create adapter based on target format
                if target_format == 'coco':
                    from ..adapters.coco_adapter import COCOAdapter
                    self.adapters[target_format] = COCOAdapter()
                elif target_format == 'yolo':
                    from ..adapters.yolo_adapter import YOLOAdapter
                    self.adapters[target_format] = YOLOAdapter()
                else:
                    self.logger.error(f"Unsupported target format: {target_format}")
                    return False
            
            # Convert to target format
            adapter = self.adapters[target_format]
            adapter.convert_and_save(master_data, target_path)
            
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
            'supported_formats': list(self.converters.keys()),
            'available_datasets': [d.name for d in self.master_data_dir.iterdir() if d.is_dir()]
        } 