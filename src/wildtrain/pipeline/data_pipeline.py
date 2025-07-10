"""
Main data pipeline orchestrator for managing the complete data workflow.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from ..validators.coco_validator import COCOValidator
from ..validators.yolo_validator import YOLOValidator
from ..converters.coco_to_master import COCOToMasterConverter
from ..converters.yolo_to_master import YOLOToMasterConverter
from .master_data_manager import MasterDataManager
from .framework_data_manager import FrameworkDataManager


class DataPipeline:
    """
    Main data pipeline orchestrator that coordinates the complete data workflow.
    
    Workflow:
    1. Validate input data format (COCO or YOLO)
    2. Convert to master format and store in master directory
    3. Create framework-specific formats using symlinks
    """
    
    def __init__(self, project_root: str):
        """
        Initialize the data pipeline.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.master_data_manager = MasterDataManager(self.project_root)
        self.framework_data_manager = FrameworkDataManager(self.project_root)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def import_dataset(self, 
                      source_path: str, 
                      format_type: str,
                      dataset_name: str,
                      validation_hints: bool = True) -> Dict[str, Any]:
        """
        Import a dataset from COCO or YOLO format into the master format.
        
        Args:
            source_path: Path to the source dataset
            format_type: Either 'coco' or 'yolo'
            dataset_name: Name for the dataset in master storage
            validation_hints: Whether to provide detailed validation hints
            
        Returns:
            Dictionary with import results and status
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {source_path}")
        
        # Step 1: Validate the input data
        self.logger.info(f"Validating {format_type.upper()} dataset at {source_path}")
        validation_result = self._validate_dataset(source_path, format_type, validation_hints)
        
        if not validation_result['is_valid']:
            return {
                'success': False,
                'error': 'Validation failed',
                'validation_errors': validation_result['errors'],
                'hints': validation_result.get('hints', [])
            }
        
        # Step 2: Convert to master format
        self.logger.info("Converting to master format")
        conversion_result = self._convert_to_master(source_path, format_type, dataset_name)
        
        if not conversion_result['success']:
            return conversion_result
        
        # Step 3: Create framework-specific formats
        self.logger.info("Creating framework-specific formats")
        framework_result = self._create_framework_formats(dataset_name)
        
        return {
            'success': True,
            'dataset_name': dataset_name,
            'master_path': conversion_result['master_path'],
            'framework_paths': framework_result['framework_paths'],
            'validation_result': validation_result,
            'conversion_result': conversion_result
        }
    
    def _validate_dataset(self, source_path: Path, format_type: str, provide_hints: bool) -> Dict[str, Any]:
        """Validate the input dataset."""
        if format_type.lower() == 'coco':
            validator = COCOValidator(str(source_path))
        elif format_type.lower() == 'yolo':
            validator = YOLOValidator(str(source_path))
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        is_valid = validator.validate()
        errors = validator.get_errors()
        
        result = {
            'is_valid': is_valid,
            'errors': errors
        }
        
        if provide_hints and not is_valid:
            result['hints'] = self._generate_validation_hints(format_type, errors)
        
        return result
    
    def _convert_to_master(self, source_path: Path, format_type: str, dataset_name: str) -> Dict[str, Any]:
        """Convert the dataset to master format."""
        try:
            if format_type.lower() == 'coco':
                converter = COCOToMasterConverter(str(source_path))
            elif format_type.lower() == 'yolo':
                converter = YOLOToMasterConverter(str(source_path))
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
            
            master_data = converter.convert()
            master_path = self.master_data_manager.store_dataset(dataset_name, master_data)
            
            return {
                'success': True,
                'master_path': master_path
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_framework_formats(self, dataset_name: str) -> Dict[str, Any]:
        """Create framework-specific formats using symlinks."""
        try:
            framework_paths = self.framework_data_manager.create_framework_formats(dataset_name)
            return {
                'success': True,
                'framework_paths': framework_paths
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_validation_hints(self, format_type: str, errors: List[str]) -> List[str]:
        """Generate helpful hints for validation errors."""
        hints = []
        
        if format_type.lower() == 'coco':
            if any('missing required field' in error for error in errors):
                hints.append("Ensure your COCO JSON file contains all required fields: images, annotations, categories")
            if any('file not found' in error for error in errors):
                hints.append("Check that all image files referenced in the COCO JSON exist in the specified paths")
        
        elif format_type.lower() == 'yolo':
            if any('missing required field' in error for error in errors):
                hints.append("Ensure your data.yaml contains required fields: path, train, names")
            if any('path' in error and 'string' in error for error in errors):
                hints.append("The 'path' field in data.yaml must be a string pointing to the dataset root directory")
            if any('train' in error for error in errors):
                hints.append("The 'train' field in data.yaml must be a string or list of strings pointing to training images")
        
        return hints
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets in the master storage."""
        return self.master_data_manager.list_datasets()
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset."""
        return self.master_data_manager.get_dataset_info(dataset_name)
    
    def export_framework_format(self, dataset_name: str, framework: str) -> Dict[str, Any]:
        """Export a dataset to a specific framework format."""
        return self.framework_data_manager.export_framework_format(dataset_name, framework) 