"""
Tests for the data pipeline using synthetic data.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json

from wildtrain.pipeline.data_pipeline import DataPipeline


class TestDataPipelineSynthetic:
    """Test the data pipeline with synthetic data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create test data structure
        self.test_data_dir = self.project_root / "data"
        self.test_data_dir.mkdir()
        
        # Create synthetic COCO data
        self.coco_test_file = self.test_data_dir / "annotations_train.json"
        self._create_synthetic_coco_data()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_synthetic_coco_data(self):
        """Create synthetic COCO annotation data."""
        coco_data = {
            "images": [
                {
                    "id": 1,
                    "file_name": "test_image_1.jpg",
                    "width": 640,
                    "height": 480
                },
                {
                    "id": 2,
                    "file_name": "test_image_2.jpg",
                    "width": 800,
                    "height": 600
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 150],
                    "area": 30000,
                    "iscrowd": 0
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [150, 150, 250, 200],
                    "area": 25000,
                    "iscrowd": 0
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "test_category",
                    "supercategory": "test"
                }
            ]
        }
        
        with open(self.coco_test_file, 'w') as f:
            json.dump(coco_data, f)
    
    def test_import_coco_synthetic_data(self):
        """Test importing synthetic COCO dataset."""
        pipeline = DataPipeline(str(self.test_data_dir))
        
        # Test import
        result = pipeline.import_dataset(
            source_path=str(self.coco_test_file),
            source_format="coco",
            dataset_name="test_coco_dataset"
        )
        
        assert result['success'] is True
        assert result['dataset_name'] == "test_coco_dataset"
        assert 'master_path' in result
        assert 'framework_paths' in result
        
        # Check that master data was created
        master_file = Path(result['master_path'])
        assert master_file.exists()
        
        # Load and validate master data
        with open(master_file, 'r') as f:
            master_data = json.load(f)
        
        assert 'dataset_info' in master_data
        assert 'images' in master_data
        assert 'annotations' in master_data
        assert len(master_data['images']) > 0
        assert len(master_data['annotations']) > 0
        
        # Check dataset info
        dataset_info = master_data['dataset_info']
        assert dataset_info['name'] == "test_coco_dataset"
        assert dataset_info['task_type'] == "detection"
        assert len(dataset_info['classes']) > 0
    
    def test_list_datasets(self):
        """Test listing datasets."""
        pipeline = DataPipeline(str(self.test_data_dir))
        
        # Import a dataset first
        result = pipeline.import_dataset(
            source_path=str(self.coco_test_file),
            source_format="coco",
            dataset_name="test_coco_dataset"
        )
        
        assert result['success'] is True
        
        # List datasets
        datasets = pipeline.list_datasets()
        assert len(datasets) > 0
        
        # Find our test dataset
        test_dataset = None
        for dataset in datasets:
            if dataset['dataset_name'] == 'test_coco_dataset':
                test_dataset = dataset
                break
        
        assert test_dataset is not None
        assert test_dataset['total_images'] > 0
        assert test_dataset['total_annotations'] > 0
    
    def test_get_dataset_info(self):
        """Test getting dataset info."""
        pipeline = DataPipeline(str(self.test_data_dir))
        
        # Import a dataset first
        result = pipeline.import_dataset(
            source_path=str(self.coco_test_file),
            source_format="coco",
            dataset_name="test_coco_dataset"
        )
        
        assert result['success'] is True
        
        # Get dataset info
        info = pipeline.get_dataset_info("test_coco_dataset")
        
        assert info['dataset_name'] == "test_coco_dataset"
        assert info['total_images'] > 0
        assert info['total_annotations'] > 0
        assert 'images_by_split' in info
        assert 'annotations_by_type' in info
        assert 'categories' in info
    
    def test_export_framework_format(self):
        """Test exporting to framework format."""
        pipeline = DataPipeline(str(self.test_data_dir))
        
        # Import a dataset first
        result = pipeline.import_dataset(
            source_path=str(self.coco_test_file),
            source_format="coco",
            dataset_name="test_coco_dataset"
        )
        
        assert result['success'] is True
        
        # Export to COCO format
        export_result = pipeline.export_framework_format("test_coco_dataset", "coco")
        
        assert export_result['framework'] == 'coco'
        assert 'output_path' in export_result
        assert 'data_dir' in export_result
        assert 'annotations_file' in export_result
        
        # Check that files were created
        output_path = Path(export_result['output_path'])
        assert output_path.exists()
        
        annotations_file = Path(export_result['annotations_file'])
        assert annotations_file.exists()
    
    def test_pipeline_status(self):
        """Test getting pipeline status."""
        pipeline = DataPipeline(str(self.test_data_dir))
        
        status = pipeline.get_pipeline_status()
        
        assert 'master_data_dir' in status
        assert 'transformation_pipeline' in status
        assert 'supported_formats' in status
        assert 'available_datasets' in status
        assert 'coco' in status['supported_formats']
        assert 'yolo' in status['supported_formats'] 