"""
Tests for the data pipeline using real data.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json

from wildtrain.pipeline.data_pipeline import DataPipeline


class TestDataPipelineRealData:
    """Test the data pipeline with real COCO data."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create test data structure
        self.test_data_dir = self.project_root / "data"
        self.test_data_dir.mkdir()
        
        # Copy real COCO data
        self.coco_source = Path("data/general_dataset/coco/annotations/annotations_train.json")
        if self.coco_source.exists():
            # Create a copy for testing
            self.coco_test_file = self.test_data_dir / "annotations_train.json"
            shutil.copy2(self.coco_source, self.coco_test_file)
        else:
            pytest.skip("Real COCO data not found")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_import_coco_real_data(self):
        """Test importing real COCO dataset."""
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