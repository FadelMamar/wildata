"""
Tests for the data pipeline components.
"""

import pytest
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from wildtrain.pipeline.data_pipeline import DataPipeline
from wildtrain.pipeline.master_data_manager import MasterDataManager
from wildtrain.pipeline.framework_data_manager import FrameworkDataManager


class TestDataPipeline:
    """Test the main data pipeline orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        
        # Create test data structure
        self.test_data_dir = self.project_root / "test_data"
        self.test_data_dir.mkdir()
        
        # Create mock COCO dataset
        self.coco_dir = self.test_data_dir / "coco_dataset"
        self.coco_dir.mkdir()
        
        # Create mock images
        self.images_dir = self.coco_dir / "images"
        self.images_dir.mkdir()
        
        # Create a mock image file
        (self.images_dir / "test_image.jpg").write_text("mock image data")
        
        # Create mock COCO annotations
        self.coco_annotations = {
            "images": [
                {
                    "id": 1,
                    "file_name": "test_image.jpg",
                    "width": 640,
                    "height": 480
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
        
        with open(self.coco_dir / "annotations.json", "w") as f:
            json.dump(self.coco_annotations, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('wildtrain.pipeline.data_pipeline.COCOValidator')
    @patch('wildtrain.pipeline.data_pipeline.COCOToMasterConverter')
    def test_import_coco_dataset_success(self, mock_converter, mock_validator):
        """Test successful import of a COCO dataset."""
        # Mock validator
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.return_value = (True, [], [])
        mock_validator.return_value = mock_validator_instance
        
        # Mock converter
        mock_converter_instance = MagicMock()
        mock_converter_instance.convert_to_master.return_value = {
            "dataset_info": {
                "name": "test_dataset",
                "version": "1.0",
                "schema_version": "1.0",
                "task_type": "detection",
                "classes": [
                    {
                        "id": 1,
                        "name": "test_category"
                    }
                ]
            },
            "images": [
                {
                    "id": 1,
                    "file_name": "test_image.jpg",
                    "file_path": str(self.images_dir / "test_image.jpg"),
                    "width": 640,
                    "height": 480,
                    "split": "train"
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 150],
                    "type": "detection"
                }
            ]
        }
        mock_converter.return_value = mock_converter_instance
        
        # Create pipeline
        pipeline = DataPipeline(str(self.project_root))
        
        # Test import
        result = pipeline.import_dataset(
            source_path=str(self.coco_dir / "annotations.json"),
            source_format="coco",
            dataset_name="test_dataset"
        )
        
        assert result is True
    
    @patch('wildtrain.pipeline.data_pipeline.COCOValidator')
    def test_import_coco_dataset_validation_failure(self, mock_validator):
        """Test import failure due to validation errors."""
        # Mock validator with errors
        mock_validator_instance = MagicMock()
        mock_validator_instance.validate.return_value = (False, ["Missing required field: images"], [])
        mock_validator.return_value = mock_validator_instance
        
        # Create pipeline
        pipeline = DataPipeline(str(self.project_root))
        
        # Test import
        result = pipeline.import_dataset(
            source_path=str(self.coco_dir / "annotations.json"),
            source_format="coco",
            dataset_name="test_dataset"
        )
        
        assert result is False


class TestMasterDataManager:
    """Test the master data manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        self.manager = MasterDataManager(self.project_root)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_store_dataset(self):
        """Test storing a dataset in master format."""
        master_data = {
            "dataset_info": {
                "name": "test_dataset",
                "version": "1.0",
                "schema_version": "1.0",
                "task_type": "detection",
                "classes": [
                    {
                        "id": 1,
                        "name": "test_category"
                    }
                ]
            },
            "images": [
                {
                    "id": 1,
                    "file_name": "test_image.jpg",
                    "file_path": str(self.project_root / "test_image.jpg"),
                    "width": 640,
                    "height": 480,
                    "split": "train"
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 150],
                    "type": "detection"
                }
            ]
        }
        
        # Create a mock image file
        (self.project_root / "test_image.jpg").write_text("mock image data")
        
        # Store dataset
        annotations_path = self.manager.store_dataset("test_dataset", master_data)
        
        # Verify the file was created
        assert Path(annotations_path).exists()
        
        # Verify the data was stored correctly
        with open(annotations_path, 'r') as f:
            stored_data = json.load(f)
        
        assert stored_data['images'][0]['file_name'] == "test_image.jpg"
        assert stored_data['annotations'][0]['type'] == "detection"
    
    def test_list_datasets_empty(self):
        """Test listing datasets when none exist."""
        datasets = self.manager.list_datasets()
        assert datasets == []
    
    def test_get_dataset_info_not_found(self):
        """Test getting info for a non-existent dataset."""
        with pytest.raises(FileNotFoundError):
            self.manager.get_dataset_info("non_existent")


class TestFrameworkDataManager:
    """Test the framework data manager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir)
        self.manager = FrameworkDataManager(self.project_root)
        
        # Create master data structure
        self.master_data_dir = self.project_root / "data"
        self.master_images_dir = self.master_data_dir / "images" / "test_dataset"
        self.master_annotations_dir = self.master_data_dir / "annotations" / "master" / "test_dataset"
        
        self.master_images_dir.mkdir(parents=True)
        self.master_annotations_dir.mkdir(parents=True)
        
        # Create test image
        (self.master_images_dir / "train").mkdir()
        (self.master_images_dir / "train" / "test_image.jpg").write_text("mock image data")
        
        # Create master annotations
        master_data = {
            "dataset_info": {
                "name": "test_dataset",
                "version": "1.0",
                "schema_version": "1.0",
                "task_type": "detection",
                "classes": [
                    {
                        "id": 1,
                        "name": "test_category"
                    }
                ]
            },
            "images": [
                {
                    "id": 1,
                    "file_name": "test_image.jpg",
                    "file_path": str(self.master_images_dir / "train" / "test_image.jpg"),
                    "width": 640,
                    "height": 480,
                    "split": "train"
                }
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 100, 200, 150],
                    "type": "detection"
                }
            ]
        }
        
        with open(self.master_annotations_dir / "annotations.json", "w") as f:
            json.dump(master_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('wildtrain.pipeline.framework_data_manager.COCOAdapter')
    def test_create_coco_format(self, mock_adapter):
        """Test creating COCO format."""
        # Mock adapter
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.convert_to_format.return_value = {
            "images": [{"id": 1, "file_name": "test_image.jpg"}],
            "annotations": [{"id": 1, "image_id": 1}],
            "categories": [{"id": 1, "name": "test_category"}]
        }
        mock_adapter.return_value = mock_adapter_instance
        
        # Create COCO format
        coco_path = self.manager._create_coco_format("test_dataset")
        
        # Verify the directory structure was created
        coco_dir = Path(coco_path)
        assert coco_dir.exists()
        assert (coco_dir / "data").exists()
        assert (coco_dir / "annotations").exists()
        assert (coco_dir / "annotations" / "instances_train.json").exists()
    
    @patch('wildtrain.pipeline.framework_data_manager.YOLOAdapter')
    def test_create_yolo_format(self, mock_adapter):
        """Test creating YOLO format."""
        # Mock adapter
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.convert_to_format.return_value = {
            "annotations": {
                "train": {
                    "test_image.jpg": [
                        {"class_id": 0, "bbox": [0.5, 0.5, 0.2, 0.3]}
                    ]
                }
            },
            "names": {0: "test_category"}
        }
        mock_adapter.return_value = mock_adapter_instance
        
        # Create YOLO format
        yolo_path = self.manager._create_yolo_format("test_dataset")
        
        # Verify the directory structure was created
        yolo_dir = Path(yolo_path)
        assert yolo_dir.exists()
        assert (yolo_dir / "images").exists()
        assert (yolo_dir / "labels").exists()
        assert (yolo_dir / "data.yaml").exists()
        
        # Verify data.yaml content
        with open(yolo_dir / "data.yaml", 'r') as f:
            data_yaml = yaml.safe_load(f)
        
        assert 'path' in data_yaml
        assert 'train' in data_yaml
        assert 'names' in data_yaml 