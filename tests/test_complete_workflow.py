"""
Test the complete data pipeline workflow: Extract -> Transform -> Save -> Load using adapters.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json
import yaml

from wildtrain.pipeline.data_pipeline import DataPipeline


class TestCompleteWorkflow:
    """Test the complete data pipeline workflow."""
    
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
    
    def test_complete_workflow_coco_to_yolo(self):
        """
        Test complete workflow: COCO -> Master -> YOLO
        Extract -> Transform -> Save -> Load using adapters
        """
        pipeline = DataPipeline(str(self.test_data_dir))
        
        # Step 1: Extract (Import COCO dataset)
        print("Step 1: Extracting COCO dataset...")
        result = pipeline.import_dataset(
            source_path=str(self.coco_test_file),
            source_format="coco",
            dataset_name="test_workflow_dataset"
        )
        
        assert result['success'] is True
        assert result['dataset_name'] == "test_workflow_dataset"
        assert 'master_path' in result
        assert 'framework_paths' in result
        
        # Verify master data was created
        master_file = Path(result['master_path'])
        assert master_file.exists()
        
        # Load and verify master data structure
        with open(master_file, 'r') as f:
            master_data = json.load(f)
        
        assert 'dataset_info' in master_data
        assert 'images' in master_data
        assert 'annotations' in master_data
        assert len(master_data['images']) == 2
        assert len(master_data['annotations']) == 2
        
        print(f"✓ Extracted {len(master_data['images'])} images and {len(master_data['annotations'])} annotations")
        
        # Step 2: Transform (Verify data transformation capabilities)
        print("Step 2: Verifying transformation capabilities...")
        
        # Check that we can add transformations
        from wildtrain.transformations import AugmentationTransformer
        from wildtrain.config import AugmentationConfig
        
        config = AugmentationConfig(
            rotation_range=(-10, 10),
            probability=0.5,
            brightness_range=(0.9, 1.1)
        )
        
        augmentation_transformer = AugmentationTransformer(config)
        pipeline.add_transformation(augmentation_transformer)
        
        # Verify transformation was added
        status = pipeline.get_pipeline_status()
        assert len(status['transformation_pipeline']['transformer_types']) > 0
        
        print("✓ Transformation pipeline configured")
        
        # Step 3: Save (Verify framework formats were created)
        print("Step 3: Verifying framework formats...")
        
        # Check that COCO format was created
        coco_path = result['framework_paths'].get('coco')
        assert coco_path is not None
        
        coco_dir = Path(coco_path)
        assert coco_dir.exists()
        
        # Check COCO annotations file
        coco_annotations_file = coco_dir / "annotations" / "instances_train.json"
        assert coco_annotations_file.exists()
        
        # Load and verify COCO format
        with open(coco_annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        assert 'images' in coco_data
        assert 'annotations' in coco_data
        assert 'categories' in coco_data
        assert len(coco_data['images']) > 0
        assert len(coco_data['annotations']) > 0
        
        print(f"✓ COCO format created with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
        
        # Check that YOLO format was created
        yolo_path = result['framework_paths'].get('yolo')
        assert yolo_path is not None
        
        yolo_dir = Path(yolo_path)
        assert yolo_dir.exists()
        
        # Check YOLO data.yaml
        yolo_data_yaml = yolo_dir / "data.yaml"
        assert yolo_data_yaml.exists()
        
        # Load and verify YOLO data.yaml
        with open(yolo_data_yaml, 'r') as f:
            yolo_config = yaml.safe_load(f)
        
        assert 'path' in yolo_config
        assert 'train' in yolo_config
        assert 'names' in yolo_config
        
        print("✓ YOLO format created with data.yaml")
        
        # Step 4: Load using adapters (Export to different formats)
        print("Step 4: Testing adapter-based exports...")
        
        # Export to COCO format using adapter
        coco_export_result = pipeline.export_framework_format("test_workflow_dataset", "coco")
        assert coco_export_result['framework'] == 'coco'
        assert 'output_path' in coco_export_result
        assert 'annotations_file' in coco_export_result
        
        # Export to YOLO format using adapter
        yolo_export_result = pipeline.export_framework_format("test_workflow_dataset", "yolo")
        assert yolo_export_result['framework'] == 'yolo'
        assert 'output_path' in yolo_export_result
        assert 'images_dir' in yolo_export_result
        assert 'labels_dir' in yolo_export_result
        assert 'data_yaml' in yolo_export_result
        
        print("✓ Adapter-based exports successful")
        
        # Step 5: Verify data integrity
        print("Step 5: Verifying data integrity...")
        
        # Get dataset info
        info = pipeline.get_dataset_info("test_workflow_dataset")
        assert info['dataset_name'] == "test_workflow_dataset"
        assert info['total_images'] == 2
        assert info['total_annotations'] == 2
        
        # List datasets
        datasets = pipeline.list_datasets()
        assert len(datasets) > 0
        
        # Find our test dataset
        test_dataset = None
        for dataset in datasets:
            if dataset['dataset_name'] == 'test_workflow_dataset':
                test_dataset = dataset
                break
        
        assert test_dataset is not None
        assert test_dataset['total_images'] == 2
        assert test_dataset['total_annotations'] == 2
        
        print("✓ Data integrity verified")
        
        print("\n🎉 Complete workflow test passed!")
        print("✅ Extract: COCO -> Master format")
        print("✅ Transform: Transformation pipeline configured")
        print("✅ Save: Framework formats created")
        print("✅ Load: Adapter-based exports successful")
        print("✅ Verify: Data integrity maintained")
    
    def test_workflow_with_transformations(self):
        """
        Test workflow with actual transformations applied.
        """
        pipeline = DataPipeline(str(self.test_data_dir))
        
        # Import dataset
        result = pipeline.import_dataset(
            source_path=str(self.coco_test_file),
            source_format="coco",
            dataset_name="test_transformed_dataset",
            apply_transformations=True  # This would apply transformations during import
        )
        
        assert result['success'] is True
        
        # Verify that the dataset was created
        info = pipeline.get_dataset_info("test_transformed_dataset")
        assert info['dataset_name'] == "test_transformed_dataset"
        
        print("✓ Workflow with transformations test passed!")
    
    def test_error_handling(self):
        """
        Test error handling in the pipeline.
        """
        pipeline = DataPipeline(str(self.test_data_dir))
        
        # Test with non-existent file
        result = pipeline.import_dataset(
            source_path="non_existent_file.json",
            source_format="coco",
            dataset_name="test_error_dataset"
        )
        
        assert result['success'] is False
        assert 'error' in result
        
        # Test with unsupported format
        result = pipeline.import_dataset(
            source_path=str(self.coco_test_file),
            source_format="unsupported_format",
            dataset_name="test_error_dataset"
        )
        
        assert result['success'] is False
        assert 'error' in result
        
        print("✓ Error handling test passed!") 