import json
import pytest
import tempfile
from pathlib import Path
from wildtrain.validators.master_validator import MasterValidator
from wildtrain.validators.coco_validator import COCOValidator
from wildtrain.converters.coco_to_master import COCOToMasterConverter
from wildtrain.validators.yolo_validator import YOLOValidator
from wildtrain.converters.yolo_to_master import YOLOToMasterConverter
import shutil
import os
from PIL import Image


def create_invalid_coco_data():
    """Create COCO data with invalid annotations."""
    return {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600}
        ],
        "annotations": [
            # Valid annotation
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000},
            # Invalid: missing required field
            {"id": 2, "image_id": 1, "bbox": [100, 100, 200, 150]},  # missing category_id
            # Invalid: references non-existent image
            {"id": 3, "image_id": 999, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000},
            # Invalid: negative area
            {"id": 4, "image_id": 2, "category_id": 1, "bbox": [100, 100, 200, 150], "area": -1000},
            # Invalid: invalid bbox dimensions
            {"id": 5, "image_id": 2, "category_id": 1, "bbox": [100, 100, 0, 150], "area": 0},  # width = 0
        ],
        "categories": [
            {"id": 1, "name": "test_category", "supercategory": "test"}
        ]
    }


def create_invalid_master_data():
    """Create master data with invalid annotations."""
    return {
        "dataset_info": {
            "name": "test_dataset",
            "version": "1.0",
            "schema_version": "1.0",
            "task_type": "detection",
            "classes": [
                {"id": 1, "name": "test_category", "supercategory": "test"}
            ]
        },
        "images": [
            {"id": 1, "file_name": "image1.jpg", "width": 640, "height": 480, "split": "train", "path": "/path/to/image1.jpg"},
            {"id": 2, "file_name": "image2.jpg", "width": 800, "height": 600, "split": "train", "path": "/path/to/image2.jpg"}
        ],
        "annotations": [
            # Valid annotation
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0, "segmentation": [], "keypoints": [], "attributes": {}},
            # Invalid: missing required field
            {"image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0, "segmentation": [], "keypoints": [], "attributes": {}},  # missing id
            # Invalid: references non-existent image
            {"id": 3, "image_id": 999, "category_id": 1, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0, "segmentation": [], "keypoints": [], "attributes": {}},
            # Invalid: references non-existent category
            {"id": 4, "image_id": 2, "category_id": 999, "bbox": [100, 100, 200, 150], "area": 30000, "iscrowd": 0, "segmentation": [], "keypoints": [], "attributes": {}},
            # Invalid: negative area
            {"id": 5, "image_id": 2, "category_id": 1, "bbox": [100, 100, 200, 150], "area": -1000, "iscrowd": 0, "segmentation": [], "keypoints": [], "attributes": {}},
        ]
    }


class TestInvalidAnnotationFiltering:
    """Test the invalid annotation filtering functionality."""
    
    def test_master_validator_with_filtering(self):
        """Test that MasterValidator can filter invalid annotations."""
        master_data = create_invalid_master_data()
        
        # Test without filtering (should fail)
        validator = MasterValidator(filter_invalid_annotations=False)
        is_valid, errors, warnings = validator.validate_data(master_data)
        assert not is_valid
        assert len(errors) > 0
        
        # Test with filtering (should pass)
        validator = MasterValidator(filter_invalid_annotations=True)
        is_valid, errors, warnings = validator.validate_data(master_data)
        assert is_valid
        assert len(errors) == 0
        assert len(warnings) > 0
        
        # Check that invalid annotations were filtered out
        skipped_count = validator.get_skipped_count()
        assert skipped_count == 4  # 4 invalid annotations
        
        # Check that only valid annotations remain
        filtered_data = validator.master_data
        assert filtered_data is not None
        assert len(filtered_data['annotations']) == 1  # Only the valid one
        
        # Check skipped annotation details
        skipped = validator.get_skipped_annotations()
        assert len(skipped) == 4
        assert any("missing 'id' field" in s.get('reason', '') for s in skipped)
        assert any("references non-existent image_id" in s.get('reason', '') for s in skipped)
        assert any("references non-existent category_id" in s.get('reason', '') for s in skipped)
        assert any("negative area" in s.get('reason', '') for s in skipped)
    
    def test_coco_validator_with_filtering(self):
        """Test that COCOValidator can filter invalid annotations."""
        coco_data = create_invalid_coco_data()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_data, f)
            temp_file = f.name
        
        try:
            # Test without filtering (should fail)
            validator = COCOValidator(temp_file, filter_invalid_annotations=False)
            is_valid, errors, warnings = validator.validate()
            assert not is_valid
            assert len(errors) > 0
            
            # Test with filtering (should pass)
            validator = COCOValidator(temp_file, filter_invalid_annotations=True)
            is_valid, errors, warnings = validator.validate()
            assert is_valid
            assert len(errors) == 0
            assert len(warnings) > 0
            
            # Check that invalid annotations were filtered out
            skipped_count = validator.get_skipped_count()
            assert skipped_count > 0
            
            # Check that only valid annotations remain
            filtered_data = validator.coco_data
            assert len(filtered_data['annotations']) == 1  # Only the valid one
            
        finally:
            Path(temp_file).unlink()
    
    def test_coco_converter_with_filtering(self):
        """Test that COCOToMasterConverter can filter invalid annotations."""
        coco_data = create_invalid_coco_data()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(coco_data, f)
            temp_file = f.name
        
        try:
            # Test without filtering (should fail)
            converter = COCOToMasterConverter(temp_file)
            with pytest.raises(ValueError):
                converter.load_coco_annotation(filter_invalid_annotations=False)
            
            # Test with filtering (should pass)
            converter = COCOToMasterConverter(temp_file)
            converter.load_coco_annotation(filter_invalid_annotations=True)
            
            # Convert to master format
            master_data = converter.convert_to_master(
                "test_dataset", 
                filter_invalid_annotations=True
            )
            
            # Check that only valid annotations remain
            assert len(master_data['annotations']) == 1
            assert master_data['annotations'][0]['id'] == 1  # Only the valid annotation
            
        finally:
            Path(temp_file).unlink()
    
    def test_filtering_warning_messages(self):
        """Test that appropriate warning messages are displayed."""
        master_data = create_invalid_master_data()
        
        validator = MasterValidator(filter_invalid_annotations=True)
        is_valid, errors, warnings = validator.validate_data(master_data)
        
        # Check that warnings include skipped count
        warning_messages = " ".join(warnings)
        assert "Skipped" in warning_messages
        assert "invalid annotations" in warning_messages
        
        # Check that we can get detailed information
        skipped = validator.get_skipped_annotations()
        assert len(skipped) == 4
        
        # Check that each skipped annotation has the required fields
        for skip_info in skipped:
            assert 'index' in skip_info
            assert 'annotation' in skip_info
            assert 'reason' in skip_info
            assert isinstance(skip_info['reason'], str)
    
    def test_empty_dataset_handling(self):
        """Test that empty datasets are handled correctly."""
        empty_data = {
            "dataset_info": {
                "name": "empty_dataset",
                "version": "1.0",
                "schema_version": "1.0",
                "task_type": "detection",
                "classes": []
            },
            "images": [],
            "annotations": []
        }
        
        validator = MasterValidator(filter_invalid_annotations=True)
        is_valid, errors, warnings = validator.validate_data(empty_data)
        
        assert is_valid
        assert len(errors) == 0
        assert len(warnings) > 0  # Should warn about empty dataset
        assert validator.get_skipped_count() == 0 

    def test_yolo_validator_and_converter_with_filtering(self):
        """Test that YOLOValidator and YOLOToMasterConverter can filter invalid annotation lines."""
        real_image_path = Path(r"D:/workspace/savmap/yolo/train/images/00a033fefe644429a1e0fcffe88f8b39_0_4_0_1024_640_1664.jpg")
        if not real_image_path.exists():
            pytest.skip(f"Real image not found: {real_image_path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup YOLO directory structure
            yolo_root = Path(temp_dir) / "yolo"
            images_dir = yolo_root / "train" / "images"
            labels_dir = yolo_root / "train" / "labels"
            images_dir.mkdir(parents=True)
            labels_dir.mkdir(parents=True)

            # Copy real image as img1.jpg
            shutil.copy(real_image_path, images_dir / "img1.jpg")
            # Create a valid dummy image for img2.jpg
            img2_path = images_dir / "img2.jpg"
            img = Image.new("RGB", (10, 10), color=(255, 255, 255))
            img.save(img2_path)

            # Create label files: one valid, one with multiple invalid lines
            valid_label = "0 0.5 0.5 0.2 0.2\n"
            invalid_label = "0 0.5 0.5 0.2\n"  # too few values
            invalid_label += "1 1.5 0.5 0.2 0.2\n"  # coord out of range
            invalid_label += "a 0.5 0.5 0.2 0.2\n"  # invalid class id
            invalid_label += "0 0.5 0.5 0.2 0.2 0.1\n"  # odd number of seg points
            (labels_dir / "img1.txt").write_text(valid_label)
            (labels_dir / "img2.txt").write_text(invalid_label)

            # Create data.yaml
            data_yaml = yolo_root / "data.yaml"
            data_yaml.write_text(
                f"path: {yolo_root}\n"
                "train: train/images\n"
                "names:\n"
                "  0: class0\n"
                "  1: class1\n"
            )

            # Test validator without filtering (should fail)
            validator = YOLOValidator(str(data_yaml), filter_invalid_annotations=False)
            is_valid, errors, warnings = validator.validate()
            assert not is_valid
            assert len(errors) > 0

            # Test validator with filtering (should pass, and skip 4 lines)
            validator = YOLOValidator(str(data_yaml), filter_invalid_annotations=True)
            is_valid, errors, warnings = validator.validate()
            print('YOLOValidator errors (filtering):', errors)
            print('YOLOValidator warnings (filtering):', warnings)
            assert is_valid
            assert len(errors) == 0
            assert validator.get_skipped_count() == 4
            skipped = validator.get_skipped_annotations()
            assert len(skipped) == 4
            reasons = [s['reason'] for s in skipped]
            assert any('too few values' in r for r in reasons)
            assert any('coordinate 1 out of range' in r for r in reasons)
            assert any('invalid class id' in r for r in reasons)
            assert any('invalid segmentation count' in r for r in reasons)

            # Test converter with filtering (should only include valid annotation)
            converter = YOLOToMasterConverter(str(data_yaml))
            converter.load_yolo_data(filter_invalid_annotations=True)
            master_data = converter.convert_to_master("test_dataset", filter_invalid_annotations=True)
            assert 'annotations' in master_data
            # Only two valid annotations (one from img1.txt, one from img2.txt)
            assert len(master_data['annotations']) == 2
            anns = master_data['annotations']
            # Check that both annotations are for category_id 0
            assert all(ann['category_id'] == 0 for ann in anns)
            # Check that the images are correct
            img_ids = [ann['image_id'] for ann in anns]
            img_files = {img['id']: img['file_name'] for img in master_data['images']}
            assert set(img_files[img_id] for img_id in img_ids) == {'img1.jpg', 'img2.jpg'}
            # Check bbox for img1.jpg (from img1.txt)
            img1 = next(img for img in master_data['images'] if img['file_name'] == 'img1.jpg')
            ann1 = next(ann for ann in anns if ann['image_id'] == img1['id'])
            width1, height1 = img1['width'], img1['height']
            expected_bbox1 = [0.5 * width1 - 0.2 * width1 / 2, 0.5 * height1 - 0.2 * height1 / 2, 0.2 * width1, 0.2 * height1]
            for a, b in zip(ann1['bbox'], expected_bbox1):
                assert abs(a - b) < 1e-3
            # Check bbox for img2.jpg (from first valid line in img2.txt)
            img2 = next(img for img in master_data['images'] if img['file_name'] == 'img2.jpg')
            ann2 = next(ann for ann in anns if ann['image_id'] == img2['id'])
            width2, height2 = img2['width'], img2['height']
            expected_bbox2 = [0.5 * width2 - 0.2 * width2 / 2, 0.5 * height2 - 0.2 * height2 / 2, 0.2 * width2, 0.2 * height2]
            for a, b in zip(ann2['bbox'], expected_bbox2):
                assert abs(a - b) < 1e-3 