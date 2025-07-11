import json
import os
import tempfile

import pytest
from wildtrain.validators.master_validator import MasterValidator


def create_valid_master_data():
    """Create a valid master annotation data structure."""
    return {
        "dataset_info": {
            "name": "test_dataset",
            "version": "1.0",
            "schema_version": "1.0",
            "task_type": "detection",
            "classes": [
                {"id": 1, "name": "class1", "supercategory": "test"},
                {"id": 2, "name": "class2", "supercategory": "test"},
            ],
        },
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "width": 640,
                "height": 480,
                "split": "train",
                "path": "/path/to/image1.jpg",
            },
            {
                "id": 2,
                "file_name": "image2.jpg",
                "width": 800,
                "height": 600,
                "split": "val",
                "path": "/path/to/image2.jpg",
            },
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
                "iscrowd": 0,
                "segmentation": [],
                "keypoints": [],
                "attributes": {},
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 2,
                "bbox": [150, 150, 100, 100],
                "area": 10000,
                "iscrowd": 0,
                "segmentation": [],
                "keypoints": [],
                "attributes": {},
            },
        ],
    }


def create_invalid_master_data():
    """Create an invalid master annotation data structure."""
    return {
        "dataset_info": {
            "name": "test_dataset",
            "version": "1.0",
            "schema_version": "1.0",
            "task_type": "detection",
            "classes": [
                {"id": 1, "name": "class1"},  # Missing supercategory
                {"id": 1, "name": "class2"},  # Duplicate ID
            ],
        },
        "images": [
            {
                "id": 1,
                "file_name": "image1.jpg",
                "width": -1,  # Invalid width
                "height": 480,
                "split": "train",
                "path": "/path/to/image1.jpg",
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 999,  # Non-existent image_id
                "category_id": 999,  # Non-existent category_id
                "bbox": [100, 100, 200],  # Invalid bbox length
                "area": -1000,  # Negative area
                "iscrowd": 0,
                "segmentation": [],
                "keypoints": [],
                "attributes": {},
            }
        ],
    }


def test_master_validator_with_valid_data():
    """Test master validator with valid data."""
    validator = MasterValidator()
    valid_data = create_valid_master_data()

    is_valid, errors, warnings = validator.validate_data(valid_data)

    assert is_valid
    assert len(errors) == 0
    assert len(warnings) == 0


def test_master_validator_with_invalid_data():
    """Test master validator with invalid data."""
    validator = MasterValidator()
    invalid_data = create_invalid_master_data()

    is_valid, errors, warnings = validator.validate_data(invalid_data)

    assert not is_valid
    assert len(errors) > 0
    # Should have errors for invalid bbox, schema validation errors, etc.
    assert any("bbox" in error.lower() for error in errors)


def test_master_validator_with_file():
    """Test master validator with a file."""
    validator = MasterValidator()
    valid_data = create_valid_master_data()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(valid_data, f)
        temp_file = f.name

    try:
        validator = MasterValidator(master_annotation_path=temp_file)
        is_valid, errors, warnings = validator.validate()

        assert is_valid
        assert len(errors) == 0
        assert len(warnings) == 0
    finally:
        os.unlink(temp_file)


def test_master_validator_missing_required_fields():
    """Test master validator with missing required fields."""
    validator = MasterValidator()

    # Missing dataset_info
    invalid_data = {"images": [], "annotations": []}

    is_valid, errors, warnings = validator.validate_data(invalid_data)

    assert not is_valid
    assert len(errors) > 0
    assert any("dataset_info" in error.lower() for error in errors)


def test_master_validator_empty_dataset():
    """Test master validator with empty dataset."""
    validator = MasterValidator()

    empty_data = {
        "dataset_info": {
            "name": "empty_dataset",
            "version": "1.0",
            "schema_version": "1.0",
            "task_type": "detection",
            "classes": [],
        },
        "images": [],
        "annotations": [],
    }

    is_valid, errors, warnings = validator.validate_data(empty_data)

    assert is_valid  # Empty dataset should be valid
    assert len(errors) == 0
    assert len(warnings) > 0  # Should warn about empty dataset


def test_master_validator_validation_summary():
    """Test master validator validation summary."""
    validator = MasterValidator()
    valid_data = create_valid_master_data()

    summary = validator.get_validation_summary()

    # Test with no data loaded
    assert "error" in summary

    # Test with data loaded
    validator.validate_data(valid_data)
    summary = validator.get_validation_summary()

    assert "dataset_info" in summary
    assert "statistics" in summary
    assert "issues" in summary

    # Check dataset info
    dataset_info = summary["dataset_info"]
    assert dataset_info["name"] == "test_dataset"
    assert dataset_info["version"] == "1.0"
    assert dataset_info["task_type"] == "detection"
    assert dataset_info["num_classes"] == 2

    # Check statistics
    stats = summary["statistics"]
    assert stats["total_images"] == 2
    assert stats["total_annotations"] == 2
    assert "train" in stats["images_per_split"]
    assert "val" in stats["images_per_split"]

    # Check issues
    issues = summary["issues"]
    assert issues["is_valid"] is True
    assert issues["error_count"] == 0


def test_master_validator_schema_loading():
    """Test master validator schema loading."""
    validator = MasterValidator()

    # Test schema loading
    validator.load_schema()
    assert validator.schema is not None
    assert "properties" in validator.schema
    assert "dataset_info" in validator.schema["properties"]


def test_master_validator_custom_schema_path():
    """Test master validator with custom schema path."""
    # Create a custom schema file
    custom_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["dataset_info"],
        "properties": {
            "dataset_info": {
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            }
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(custom_schema, f)
        schema_file = f.name

    try:
        validator = MasterValidator(schema_path=schema_file)
        validator.load_schema()

        # Test with minimal valid data
        minimal_data = {"dataset_info": {"name": "test"}}

        is_valid, errors, warnings = validator.validate_data(minimal_data)
        assert is_valid
        assert len(errors) == 0
    finally:
        os.unlink(schema_file)


def test_master_validator_invalid_schema_file():
    """Test master validator with invalid schema file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json content")
        invalid_schema_file = f.name

    try:
        validator = MasterValidator(schema_path=invalid_schema_file)
        with pytest.raises(ValueError):
            validator.load_schema()
    finally:
        os.unlink(invalid_schema_file)


def test_master_validator_missing_schema_file():
    """Test master validator with missing schema file."""
    validator = MasterValidator(schema_path="nonexistent_schema.json")
    with pytest.raises(FileNotFoundError):
        validator.load_schema()


def test_master_validator_with_real_converted_data():
    """Test master validator with real converted data from converters."""
    # This test will be skipped if no real data is available
    coco_data_dir = os.getenv("COCO_DATA_DIR", "data/savmap/coco")
    if not os.path.exists(coco_data_dir):
        pytest.skip(f"COCO data directory not found: {coco_data_dir}")

    # Import converters
    from wildtrain.converters.coco_to_master import COCOToMasterConverter

    # Find a COCO file
    coco_files = [f for f in os.listdir(coco_data_dir) if f.endswith(".json")]
    if not coco_files:
        pytest.skip("No COCO annotation files found")

    coco_file = os.path.join(coco_data_dir, coco_files[0])

    # Convert to master format
    converter = COCOToMasterConverter(coco_file)
    converter.load_coco_annotation()
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection")

    # Validate the converted data
    validator = MasterValidator()
    is_valid, errors, warnings = validator.validate_data(master_data)

    assert is_valid
    assert len(errors) == 0

    # Get validation summary
    summary = validator.get_validation_summary()
    assert summary["issues"]["is_valid"] is True
    assert summary["statistics"]["total_images"] > 0
    assert summary["statistics"]["total_annotations"] > 0
