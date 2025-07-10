import os
import json
import pytest
from wildtrain.converters.coco_to_master import COCOToMasterConverter
from wildtrain.converters.yolo_to_master import YOLOToMasterConverter

COCO_DATA_DIR = os.getenv('COCO_DATA_DIR', 'data/savmap/coco')
YOLO_DATA_DIR = os.getenv('YOLO_DATA_DIR', 'data/savmap/yolo')

def test_coco_to_master_conversion(tmp_path):
    if not os.path.exists(COCO_DATA_DIR):
        pytest.skip(f"COCO data directory not found: {COCO_DATA_DIR}")
    coco_files = [f for f in os.listdir(COCO_DATA_DIR) if f.endswith('.json')]
    if not coco_files:
        pytest.skip("No COCO annotation files found")
    coco_file = os.path.join(COCO_DATA_DIR, coco_files[0])
    converter = COCOToMasterConverter(coco_file)
    converter.load_coco_annotation()
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection", validate_output=True)
    master_file = tmp_path / "master.json"
    converter.save_master_annotation(master_data, str(master_file))
    assert os.path.exists(master_file)
    with open(master_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert 'dataset_info' in loaded
    assert 'images' in loaded
    assert 'annotations' in loaded
    assert len(loaded['images']) > 0
    assert len(loaded['annotations']) > 0

def test_yolo_to_master_conversion(tmp_path):
    if not os.path.exists(YOLO_DATA_DIR):
        pytest.skip(f"YOLO data directory not found: {YOLO_DATA_DIR}")
    data_yaml_path = os.path.join(YOLO_DATA_DIR, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        pytest.skip("data.yaml not found in YOLO directory")
    converter = YOLOToMasterConverter(data_yaml_path)
    converter.load_yolo_data()
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection", validate_output=True)
    master_file = tmp_path / "master.json"
    converter.save_master_annotation(master_data, str(master_file))
    assert os.path.exists(master_file)
    with open(master_file, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    assert 'dataset_info' in loaded
    assert 'images' in loaded
    assert 'annotations' in loaded
    assert len(loaded['images']) > 0

def test_converter_validation_disabled():
    """Test that converters work when validation is disabled."""
    if not os.path.exists(COCO_DATA_DIR):
        pytest.skip(f"COCO data directory not found: {COCO_DATA_DIR}")
    coco_files = [f for f in os.listdir(COCO_DATA_DIR) if f.endswith('.json')]
    if not coco_files:
        pytest.skip("No COCO annotation files found")
    coco_file = os.path.join(COCO_DATA_DIR, coco_files[0])
    converter = COCOToMasterConverter(coco_file)
    converter.load_coco_annotation()
    # This should work without validation
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection", validate_output=False)
    assert 'dataset_info' in master_data
    assert 'images' in master_data
    assert 'annotations' in master_data 