import os
import json
import tempfile
import pytest
from wildtrain.adapters.coco_adapter import COCOAdapter
from wildtrain.adapters.yolo_adapter import YOLOAdapter
from wildtrain.converters.coco_to_master import COCOToMasterConverter
from wildtrain.converters.yolo_to_master import YOLOToMasterConverter

# Real data paths (can be overridden with environment variables)
COCO_DATA_DIR = os.getenv('COCO_DATA_DIR', "D:/workspace/savmap/coco")
YOLO_DATA_DIR = os.getenv('YOLO_DATA_DIR', "D:/workspace/savmap/yolo")

def test_coco_to_master_converter(tmp_path):
    """Test COCO to master annotation conversion using real data."""
    if not os.path.exists(COCO_DATA_DIR):
        pytest.skip(f"COCO data directory not found: {COCO_DATA_DIR}")
    
    # Find COCO annotation file
    coco_files = [f for f in os.listdir(COCO_DATA_DIR) if f.endswith('.json')]
    if not coco_files:
        pytest.skip("No COCO annotation files found")
    
    coco_file = os.path.join(COCO_DATA_DIR, coco_files[0])
    
    # Convert COCO to master
    converter = COCOToMasterConverter(coco_file)
    converter.load_coco_annotation()
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection")
    
    # Save master annotation
    master_file = tmp_path / "master.json"
    converter.save_master_annotation(master_data, str(master_file))
    
    # Verify structure
    assert 'dataset_info' in master_data
    assert 'images' in master_data
    assert 'annotations' in master_data
    assert len(master_data['images']) > 0
    assert len(master_data['annotations']) > 0

def test_yolo_to_master_converter(tmp_path):
    """Test YOLO to master annotation conversion using real data."""
    if not os.path.exists(YOLO_DATA_DIR):
        pytest.skip(f"YOLO data directory not found: {YOLO_DATA_DIR}")
    
    # Find data.yaml file
    data_yaml_path = os.path.join(YOLO_DATA_DIR, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        pytest.skip("data.yaml not found in YOLO directory")
    
    # Convert YOLO to master
    converter = YOLOToMasterConverter(data_yaml_path)
    converter.load_yolo_data()
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection")
    
    # Save master annotation
    master_file = tmp_path / "master.json"
    converter.save_master_annotation(master_data, str(master_file))
    
    # Verify structure
    assert 'dataset_info' in master_data
    assert 'images' in master_data
    assert 'annotations' in master_data
    assert len(master_data['images']) > 0

def test_coco_adapter_with_real_data(tmp_path):
    """Test COCO adapter using master annotation generated from real COCO data."""
    if not os.path.exists(COCO_DATA_DIR):
        pytest.skip(f"COCO data directory not found: {COCO_DATA_DIR}")
    
    # Generate master annotation from COCO
    coco_files = [f for f in os.listdir(COCO_DATA_DIR) if f.endswith('.json')]
    if not coco_files:
        pytest.skip("No COCO annotation files found")
    
    coco_file = os.path.join(COCO_DATA_DIR, coco_files[0])
    converter = COCOToMasterConverter(coco_file)
    converter.load_coco_annotation()
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection")
    
    # Save master annotation
    master_file = tmp_path / "master.json"
    converter.save_master_annotation(master_data, str(master_file))
    
    # Test COCO adapter
    adapter = COCOAdapter(str(master_file))
    adapter.load_master_annotation()
    
    # Convert to COCO format
    coco_data = adapter.convert('train')
    
    # Verify COCO structure
    assert 'images' in coco_data
    assert 'annotations' in coco_data
    assert 'categories' in coco_data
    
    # Save COCO output
    coco_output = tmp_path / "coco_output.json"
    adapter.save(coco_data, str(coco_output))
    
    # Verify file was created
    assert coco_output.exists()

def test_yolo_adapter_with_real_data(tmp_path):
    """Test YOLO adapter using master annotation generated from real YOLO data."""
    if not os.path.exists(YOLO_DATA_DIR):
        raise FileNotFoundError(f"YOLO data directory not found: {YOLO_DATA_DIR}")
    
    # Find data.yaml file
    data_yaml_path = os.path.join(YOLO_DATA_DIR, 'data.yaml')
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError("data.yaml not found in YOLO directory")
    
    # Convert YOLO to master
    converter = YOLOToMasterConverter(data_yaml_path)
    converter.load_yolo_data()
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection")
    
    # Save master annotation
    master_file = tmp_path / "master.json"
    converter.save_master_annotation(master_data, str(master_file))
    
    # Test YOLO adapter
    adapter = YOLOAdapter(str(master_file))
    adapter.load_master_annotation()
    
    # Convert to YOLO format
    yolo_data = adapter.convert('train')
    
    # Debug output
    print(f"YOLO data keys: {list(yolo_data.keys())}")
    print(f"Number of images with labels: {len(yolo_data)}")
    if yolo_data:
        first_key = list(yolo_data.keys())[0]
        print(f"First image: {first_key}")
        print(f"Labels for first image: {yolo_data[first_key]}")
    
    # Verify YOLO structure
    assert len(yolo_data) > 0
    
    # Save YOLO output
    adapter.save(yolo_data)
    
def test_round_trip_conversion(tmp_path):
    """Test round-trip conversion: COCO -> Master -> COCO."""
    if not os.path.exists(COCO_DATA_DIR):
        pytest.skip(f"COCO data directory not found: {COCO_DATA_DIR}")
    
    # Generate master from COCO
    coco_files = [f for f in os.listdir(COCO_DATA_DIR) if f.endswith('.json')]
    if not coco_files:
        pytest.skip("No COCO annotation files found")
    
    coco_file = os.path.join(COCO_DATA_DIR, coco_files[0])
    converter = COCOToMasterConverter(coco_file)
    converter.load_coco_annotation()
    master_data = converter.convert_to_master("test_dataset", "1.0", "detection")
    
    # Save master
    master_file = tmp_path / "master.json"
    converter.save_master_annotation(master_data, str(master_file))
    
    # Convert back to COCO
    adapter = COCOAdapter(str(master_file))
    adapter.load_master_annotation()
    coco_data = adapter.convert('train')
    
    # Verify we have the same number of images and annotations
    original_coco = converter.coco_data
    assert len(coco_data['images']) == len(original_coco['images'])
    assert len(coco_data['annotations']) == len(original_coco['annotations']) 