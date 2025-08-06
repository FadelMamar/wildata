"""
Test fixtures for API tests.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from wildata.api.main import app
from wildata.config import ROOT


@pytest.fixture
def test_client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_import_request() -> dict:
    """Sample import dataset request data."""
    return {
        "source_path": r"D:\workspace\savmap\coco\annotations\train.json",
        "source_format": "coco",
        "dataset_name": "savmap",
        "root": str(ROOT / "data" / "api-testing-dataset"),
        "split_name": "train",
        "processing_mode": "batch",
        "track_with_dvc": False,
        "bbox_tolerance": 5,
        "disable_roi": False,
    }


@pytest.fixture
def sample_bulk_import_request() -> dict:
    """Sample bulk import request data."""
    return {
        "source_paths": [
            r"D:\workspace\data\general_dataset\tiled-data\coco-dataset\annotations\annotations_val.json",
            r"D:\workspace\data\general_dataset\tiled-data\coco-dataset\annotations\annotations_train.json",
        ],
        "source_format": "coco",
        "root": str(ROOT / "data" / "api-testing-dataset"),
        "split_name": "train",
        "processing_mode": "batch",
        "track_with_dvc": False,
        "bbox_tolerance": 5,
        "disable_roi": False,
    }


@pytest.fixture
def sample_roi_request() -> dict:
    """Sample ROI creation request data."""
    return {
        "source_path": r"D:\workspace\data\general_dataset\tiled-data\coco-dataset\annotations\annotations_val.json",
        "source_format": "coco",
        "dataset_name": "test_roi_dataset",
        "root": str(ROOT / "data" / "api-testing-dataset"),
        "split_name": "val",
        "bbox_tolerance": 5,
        "roi_config": {
            "random_roi_count": 2,
            "roi_box_size": 384,
            "min_roi_size": 32,
            "dark_threshold": 0.7,
            "background_class": "background",
            "save_format": "jpg",
            "quality": 85,
            "sample_background": True,
        },
        "draw_original_bboxes": False,
    }


@pytest.fixture
def sample_gps_request() -> dict:
    """Sample GPS update request data."""
    return {
        "image_folder": r"D:\workspace\data\savmap_dataset_v2\images_splits",
        "csv_path": str(ROOT / "examples" / "mock_csv.csv"),
        "output_dir": r"D:\workspace\data\savmap_dataset_v2_splits_with_gps",
        "skip_rows": 0,
        "filename_col": "filename",
        "lat_col": "latitude",
        "lon_col": "longitude",
        "alt_col": "altitude",
    }


@pytest.fixture
def sample_visualize_request() -> dict:
    """Sample visualization request data."""
    return {
        "dataset_name": "test_dataset",
        "root_data_directory": "/path/to/data",
        "split": "train",
        "load_as_single_class": False,
        "background_class_name": "background",
        "single_class_name": "wildlife",
        "keep_classes": ["animal", "bird"],
        "discard_classes": ["vehicle"],
    }


@pytest.fixture
def mock_job_id() -> str:
    """Mock job ID for testing."""
    return "test-job-12345"
