"""
Tests for DVC integration functionality.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the modules we want to test
from wildtrain.pipeline.dvc_manager import DVCConfig, DVCManager, DVCStorageType
from wildtrain.pipeline.master_data_manager import MasterDataManager


class TestDVCManager:
    """Test DVC manager functionality."""

    def test_dvc_manager_initialization(self):
        """Test DVC manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test initialization without DVC installed (should handle gracefully)
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("dvc not found")

                with pytest.raises(RuntimeError, match="DVC is not installed"):
                    DVCManager(temp_path)

    def test_dvc_config(self):
        """Test DVC configuration."""
        config = DVCConfig(
            storage_type=DVCStorageType.S3,
            storage_path="s3://bucket/datasets",
            remote_name="s3-remote",
            auto_push=True,
        )

        assert config.storage_type == DVCStorageType.S3
        assert config.storage_path == "s3://bucket/datasets"
        assert config.remote_name == "s3-remote"
        assert config.auto_push is True

    def test_storage_type_enum(self):
        """Test storage type enumeration."""
        assert DVCStorageType.LOCAL.value == "local"
        assert DVCStorageType.S3.value == "s3"
        assert DVCStorageType.GCS.value == "gcs"
        assert DVCStorageType.AZURE.value == "azure"
        assert DVCStorageType.SSH.value == "ssh"
        assert DVCStorageType.HDFS.value == "hdfs"


class TestMasterDataManagerWithDVC:
    """Test master data manager with DVC integration."""

    def test_master_data_manager_with_dvc(self):
        """Test master data manager with DVC enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test with DVC disabled
            manager = MasterDataManager(temp_path, enable_dvc=False)
            assert manager.dvc_manager is None

            # Test with DVC enabled (should handle gracefully if DVC not installed)
            manager = MasterDataManager(temp_path, enable_dvc=True)
            # Should not raise an exception even if DVC is not available

    def test_store_dataset_with_dvc(self):
        """Test storing dataset with DVC tracking."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            manager = MasterDataManager(temp_path, enable_dvc=False)

            # Create test master data
            master_data = {
                "dataset_info": {
                    "name": "test_dataset",
                    "version": "1.0",
                    "schema_version": "1.0",
                    "task_type": "detection",
                    "classes": [{"id": 1, "name": "test_class"}],
                },
                "images": [
                    {
                        "id": 1,
                        "file_name": "test_image.jpg",
                        "width": 640,
                        "height": 480,
                        "split": "train",
                        "file_path": str(temp_path / "test_image.jpg"),
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [100, 100, 200, 150],
                        "area": 30000,
                        "iscrowd": 0,
                    }
                ],
            }

            # Create test image file
            test_image_path = temp_path / "test_image.jpg"
            with open(test_image_path, "wb") as f:
                f.write(b"fake_image_data")

            # Store dataset
            annotations_path = manager.store_dataset(
                "test_dataset", master_data, track_with_dvc=False
            )

            assert Path(annotations_path).exists()
            assert "test_dataset" in annotations_path

    def test_get_dataset_info_with_dvc(self):
        """Test getting dataset info with DVC information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            manager = MasterDataManager(temp_path, enable_dvc=False)

            # Create test dataset
            dataset_dir = temp_path / "data" / "annotations" / "master" / "test_dataset"
            dataset_dir.mkdir(parents=True, exist_ok=True)

            test_data = {
                "dataset_info": {
                    "name": "test_dataset",
                    "version": "1.0",
                    "schema_version": "1.0",
                    "task_type": "detection",
                    "classes": [{"id": 1, "name": "test_class"}],
                },
                "images": [
                    {
                        "id": 1,
                        "file_name": "test_image.jpg",
                        "width": 640,
                        "height": 480,
                        "split": "train",
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [100, 100, 200, 150],
                        "area": 30000,
                        "iscrowd": 0,
                    }
                ],
            }

            # Save test data
            import json

            with open(dataset_dir / "annotations.json", "w") as f:
                json.dump(test_data, f)

            # Get dataset info
            info = manager.get_dataset_info("test_dataset")

            assert info["dataset_name"] == "test_dataset"
            assert info["total_images"] == 1
            assert info["total_annotations"] == 1
            assert "dvc_info" in info

    def test_list_datasets_with_dvc(self):
        """Test listing datasets with DVC information."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            manager = MasterDataManager(temp_path, enable_dvc=False)

            # Create test datasets
            for dataset_name in ["dataset1", "dataset2"]:
                dataset_dir = (
                    temp_path / "data" / "annotations" / "master" / dataset_name
                )
                dataset_dir.mkdir(parents=True, exist_ok=True)

                test_data = {
                    "dataset_info": {
                        "name": dataset_name,
                        "version": "1.0",
                        "schema_version": "1.0",
                        "task_type": "detection",
                        "classes": [{"id": 1, "name": "test_class"}],
                    },
                    "images": [
                        {
                            "id": 1,
                            "file_name": "test_image.jpg",
                            "width": 640,
                            "height": 480,
                            "split": "train",
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [100, 100, 200, 150],
                            "area": 30000,
                            "iscrowd": 0,
                        }
                    ],
                }

                # Save test data
                import json

                with open(dataset_dir / "annotations.json", "w") as f:
                    json.dump(test_data, f)

            # List datasets
            datasets = manager.list_datasets()

            assert len(datasets) == 2
            dataset_names = [d["dataset_name"] for d in datasets]
            assert "dataset1" in dataset_names
            assert "dataset2" in dataset_names


class TestDVCIntegrationFeatures:
    """Test DVC integration features."""

    def test_remote_storage_setup(self):
        """Test remote storage setup functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock DVC manager
            with patch("wildtrain.pipeline.dvc_manager.DVCManager") as mock_dvc_manager:
                mock_instance = MagicMock()
                mock_dvc_manager.return_value = mock_instance
                mock_instance.setup_remote_storage.return_value = True

                manager = MasterDataManager(temp_path, enable_dvc=True)

                # Test remote storage setup
                result = manager.setup_remote_storage(
                    DVCStorageType.LOCAL, str(temp_path / "dvc_storage")
                )

                assert result is True

    def test_pull_dataset(self):
        """Test pulling dataset from remote storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock DVC manager
            with patch("wildtrain.pipeline.dvc_manager.DVCManager") as mock_dvc_manager:
                mock_instance = MagicMock()
                mock_dvc_manager.return_value = mock_instance
                mock_instance.pull_data.return_value = True

                manager = MasterDataManager(temp_path, enable_dvc=True)

                # Test pulling dataset
                result = manager.pull_dataset("test_dataset")

                assert result is True

    def test_create_data_pipeline(self):
        """Test creating data pipeline."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock DVC manager
            with patch("wildtrain.pipeline.dvc_manager.DVCManager") as mock_dvc_manager:
                mock_instance = MagicMock()
                mock_dvc_manager.return_value = mock_instance
                mock_instance.create_pipeline.return_value = True

                manager = MasterDataManager(temp_path, enable_dvc=True)

                # Test creating pipeline
                stages = [
                    {
                        "name": "import",
                        "command": "wildtrain dataset import data/raw coco dataset",
                        "deps": ["data/raw"],
                        "outs": ["data/processed"],
                    }
                ]

                result = manager.create_data_pipeline("test_pipeline", stages)

                assert result is True

    def test_get_dvc_status(self):
        """Test getting DVC status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Mock DVC manager
            with patch("wildtrain.pipeline.dvc_manager.DVCManager") as mock_dvc_manager:
                mock_instance = MagicMock()
                mock_dvc_manager.return_value = mock_instance
                mock_instance.get_status.return_value = {
                    "dvc_initialized": True,
                    "remote_configured": True,
                    "data_tracked": True,
                }

                manager = MasterDataManager(temp_path, enable_dvc=True)

                # Test getting DVC status
                status = manager.get_dvc_status()

                assert status["dvc_enabled"] is True
                assert status["dvc_initialized"] is True
                assert status["remote_configured"] is True
                assert status["data_tracked"] is True


if __name__ == "__main__":
    pytest.main([__file__])
