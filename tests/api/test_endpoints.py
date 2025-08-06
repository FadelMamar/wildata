"""
Test API endpoints.
"""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient
from wildata.api.models.jobs import JobStatus


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, test_client: TestClient):
        """Test basic health check endpoint."""
        response = test_client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_detailed_health_check(self, test_client: TestClient):
        """Test detailed health check endpoint."""
        response = test_client.get("/api/v1/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data

    def test_metrics_endpoint(self, test_client: TestClient):
        """Test metrics endpoint."""
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "requests_per_second" in data
        assert "response_time_avg" in data


class TestDatasetEndpoints:
    """Test dataset management endpoints."""

    def test_import_dataset_endpoint(
        self, test_client: TestClient, sample_import_request: dict
    ):
        """Test dataset import endpoint."""
        with patch(
            "wildata.api.services.task_handlers.handle_import_dataset"
        ) as mock_handler:
            mock_handler.return_value = Mock(
                success=True,
                message="Dataset imported successfully",
                data={"dataset_name": "test_dataset"},
            )

            response = test_client.post(
                "/api/v1/datasets/import", json=sample_import_request
            )
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["message"] == "Dataset import job started"

    def test_bulk_import_dataset_endpoint(
        self, test_client: TestClient, sample_bulk_import_request: dict
    ):
        """Test bulk dataset import endpoint."""
        with patch(
            "wildata.api.services.task_handlers.handle_bulk_import"
        ) as mock_handler:
            mock_handler.return_value = Mock(
                success=True,
                message="Bulk import completed",
                data={"total_files": 2, "successful_imports": 2},
            )

            response = test_client.post(
                "/api/v1/datasets/import/bulk", json=sample_bulk_import_request
            )
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["message"] == "Bulk import job started"

    def test_list_datasets_endpoint(self, test_client: TestClient):
        """Test list datasets endpoint."""
        with patch("wildata.pipeline.DataPipeline") as mock_pipeline:
            mock_pipeline.return_value.list_datasets.return_value = [
                {"name": "dataset1", "type": "detection"},
                {"name": "dataset2", "type": "classification"},
            ]

            response = test_client.get("/api/v1/datasets")
            assert response.status_code == 200
            data = response.json()
            assert "datasets" in data
            assert len(data["datasets"]) == 2

    def test_get_dataset_info_endpoint(self, test_client: TestClient):
        """Test get dataset info endpoint."""
        with patch("wildata.pipeline.DataPipeline") as mock_pipeline:
            mock_pipeline.return_value.get_dataset_info.return_value = {
                "name": "test_dataset",
                "type": "detection",
                "num_images": 100,
                "num_annotations": 500,
            }

            response = test_client.get("/api/v1/datasets/test_dataset")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test_dataset"
            assert data["type"] == "detection"

    def test_export_dataset_endpoint(self, test_client: TestClient):
        """Test export dataset endpoint."""
        with patch(
            "wildata.api.services.task_handlers.handle_export_dataset"
        ) as mock_handler:
            mock_handler.return_value = Mock(
                success=True,
                message="Dataset exported successfully",
                data={"dataset_name": "test_dataset", "target_format": "coco"},
            )

            response = test_client.post(
                "/api/v1/datasets/test_dataset/export",
                json={
                    "target_format": "coco",
                    "target_path": "/path/to/export",
                    "root": "data",
                },
            )
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["message"] == "Dataset export job started"


class TestROIEndpoints:
    """Test ROI dataset endpoints."""

    def test_create_roi_endpoint(
        self, test_client: TestClient, sample_roi_request: dict
    ):
        """Test ROI dataset creation endpoint."""
        with patch(
            "wildata.api.services.task_handlers.handle_create_roi"
        ) as mock_handler:
            mock_handler.return_value = Mock(
                success=True,
                message="ROI dataset created successfully",
                data={"dataset_name": "test_roi_dataset"},
            )

            response = test_client.post("/api/v1/roi/create", json=sample_roi_request)
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["message"] == "ROI dataset creation job started"

    def test_bulk_create_roi_endpoint(
        self, test_client: TestClient, sample_bulk_import_request: dict
    ):
        """Test bulk ROI dataset creation endpoint."""
        # Add roi_config to the bulk request
        sample_bulk_import_request["roi_config"] = {
            "width": 512,
            "height": 512,
            "stride": 256,
            "min_visibility": 0.1,
        }

        with patch(
            "wildata.api.services.task_handlers.handle_bulk_create_roi"
        ) as mock_handler:
            mock_handler.return_value = Mock(
                success=True,
                message="Bulk ROI creation completed",
                data={"total_files": 2, "successful_creations": 2},
            )

            response = test_client.post(
                "/api/v1/roi/create/bulk", json=sample_bulk_import_request
            )
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["message"] == "Bulk ROI creation job started"

    def test_get_roi_dataset_info_endpoint(self, test_client: TestClient):
        """Test get ROI dataset info endpoint."""
        with patch("wildata.pipeline.DataPipeline") as mock_pipeline:
            mock_pipeline.return_value.get_dataset_info.return_value = {
                "name": "test_roi_dataset",
                "type": "roi",
                "num_images": 50,
                "roi_config": {"width": 512, "height": 512},
            }

            response = test_client.get("/api/v1/roi/test_roi_dataset")
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "test_roi_dataset"
            assert data["type"] == "roi"


class TestGPSEndpoints:
    """Test GPS update endpoints."""

    def test_update_gps_endpoint(
        self, test_client: TestClient, sample_gps_request: dict
    ):
        """Test GPS update endpoint."""
        with patch(
            "wildata.api.services.task_handlers.handle_update_gps"
        ) as mock_handler:
            mock_handler.return_value = Mock(
                success=True,
                message="GPS data updated successfully",
                data={"image_folder": "/path/to/images"},
            )

            response = test_client.post("/api/v1/gps/update", json=sample_gps_request)
            assert response.status_code == 202
            data = response.json()
            assert "job_id" in data
            assert data["message"] == "GPS update job started"


class TestVisualizationEndpoints:
    """Test visualization endpoints."""

    def test_visualize_classification_endpoint(
        self, test_client: TestClient, sample_visualize_request: dict
    ):
        """Test classification visualization endpoint."""
        with patch(
            "wildata.visualization.visualize_classification_dataset"
        ) as mock_viz:
            mock_viz.return_value = "http://localhost:5151"

            response = test_client.post(
                "/api/v1/visualize/classification", json=sample_visualize_request
            )
            assert response.status_code == 200
            data = response.json()
            assert "visualization_url" in data
            assert data["visualization_url"] == "http://localhost:5151"

    def test_visualize_detection_endpoint(
        self, test_client: TestClient, sample_visualize_request: dict
    ):
        """Test detection visualization endpoint."""
        with patch("wildata.visualization.visualize_detection_dataset") as mock_viz:
            mock_viz.return_value = "http://localhost:5151"

            response = test_client.post(
                "/api/v1/visualize/detection", json=sample_visualize_request
            )
            assert response.status_code == 200
            data = response.json()
            assert "visualization_url" in data
            assert data["visualization_url"] == "http://localhost:5151"


class TestJobEndpoints:
    """Test job management endpoints."""

    def test_get_job_status_endpoint(self, test_client: TestClient, mock_job_id: str):
        """Test get job status endpoint."""
        with patch("wildata.api.services.job_queue.JobQueue.get_job") as mock_get_job:
            mock_job = Mock()
            mock_job.job_id = mock_job_id
            mock_job.status = JobStatus.COMPLETED
            mock_job.progress = 100.0
            mock_job.result = Mock(success=True, message="Job completed successfully")
            mock_get_job.return_value = mock_job

            response = test_client.get(f"/api/v1/jobs/{mock_job_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["job_id"] == mock_job_id
            assert data["status"] == "completed"
            assert data["progress"] == 100.0

    def test_list_jobs_endpoint(self, test_client: TestClient):
        """Test list jobs endpoint."""
        with patch(
            "wildata.api.services.job_queue.JobQueue.list_jobs"
        ) as mock_list_jobs:
            mock_jobs = [
                Mock(
                    job_id="job1", status=JobStatus.COMPLETED, job_type="import_dataset"
                ),
                Mock(job_id="job2", status=JobStatus.RUNNING, job_type="bulk_import"),
            ]
            mock_list_jobs.return_value = mock_jobs

            response = test_client.get("/api/v1/jobs")
            assert response.status_code == 200
            data = response.json()
            assert "jobs" in data
            assert len(data["jobs"]) == 2

    def test_cancel_job_endpoint(self, test_client: TestClient, mock_job_id: str):
        """Test cancel job endpoint."""
        with patch("wildata.api.services.job_queue.JobQueue.cancel_job") as mock_cancel:
            mock_cancel.return_value = True

            response = test_client.delete(f"/api/v1/jobs/{mock_job_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == f"Job {mock_job_id} cancelled successfully"


class TestErrorHandling:
    """Test error handling in endpoints."""

    def test_invalid_request_data(self, test_client: TestClient):
        """Test handling of invalid request data."""
        response = test_client.post("/api/v1/datasets/import", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error

    def test_job_not_found(self, test_client: TestClient, mock_job_id: str):
        """Test handling of job not found."""
        with patch("wildata.api.services.job_queue.JobQueue.get_job") as mock_get_job:
            mock_get_job.return_value = None

            response = test_client.get(f"/api/v1/jobs/{mock_job_id}")
            assert response.status_code == 404
            data = response.json()
            assert "JOB_NOT_FOUND" in data["error_code"]
