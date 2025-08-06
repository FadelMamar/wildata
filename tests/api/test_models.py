"""
Test API models.
"""

from datetime import datetime, timezone
from typing import Dict, List

import pytest
from pydantic import ValidationError
from wildata.api.models.jobs import BackgroundJob, JobResult, JobStatus
from wildata.api.models.requests import (
    BulkCreateROIRequest,
    BulkImportRequest,
    CreateROIRequest,
    ExportDatasetRequest,
    ImportDatasetRequest,
    UpdateGPSRequest,
    VisualizeRequest,
)
from wildata.api.models.responses import (
    BulkCreateROIResponse,
    BulkImportResponse,
    CreateROIResponse,
    DatasetInfo,
    DatasetListResponse,
    ErrorResponse,
    ExportDatasetResponse,
    ImportDatasetResponse,
    JobStatusResponse,
    UpdateGPSResponse,
)


class TestJobModels:
    """Test job-related models."""

    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_job_result_creation(self):
        """Test JobResult model creation."""
        # Successful result
        success_result = JobResult(
            success=True,
            message="Operation completed successfully",
            data={"dataset_name": "test_dataset"},
        )
        assert success_result.success is True
        assert success_result.message == "Operation completed successfully"
        assert success_result.data["dataset_name"] == "test_dataset"
        assert success_result.error is None

        # Failed result
        failed_result = JobResult(
            success=False, error="Operation failed due to invalid input"
        )
        assert failed_result.success is False
        assert failed_result.error == "Operation failed due to invalid input"
        assert failed_result.message is None
        assert failed_result.data is None

    def test_background_job_creation(self):
        """Test BackgroundJob model creation."""
        job = BackgroundJob(
            job_id="test-job-123",
            job_type="import_dataset",
            status=JobStatus.PENDING,
            progress=0.0,
            parameters={"dataset_name": "test_dataset"},
            user_id="test_user",
        )

        assert job.job_id == "test-job-123"
        assert job.job_type == "import_dataset"
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0
        assert job.parameters["dataset_name"] == "test_dataset"
        assert job.user_id == "test_user"
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.result is None

    def test_background_job_with_result(self):
        """Test BackgroundJob with result."""
        result = JobResult(
            success=True,
            message="Dataset imported successfully",
            data={"dataset_name": "test_dataset"},
        )

        job = BackgroundJob(
            job_id="test-job-123",
            job_type="import_dataset",
            status=JobStatus.COMPLETED,
            progress=100.0,
            result=result,
            parameters={"dataset_name": "test_dataset"},
        )

        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.result.success is True
        assert job.result.message == "Dataset imported successfully"

    def test_background_job_timestamps(self):
        """Test BackgroundJob timestamp handling."""
        now = datetime.now(timezone.utc)
        started_at = now.replace(second=now.second + 10)
        completed_at = now.replace(second=now.second + 30)

        job = BackgroundJob(
            job_id="test-job-123",
            job_type="import_dataset",
            started_at=started_at,
            completed_at=completed_at,
        )

        assert job.started_at == started_at
        assert job.completed_at == completed_at


class TestRequestModels:
    """Test request models."""

    def test_import_dataset_request_creation(self):
        """Test ImportDatasetRequest model creation."""
        request = ImportDatasetRequest(
            source_path="/path/to/dataset",
            source_format="coco",
            dataset_name="test_dataset",
            root="data",
            split_name="train",
            processing_mode="batch",
            track_with_dvc=False,
            bbox_tolerance=5,
            disable_roi=False,
        )

        assert request.source_path == "/path/to/dataset"
        assert request.source_format == "coco"
        assert request.dataset_name == "test_dataset"
        assert request.root == "data"
        assert request.split_name == "train"
        assert request.processing_mode == "batch"
        assert request.track_with_dvc is False
        assert request.bbox_tolerance == 5
        assert request.disable_roi is False

    def test_import_dataset_request_defaults(self):
        """Test ImportDatasetRequest default values."""
        request = ImportDatasetRequest(
            source_path="/path/to/dataset",
            source_format="coco",
            dataset_name="test_dataset",
        )

        assert request.root == "data"
        assert request.split_name == "train"
        assert request.processing_mode == "batch"
        assert request.track_with_dvc is False
        assert request.bbox_tolerance == 5
        assert request.disable_roi is False

    def test_bulk_import_request_creation(self):
        """Test BulkImportRequest model creation."""
        request = BulkImportRequest(
            source_paths=["/path/to/dataset1", "/path/to/dataset2"],
            source_format="coco",
            root="data",
            split_name="train",
            processing_mode="batch",
            track_with_dvc=False,
            bbox_tolerance=5,
            disable_roi=False,
        )

        assert request.source_paths == ["/path/to/dataset1", "/path/to/dataset2"]
        assert request.source_format == "coco"
        assert request.root == "data"
        assert request.split_name == "train"
        assert request.processing_mode == "batch"
        assert request.track_with_dvc is False
        assert request.bbox_tolerance == 5
        assert request.disable_roi is False

    def test_create_roi_request_creation(self):
        """Test CreateROIRequest model creation."""
        roi_config = {"width": 512, "height": 512, "stride": 256, "min_visibility": 0.1}

        request = CreateROIRequest(
            source_path="/path/to/dataset",
            source_format="coco",
            dataset_name="test_roi_dataset",
            root="data",
            split_name="val",
            bbox_tolerance=5,
            roi_config=roi_config,
            draw_original_bboxes=False,
        )

        assert request.source_path == "/path/to/dataset"
        assert request.source_format == "coco"
        assert request.dataset_name == "test_roi_dataset"
        assert request.root == "data"
        assert request.split_name == "val"
        assert request.bbox_tolerance == 5
        assert request.roi_config == roi_config
        assert request.draw_original_bboxes is False

    def test_bulk_create_roi_request_creation(self):
        """Test BulkCreateROIRequest model creation."""
        roi_config = {"width": 512, "height": 512, "stride": 256, "min_visibility": 0.1}

        request = BulkCreateROIRequest(
            source_paths=["/path/to/dataset1", "/path/to/dataset2"],
            source_format="coco",
            root="data",
            split_name="val",
            bbox_tolerance=5,
            roi_config=roi_config,
        )

        assert request.source_paths == ["/path/to/dataset1", "/path/to/dataset2"]
        assert request.source_format == "coco"
        assert request.root == "data"
        assert request.split_name == "val"
        assert request.bbox_tolerance == 5
        assert request.roi_config == roi_config

    def test_export_dataset_request_creation(self):
        """Test ExportDatasetRequest model creation."""
        request = ExportDatasetRequest(
            dataset_name="test_dataset",
            target_format="coco",
            target_path="/path/to/export",
            root="data",
        )

        assert request.dataset_name == "test_dataset"
        assert request.target_format == "coco"
        assert request.target_path == "/path/to/export"
        assert request.root == "data"

    def test_update_gps_request_creation(self):
        """Test UpdateGPSRequest model creation."""
        request = UpdateGPSRequest(
            image_folder="/path/to/images",
            csv_path="/path/to/gps.csv",
            output_dir="/path/to/output",
            skip_rows=0,
            filename_col="filename",
            lat_col="latitude",
            lon_col="longitude",
            alt_col="altitude",
        )

        assert request.image_folder == "/path/to/images"
        assert request.csv_path == "/path/to/gps.csv"
        assert request.output_dir == "/path/to/output"
        assert request.skip_rows == 0
        assert request.filename_col == "filename"
        assert request.lat_col == "latitude"
        assert request.lon_col == "longitude"
        assert request.alt_col == "altitude"

    def test_update_gps_request_defaults(self):
        """Test UpdateGPSRequest default values."""
        request = UpdateGPSRequest(
            image_folder="/path/to/images",
            csv_path="/path/to/gps.csv",
            output_dir="/path/to/output",
        )

        assert request.skip_rows == 0
        assert request.filename_col == "filename"
        assert request.lat_col == "latitude"
        assert request.lon_col == "longitude"
        assert request.alt_col == "altitude"

    def test_visualize_request_creation(self):
        """Test VisualizeRequest model creation."""
        request = VisualizeRequest(
            dataset_name="test_dataset",
            root_data_directory="/path/to/data",
            split="train",
            load_as_single_class=False,
            background_class_name="background",
            single_class_name="wildlife",
            keep_classes=["animal", "bird"],
            discard_classes=["vehicle"],
        )

        assert request.dataset_name == "test_dataset"
        assert request.root_data_directory == "/path/to/data"
        assert request.split == "train"
        assert request.load_as_single_class is False
        assert request.background_class_name == "background"
        assert request.single_class_name == "wildlife"
        assert request.keep_classes == ["animal", "bird"]
        assert request.discard_classes == ["vehicle"]

    def test_visualize_request_defaults(self):
        """Test VisualizeRequest default values."""
        request = VisualizeRequest(dataset_name="test_dataset")

        assert request.root_data_directory is None
        assert request.split == "train"
        assert request.load_as_single_class is False
        assert request.background_class_name == "background"
        assert request.single_class_name == "wildlife"
        assert request.keep_classes is None
        assert request.discard_classes is None


class TestResponseModels:
    """Test response models."""

    def test_error_response_creation(self):
        """Test ErrorResponse model creation."""
        response = ErrorResponse(
            message="An error occurred",
            error_code="VALIDATION_ERROR",
            details={"field": "source_path", "issue": "Path does not exist"},
        )

        assert response.message == "An error occurred"
        assert response.error_code == "VALIDATION_ERROR"
        assert response.details["field"] == "source_path"
        assert response.details["issue"] == "Path does not exist"

    def test_job_status_response_creation(self):
        """Test JobStatusResponse model creation."""
        response = JobStatusResponse(
            job_id="test-job-123",
            status="completed",
            progress=100.0,
            message="Job completed successfully",
            data={"dataset_name": "test_dataset"},
        )

        assert response.job_id == "test-job-123"
        assert response.status == "completed"
        assert response.progress == 100.0
        assert response.message == "Job completed successfully"
        assert response.data["dataset_name"] == "test_dataset"

    def test_import_dataset_response_creation(self):
        """Test ImportDatasetResponse model creation."""
        response = ImportDatasetResponse(
            job_id="test-job-123",
            message="Dataset import job started",
            dataset_name="test_dataset",
        )

        assert response.job_id == "test-job-123"
        assert response.message == "Dataset import job started"
        assert response.dataset_name == "test_dataset"

    def test_bulk_import_response_creation(self):
        """Test BulkImportResponse model creation."""
        response = BulkImportResponse(
            job_id="test-job-123",
            message="Bulk import job started",
            total_files=5,
            source_paths=["/path/to/dataset1", "/path/to/dataset2"],
        )

        assert response.job_id == "test-job-123"
        assert response.message == "Bulk import job started"
        assert response.total_files == 5
        assert response.source_paths == ["/path/to/dataset1", "/path/to/dataset2"]

    def test_create_roi_response_creation(self):
        """Test CreateROIResponse model creation."""
        response = CreateROIResponse(
            job_id="test-job-123",
            message="ROI dataset creation job started",
            dataset_name="test_roi_dataset",
        )

        assert response.job_id == "test-job-123"
        assert response.message == "ROI dataset creation job started"
        assert response.dataset_name == "test_roi_dataset"

    def test_bulk_create_roi_response_creation(self):
        """Test BulkCreateROIResponse model creation."""
        response = BulkCreateROIResponse(
            job_id="test-job-123",
            message="Bulk ROI creation job started",
            total_files=3,
            source_paths=[
                "/path/to/dataset1",
                "/path/to/dataset2",
                "/path/to/dataset3",
            ],
        )

        assert response.job_id == "test-job-123"
        assert response.message == "Bulk ROI creation job started"
        assert response.total_files == 3
        assert response.source_paths == [
            "/path/to/dataset1",
            "/path/to/dataset2",
            "/path/to/dataset3",
        ]

    def test_export_dataset_response_creation(self):
        """Test ExportDatasetResponse model creation."""
        response = ExportDatasetResponse(
            job_id="test-job-123",
            message="Dataset export job started",
            dataset_name="test_dataset",
            target_format="coco",
            target_path="/path/to/export",
        )

        assert response.job_id == "test-job-123"
        assert response.message == "Dataset export job started"
        assert response.dataset_name == "test_dataset"
        assert response.target_format == "coco"
        assert response.target_path == "/path/to/export"

    def test_update_gps_response_creation(self):
        """Test UpdateGPSResponse model creation."""
        response = UpdateGPSResponse(
            job_id="test-job-123",
            message="GPS update job started",
            image_folder="/path/to/images",
            csv_path="/path/to/gps.csv",
            output_dir="/path/to/output",
        )

        assert response.job_id == "test-job-123"
        assert response.message == "GPS update job started"
        assert response.image_folder == "/path/to/images"
        assert response.csv_path == "/path/to/gps.csv"
        assert response.output_dir == "/path/to/output"

    def test_dataset_info_creation(self):
        """Test DatasetInfo model creation."""
        info = DatasetInfo(
            name="test_dataset",
            type="detection",
            num_images=100,
            num_annotations=500,
            classes=["animal", "bird", "vehicle"],
            created_at="2024-01-01T12:00:00Z",
            last_modified="2024-01-02T15:30:00Z",
        )

        assert info.name == "test_dataset"
        assert info.type == "detection"
        assert info.num_images == 100
        assert info.num_annotations == 500
        assert info.classes == ["animal", "bird", "vehicle"]
        assert info.created_at == "2024-01-01T12:00:00Z"
        assert info.last_modified == "2024-01-02T15:30:00Z"

    def test_dataset_list_response_creation(self):
        """Test DatasetListResponse model creation."""
        datasets = [
            DatasetInfo(
                name="dataset1",
                type="detection",
                num_images=50,
                num_annotations=200,
                classes=["animal", "bird"],
            ),
            DatasetInfo(
                name="dataset2",
                type="classification",
                num_images=75,
                num_annotations=75,
                classes=["cat", "dog", "bird"],
            ),
        ]

        response = DatasetListResponse(datasets=datasets, total_count=2)

        assert len(response.datasets) == 2
        assert response.total_count == 2
        assert response.datasets[0].name == "dataset1"
        assert response.datasets[1].name == "dataset2"


class TestModelValidation:
    """Test model validation."""

    def test_import_dataset_request_validation(self):
        """Test ImportDatasetRequest validation."""
        # Valid request
        request = ImportDatasetRequest(
            source_path="/path/to/dataset",
            source_format="coco",
            dataset_name="test_dataset",
        )
        assert request.source_path == "/path/to/dataset"
        assert request.source_format == "coco"
        assert request.dataset_name == "test_dataset"

        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            ImportDatasetRequest(
                source_path="/path/to/dataset"
                # Missing source_format and dataset_name
            )

    def test_bulk_import_request_validation(self):
        """Test BulkImportRequest validation."""
        # Valid request
        request = BulkImportRequest(
            source_paths=["/path/to/dataset1", "/path/to/dataset2"],
            source_format="coco",
        )
        assert len(request.source_paths) == 2
        assert request.source_format == "coco"

        # Empty source_paths should raise ValidationError
        with pytest.raises(ValidationError):
            BulkImportRequest(source_paths=[], source_format="coco")

    def test_create_roi_request_validation(self):
        """Test CreateROIRequest validation."""
        roi_config = {"width": 512, "height": 512}

        # Valid request
        request = CreateROIRequest(
            source_path="/path/to/dataset",
            source_format="coco",
            dataset_name="test_roi_dataset",
            roi_config=roi_config,
        )
        assert request.roi_config == roi_config

        # Missing roi_config should raise ValidationError
        with pytest.raises(ValidationError):
            CreateROIRequest(
                source_path="/path/to/dataset",
                source_format="coco",
                dataset_name="test_roi_dataset",
                # Missing roi_config
            )

    def test_update_gps_request_validation(self):
        """Test UpdateGPSRequest validation."""
        # Valid request
        request = UpdateGPSRequest(
            image_folder="/path/to/images",
            csv_path="/path/to/gps.csv",
            output_dir="/path/to/output",
        )
        assert request.image_folder == "/path/to/images"
        assert request.csv_path == "/path/to/gps.csv"
        assert request.output_dir == "/path/to/output"

        # Missing required fields should raise ValidationError
        with pytest.raises(ValidationError):
            UpdateGPSRequest(
                image_folder="/path/to/images"
                # Missing csv_path and output_dir
            )

    def test_job_result_validation(self):
        """Test JobResult validation."""
        # Valid success result
        success_result = JobResult(success=True, message="Operation completed")
        assert success_result.success is True
        assert success_result.message == "Operation completed"

        # Valid failure result
        failure_result = JobResult(success=False, error="Operation failed")
        assert failure_result.success is False
        assert failure_result.error == "Operation failed"

        # Invalid: both success and error
        with pytest.raises(ValidationError):
            JobResult(
                success=True,
                message="Success",
                error="Error",  # Should not have both message and error
            )

    def test_background_job_validation(self):
        """Test BackgroundJob validation."""
        # Valid job
        job = BackgroundJob(job_id="test-job-123", job_type="import_dataset")
        assert job.job_id == "test-job-123"
        assert job.job_type == "import_dataset"
        assert job.status == JobStatus.PENDING
        assert job.progress == 0.0

        # Invalid: missing required fields
        with pytest.raises(ValidationError):
            BackgroundJob(
                job_id="test-job-123"
                # Missing job_type
            )
