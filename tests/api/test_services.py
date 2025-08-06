"""
Test API services.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from wildata.api.models.jobs import JobResult, JobStatus
from wildata.api.services.job_queue import JobQueue
from wildata.api.services.task_handlers import (
    handle_bulk_create_roi,
    handle_bulk_import,
    handle_create_roi,
    handle_export_dataset,
    handle_import_dataset,
    handle_update_gps,
)


class TestJobQueue:
    """Test job queue functionality."""

    @pytest.fixture
    def job_queue(self):
        """Create a job queue instance."""
        return JobQueue()

    @pytest.mark.asyncio
    async def test_create_job(self, job_queue: JobQueue):
        """Test creating a job."""
        job_id = await job_queue.create_job(
            job_type="import_dataset",
            parameters={"dataset_name": "test_dataset"},
            user_id="test_user",
        )

        assert job_id is not None
        assert len(job_id) > 0

        job = await job_queue.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id
        assert job.job_type == "import_dataset"
        assert job.status == JobStatus.PENDING
        assert job.user_id == "test_user"

    @pytest.mark.asyncio
    async def test_update_job_status(self, job_queue: JobQueue):
        """Test updating job status."""
        job_id = await job_queue.create_job(
            job_type="import_dataset", parameters={"dataset_name": "test_dataset"}
        )

        await job_queue.update_job_status(job_id, JobStatus.RUNNING, progress=50.0)

        job = await job_queue.get_job(job_id)
        assert job.status == JobStatus.RUNNING
        assert job.progress == 50.0
        assert job.started_at is not None

    @pytest.mark.asyncio
    async def test_complete_job(self, job_queue: JobQueue):
        """Test completing a job."""
        job_id = await job_queue.create_job(
            job_type="import_dataset", parameters={"dataset_name": "test_dataset"}
        )

        result = JobResult(
            success=True,
            message="Dataset imported successfully",
            data={"dataset_name": "test_dataset"},
        )

        await job_queue.update_job_status(
            job_id, JobStatus.COMPLETED, progress=100.0, result=result
        )

        job = await job_queue.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.completed_at is not None
        assert job.result.success is True

    @pytest.mark.asyncio
    async def test_cancel_job(self, job_queue: JobQueue):
        """Test canceling a job."""
        job_id = await job_queue.create_job(
            job_type="import_dataset", parameters={"dataset_name": "test_dataset"}
        )

        success = await job_queue.cancel_job(job_id)
        assert success is True

        job = await job_queue.get_job(job_id)
        assert job.status == JobStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_list_jobs(self, job_queue: JobQueue):
        """Test listing jobs."""
        # Create multiple jobs
        job1_id = await job_queue.create_job(
            job_type="import_dataset", parameters={"dataset_name": "dataset1"}
        )
        job2_id = await job_queue.create_job(
            job_type="bulk_import", parameters={"source_paths": ["path1", "path2"]}
        )

        jobs = await job_queue.list_jobs()
        assert len(jobs) == 2

        job_ids = [job.job_id for job in jobs]
        assert job1_id in job_ids
        assert job2_id in job_ids

    @pytest.mark.asyncio
    async def test_list_jobs_with_filters(self, job_queue: JobQueue):
        """Test listing jobs with filters."""
        # Create jobs with different types
        await job_queue.create_job(
            job_type="import_dataset", parameters={"dataset_name": "dataset1"}
        )
        await job_queue.create_job(
            job_type="bulk_import", parameters={"source_paths": ["path1", "path2"]}
        )

        # Filter by job type
        import_jobs = await job_queue.list_jobs(job_type="import_dataset")
        assert len(import_jobs) == 1
        assert import_jobs[0].job_type == "import_dataset"

        # Filter by status
        pending_jobs = await job_queue.list_jobs(status=JobStatus.PENDING)
        assert len(pending_jobs) == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, job_queue: JobQueue):
        """Test getting a job that doesn't exist."""
        job = await job_queue.get_job("nonexistent-job-id")
        assert job is None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, job_queue: JobQueue):
        """Test canceling a job that doesn't exist."""
        success = await job_queue.cancel_job("nonexistent-job-id")
        assert success is False


class TestTaskHandlers:
    """Test background task handlers."""

    @pytest.mark.asyncio
    async def test_handle_import_dataset_success(self):
        """Test successful dataset import."""
        with patch("wildata.cli.import_logic._import_dataset_core") as mock_import:
            mock_import.return_value = True

            with patch(
                "wildata.api.services.job_queue.get_job_queue"
            ) as mock_get_queue:
                mock_queue = AsyncMock()
                mock_get_queue.return_value = mock_queue

                result = await handle_import_dataset(
                    job_id="test-job",
                    config=Mock(
                        dataset_name="test_dataset",
                        source_path="/path/to/dataset",
                        source_format="coco",
                    ),
                    verbose=False,
                )

                assert result.success is True
                assert "Successfully imported dataset" in result.message
                assert result.data["dataset_name"] == "test_dataset"

    @pytest.mark.asyncio
    async def test_handle_import_dataset_failure(self):
        """Test failed dataset import."""
        with patch("wildata.cli.import_logic._import_dataset_core") as mock_import:
            mock_import.return_value = False

            with patch(
                "wildata.api.services.job_queue.get_job_queue"
            ) as mock_get_queue:
                mock_queue = AsyncMock()
                mock_get_queue.return_value = mock_queue

                result = await handle_import_dataset(
                    job_id="test-job",
                    config=Mock(
                        dataset_name="test_dataset",
                        source_path="/path/to/dataset",
                        source_format="coco",
                    ),
                    verbose=False,
                )

                assert result.success is False
                assert "Dataset import failed" in result.error

    @pytest.mark.asyncio
    async def test_handle_bulk_import_success(self):
        """Test successful bulk import."""
        with patch("wildata.cli.import_logic.import_one_worker") as mock_worker:
            mock_worker.return_value = (0, "test_dataset", True, "Success")

            with patch(
                "wildata.api.services.job_queue.get_job_queue"
            ) as mock_get_queue:
                mock_queue = AsyncMock()
                mock_get_queue.return_value = mock_queue

                config = Mock(
                    source_paths=["/path/to/dataset1", "/path/to/dataset2"],
                    source_format="coco",
                    model_dump=lambda: {"source_format": "coco"},
                )

                result = await handle_bulk_import(
                    job_id="test-job", config=config, verbose=False
                )

                assert result.success is True
                assert "Bulk import complete" in result.message

    @pytest.mark.asyncio
    async def test_handle_create_roi_success(self):
        """Test successful ROI creation."""
        with patch("wildata.cli.roi_logic.create_roi_dataset_core") as mock_create:
            mock_create.return_value = True

            with patch(
                "wildata.api.services.job_queue.get_job_queue"
            ) as mock_get_queue:
                mock_queue = AsyncMock()
                mock_get_queue.return_value = mock_queue

                result = await handle_create_roi(
                    job_id="test-job",
                    config=Mock(
                        dataset_name="test_roi_dataset",
                        source_path="/path/to/dataset",
                        source_format="coco",
                    ),
                    verbose=False,
                )

                assert result.success is True
                assert "Successfully created ROI dataset" in result.message
                assert result.data["dataset_name"] == "test_roi_dataset"

    @pytest.mark.asyncio
    async def test_handle_bulk_create_roi_success(self):
        """Test successful bulk ROI creation."""
        with patch("wildata.cli.roi_logic.create_roi_one_worker") as mock_worker:
            mock_worker.return_value = (0, "test_roi_dataset", True, "Success")

            with patch(
                "wildata.api.services.job_queue.get_job_queue"
            ) as mock_get_queue:
                mock_queue = AsyncMock()
                mock_get_queue.return_value = mock_queue

                config = Mock(
                    source_paths=["/path/to/dataset1", "/path/to/dataset2"],
                    source_format="coco",
                    model_dump=lambda: {"source_format": "coco"},
                )

                result = await handle_bulk_create_roi(
                    job_id="test-job", config=config, verbose=False
                )

                assert result.success is True
                assert "Bulk ROI creation complete" in result.message

    @pytest.mark.asyncio
    async def test_handle_export_dataset_success(self):
        """Test successful dataset export."""
        with patch("wildata.pipeline.DataPipeline") as mock_pipeline:
            mock_pipeline.return_value.export_dataset.return_value = True

            with patch(
                "wildata.api.services.job_queue.get_job_queue"
            ) as mock_get_queue:
                mock_queue = AsyncMock()
                mock_get_queue.return_value = mock_queue

                result = await handle_export_dataset(
                    job_id="test-job",
                    dataset_name="test_dataset",
                    target_format="coco",
                    target_path="/path/to/export",
                    root="data",
                )

                assert result.success is True
                assert "Successfully exported dataset" in result.message
                assert result.data["dataset_name"] == "test_dataset"

    @pytest.mark.asyncio
    async def test_handle_update_gps_success(self):
        """Test successful GPS update."""
        with patch("wildata.adapters.utils.ExifGPSManager") as mock_gps_manager:
            mock_manager = Mock()
            mock_gps_manager.return_value = mock_manager

            with patch(
                "wildata.api.services.job_queue.get_job_queue"
            ) as mock_get_queue:
                mock_queue = AsyncMock()
                mock_get_queue.return_value = mock_queue

                config = Mock(
                    image_folder="/path/to/images",
                    csv_path="/path/to/gps.csv",
                    output_dir="/path/to/output",
                    skip_rows=0,
                    filename_col="filename",
                    lat_col="latitude",
                    lon_col="longitude",
                    alt_col="altitude",
                )

                result = await handle_update_gps(
                    job_id="test-job", config=config, verbose=False
                )

                assert result.success is True
                assert "Successfully updated GPS data" in result.message
                assert result.data["image_folder"] == "/path/to/images"
