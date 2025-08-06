"""
Health check endpoints.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends

from ..dependencies import get_api_config
from ..models.responses import ErrorResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "wildata-api",
    }


@router.get("/detailed")
async def detailed_health_check(config=Depends(get_api_config)):
    """Detailed health check with system information."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "wildata-api",
        "version": "0.1.0",
        "config": {
            "host": config.host,
            "port": config.port,
            "debug": config.debug,
            "max_file_size": config.max_file_size,
            "job_queue_size": config.job_queue_size,
        },
    }


@router.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    # TODO: Implement actual metrics collection
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "active_jobs": 0,
            "total_requests": 0,
            "error_rate": 0.0,
            "response_time_avg": 0.0,
        },
    }
