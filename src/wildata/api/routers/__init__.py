"""
API routers package.
"""

from .datasets import router as datasets_router
from .health import router as health_router
from .jobs import router as jobs_router

__all__ = [
    "datasets_router",
    "jobs_router",
    "health_router",
]
