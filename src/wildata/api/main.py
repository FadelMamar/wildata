"""
Main FastAPI application for WildData API.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import api_config
from .dependencies import handle_api_exception
from .exceptions import WildDataAPIException
from .routers import datasets_router, health_router, jobs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    print("Starting WildData API...")

    yield

    # Shutdown
    print("Shutting down WildData API...")


# Create FastAPI application
app = FastAPI(
    title="WildData API",
    description="RESTful API for WildData dataset management",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_config.cors_origins,
    allow_credentials=api_config.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add exception handler for custom exceptions
@app.exception_handler(WildDataAPIException)
async def wilddata_exception_handler(request: Request, exc: WildDataAPIException):
    """Handle WildData API exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
        },
    )


# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(datasets_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "WildData API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "WildData API",
        "version": "0.1.0",
        "description": "RESTful API for WildData dataset management",
        "endpoints": {
            "health": "/api/v1/health",
            "datasets": "/api/v1/datasets",
            "jobs": "/api/v1/jobs",
        },
    }
