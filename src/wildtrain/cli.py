"""
Command-line interface for the WildTrain data pipeline using Typer.
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
import yaml
from pydantic import BaseModel, Field, ValidationError

from .config import ROOT, AugmentationConfig, TilingConfig
from .logging_config import setup_logging
from .pipeline.data_pipeline import DataPipeline
from .pipeline.dvc_manager import DVCConfig, DVCManager, DVCStorageType
from .transformations import (
    AugmentationTransformer,
    TilingTransformer,
    TransformationPipeline,
)

__version__ = "0.1.0"

app = typer.Typer(
    name="wildtrain",
    help="WildTrain Data Pipeline - Manage datasets in master format and create framework-specific formats",
    add_completion=False,
    rich_markup_mode="rich",
)


class AppState:
    data_dir: Path = ROOT / "data"


state = AppState()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Show the application's version and exit.",
        is_eager=True,
        callback=lambda v: (
            typer.echo(f"wildtrain version {__version__}") or raise_(typer.Exit())
            if v
            else None
        ),
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", help="Directory to store master data (default: ./data)"
    ),
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Enable verbose logging (DEBUG level)"
    ),
):
    """WildTrain Data Pipeline CLI."""
    # Setup logging
    if verbose:
        log_level = "DEBUG"
    setup_logging(level=log_level)

    if data_dir:
        state.data_dir = data_dir
    else:
        state.data_dir = Path.cwd() / "data"


def raise_(ex):
    raise ex


# Helper to get pipeline with correct split and optional transformation pipeline


def get_pipeline(
    split_name: str, transformation_pipeline=None, enable_dvc=True, filter_pipeline=None
):
    return DataPipeline(
        str(state.data_dir),
        split_name,
        transformation_pipeline=transformation_pipeline,
        enable_dvc=enable_dvc,
        filter_pipeline=filter_pipeline,
    )


# Dataset command group

dataset_app = typer.Typer(help="Dataset management commands.")


class ImportConfig(BaseModel):
    source_path: str
    format_type: str
    dataset_name: str
    track_with_dvc: bool = False
    augment: bool = False
    rotation_range: Tuple[float, float] = (-10.0, 10.0)
    probability: float = 0.5
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)
    noise_std: Tuple[float, float] = (0.01, 0.1)
    tile: bool = False
    tile_size: int = 512
    stride: int = 256
    min_visibility: float = 0.1
    max_negative_tiles: int = 3
    negative_positive_ratio: float = 1.0


@dataset_app.command("import")
def import_dataset(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML config file for import (see docs for structure)",
    ),
    source_path: Optional[str] = typer.Option(
        None, help="Path to the source dataset (overrides config)"
    ),
    format_type: Optional[str] = typer.Option(
        None, help="Format type of the source dataset (overrides config)"
    ),
    dataset_name: Optional[str] = typer.Option(
        None, help="Name for the dataset in master storage (overrides config)"
    ),
    track_with_dvc: Optional[bool] = typer.Option(None, help="Track dataset with DVC"),
    augment: Optional[bool] = typer.Option(None, help="Apply data augmentation"),
    rotation_range: Optional[Tuple[float, float]] = typer.Option(
        None, help="Rotation range in degrees"
    ),
    probability: Optional[float] = typer.Option(
        None, help="Augmentation probability (0-1)"
    ),
    brightness_range: Optional[Tuple[float, float]] = typer.Option(
        None, help="Brightness range"
    ),
    contrast_range: Optional[Tuple[float, float]] = typer.Option(
        None, help="Contrast range"
    ),
    noise_std: Optional[Tuple[float, float]] = typer.Option(
        None, help="Noise standard deviation range (min,max)"
    ),
    tile: Optional[bool] = typer.Option(None, help="Apply image tiling"),
    tile_size: Optional[int] = typer.Option(None, help="Tile size in pixels"),
    stride: Optional[int] = typer.Option(None, help="Stride between tiles"),
    min_visibility: Optional[float] = typer.Option(
        None, help="Minimum object visibility in tiles (0-1)"
    ),
    max_negative_tiles: Optional[int] = typer.Option(
        None, help="Maximum negative tiles per image"
    ),
    negative_positive_ratio: Optional[float] = typer.Option(
        None, help="Negative to positive tile ratio"
    ),
):
    """
    Import a dataset from COCO or YOLO format with optional transformations and DVC tracking.
    You can specify all arguments in a YAML config file using --config/-c. CLI arguments override config values.
    """
    # 1. Load YAML config if provided
    config_dict = {}
    if config:
        try:
            with open(config, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
            typer.echo(f"[CONFIG] Loaded config from {config}")
        except Exception as e:
            typer.echo(f"❌ Error loading config file: {e}")
            raise typer.Exit(1)
    # 2. Build CLI args dict (only those not None)
    cli_args = {
        k: v
        for k, v in {
            "source_path": source_path,
            "format_type": format_type,
            "dataset_name": dataset_name,
            "track_with_dvc": track_with_dvc,
            "augment": augment,
            "rotation_range": rotation_range,
            "probability": probability,
            "brightness_range": brightness_range,
            "contrast_range": contrast_range,
            "noise_std": noise_std,
            "tile": tile,
            "tile_size": tile_size,
            "stride": stride,
            "min_visibility": min_visibility,
            "max_negative_tiles": max_negative_tiles,
            "negative_positive_ratio": negative_positive_ratio,
        }.items()
        if v is not None
    }
    # 3. Merge CLI args over config_dict
    merged = {**config_dict, **cli_args}
    # 4. Validate and coerce with Pydantic
    try:
        import_config = ImportConfig(**merged)
    except ValidationError as e:
        typer.echo(f"❌ Config validation error:\n{e}")
        raise typer.Exit(1)
    # 5. Use import_config in your logic
    if import_config.format_type.lower() not in ["coco", "yolo"]:
        typer.echo(
            f"❌ Error: Format type must be 'coco' or 'yolo', got '{import_config.format_type}'"
        )
        raise typer.Exit(1)
    split_name = "train"
    transformation_pipeline = None
    if import_config.augment or import_config.tile:
        transformation_pipeline = TransformationPipeline()
        if import_config.augment:
            try:
                aug_config = AugmentationConfig(
                    rotation_range=import_config.rotation_range,
                    probability=import_config.probability,
                    brightness_range=import_config.brightness_range,
                    contrast_range=import_config.contrast_range,
                    noise_std=import_config.noise_std,
                )
                aug_transformer = AugmentationTransformer(aug_config)
                transformation_pipeline.add_transformer(aug_transformer)
                typer.echo(
                    f"🔧 Added augmentation transformer (probability: {import_config.probability})"
                )
            except Exception as e:
                typer.echo(f"❌ Error setting up augmentation: {e}")
                raise typer.Exit(1)
        if import_config.tile:
            try:
                tile_config = TilingConfig(
                    tile_size=import_config.tile_size,
                    stride=import_config.stride,
                    min_visibility=import_config.min_visibility,
                    max_negative_tiles_in_negative_image=import_config.max_negative_tiles,
                    negative_positive_ratio=import_config.negative_positive_ratio,
                )
                tile_transformer = TilingTransformer(tile_config)
                transformation_pipeline.add_transformer(tile_transformer)
                typer.echo(
                    f"🔧 Added tiling transformer (tile size: {import_config.tile_size}, stride: {import_config.stride})"
                )
            except Exception as e:
                typer.echo(f"❌ Error setting up tiling: {e}")
                raise typer.Exit(1)
    pipeline = get_pipeline(split_name, transformation_pipeline)
    try:
        typer.echo(
            f"🚀 Importing {import_config.format_type.upper()} dataset from {import_config.source_path}"
        )
        typer.echo(f"📝 Dataset name: {import_config.dataset_name}")
        if import_config.augment or import_config.tile:
            typer.echo(
                f"🔧 Applying transformations: {'augmentation' if import_config.augment else ''}{' + tiling' if import_config.tile else ''}"
            )
        if import_config.track_with_dvc:
            typer.echo("📦 DVC tracking enabled")
        typer.echo("─" * 50)
        result = pipeline.import_dataset(
            source_path=import_config.source_path,
            source_format=import_config.format_type.lower(),
            dataset_name=import_config.dataset_name,
            track_with_dvc=import_config.track_with_dvc,
        )
        if result["success"]:
            typer.echo("✅ Import successful!")
            typer.echo(
                f"📄 Master annotations: {result.get('dataset_info_path', result.get('master_path', 'N/A'))}"
            )
            typer.echo("🔧 Framework formats created:")
            for framework, path in result["framework_paths"].items():
                typer.echo(f"  - {framework.upper()}: {path}")
            if import_config.track_with_dvc and result.get("dvc_tracked"):
                typer.echo("📦 Dataset tracked with DVC")
        else:
            typer.echo("❌ Import failed!")
            typer.echo(f"💥 Error: {result.get('error', 'Unknown error')}")
            if "validation_errors" in result:
                typer.echo("\n🔍 Validation errors:")
                for error in result["validation_errors"]:
                    typer.echo(f"  - {error}")
            if "hints" in result:
                typer.echo("\n💡 Hints:")
                for hint in result["hints"]:
                    typer.echo(f"  - {hint}")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@dataset_app.command("list")
def list_datasets():
    pipeline = get_pipeline("train")
    try:
        datasets = pipeline.list_datasets()
        if not datasets:
            typer.echo("📭 No datasets found in master storage.")
            return
        typer.echo(f"📋 Found {len(datasets)} dataset(s):")
        typer.echo("─" * 50)
        for dataset in datasets:
            typer.echo(f"📁 Dataset: {dataset['dataset_name']}")
            typer.echo(f"  📊 Total images: {dataset['total_images']}")
            typer.echo(f"  🏷️  Total annotations: {dataset['total_annotations']}")
            typer.echo(f"  📂 Images by split: {dataset['images_by_split']}")
            typer.echo(f"  🎯 Annotations by type: {dataset['annotations_by_type']}")
            typer.echo()
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@dataset_app.command()
def info(dataset_name: str = typer.Argument(..., help="Name of the dataset")):
    pipeline = get_pipeline("train")
    try:
        info = pipeline.get_dataset_info(dataset_name)
        typer.echo(f"📁 Dataset: {info['dataset_name']}")
        typer.echo("─" * 50)
        typer.echo(f"📄 Master annotations: {info['master_annotations_file']}")
        typer.echo(f"📊 Total images: {info['total_images']}")
        typer.echo(f"🏷️  Total annotations: {info['total_annotations']}")
        typer.echo(f"📂 Images by split: {info['images_by_split']}")
        typer.echo(f"🎯 Annotations by type: {info['annotations_by_type']}")
        typer.echo(f"🏷️  Categories: {len(info['categories'])}")
        # Check for framework formats
        framework_formats = []
        coco_dir = state.data_dir / "framework_formats" / "coco" / dataset_name
        yolo_dir = state.data_dir / "framework_formats" / "yolo" / dataset_name
        if coco_dir.exists():
            framework_formats.append({"framework": "coco", "path": str(coco_dir)})
        if yolo_dir.exists():
            framework_formats.append({"framework": "yolo", "path": str(yolo_dir)})
        if framework_formats:
            typer.echo("\n🔧 Available framework formats:")
            for fmt in framework_formats:
                typer.echo(f"  - {fmt['framework'].upper()}: {fmt['path']}")
        else:
            typer.echo("\n⚠️  No framework formats created yet.")
    except FileNotFoundError:
        typer.echo(f"❌ Dataset '{dataset_name}' not found.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@dataset_app.command()
def export(
    dataset_name: str = typer.Argument(..., help="Name of the dataset"),
    framework: str = typer.Argument(
        ..., help="Target framework format", case_sensitive=False
    ),
):
    if framework.lower() not in ["coco", "yolo"]:
        typer.echo(f"❌ Error: Framework must be 'coco' or 'yolo', got '{framework}'")
        raise typer.Exit(1)
    pipeline = get_pipeline("train")
    try:
        result = pipeline.export_framework_format(dataset_name, framework.lower())
        typer.echo(
            f"✅ Exported dataset '{dataset_name}' to {framework.upper()} format"
        )
        typer.echo(f"📁 Output path: {result['output_path']}")
        if framework.lower() == "coco":
            typer.echo(f"📂 Data directory: {result.get('data_dir', 'N/A')}")
            typer.echo(f"📄 Annotations file: {result.get('annotations_file', 'N/A')}")
        elif framework.lower() == "yolo":
            typer.echo(f"🖼️  Images directory: {result.get('images_dir', 'N/A')}")
            typer.echo(f"🏷️  Labels directory: {result.get('labels_dir', 'N/A')}")
            typer.echo(f"⚙️  Data YAML: {result.get('data_yaml', 'N/A')}")
    except FileNotFoundError:
        typer.echo(f"❌ Dataset '{dataset_name}' not found.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Export failed: {e}")
        raise typer.Exit(1)


@dataset_app.command()
def delete(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to delete"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force deletion without confirmation"
    ),
):
    """Delete a dataset from master storage."""
    try:
        if not force:
            confirm = typer.confirm(
                f"Are you sure you want to delete dataset '{dataset_name}'?"
            )
            if not confirm:
                typer.echo("🗑️  Deletion cancelled.")
                return
        # Delete dataset directory
        dataset_dir = state.data_dir / dataset_name
        if dataset_dir.exists():
            import shutil

            shutil.rmtree(dataset_dir)
            typer.echo(f"✅ Dataset '{dataset_name}' deleted successfully.")
        else:
            typer.echo(f"❌ Dataset '{dataset_name}' not found.")
            raise typer.Exit(1)
    except FileNotFoundError:
        typer.echo(f"❌ Dataset '{dataset_name}' not found.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


# Add the dataset group to the main app
app.add_typer(dataset_app, name="dataset")


@app.command()
def validate(
    source_path: str = typer.Argument(..., help="Path to the source dataset"),
    format_type: str = typer.Argument(
        ..., help="Format type to validate", case_sensitive=False
    ),
):
    if format_type.lower() not in ["coco", "yolo"]:
        typer.echo(
            f"❌ Error: Format type must be 'coco' or 'yolo', got '{format_type}'"
        )
        raise typer.Exit(1)
    try:
        typer.echo(f"🔍 Validating {format_type.upper()} dataset at {source_path}")
        typer.echo("─" * 50)
        # Validate dataset
        if format_type.lower() == "coco":
            from wildtrain.validators.coco_validator import COCOValidator

            validator = COCOValidator(source_path)
            is_valid, errors, warnings = validator.validate()
        else:
            from wildtrain.validators.yolo_validator import YOLOValidator

            validator = YOLOValidator(source_path)
            is_valid, errors, warnings = validator.validate()
        if is_valid:
            typer.echo("✅ Validation passed!")
            typer.echo("🎉 Dataset is ready for import.")
        else:
            typer.echo("❌ Validation failed!")
            typer.echo("\n🔍 Validation errors:")
            for error in errors:
                typer.echo(f"  - {error}")
            if warnings:
                typer.echo("\n💡 Hints:")
                for hint in warnings:
                    typer.echo(f"  - {hint}")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    try:
        typer.echo("📊 WildTrain Pipeline Status")
        typer.echo("─" * 50)
        data_dir = state.data_dir
        if data_dir.exists():
            typer.echo(f"📁 Data directory: {data_dir} ✅")
        else:
            typer.echo(f"📁 Data directory: {data_dir} ❌ (not found)")
        framework_dir = data_dir / "framework_formats"
        if framework_dir.exists():
            typer.echo(f"🔧 Framework formats: {framework_dir} ✅")
        else:
            typer.echo(f"🔧 Framework formats: {framework_dir} ❌ (not found)")
        pipeline = get_pipeline("train")
        datasets = pipeline.list_datasets()
        typer.echo(f"📋 Datasets: {len(datasets)} found")
        if datasets:
            typer.echo("\n📁 Available datasets:")
            for dataset in datasets:
                typer.echo(
                    f"  - {dataset['dataset_name']}: {dataset['total_images']} images"
                )
        typer.echo("\n✨ Pipeline is ready!")
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


# DVC command group
dvc_app = typer.Typer(help="DVC data versioning commands.")


@dvc_app.command("setup")
def setup_dvc(
    storage_type: str = typer.Option(
        "local", "--storage-type", help="Storage type (local, s3, gcs, azure)"
    ),
    storage_path: str = typer.Option(
        None, "--storage-path", help="Path to remote storage"
    ),
    remote_name: str = typer.Option(
        "origin", "--remote-name", help="Name for the remote"
    ),
    force: bool = typer.Option(False, "--force", help="Force reconfiguration"),
):
    """Setup DVC remote storage."""
    try:
        # Map storage type
        storage_type_enum = DVCStorageType.S3
        if storage_type.lower() != "s3":
            typer.echo(f"❌ Error: Unsupported storage type: {storage_type}")
            raise typer.Exit(1)

        # Create DVC configuration
        config = DVCConfig(
            storage_type=storage_type_enum,
            storage_path=storage_path,
            remote_name=remote_name,
        )

        # Initialize DVC manager
        dvc_manager = DVCManager(state.data_dir, config)

        # Setup remote storage
        if dvc_manager.setup_remote_storage(force):
            typer.echo(f"✅ DVC remote storage setup successful!")
            typer.echo(f"📦 Storage type: {storage_type}")
            typer.echo(f"📁 Storage path: {storage_path}")
            typer.echo(f"🔗 Remote name: {remote_name}")
        else:
            typer.echo("❌ Failed to setup DVC remote storage")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Error setting up DVC: {e}")
        raise typer.Exit(1)


@dvc_app.command("status")
def dvc_status():
    """Show DVC status information."""
    try:
        dvc_manager = DVCManager(state.data_dir)
        status = dvc_manager.get_status()

        typer.echo("📊 DVC Status")
        typer.echo("─" * 50)
        typer.echo(
            f"🔧 DVC Initialized: {'✅' if status.get('dvc_initialized', False) else '❌'}"
        )
        typer.echo(
            f"📦 Remote Configured: {'✅' if status.get('remote_configured', False) else '❌'}"
        )
        typer.echo(
            f"📁 Data Tracked: {'✅' if status.get('data_tracked', False) else '❌'}"
        )

        if status.get("status_output"):
            typer.echo("\n📋 Status Details:")
            typer.echo(status["status_output"])

    except Exception as e:
        typer.echo(f"❌ Error getting DVC status: {e}")
        raise typer.Exit(1)


@dvc_app.command("pull")
def pull_data(
    dataset_name: Optional[str] = typer.Argument(
        None, help="Specific dataset to pull (all if not specified)"
    ),
):
    """Pull data from DVC remote storage."""
    try:
        dvc_manager = DVCManager(state.data_dir)

        if dataset_name:
            typer.echo(f"📥 Pulling dataset '{dataset_name}' from remote storage...")
            success = dvc_manager.pull_data(dataset_name)
        else:
            typer.echo("📥 Pulling all data from remote storage...")
            success = dvc_manager.pull_data()

        if success:
            typer.echo("✅ Data pull successful!")
        else:
            typer.echo("❌ Failed to pull data from remote storage")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Error pulling data: {e}")
        raise typer.Exit(1)


@dvc_app.command("push")
def push_data():
    """Push data to DVC remote storage."""
    try:
        dvc_manager = DVCManager(state.data_dir)

        typer.echo("📤 Pushing data to remote storage...")
        returncode, stdout, stderr = dvc_manager._run_dvc_command(["push"])

        if returncode == 0:
            typer.echo("✅ Data push successful!")
        else:
            typer.echo(f"❌ Failed to push data: {stderr}")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Error pushing data: {e}")
        raise typer.Exit(1)


@dvc_app.command("pipeline")
def create_pipeline(
    pipeline_name: str = typer.Argument(..., help="Name of the pipeline"),
    stages_file: str = typer.Option(
        None, "--stages-file", help="Path to stages configuration file"
    ),
):
    """Create a DVC pipeline for data processing."""
    try:
        dvc_manager = DVCManager(state.data_dir)

        if stages_file:
            # Load stages from file
            with open(stages_file, "r", encoding="utf-8") as f:
                stages = yaml.safe_load(f)
            if stages is None:
                stages = []
        else:
            # Create default pipeline stages
            stages = [
                {
                    "name": "import",
                    "command": "wildtrain dataset import",
                    "deps": ["data/raw"],
                    "outs": ["data/processed"],
                },
                {
                    "name": "transform",
                    "command": "wildtrain dataset transform",
                    "deps": ["data/processed"],
                    "outs": ["data/transformed"],
                },
            ]

        if dvc_manager.create_pipeline(pipeline_name, stages):
            typer.echo(f"✅ Created DVC pipeline: {pipeline_name}")
            typer.echo("📋 Pipeline stages:")
            if stages:
                for stage in stages:
                    if (
                        isinstance(stage, dict)
                        and "name" in stage
                        and "command" in stage
                    ):
                        typer.echo(f"  - {stage['name']}: {stage['command']}")
        else:
            typer.echo("❌ Failed to create DVC pipeline")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Error creating pipeline: {e}")
        raise typer.Exit(1)


@dvc_app.command("run")
def run_pipeline(
    pipeline_name: str = typer.Argument(..., help="Name of the pipeline to run"),
):
    """Run a DVC pipeline."""
    try:
        dvc_manager = DVCManager(state.data_dir)

        typer.echo(f"🚀 Running DVC pipeline: {pipeline_name}")
        if dvc_manager.run_pipeline(pipeline_name):
            typer.echo("✅ Pipeline execution successful!")
        else:
            typer.echo("❌ Pipeline execution failed")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Error running pipeline: {e}")
        raise typer.Exit(1)


# Add the DVC group to the main app
app.add_typer(dvc_app, name="dvc")
