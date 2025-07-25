"""
Command-line interface for the WildTrain data pipeline using Typer.
"""

import ast
import datetime
import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import typer
import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import ROOT, AugmentationConfig, ROIConfig, TilingConfig

from .logging_config import setup_logging

setup_logging()

from .pipeline import DataPipeline, Loader, PathManager, FrameworkDataManager
from .pipeline.dvc_manager import DVCConfig, DVCManager, DVCStorageType
from .transformations import (
    AugmentationTransformer,
    BoundingBoxClippingTransformer,
    TilingTransformer,
    TransformationPipeline,
)
from .visualization import FiftyOneManager

import concurrent.futures
from concurrent.futures import as_completed, ProcessPoolExecutor


class ROIConfigCLI(BaseModel):
    """ROI configuration for CLI."""

    random_roi_count: int = Field(default=1, description="Number of random ROIs")
    roi_box_size: int = Field(default=128, description="ROI box size")
    min_roi_size: int = Field(default=32, description="Minimum ROI size")
    dark_threshold: float = Field(default=0.5, description="Dark threshold")
    background_class: str = Field(
        default="background", description="Background class name"
    )
    save_format: str = Field(default="jpg", description="Save format")
    quality: int = Field(default=95, description="Image quality")

    @field_validator("random_roi_count", mode="before")
    @classmethod
    def validate_random_roi_count(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("roi_box_size", mode="before")
    @classmethod
    def validate_roi_box_size(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("min_roi_size", mode="before")
    @classmethod
    def validate_min_roi_size(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("quality", mode="before")
    @classmethod
    def validate_quality(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("dark_threshold", mode="before")
    @classmethod
    def validate_dark_threshold(cls, v: Any) -> float:
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("dark_threshold must be between 0 and 1")
        return v

    @field_validator("save_format", mode="before")
    @classmethod
    def validate_save_format(cls, v: Any) -> str:
        if v not in ["jpg", "jpeg", "png"]:
            raise ValueError("save_format must be one of: jpg, jpeg, png")
        return v


class TilingConfigCLI(BaseModel):
    """Tiling configuration for CLI."""

    tile_size: int = Field(default=512, description="Tile size")
    stride: int = Field(default=416, description="Stride between tiles")
    min_visibility: float = Field(default=0.1, description="Minimum visibility ratio")
    max_negative_tiles_in_negative_image: int = Field(
        default=3, description="Max negative tiles in negative image"
    )
    negative_positive_ratio: float = Field(
        default=1.0, description="Negative to positive ratio"
    )
    dark_threshold: float = Field(default=0.5, description="Dark threshold")

    @field_validator("tile_size", mode="before")
    @classmethod
    def validate_tile_size(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("stride", mode="before")
    @classmethod
    def validate_stride(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("max_negative_tiles_in_negative_image", mode="before")
    @classmethod
    def validate_max_negative_tiles_in_negative_image(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("min_visibility", mode="before")
    @classmethod
    def validate_min_visibility(cls, v: Any) -> float:
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("Value must be between 0 and 1")
        return v

    @field_validator("dark_threshold", mode="before")
    @classmethod
    def validate_dark_threshold(cls, v: Any) -> float:
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("Value must be between 0 and 1")
        return v

    @field_validator("negative_positive_ratio", mode="before")
    @classmethod
    def validate_negative_positive_ratio(cls, v: Any) -> float:
        v = float(v)
        if v < 0:
            raise ValueError("Ratio must be non-negative")
        return v


class AugmentationConfigCLI(BaseModel):
    """Augmentation configuration for CLI."""

    rotation_range: Tuple[float, float] = Field(
        default=(-45, 45), description="Rotation range"
    )
    probability: float = Field(
        default=1.0, description="Probability of applying augmentation"
    )
    brightness_range: Tuple[float, float] = Field(
        default=(-0.2, 0.4), description="Brightness range"
    )
    scale: Tuple[float, float] = Field(default=(1.0, 2.0), description="Scale range")
    translate: Tuple[float, float] = Field(
        default=(-0.1, 0.2), description="Translation range"
    )
    shear: Tuple[float, float] = Field(default=(-5, 5), description="Shear range")
    contrast_range: Tuple[float, float] = Field(
        default=(-0.2, 0.4), description="Contrast range"
    )
    noise_std: Tuple[float, float] = Field(
        default=(0.01, 0.1), description="Noise standard deviation range"
    )
    seed: int = Field(default=41, description="Random seed")
    num_transforms: int = Field(default=2, description="Number of transformations")

    @field_validator("probability", mode="before")
    @classmethod
    def validate_probability(cls, v: Any) -> float:
        v = float(v)
        if not 0 <= v <= 1:
            raise ValueError("Probability must be between 0 and 1")
        return v

    @field_validator("num_transforms", mode="before")
    @classmethod
    def validate_num_transforms(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed(cls, v: Any) -> int:
        v = int(v)
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class BboxClippingConfigCLI(BaseModel):
    """Bounding box clipping configuration for CLI."""

    tolerance: int = Field(default=5, description="Tolerance for clipping")
    skip_invalid: bool = Field(default=False, description="Skip invalid annotations")

    @field_validator("tolerance", mode="before")
    @classmethod
    def validate_tolerance(cls, v: Any) -> int:
        v = int(v)
        if v < 0:
            raise ValueError("Tolerance must be non-negative")
        return v


class TransformationConfigCLI(BaseModel):
    """Transformation pipeline configuration for CLI."""

    enable_bbox_clipping: bool = Field(default=True, description="Enable bbox clipping")
    bbox_clipping: Optional[BboxClippingConfigCLI] = Field(
        default=None, description="Bbox clipping config"
    )

    enable_augmentation: bool = Field(default=False, description="Enable augmentation")
    augmentation: Optional[AugmentationConfigCLI] = Field(
        default=None, description="Augmentation config"
    )

    enable_tiling: bool = Field(default=False, description="Enable tiling")
    tiling: Optional[TilingConfigCLI] = Field(default=None, description="Tiling config")


class ImportDatasetConfig(BaseModel):
    """Configuration for importing datasets."""

    # Required parameters
    source_path: str = Field(..., description="Path to source dataset")
    source_format: str = Field(..., description="Source format (coco/yolo)")
    dataset_name: str = Field(..., description="Name for the dataset")

    # Pipeline configuration
    root: str = Field(default="data", description="Root directory for data storage")
    split_name: str = Field(default="train", description="Split name (train/val/test)")
    enable_dvc: bool = Field(default=False, description="Enable DVC integration")

    # Processing options
    processing_mode: str = Field(
        default="batch", description="Processing mode (streaming/batch)"
    )
    track_with_dvc: bool = Field(default=False, description="Track dataset with DVC")
    bbox_tolerance: int = Field(default=5, description="Bbox validation tolerance")

    # Label Studio options
    dotenv_path: Optional[str] = Field(default=None, description="Path to .env file")
    ls_xml_config: Optional[str] = Field(
        default=None, description="Label Studio XML config path"
    )
    ls_parse_config: bool = Field(
        default=False, description="Parse Label Studio config"
    )

    # ROI configuration
    roi_config: Optional[ROIConfigCLI] = Field(
        default=None, description="ROI configuration"
    )
    disable_roi: bool = Field(default=False, description="Disable ROI extraction")

    # Transformation pipeline configuration
    transformations: Optional[TransformationConfigCLI] = Field(
        default=None, description="Transformation pipeline config"
    )

    # Validation methods
    @field_validator("source_format", mode="before")
    @classmethod
    def validate_source_format(cls, v: Any) -> str:
        if v not in ["coco", "yolo", "ls"]:
            raise ValueError('source_format must be either "coco" or "yolo"')
        return v

    @field_validator("split_name", mode="before")
    @classmethod
    def validate_split_name(cls, v: Any) -> str:
        if v not in ["train", "val", "test"]:
            raise ValueError("split_name must be one of: train, val, test")
        return v

    @field_validator("processing_mode", mode="before")
    @classmethod
    def validate_processing_mode(cls, v: Any) -> str:
        if v not in ["streaming", "batch"]:
            raise ValueError('processing_mode must be either "streaming" or "batch"')
        return v

    @field_validator("source_path", mode="before")
    @classmethod
    def validate_source_path(cls, v: Any) -> str:
        if not Path(v).exists():
            raise ValueError(f"Source path does not exist: {v}")
        return v

    @field_validator("dotenv_path", mode="before")
    @classmethod
    def validate_dotenv_path(cls, v: Any) -> str:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Dotenv path does not exist: {v}")
        return v

    @field_validator("ls_xml_config", mode="before")
    @classmethod
    def validate_ls_xml_config(cls, v: Any) -> str:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Label Studio XML config path does not exist: {v}")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "ImportDatasetConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)


class ROIDatasetConfig(BaseModel):
    """Configuration for creating ROI datasets."""
    source_path: str = Field(..., description="Path to source dataset")
    source_format: str = Field(..., description="Source format (coco/yolo)")
    dataset_name: str = Field(..., description="Name for the dataset")
    root: str = Field(default="data", description="Root directory for data storage")
    split_name: str = Field(default="val", description="Split name (train/val/test)")
    bbox_tolerance: int = Field(default=5, description="Bbox validation tolerance")
    roi_config: ROIConfigCLI = Field(..., description="ROI configuration")

    @field_validator("source_format", mode="before")
    @classmethod
    def validate_source_format(cls, v: Any) -> str:
        if v not in ["coco", "yolo"]:
            raise ValueError('source_format must be either "coco" or "yolo"')
        return v

    @field_validator("split_name", mode="before")
    @classmethod
    def validate_split_name(cls, v: Any) -> str:
        if v not in ["train", "val", "test"]:
            raise ValueError("split_name must be one of: train, val, test")
        return v

    @field_validator("source_path", mode="before")
    @classmethod
    def validate_source_path(cls, v: Any) -> str:
        if not Path(v).exists():
            raise ValueError(f"Source path does not exist: {v}")
        return v

    @classmethod
    def from_yaml(cls, path: str) -> "ROIDatasetConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)


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


__version__ = "0.1.0"

app = typer.Typer(
    name="wildata",
    help="Data Pipeline - Manage datasets in master format and create framework-specific formats",
    add_completion=False,
    rich_markup_mode="rich",
)


def _import_dataset_core(config: ImportDatasetConfig, verbose: bool = False):
    """Core logic for importing a dataset, shared by CLI and bulk import."""
    # Convert ROI config if provided
    roi_config = None
    if not config.disable_roi and config.roi_config:
        roi_config = ROIConfig(
            random_roi_count=config.roi_config.random_roi_count,
            roi_box_size=config.roi_config.roi_box_size,
            min_roi_size=config.roi_config.min_roi_size,
            dark_threshold=config.roi_config.dark_threshold,
            background_class=config.roi_config.background_class,
            save_format=config.roi_config.save_format,
            quality=config.roi_config.quality,
        )

    # Create transformation pipeline if configured
    transformation_pipeline = None
    if config.transformations:
        if verbose:
            typer.echo(f"🔧 Creating transformation pipeline...")

        transformation_pipeline = TransformationPipeline()

        # Add bbox clipping transformer
        if (
            config.transformations.enable_bbox_clipping
            and config.transformations.bbox_clipping
        ):
            bbox_config = config.transformations.bbox_clipping
            bbox_transformer = BoundingBoxClippingTransformer(
                tolerance=bbox_config.tolerance, skip_invalid=bbox_config.skip_invalid
            )
            transformation_pipeline.add_transformer(bbox_transformer)
            if verbose:
                typer.echo(
                    f"   Added BoundingBoxClippingTransformer (tolerance: {bbox_config.tolerance})"
                )

        # Add augmentation transformer
        if (
            config.transformations.enable_augmentation
            and config.transformations.augmentation
        ):
            aug_config = config.transformations.augmentation
            aug_transformer = AugmentationTransformer(
                config=AugmentationConfig(
                    rotation_range=aug_config.rotation_range,
                    probability=aug_config.probability,
                    brightness_range=aug_config.brightness_range,
                    scale=aug_config.scale,
                    translate=aug_config.translate,
                    shear=aug_config.shear,
                    contrast_range=aug_config.contrast_range,
                    noise_std=aug_config.noise_std,
                    seed=aug_config.seed,
                    num_transforms=aug_config.num_transforms,
                )
            )
            transformation_pipeline.add_transformer(aug_transformer)
            if verbose:
                typer.echo(
                    f"   Added AugmentationTransformer (num_transforms: {aug_config.num_transforms})"
                )

        # Add tiling transformer
        if config.transformations.enable_tiling and config.transformations.tiling:
            tiling_config = config.transformations.tiling
            tiling_transformer = TilingTransformer(
                config=TilingConfig(
                    tile_size=tiling_config.tile_size,
                    stride=tiling_config.stride,
                    min_visibility=tiling_config.min_visibility,
                    max_negative_tiles_in_negative_image=tiling_config.max_negative_tiles_in_negative_image,
                    negative_positive_ratio=tiling_config.negative_positive_ratio,
                    dark_threshold=tiling_config.dark_threshold,
                )
            )
            transformation_pipeline.add_transformer(tiling_transformer)
            if verbose:
                typer.echo(
                    f"   Added TilingTransformer (tile_size: {tiling_config.tile_size}, stride: {tiling_config.stride})"
                )

    # Execute import
    try:
        if verbose:
            typer.echo(f"🔧 Creating data pipeline...")
            typer.echo(f"   Root: {config.root}")
            typer.echo(f"   Split: {config.split_name}")
            typer.echo(f"   DVC enabled: {config.enable_dvc}")
            if transformation_pipeline:
                typer.echo(f"   Transformers: {len(transformation_pipeline)}")

        pipeline = DataPipeline(
            root=config.root,
            split_name=config.split_name,
            enable_dvc=config.enable_dvc,
            transformation_pipeline=transformation_pipeline,
        )

        if verbose:
            typer.echo(f"📥 Importing dataset...")
            typer.echo(f"   Source: {config.source_path}")
            typer.echo(f"   Format: {config.source_format}")
            typer.echo(f"   Name: {config.dataset_name}")
            typer.echo(f"   Mode: {config.processing_mode}")

        result = pipeline.import_dataset(
            source_path=config.source_path,
            source_format=config.source_format,
            dataset_name=config.dataset_name,
            processing_mode=config.processing_mode,
            track_with_dvc=config.track_with_dvc,
            bbox_tolerance=config.bbox_tolerance,
            roi_config=roi_config,
            dotenv_path=config.dotenv_path,
            ls_xml_config=config.ls_xml_config,
            ls_parse_config=config.ls_parse_config,
        )

        if result["success"]:
            typer.echo(f"✅ Successfully imported dataset '{config.dataset_name}'")
            if verbose:
                typer.echo(f"   Dataset info: {result['dataset_info_path']}")
                typer.echo(f"   Framework paths: {result['framework_paths']}")
                typer.echo(f"   Processing mode: {result['processing_mode']}")
                typer.echo(f"   DVC tracked: {result['dvc_tracked']}")
        else:
            typer.echo(f"❌ Failed to import dataset: {result['error']}")
            if "validation_errors" in result and result["validation_errors"]:
                typer.echo("   Validation errors:")
                for error in result["validation_errors"]:
                    typer.echo(f"     - {error}")
            if "hints" in result and result["hints"]:
                typer.echo("   Hints:")
                for hint in result["hints"]:
                    typer.echo(f"     - {hint}")
            return False
        return True
    except ValidationError as e:
        typer.echo(f"❌ Configuration validation error:")
        for error in e.errors():
            typer.echo(f"   {error['loc'][0]}: {error['msg']}")
        return False
    except Exception as e:
        typer.echo(f"❌ Import failed: {str(e)}")
        if verbose:
            typer.echo(f"   Traceback: {traceback.format_exc()}")
        return False

def import_one_worker(args):
    """Top-level worker for ProcessPoolExecutor in bulk_import_datasets."""
    i, src, name, fmt, config_dict, verbose = args
    from pydantic import ValidationError
    import traceback
    import typer
    try:
        typer.echo(f"\n=== Importing [{i+1}]: {name} ===")
        single_config = ImportDatasetConfig(
            source_path=src,
            source_format=fmt,
            dataset_name=name,
            root=config_dict['root'],
            split_name=config_dict['split_name'],
            enable_dvc=config_dict['enable_dvc'],
            processing_mode=config_dict['processing_mode'],
            track_with_dvc=config_dict['track_with_dvc'],
            bbox_tolerance=config_dict['bbox_tolerance'],
            dotenv_path=config_dict['dotenv_path'],
            ls_xml_config=config_dict['ls_xml_config'],
            ls_parse_config=config_dict['ls_parse_config'],
            roi_config=config_dict['roi_config'],
            disable_roi=config_dict['disable_roi'],
            transformations=config_dict['transformations'],
        )
        success = _import_dataset_core(single_config, verbose)
        return (i, name, success, None)
    except ValidationError as e:
        msg = f"❌ Configuration validation error for '{name}':\n" + "\n".join(f"   {error['loc'][0]}: {error['msg']}" for error in e.errors())
        return (i, name, False, msg)
    except Exception as e:
        msg = f"❌ Unexpected error for '{name}': {str(e)}"
        if verbose:
            msg += f"\n   Traceback: {traceback.format_exc()}"
        return (i, name, False, msg)

@app.command()
def import_dataset(
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to YAML config file"
    ),
    source_path: Optional[str] = typer.Argument(None, help="Path to source dataset"),
    source_format: Optional[str] = typer.Option(
        None, "--format", "-f", help="Source format (coco/yolo/ls)"
    ),
    dataset_name: Optional[str] = typer.Option(
        None, "--name", "-n", help="Dataset name"
    ),
    root: Optional[str] = typer.Option(
        None, "--root", "-r", help="Root directory for data storage"
    ),
    split_name: Optional[str] = typer.Option(
        None, "--split", "-s", help="Split name (train/val/test)"
    ),
    processing_mode: Optional[str] = typer.Option(
        None, "--mode", "-m", help="Processing mode (streaming/batch)"
    ),
    track_with_dvc: Optional[bool] = typer.Option(
        None, "--track-dvc", help="Track dataset with DVC"
    ),
    bbox_tolerance: Optional[int] = typer.Option(
        None, "--bbox-tolerance", help="Bbox validation tolerance"
    ),
    dotenv_path: Optional[str] = typer.Option(
        None, "--dotenv", help="Path to .env file"
    ),
    ls_xml_config: Optional[str] = typer.Option(
        None, "--ls-config", help="Label Studio XML config path"
    ),
    ls_parse_config: Optional[bool] = typer.Option(
        None, "--parse-ls-config", help="Parse Label Studio config"
    ),
    # Transformation pipeline options
    enable_bbox_clipping: Optional[bool] = typer.Option(
        None, "--enable-bbox-clipping", help="Enable bbox clipping"
    ),
    bbox_clipping_tolerance: Optional[int] = typer.Option(
        None, "--bbox-clipping-tolerance", help="Bbox clipping tolerance"
    ),
    skip_invalid_bbox: Optional[bool] = typer.Option(
        None, "--skip-invalid-bbox", help="Skip invalid bboxes"
    ),
    enable_augmentation: Optional[bool] = typer.Option(
        None, "--enable-augmentation", help="Enable data augmentation"
    ),
    augmentation_probability: Optional[float] = typer.Option(
        None, "--aug-prob", help="Augmentation probability"
    ),
    num_augmentations: Optional[int] = typer.Option(
        None, "--num-augs", help="Number of augmentations per image"
    ),
    enable_tiling: Optional[bool] = typer.Option(
        None, "--enable-tiling", help="Enable image tiling"
    ),
    tile_size: Optional[int] = typer.Option(None, "--tile-size", help="Tile size"),
    tile_stride: Optional[int] = typer.Option(
        None, "--tile-stride", help="Tile stride"
    ),
    min_visibility: Optional[float] = typer.Option(
        None, "--min-visibility", help="Minimum visibility ratio"
    ),
    disable_roi: Optional[bool] = typer.Option(
        None, "--disable-roi", help="Disable ROI extraction"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Import a dataset from various formats into the WildData pipeline."""

    # Enforce mutual exclusivity
    if config_file:
        # If config is given, do not allow any other required args
        if any([source_path, source_format, dataset_name]):
            typer.echo(
                "❌ If --config is provided, do not provide other arguments.", err=True
            )
            raise typer.Exit(1)
        try:
            config = ImportDatasetConfig.from_yaml(config_file)
        except Exception as e:
            typer.echo(f"❌ Failed to load config file: {traceback.format_exc()}")
            raise typer.Exit(1)
    else:
        # If config is not given, require all required args
        missing = []
        if not source_path:
            missing.append("source_path")
        if not source_format:
            missing.append("source_format")
        if not dataset_name:
            missing.append("dataset_name")
        if missing:
            typer.echo(f"❌ Missing required arguments: {', '.join(missing)}", err=True)
            raise typer.Exit(1)
        # Create transformation config from command-line arguments
        transformation_config = None
        if enable_bbox_clipping or enable_augmentation or enable_tiling:
            transformation_config = TransformationConfigCLI(
                enable_bbox_clipping=enable_bbox_clipping
                if enable_bbox_clipping is not None
                else True,
                bbox_clipping=BboxClippingConfigCLI(
                    tolerance=bbox_clipping_tolerance
                    if bbox_clipping_tolerance is not None
                    else 5,
                    skip_invalid=skip_invalid_bbox
                    if skip_invalid_bbox is not None
                    else False,
                )
                if enable_bbox_clipping
                else None,
                enable_augmentation=enable_augmentation
                if enable_augmentation is not None
                else False,
                augmentation=AugmentationConfigCLI(
                    probability=augmentation_probability
                    if augmentation_probability is not None
                    else 1.0,
                    num_transforms=num_augmentations
                    if num_augmentations is not None
                    else 2,
                )
                if enable_augmentation
                else None,
                enable_tiling=enable_tiling if enable_tiling is not None else False,
                tiling=TilingConfigCLI(
                    tile_size=tile_size if tile_size is not None else 512,
                    stride=tile_stride if tile_stride is not None else 416,
                    min_visibility=min_visibility
                    if min_visibility is not None
                    else 0.1,
                )
                if enable_tiling
                else None,
            )
        # Create config from command-line arguments
        config_data = {
            "source_path": source_path,
            "source_format": source_format,
            "dataset_name": dataset_name,
            "root": root if root is not None else "data",
            "split_name": split_name if split_name is not None else "train",
            "processing_mode": processing_mode
            if processing_mode is not None
            else "batch",
            "track_with_dvc": track_with_dvc if track_with_dvc is not None else False,
            "bbox_tolerance": bbox_tolerance if bbox_tolerance is not None else 5,
            "dotenv_path": dotenv_path,
            "ls_xml_config": ls_xml_config,
            "ls_parse_config": ls_parse_config
            if ls_parse_config is not None
            else False,
            "transformations": transformation_config,
            "disable_roi": disable_roi if disable_roi is not None else False,
        }
        try:
            config = ImportDatasetConfig(**config_data)
        except ValidationError as e:
            typer.echo(f"❌ Configuration validation error:")
            for error in e.errors():
                typer.echo(f"   {error['loc'][0]}: {error['msg']}")
            raise typer.Exit(1)

    _import_dataset_core(config, verbose)


class BulkImportDatasetConfig(BaseModel):
    """Configuration for bulk importing datasets from a directory."""
    source_path: str = Field(..., description="Directory containing source dataset files")
    source_format: str = Field(..., description="Source format (coco/yolo/ls)")
    root: str = Field(default="data", description="Root directory for data storage")
    split_name: str = Field(default="train", description="Split name (train/val/test)")
    enable_dvc: bool = Field(default=False, description="Enable DVC integration")
    processing_mode: str = Field(default="batch", description="Processing mode (streaming/batch)")
    track_with_dvc: bool = Field(default=False, description="Track dataset with DVC")
    bbox_tolerance: int = Field(default=5, description="Bbox validation tolerance")
    dotenv_path: Optional[str] = Field(default=None, description="Path to .env file")
    ls_xml_config: Optional[str] = Field(default=None, description="Label Studio XML config path")
    ls_parse_config: bool = Field(default=False, description="Parse Label Studio config")
    roi_config: Optional[ROIConfigCLI] = Field(default=None, description="ROI configuration")
    disable_roi: bool = Field(default=False, description="Disable ROI extraction")
    transformations: Optional[TransformationConfigCLI] = Field(default=None, description="Transformation pipeline config")

    @classmethod
    def from_yaml(cls, path: str) -> "BulkImportDatasetConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

@app.command()
def bulk_import_datasets(
    config_file: str = typer.Option(..., "--config", "-c", help="Path to YAML config file (YAML only)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    num_workers: int = typer.Option(4, "--num-workers", "-n", help="Number of workers to use for bulk import"),
):
    """Bulk import multiple datasets from all files in a directory.

    The config YAML should contain:
      source_path: path/to/directory  # directory containing dataset files
      source_format: yolo  # or coco, ls
      ... (other config fields)

    Each file in the directory will be imported as a dataset, with the dataset name derived from the filename (without extension).
    """
    import os
    if not (config_file.endswith('.yaml') or config_file.endswith('.yml')):
        typer.echo("❌ Only YAML config files are supported for bulk import. Please provide a .yaml or .yml file.")
        raise typer.Exit(1)
    try:
        config = BulkImportDatasetConfig.from_yaml(config_file)
    except Exception as e:
        typer.echo(f"❌ Failed to load YAML config file: {traceback.format_exc()}")
        raise typer.Exit(1)

    # Validate directory
    if not os.path.isdir(config.source_path):
        typer.echo(f"❌ source_path must be a directory: {config.source_path}")
        raise typer.Exit(1)
    # List files in directory (ignore hidden files)
    files = [f for f in os.listdir(config.source_path) if os.path.isfile(os.path.join(config.source_path, f)) and not f.startswith('.')]
    if not files:
        typer.echo(f"❌ No files found in directory: {config.source_path}")
        raise typer.Exit(1)
    source_paths = [os.path.join(config.source_path, f) for f in files]
    dataset_names = [Path(f).stem.replace(" ", "").replace(",", "-").lower() for f in files]
    formats = [config.source_format] * len(source_paths)

    # Convert config to dict for pickling
    config_dict = config.model_dump()
    args_list = [
        (i, src, name, fmt, config_dict, verbose)
        for i, (src, name, fmt) in enumerate(zip(source_paths, dataset_names, formats))
    ]

    results = [None] * len(source_paths)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(import_one_worker, args): i
            for i, args in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            i, name, success, msg = future.result()
            results[i] = success
            if msg:
                typer.echo(msg)
            elif success:
                typer.echo(f"✅ Import finished for '{name}' [{i+1}/{len(source_paths)}]")
            else:
                typer.echo(f"❌ Import failed for '{name}' [{i+1}/{len(source_paths)}]")
    typer.echo(f"\nBulk import complete. {sum(results)}/{len(results)} succeeded.")


@app.command()
def create_roi_dataset(
    config_file: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Create an ROI dataset from a source dataset using a YAML config file."""
    # Only config file is allowed
    try:
        config = ROIDatasetConfig.from_yaml(config_file)
    except Exception as e:
        typer.echo(f"❌ Failed to load config file: {traceback.format_exc()}")
        raise typer.Exit(1)

    try:
        if verbose:
            typer.echo(f"🔧 Creating dataset...")
            typer.echo(f"   Source: {config.source_path}")
            typer.echo(f"   Format: {config.source_format}")
            typer.echo(f"   Name: {config.dataset_name}")
            typer.echo(f"   Split: {config.split_name}")
            typer.echo(f"   Output root: {config.root}")
            typer.echo(f"   ROI config: {config.roi_config}")

        loader = Loader()
        dataset_info, split_coco_data = loader.load(
            config.source_path,
            config.source_format,
            config.dataset_name,
            config.bbox_tolerance,
            config.split_name,
        )

        path_manager = PathManager(Path(config.root))
        framework_data_manager = FrameworkDataManager(path_manager)
        roi_config = ROIConfig(
            random_roi_count=config.roi_config.random_roi_count,
            roi_box_size=config.roi_config.roi_box_size,
            min_roi_size=config.roi_config.min_roi_size,
            dark_threshold=config.roi_config.dark_threshold,
            roi_callback=None,  # Not supported via CLI config
            background_class=config.roi_config.background_class,
            save_format=config.roi_config.save_format,
            quality=config.roi_config.quality,
        )
        framework_data_manager.create_roi_format(
            dataset_name=config.dataset_name,
            coco_data=split_coco_data[config.split_name],
            split=config.split_name,
            roi_config=roi_config,
        )
        typer.echo(f"✅ Successfully created ROI dataset for '{config.dataset_name}' (split: {config.split_name}) at {config.root}")
    except ValidationError as e:
        typer.echo(f"❌ Configuration validation error:")
        for error in e.errors():
            typer.echo(f"   {error['loc'][0]}: {error['msg']}")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Failed to create ROI dataset: {str(e)}")
        if verbose:
            typer.echo(f"   Traceback: {traceback.format_exc()}")
        raise typer.Exit(1)


@app.command()
def list_datasets(
    root: str = typer.Option(
        "data", "--root", "-r", help="Root directory for data storage"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """List all available datasets."""
    try:
        pipeline = DataPipeline(root=root, split_name="train")
        datasets = pipeline.list_datasets()

        if not datasets:
            typer.echo("📭 No datasets found")
            return

        typer.echo(f"📚 Found {len(datasets)} dataset(s):")
        for dataset in datasets:
            typer.echo(f"   • {dataset['name']}")
            if verbose:
                typer.echo(f"     Created: {dataset.get('created_at', 'Unknown')}")
                typer.echo(f"     Size: {dataset.get('size', 'Unknown')}")
                typer.echo(f"     Format: {dataset.get('format', 'Unknown')}")

    except Exception as e:
        typer.echo(f"❌ Failed to list datasets: {str(e)}")
        raise typer.Exit(1)


@app.command()
def export_dataset(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to export"),
    target_format: str = typer.Option(
        ..., "--format", "-f", help="Target format (coco/yolo)"
    ),
    target_path: str = typer.Option(..., "--output", "-o", help="Output path"),
    root: str = typer.Option(
        "data", "--root", "-r", help="Root directory for data storage"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Export a dataset to a specific format."""
    try:
        pipeline = DataPipeline(root=root, split_name="train")

        if verbose:
            typer.echo(
                f"📤 Exporting dataset '{dataset_name}' to {target_format} format..."
            )

        success = pipeline.export_dataset(dataset_name, target_format, target_path)

        if success:
            typer.echo(
                f"✅ Successfully exported dataset '{dataset_name}' to {target_path}"
            )
        else:
            typer.echo(f"❌ Failed to export dataset '{dataset_name}'")
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Export failed: {str(e)}")
        if verbose:
            typer.echo(f"   Traceback: {traceback.format_exc()}")
        raise typer.Exit(1)


@app.command()
def visualize_classification(
    dataset_name: str = typer.Argument(..., help="Name for the FiftyOne dataset"),
    root_data_directory: Optional[str] = typer.Option(None, "--root", help="Root data directory for classification dataset"),
    load_as_single_class: Optional[bool] = typer.Option(False, "--single-class", help="Load as single class (True/False)"),
    background_class_name: Optional[str] = typer.Option("background", "--background-class", help="Background class name"),
    single_class_name: Optional[str] = typer.Option("wildlife", "--single-class-name", help="Single class name"),
    keep_classes: Optional[str] = typer.Option(None, "--keep-classes", help="Comma-separated list of classes to keep"),
    discard_classes: Optional[str] = typer.Option(None, "--discard-classes", help="Comma-separated list of classes to discard"),
    split: str = typer.Option("train", "--split", help="Dataset split (train/val/test)"),
):
    """Visualize a classification dataset in FiftyOne (wraps import_classification_data)."""
    # Parse keep/discard classes if provided
    keep_classes_list = keep_classes.split(",") if keep_classes else None
    discard_classes_list = discard_classes.split(",") if discard_classes else None
    
    mgr = FiftyOneManager()
    mgr.import_classification_data(
        root_data_directory=root_data_directory,
        dataset_name=dataset_name,
        load_as_single_class=load_as_single_class,
        background_class_name=background_class_name,
        single_class_name=single_class_name,
        keep_classes=keep_classes_list,
        discard_classes=discard_classes_list,
        split=split,
    )
    typer.echo(f"✅ Visualization launched in FiftyOne for dataset '{dataset_name}' (split: {split})")


@app.command()
def version():
    """Show version information."""
    typer.echo(f"wildata version {__version__}")


if __name__ == "__main__":
    app()
