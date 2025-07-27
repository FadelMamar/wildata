"""
Command-line interface for the WildTrain data pipeline using Typer.
"""

import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from pydantic import ValidationError

from ..logging_config import setup_logging
from ..pipeline import DataPipeline
from ..visualization import FiftyOneManager
from .import_logic import _import_dataset_core, import_one_worker
from .models import (
    AugmentationConfigCLI,
    BboxClippingConfigCLI,
    BulkCreateROIDatasetConfig,
    BulkImportDatasetConfig,
    ImportDatasetConfig,
    ROIDatasetConfig,
    TilingConfigCLI,
    TransformationConfigCLI,
)
from .roi_logic import create_roi_dataset_core, create_roi_one_worker
from .utils import create_dataset_name

setup_logging()

__version__ = "0.1.0"

app = typer.Typer(
    name="wildata",
    help="Data Pipeline - Manage datasets in master format and create framework-specific formats",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def version():
    """Show version information."""
    typer.echo(f"wildata version {__version__}")


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


@app.command()
def bulk_import_datasets(
    config_file: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file (YAML only)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    num_workers: int = typer.Option(
        2, "--num-workers", "-n", help="Number of workers to use for bulk import"
    ),
):
    """Bulk import multiple datasets from all files in a directory.

    The config YAML should contain:
      source_path: path/to/directory  # directory containing dataset files
      source_format: yolo  # or coco, ls
      ... (other config fields)

    Each file in the directory will be imported as a dataset, with the dataset name derived from the filename (without extension).
    """
    if not (config_file.endswith(".yaml") or config_file.endswith(".yml")):
        typer.echo(
            "❌ Only YAML config files are supported for bulk import. Please provide a .yaml or .yml file."
        )
        raise typer.Exit(1)
    try:
        config = BulkImportDatasetConfig.from_yaml(config_file)
    except Exception as e:
        typer.echo(f"❌ Failed to load YAML config file: {traceback.format_exc()}")
        raise typer.Exit(1)

    # Validate directory
    files = []
    for source_path in config.source_paths:
        if not os.path.isdir(source_path):
            typer.echo(f"❌ source_path must be a directory: {source_path}")
            raise typer.Exit(1)
        files.extend(
            [
                f.resolve().as_posix()
                for f in Path(source_path).iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
        )

    if not files:
        typer.echo(f"❌ No files found in directories: {config.source_paths}")
        raise typer.Exit(1)
    dataset_names = [create_dataset_name(f) for f in files]
    formats = [config.source_format] * len(files)

    # Convert config to dict for pickling
    config_dict = config.model_dump()
    args_list = [
        (i, src, name, fmt, config_dict, verbose)
        for i, (src, name, fmt) in enumerate(zip(files, dataset_names, formats))
    ]

    results = [None] * len(files)
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
                typer.echo(f"✅ Import finished for '{name}' [{i+1}/{len(files)}]")
            else:
                typer.echo(f"❌ Import failed for '{name}' [{i+1}/{len(files)}]")
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

    create_roi_dataset_core(config, verbose)


@app.command()
def bulk_create_roi_datasets(
    config_file: str = typer.Option(
        ..., "--config", "-c", help="Path to YAML config file (YAML only)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    num_workers: int = typer.Option(
        2, "--num-workers", "-n", help="Number of workers to use for bulk ROI creation"
    ),
):
    """Bulk create ROI datasets from all files in a directory (multiprocessing).

    The config YAML should contain:
      source_path: path/to/directory  # directory containing dataset files
      source_format: yolo  # or coco
      root: data
      split_name: val
      bbox_tolerance: 5
      ls_xml_config: null # path to Label Studio XML config file
      ls_parse_config: false # parse Label Studio config using Server
      roi_config:
        random_roi_count: 1
        roi_box_size: 128
        min_roi_size: 32
        dark_threshold: 0.5
        background_class: background
        save_format: jpg
        quality: 95

    Each file in the directory will be used to create an ROI dataset, with the dataset name derived from the filename (without extension).
    """
    if not (config_file.endswith(".yaml") or config_file.endswith(".yml")):
        typer.echo(
            "❌ Only YAML config files are supported for bulk ROI creation. Please provide a .yaml or .yml file."
        )
        raise typer.Exit(1)
    try:
        config = BulkCreateROIDatasetConfig.from_yaml(config_file)
    except Exception as e:
        typer.echo(f"❌ Failed to load YAML config file: {traceback.format_exc()}")
        raise typer.Exit(1)

    # Validate directory
    files = []
    for source_path in config.source_paths:
        if not os.path.isdir(source_path):
            typer.echo(f"❌ source_path must be a directory: {source_path}")
            raise typer.Exit(1)
        files.extend(
            [
                f.resolve().as_posix()
                for f in Path(source_path).iterdir()
                if f.is_file() and not f.name.startswith(".")
            ]
        )
    if not files:
        typer.echo(f"❌ No files found in directories: {config.source_paths}")
        raise typer.Exit(1)
    dataset_names = [create_dataset_name(f) for f in files]

    config_dict = config.model_dump()
    args_list = [
        (i, src, name, config_dict, verbose)
        for i, (src, name) in enumerate(zip(files, dataset_names))
    ]

    results = [None] * len(files)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(create_roi_one_worker, args): i
            for i, args in enumerate(args_list)
        }
        for future in as_completed(future_to_idx):
            i, name, success, msg = future.result()
            results[i] = success
            if msg:
                typer.echo(msg)
            elif success:
                typer.echo(
                    f"✅ ROI creation finished for '{name}' [{i+1}/{len(files)}]"
                )
            else:
                typer.echo(f"❌ ROI creation failed for '{name}' [{i+1}/{len(files)}]")
    typer.echo(
        f"\nBulk ROI creation complete. {sum(results)}/{len(results)} succeeded."
    )


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
    root_data_directory: Optional[str] = typer.Option(
        None, "--root", help="Root data directory for classification dataset"
    ),
    load_as_single_class: Optional[bool] = typer.Option(
        False, "--single-class", help="Load as single class (True/False)"
    ),
    background_class_name: Optional[str] = typer.Option(
        "background", "--background-class", help="Background class name"
    ),
    single_class_name: Optional[str] = typer.Option(
        "wildlife", "--single-class-name", help="Single class name"
    ),
    keep_classes: Optional[str] = typer.Option(
        None, "--keep-classes", help="Comma-separated list of classes to keep"
    ),
    discard_classes: Optional[str] = typer.Option(
        None, "--discard-classes", help="Comma-separated list of classes to discard"
    ),
    split: str = typer.Option(
        "train", "--split", help="Dataset split (train/val/test)"
    ),
):
    """Visualize a classification dataset in FiftyOne (wraps import_classification_data)."""
    # Parse keep/discard classes if provided
    keep_classes_list = keep_classes.split(",") if keep_classes else None
    discard_classes_list = discard_classes.split(",") if discard_classes else None

    mgr = FiftyOneManager()
    mgr.import_classification_data(
        root_data_directory=root_data_directory or "",
        dataset_name=dataset_name,
        load_as_single_class=load_as_single_class or False,
        background_class_name=background_class_name or "background",
        single_class_name=single_class_name or "wildlife",
        keep_classes=keep_classes_list,
        discard_classes=discard_classes_list,
        split=split,
    )
    typer.echo(
        f"✅ Visualization launched in FiftyOne for dataset '{dataset_name}' (split: {split})"
    )


@app.command()
def visualize_detection(
    dataset_name: str = typer.Argument(..., help="Name for the FiftyOne dataset"),
    root_data_directory: str = typer.Option(
        ..., "--root", help="Root data directory for detection dataset"
    ),
    split: str = typer.Option(
        "train", "--split", help="Dataset split (train/val/test)"
    ),
):
    """Visualize a detection dataset in FiftyOne (wraps import_detection_data)."""
    mgr = FiftyOneManager()
    mgr.import_detection_data(
        root_data_directory=root_data_directory,
        dataset_name=dataset_name,
        split=split,
    )
    typer.echo(
        f"✅ Visualization launched in FiftyOne for detection dataset '{dataset_name}' (split: {split})"
    )


if __name__ == "__main__":
    app()
