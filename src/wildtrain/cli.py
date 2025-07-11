"""
Command-line interface for the WildTrain data pipeline using Typer.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

import typer

from .config import AugmentationConfig, TilingConfig
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
    data_dir: Path = Path.cwd() / "data"


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
):
    """WildTrain Data Pipeline CLI."""
    if data_dir:
        state.data_dir = data_dir
    else:
        state.data_dir = Path.cwd() / "data"


def raise_(ex):
    raise ex


# Dataset command group

dataset_app = typer.Typer(help="Dataset management commands.")


@dataset_app.command("import")
def import_dataset(
    source_path: str = typer.Argument(..., help="Path to the source dataset"),
    format_type: str = typer.Argument(
        ..., help="Format type of the source dataset", case_sensitive=False
    ),
    dataset_name: str = typer.Argument(
        ..., help="Name for the dataset in master storage"
    ),
    no_hints: bool = typer.Option(False, "--no-hints", help="Disable validation hints"),
    track_with_dvc: bool = typer.Option(
        False, "--track-with-dvc", help="Track dataset with DVC"
    ),
    # Augmentation options
    augment: bool = typer.Option(False, "--augment", help="Apply data augmentation"),
    rotation_range: Tuple[float, float] = typer.Option(
        (-10, 10), "--rotation", help="Rotation range in degrees"
    ),
    probability: float = typer.Option(
        0.5, "--probability", help="Augmentation probability (0-1)"
    ),
    brightness_range: Tuple[float, float] = typer.Option(
        (0.9, 1.1), "--brightness", help="Brightness range"
    ),
    contrast_range: Tuple[float, float] = typer.Option(
        (0.9, 1.1), "--contrast", help="Contrast range"
    ),
    noise_std: float = typer.Option(0.01, "--noise", help="Noise standard deviation"),
    # Tiling options
    tile: bool = typer.Option(False, "--tile", help="Apply image tiling"),
    tile_size: int = typer.Option(512, "--tile-size", help="Tile size in pixels"),
    stride: int = typer.Option(256, "--stride", help="Stride between tiles"),
    min_visibility: float = typer.Option(
        0.1, "--min-visibility", help="Minimum object visibility in tiles (0-1)"
    ),
    max_negative_tiles: int = typer.Option(
        3, "--max-negative-tiles", help="Maximum negative tiles per image"
    ),
    negative_positive_ratio: float = typer.Option(
        1.0, "--negative-ratio", help="Negative to positive tile ratio"
    ),
):
    print("[DEBUG] Entered CLI import_dataset command")
    """Import a dataset from COCO or YOLO format with optional transformations and DVC tracking."""
    if format_type.lower() not in ["coco", "yolo"]:
        typer.echo(
            f"❌ Error: Format type must be 'coco' or 'yolo', got '{format_type}'"
        )
        raise typer.Exit(1)

    pipeline = DataPipeline(str(state.data_dir))

    # Setup transformations if requested
    if augment or tile:
        transformation_pipeline = TransformationPipeline()

        if augment:
            try:
                aug_config = AugmentationConfig(
                    rotation_range=rotation_range,
                    probability=probability,
                    brightness_range=brightness_range,
                    contrast_range=contrast_range,
                    noise_std=noise_std,
                )
                aug_transformer = AugmentationTransformer(aug_config)
                transformation_pipeline.add_transformer(aug_transformer)
                typer.echo(
                    f"🔧 Added augmentation transformer (probability: {probability})"
                )
            except Exception as e:
                typer.echo(f"❌ Error setting up augmentation: {e}")
                raise typer.Exit(1)

        if tile:
            try:
                tile_config = TilingConfig(
                    tile_size=tile_size,
                    stride=stride,
                    min_visibility=min_visibility,
                    max_negative_tiles_in_negative_image=max_negative_tiles,
                    negative_positive_ratio=negative_positive_ratio,
                )
                tile_transformer = TilingTransformer(tile_config)
                transformation_pipeline.add_transformer(tile_transformer)
                typer.echo(
                    f"🔧 Added tiling transformer (tile size: {tile_size}, stride: {stride})"
                )
            except Exception as e:
                typer.echo(f"❌ Error setting up tiling: {e}")
                raise typer.Exit(1)

        # Create new pipeline with transformations
        pipeline = DataPipeline(str(state.data_dir), transformation_pipeline)

    try:
        typer.echo(f"🚀 Importing {format_type.upper()} dataset from {source_path}")
        typer.echo(f"📝 Dataset name: {dataset_name}")
        if augment or tile:
            typer.echo(
                f"🔧 Applying transformations: {'augmentation' if augment else ''}{' + tiling' if tile else ''}"
            )
        if track_with_dvc:
            typer.echo("📦 DVC tracking enabled")
        typer.echo("─" * 50)

        result = pipeline.import_dataset(
            source_path=source_path,
            source_format=format_type.lower(),
            dataset_name=dataset_name,
            apply_transformations=augment or tile,
            track_with_dvc=track_with_dvc,
        )

        if result["success"]:
            typer.echo("✅ Import successful!")
            typer.echo(f"📄 Master annotations: {result['master_path']}")
            typer.echo("🔧 Framework formats created:")
            for framework, path in result["framework_paths"].items():
                typer.echo(f"  - {framework.upper()}: {path}")
            if track_with_dvc and result.get("dvc_tracked"):
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
    """List all available datasets in master storage."""
    pipeline = DataPipeline(str(state.data_dir))
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
    """Get detailed information about a specific dataset."""
    pipeline = DataPipeline(str(state.data_dir))
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
    """Export a dataset to a specific framework format."""
    if framework.lower() not in ["coco", "yolo"]:
        typer.echo(f"❌ Error: Framework must be 'coco' or 'yolo', got '{framework}'")
        raise typer.Exit(1)
    pipeline = DataPipeline(str(state.data_dir))
    try:
        result = pipeline.export_framework_format(dataset_name, framework.lower())
        typer.echo(
            f"✅ Exported dataset '{dataset_name}' to {framework.upper()} format"
        )
        typer.echo(f"📁 Output path: {result['output_path']}")
        if framework.lower() == "coco":
            typer.echo(f"📂 Data directory: {result['data_dir']}")
            typer.echo(f"📄 Annotations file: {result['annotations_file']}")
        elif framework.lower() == "yolo":
            typer.echo(f"🖼️  Images directory: {result['images_dir']}")
            typer.echo(f"🏷️  Labels directory: {result['labels_dir']}")
            typer.echo(f"⚙️  Data YAML: {result['data_yaml']}")
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
    """Validate a dataset without importing it."""
    if format_type.lower() not in ["coco", "yolo"]:
        typer.echo(
            f"❌ Error: Format type must be 'coco' or 'yolo', got '{format_type}'"
        )
        raise typer.Exit(1)
    pipeline = DataPipeline(str(state.data_dir))
    try:
        typer.echo(f"🔍 Validating {format_type.upper()} dataset at {source_path}")
        typer.echo("─" * 50)
        # Validate dataset
        if format_type.lower() == "coco":
            from wildtrain.validators.coco_validator import COCOValidator

            validator = COCOValidator(source_path)
            is_valid, errors, warnings = validator.validate()
        elif format_type.lower() == "yolo":
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
    """Show the current status of the pipeline."""
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
        pipeline = DataPipeline(str(state.data_dir))
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


import json
import os
import shutil
import tempfile

import numpy as np


@dataset_app.command()
def transform(
    dataset_name: str = typer.Argument(..., help="Name of the dataset to transform"),
    # Augmentation options
    augment: bool = typer.Option(False, "--augment", help="Apply data augmentation"),
    rotation_range: Tuple[float, float] = typer.Option(
        (-10, 10), "--rotation", help="Rotation range in degrees"
    ),
    probability: float = typer.Option(
        0.5, "--probability", help="Augmentation probability (0-1)"
    ),
    brightness_range: Tuple[float, float] = typer.Option(
        (0.9, 1.1), "--brightness", help="Brightness range"
    ),
    contrast_range: Tuple[float, float] = typer.Option(
        (0.9, 1.1), "--contrast", help="Contrast range"
    ),
    noise_std: float = typer.Option(0.01, "--noise", help="Noise standard deviation"),
    # Tiling options
    tile: bool = typer.Option(False, "--tile", help="Apply image tiling"),
    tile_size: int = typer.Option(512, "--tile-size", help="Tile size in pixels"),
    stride: int = typer.Option(256, "--stride", help="Stride between tiles"),
    min_visibility: float = typer.Option(
        0.1, "--min-visibility", help="Minimum object visibility in tiles (0-1)"
    ),
    max_negative_tiles: int = typer.Option(
        3, "--max-negative-tiles", help="Maximum negative tiles per image"
    ),
    negative_positive_ratio: float = typer.Option(
        1.0, "--negative-ratio", help="Negative to positive tile ratio"
    ),
    output_name: str = typer.Option(
        None,
        "--output-name",
        help="Name for the transformed dataset (default: {dataset_name}_transformed)",
    ),
):
    """Apply transformations to an existing dataset."""
    if not augment and not tile:
        typer.echo(
            "❌ Error: Must specify at least one transformation (--augment or --tile)"
        )
        raise typer.Exit(1)

    if output_name is None:
        output_name = f"{dataset_name}_transformed"

    pipeline = DataPipeline(str(state.data_dir))

    # Setup transformations
    transformation_pipeline = TransformationPipeline()

    if augment:
        try:
            aug_config = AugmentationConfig(
                rotation_range=rotation_range,
                probability=probability,
                brightness_range=brightness_range,
                contrast_range=contrast_range,
                noise_std=noise_std,
            )
            aug_transformer = AugmentationTransformer(aug_config)
            transformation_pipeline.add_transformer(aug_transformer)
            typer.echo(
                f"🔧 Added augmentation transformer (probability: {probability})"
            )
        except Exception as e:
            typer.echo(f"❌ Error setting up augmentation: {e}")
            raise typer.Exit(1)

    if tile:
        try:
            tile_config = TilingConfig(
                tile_size=tile_size,
                stride=stride,
                min_visibility=min_visibility,
                max_negative_tiles_in_negative_image=max_negative_tiles,
                negative_positive_ratio=negative_positive_ratio,
            )
            tile_transformer = TilingTransformer(tile_config)
            transformation_pipeline.add_transformer(tile_transformer)
            typer.echo(
                f"🔧 Added tiling transformer (tile size: {tile_size}, stride: {stride})"
            )
        except Exception as e:
            typer.echo(f"❌ Error setting up tiling: {e}")
            raise typer.Exit(1)

    try:
        typer.echo(f"🔧 Applying transformations to dataset '{dataset_name}'")
        typer.echo(f"📝 Output dataset name: {output_name}")
        typer.echo("─" * 50)

        # Load the original dataset info
        original_info = pipeline.get_dataset_info(dataset_name)
        typer.echo(
            f"📊 Original dataset: {original_info['total_images']} images, {original_info['total_annotations']} annotations"
        )

        # Apply transformations
        # Note: This is a simplified approach - in a real implementation, you'd need to
        # load the master data, apply transformations, and save as a new dataset
        typer.echo("⚠️  Transformation command not yet fully implemented")
        typer.echo("💡 Use --augment or --tile with the import command instead")
        raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def demo(
    use_real_data: bool = typer.Option(
        False, "--use-real-data", help="Use real data instead of synthetic data"
    ),
    keep_temp: bool = typer.Option(
        False, "--keep-temp", help="Keep temporary files after processing"
    ),
    # Transformation options
    augment: bool = typer.Option(
        False, "--augment", help="Apply data augmentation to synthetic demo"
    ),
    tile: bool = typer.Option(
        False, "--tile", help="Apply image tiling to synthetic demo"
    ),
    rotation_range: Tuple[float, float] = typer.Option(
        (-10, 10), "--rotation", help="Rotation range in degrees"
    ),
    probability: float = typer.Option(
        0.5, "--probability", help="Augmentation probability (0-1)"
    ),
    tile_size: int = typer.Option(512, "--tile-size", help="Tile size in pixels"),
    stride: int = typer.Option(256, "--stride", help="Stride between tiles"),
):
    """Run a full workflow demo (synthetic or real data) with optional transformations."""
    # Determine data directory
    if state.data_dir and not (
        str(state.data_dir).startswith(str(Path(tempfile.gettempdir())))
    ):
        data_dir = state.data_dir
        temp_dir = None
        typer.echo(f"[INFO] Using specified data directory: {data_dir}")
        data_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp()
        data_dir = Path(temp_dir)
        typer.echo(f"[INFO] Created temp directory: {temp_dir}")

    def setup_real_data_paths():
        real_data_paths = {
            "coco": {
                "annotation_path": Path(r"D:/workspace/savmap/coco/annotations.json"),
                "description": "COCO format annotation file",
            },
            "yolo": {
                "data_yaml_path": Path(r"D:/workspace/savmap/yolo/data.yaml"),
                "description": "YOLO format data.yaml file",
            },
        }
        return real_data_paths

    def check_data_availability(real_data_paths):
        available_data = {}
        for format_name, data_info in real_data_paths.items():
            if format_name == "coco":
                path = data_info["annotation_path"]
                if path.exists():
                    available_data[format_name] = {
                        "path": path,
                        "description": data_info["description"],
                    }
                    typer.echo(f"[INFO] Found COCO data: {path}")
                else:
                    typer.echo(f"[WARNING] COCO data not found: {path}")
            elif format_name == "yolo":
                path = data_info["data_yaml_path"]
                if path.exists():
                    available_data[format_name] = {
                        "path": path,
                        "description": data_info["description"],
                    }
                    typer.echo(f"[INFO] Found YOLO data: {path}")
                else:
                    typer.echo(f"[WARNING] YOLO data not found: {path}")
        return available_data

    def create_synthetic_coco_data(images_dir: Path, annotation_file: Path):
        images = []
        for i in range(2):
            img_name = f"test_image_{i + 1}.jpg"
            img_path = images_dir / img_name
            arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            try:
                import cv2

                cv2.imwrite(str(img_path), arr)
            except ImportError:
                from PIL import Image

                Image.fromarray(arr).save(str(img_path))
            images.append(
                {
                    "id": i + 1,
                    "file_name": img_name,
                    "width": 640,
                    "height": 480,
                    "split": "train",
                }
            )
        annotations = [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [100, 100, 200, 150],
                "area": 30000,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [150, 150, 250, 200],
                "area": 25000,
                "iscrowd": 0,
            },
        ]
        categories = [{"id": 1, "name": "test_category", "supercategory": "test"}]
        coco_data = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }
        with open(annotation_file, "w") as f:
            json.dump(coco_data, f)

    try:
        typer.echo("--- WildTrain Data Pipeline CLI Demo ---")

        # Setup transformations if requested
        transformation_pipeline = None
        if augment or tile:
            transformation_pipeline = TransformationPipeline()

            if augment:
                try:
                    aug_config = AugmentationConfig(
                        rotation_range=rotation_range, probability=probability
                    )
                    aug_transformer = AugmentationTransformer(aug_config)
                    transformation_pipeline.add_transformer(aug_transformer)
                    typer.echo(
                        f"[INFO] Added augmentation transformer (probability: {probability})"
                    )
                except Exception as e:
                    typer.echo(f"[WARNING] Could not add augmentation: {e}")

            if tile:
                try:
                    tile_config = TilingConfig(tile_size=tile_size, stride=stride)
                    tile_transformer = TilingTransformer(tile_config)
                    transformation_pipeline.add_transformer(tile_transformer)
                    typer.echo(
                        f"[INFO] Added tiling transformer (tile size: {tile_size}, stride: {stride})"
                    )
                except Exception as e:
                    typer.echo(f"[WARNING] Could not add tiling: {e}")

        pipeline = DataPipeline(str(data_dir), transformation_pipeline)
        typer.echo("[INFO] DataPipeline initialized.")
        status = pipeline.get_pipeline_status()
        typer.echo(f"[INFO] Pipeline status:")
        typer.echo(f"  - Master data directory: {status['master_data_dir']}")
        typer.echo(f"  - Supported formats: {status['supported_formats']}")
        typer.echo(f"  - Available datasets: {status['available_datasets']}")

        if use_real_data:
            real_data_paths = setup_real_data_paths()
            available_data = check_data_availability(real_data_paths)
            if not available_data:
                typer.echo(
                    "[WARNING] No real data found. Falling back to synthetic data."
                )
                use_real_data = False
        if not use_real_data:
            # Synthetic demo
            typer.echo("\n--- Processing Synthetic Dataset ---")
            # Create synthetic data in the data directory
            images_dir = data_dir / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            annotation_file = data_dir / "annotations_train.json"
            create_synthetic_coco_data(images_dir, annotation_file)
            typer.echo(f"[INFO] Synthetic COCO data created at {annotation_file}")
            result = pipeline.import_dataset(
                source_path=str(annotation_file),
                source_format="coco",
                dataset_name="synthetic_demo_dataset",
                apply_transformations=augment or tile,
            )
            if result["success"]:
                typer.echo(f"[SUCCESS] Imported synthetic dataset")
                typer.echo(f"Master annotations: {result['master_path']}")
                # Export to framework formats
                for export_format in ["coco", "yolo"]:
                    try:
                        export_result = pipeline.export_framework_format(
                            "synthetic_demo_dataset", export_format
                        )
                        typer.echo(
                            f"[SUCCESS] Exported to {export_format.upper()}: {export_result['output_path']}"
                        )
                    except Exception as e:
                        typer.echo(
                            f"[WARNING] Failed to export to {export_format.upper()}: {e}"
                        )
            else:
                typer.echo(f"[ERROR] Failed to import synthetic dataset")
                typer.echo(f"Error: {result.get('error', 'Unknown error')}")
        else:
            # Real data demo
            for format_type, data_info in available_data.items():
                dataset_name = f"real_{format_type}_dataset"
                typer.echo(f"\n--- Processing {format_type.upper()} Dataset ---")
                typer.echo(f"Source: {data_info['path']}")
                typer.echo(f"Description: {data_info['description']}")
                result = pipeline.import_dataset(
                    source_path=str(data_info["path"]),
                    source_format=format_type,
                    dataset_name=dataset_name,
                    apply_transformations=augment or tile,
                )
                if result["success"]:
                    typer.echo(f"[SUCCESS] Imported dataset '{dataset_name}'")
                    typer.echo(f"Master annotations: {result['master_path']}")
                    # Export to framework formats
                    for export_format in ["coco", "yolo"]:
                        try:
                            export_result = pipeline.export_framework_format(
                                dataset_name, export_format
                            )
                            typer.echo(
                                f"[SUCCESS] Exported to {export_format.upper()}: {export_result['output_path']}"
                            )
                        except Exception as e:
                            typer.echo(
                                f"[WARNING] Failed to export to {export_format.upper()}: {e}"
                            )
                else:
                    typer.echo(f"[ERROR] Failed to import dataset '{dataset_name}'")
                    typer.echo(f"Error: {result.get('error', 'Unknown error')}")
                    if result.get("validation_errors"):
                        typer.echo("Validation errors:")
                        for error in result["validation_errors"]:
                            typer.echo(f"  - {error}")
                    if result.get("hints"):
                        typer.echo("Hints:")
                        for hint in result["hints"]:
                            typer.echo(f"  - {hint}")
        # List all datasets
        typer.echo("\n--- Available Datasets ---")
        datasets = pipeline.list_datasets()
        if datasets:
            for dataset in datasets:
                typer.echo(f"Dataset: {dataset['dataset_name']}")
                typer.echo(f"  - Total images: {dataset['total_images']}")
                typer.echo(f"  - Total annotations: {dataset['total_annotations']}")
                typer.echo(f"  - Images by split: {dataset['images_by_split']}")
                typer.echo(f"  - Annotations by type: {dataset['annotations_by_type']}")
        else:
            typer.echo("No datasets found.")
        typer.echo("\n--- Demo Complete ---")
        if keep_temp and temp_dir:
            typer.echo(f"[INFO] Keeping temp directory: {temp_dir}")
        elif temp_dir:
            typer.echo(f"[INFO] Cleaning up temp directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    except Exception as e:
        typer.echo(f"[ERROR] Demo failed: {e}")
        if temp_dir and not keep_temp:
            shutil.rmtree(temp_dir)
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
        storage_type_enum = DVCStorageType.LOCAL
        if storage_type.lower() == "s3":
            storage_type_enum = DVCStorageType.S3
        elif storage_type.lower() == "gcs":
            storage_type_enum = DVCStorageType.GCS
        elif storage_type.lower() == "azure":
            storage_type_enum = DVCStorageType.AZURE
        elif storage_type.lower() == "ssh":
            storage_type_enum = DVCStorageType.SSH
        elif storage_type.lower() == "hdfs":
            storage_type_enum = DVCStorageType.HDFS

        # Set default storage path if not provided
        if not storage_path:
            if storage_type.lower() == "local":
                storage_path = str(Path.cwd() / "dvc_storage")
            else:
                typer.echo(
                    f"❌ Error: Storage path is required for {storage_type} storage"
                )
                raise typer.Exit(1)

        # Create DVC configuration
        config = DVCConfig(
            storage_type=storage_type_enum,
            storage_path=storage_path,
            remote_name=remote_name,
        )

        # Initialize DVC manager
        dvc_manager = DVCManager(Path.cwd(), config)

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
        dvc_manager = DVCManager(Path.cwd())
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
        dvc_manager = DVCManager(Path.cwd())

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
        dvc_manager = DVCManager(Path.cwd())

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
        dvc_manager = DVCManager(Path.cwd())

        if stages_file:
            # Load stages from file
            import yaml

            with open(stages_file, "r") as f:
                stages = yaml.safe_load(f)
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
            for stage in stages:
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
        dvc_manager = DVCManager(Path.cwd())

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
