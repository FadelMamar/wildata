"""
Command-line interface for the WildTrain data pipeline using Typer.
"""

import sys
from pathlib import Path
from typing import Optional
import typer

from .pipeline.data_pipeline import DataPipeline

__version__ = "0.1.0"

app = typer.Typer(
    name="wildtrain",
    help="WildTrain Data Pipeline - Manage datasets in master format and create framework-specific formats",
    add_completion=False,
    rich_markup_mode="rich"
)

def version_callback(value: bool):
    if value:
        typer.echo(f"wildtrain version {__version__}")
        raise typer.Exit()

# Dataset command group

dataset_app = typer.Typer(help="Dataset management commands.")

@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show the application's version and exit.", is_eager=True, callback=version_callback
    )
):
    """WildTrain Data Pipeline CLI."""
    pass


@dataset_app.command("import")
def import_dataset(
    source_path: str = typer.Argument(..., help="Path to the source dataset"),
    format_type: str = typer.Argument(..., help="Format type of the source dataset", case_sensitive=False),
    dataset_name: str = typer.Argument(..., help="Name for the dataset in master storage"),
    no_hints: bool = typer.Option(False, "--no-hints", help="Disable validation hints")
):
    """Import a dataset from COCO or YOLO format."""
    if format_type.lower() not in ['coco', 'yolo']:
        typer.echo(f"❌ Error: Format type must be 'coco' or 'yolo', got '{format_type}'")
        raise typer.Exit(1)
    project_root = Path.cwd()
    pipeline = DataPipeline(str(project_root))
    try:
        typer.echo(f"🚀 Importing {format_type.upper()} dataset from {source_path}")
        typer.echo(f"📝 Dataset name: {dataset_name}")
        typer.echo("─" * 50)
        result = pipeline.import_dataset(
            source_path=source_path,
            format_type=format_type.lower(),
            dataset_name=dataset_name,
            validation_hints=not no_hints
        )
        if result['success']:
            typer.echo("✅ Import successful!")
            typer.echo(f"📄 Master annotations: {result['master_path']}")
            typer.echo("🔧 Framework formats created:")
            for framework, path in result['framework_paths'].items():
                typer.echo(f"  - {framework.upper()}: {path}")
        else:
            typer.echo("❌ Import failed!")
            typer.echo(f"💥 Error: {result.get('error', 'Unknown error')}")
            if 'validation_errors' in result:
                typer.echo("\n🔍 Validation errors:")
                for error in result['validation_errors']:
                    typer.echo(f"  - {error}")
            if 'hints' in result:
                typer.echo("\n💡 Hints:")
                for hint in result['hints']:
                    typer.echo(f"  - {hint}")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@dataset_app.command("list")
def list_datasets():
    """List all available datasets in master storage."""
    project_root = Path.cwd()
    pipeline = DataPipeline(str(project_root))
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
def info(
    dataset_name: str = typer.Argument(..., help="Name of the dataset")
):
    """Get detailed information about a specific dataset."""
    project_root = Path.cwd()
    pipeline = DataPipeline(str(project_root))
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
        framework_formats = pipeline.framework_data_manager.list_framework_formats(dataset_name)
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
    framework: str = typer.Argument(..., help="Target framework format", case_sensitive=False)
):
    """Export a dataset to a specific framework format."""
    if framework.lower() not in ['coco', 'yolo']:
        typer.echo(f"❌ Error: Framework must be 'coco' or 'yolo', got '{framework}'")
        raise typer.Exit(1)
    project_root = Path.cwd()
    pipeline = DataPipeline(str(project_root))
    try:
        result = pipeline.export_framework_format(dataset_name, framework.lower())
        typer.echo(f"✅ Exported dataset '{dataset_name}' to {framework.upper()} format")
        typer.echo(f"📁 Output path: {result['output_path']}")
        if framework.lower() == 'coco':
            typer.echo(f"📂 Data directory: {result['data_dir']}")
            typer.echo(f"📄 Annotations file: {result['annotations_file']}")
        elif framework.lower() == 'yolo':
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
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without confirmation")
):
    """Delete a dataset from master storage."""
    project_root = Path.cwd()
    pipeline = DataPipeline(str(project_root))
    try:
        if not force:
            confirm = typer.confirm(f"Are you sure you want to delete dataset '{dataset_name}'?")
            if not confirm:
                typer.echo("🗑️  Deletion cancelled.")
                return
        success = pipeline.master_data_manager.delete_dataset(dataset_name)
        if success:
            typer.echo(f"✅ Dataset '{dataset_name}' deleted successfully.")
        else:
            typer.echo(f"❌ Failed to delete dataset '{dataset_name}'.")
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
    format_type: str = typer.Argument(..., help="Format type to validate", case_sensitive=False)
):
    """Validate a dataset without importing it."""
    if format_type.lower() not in ['coco', 'yolo']:
        typer.echo(f"❌ Error: Format type must be 'coco' or 'yolo', got '{format_type}'")
        raise typer.Exit(1)
    project_root = Path.cwd()
    pipeline = DataPipeline(str(project_root))
    try:
        typer.echo(f"🔍 Validating {format_type.upper()} dataset at {source_path}")
        typer.echo("─" * 50)
        validation_result = pipeline._validate_dataset(
            Path(source_path), 
            format_type.lower(), 
            True
        )
        if validation_result['is_valid']:
            typer.echo("✅ Validation passed!")
            typer.echo("🎉 Dataset is ready for import.")
        else:
            typer.echo("❌ Validation failed!")
            typer.echo("\n🔍 Validation errors:")
            for error in validation_result['errors']:
                typer.echo(f"  - {error}")
            if 'hints' in validation_result:
                typer.echo("\n💡 Hints:")
                for hint in validation_result['hints']:
                    typer.echo(f"  - {hint}")
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """Show the current status of the pipeline."""
    project_root = Path.cwd()
    pipeline = DataPipeline(str(project_root))
    try:
        typer.echo("📊 WildTrain Pipeline Status")
        typer.echo("─" * 50)
        data_dir = project_root / "data"
        if data_dir.exists():
            typer.echo(f"📁 Data directory: {data_dir} ✅")
        else:
            typer.echo(f"📁 Data directory: {data_dir} ❌ (not found)")
        framework_dir = project_root / "framework_configs"
        if framework_dir.exists():
            typer.echo(f"🔧 Framework configs: {framework_dir} ✅")
        else:
            typer.echo(f"🔧 Framework configs: {framework_dir} ❌ (not found)")
        datasets = pipeline.list_datasets()
        typer.echo(f"📋 Datasets: {len(datasets)} found")
        if datasets:
            typer.echo("\n📁 Available datasets:")
            for dataset in datasets:
                typer.echo(f"  - {dataset['dataset_name']}: {dataset['total_images']} images")
        typer.echo("\n✨ Pipeline is ready!")
    except Exception as e:
        typer.echo(f"❌ Error: {e}")
        raise typer.Exit(1) 