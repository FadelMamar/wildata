"""
ROI-related logic for CLI commands.
"""

import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

from ..config import ENV_FILE, ROIConfig
from ..pipeline import FrameworkDataManager, Loader, PathManager
from .models import ROIDatasetConfig


def create_roi_dataset_core(config: ROIDatasetConfig, verbose: bool = False) -> bool:
    """Core logic for creating ROI datasets."""
    try:
        if verbose:
            typer.echo(f"🔧 Creating dataset...")
            typer.echo(f"   Source: {config.source_path}")
            typer.echo(f"   Format: {config.source_format}")
            typer.echo(f"   Name: {config.dataset_name}")
            typer.echo(f"   Split: {config.split_name}")
            typer.echo(f"   Output root: {config.root}")
            typer.echo(f"   ROI config: {config.roi_config}")
            typer.echo(f"   LS XML config: {config.ls_xml_config}")
            typer.echo(f"   LS parse config: {config.ls_parse_config}")

        loader = Loader()
        dataset_info, split_coco_data = loader.load(
            config.source_path,
            config.source_format,
            config.dataset_name,
            config.bbox_tolerance,
            config.split_name,
            dotenv_path=ENV_FILE,
            ls_xml_config=config.ls_xml_config,
            ls_parse_config=config.ls_parse_config,
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
        typer.echo(
            f"✅ Successfully created ROI dataset for '{config.dataset_name}' (split: {config.split_name}) at {config.root}"
        )
        return True
    except ValidationError as e:
        typer.echo(f"❌ Configuration validation error:")
        for error in e.errors():
            typer.echo(f"   {error['loc'][0]}: {error['msg']}")
        return False
    except Exception as e:
        typer.echo(f"❌ Failed to create ROI dataset: {str(e)}")
        if verbose:
            typer.echo(f"   Traceback: {traceback.format_exc()}")
        return False


def create_roi_one_worker(args) -> tuple:
    """Top-level worker for ProcessPoolExecutor in bulk_create_roi_datasets."""
    i, src, name, config_dict, verbose = args
    import traceback

    import typer
    from pydantic import ValidationError

    try:
        typer.echo(f"\n=== Creating ROI [{i+1}]: {name} ===")
        from pathlib import Path

        from .models import ROIDatasetConfig

        single_config = ROIDatasetConfig(
            source_path=src,
            source_format=config_dict["source_format"],
            dataset_name=name,
            root=config_dict["root"],
            split_name=config_dict["split_name"],
            bbox_tolerance=config_dict["bbox_tolerance"],
            roi_config=config_dict["roi_config"],
            ls_xml_config=config_dict["ls_xml_config"],
            ls_parse_config=config_dict["ls_parse_config"],
        )
        # Use the same core logic as create_roi_dataset
        from ..pipeline import FrameworkDataManager, Loader, PathManager

        loader = Loader()
        dataset_info, split_coco_data = loader.load(
            single_config.source_path,
            single_config.source_format,
            single_config.dataset_name,
            single_config.bbox_tolerance,
            single_config.split_name,
            dotenv_path=ENV_FILE,
            ls_xml_config=single_config.ls_xml_config,
            ls_parse_config=single_config.ls_parse_config,
        )
        path_manager = PathManager(Path(single_config.root))
        framework_data_manager = FrameworkDataManager(path_manager)
        roi_config = ROIConfig(
            random_roi_count=single_config.roi_config.random_roi_count,
            roi_box_size=single_config.roi_config.roi_box_size,
            min_roi_size=single_config.roi_config.min_roi_size,
            dark_threshold=single_config.roi_config.dark_threshold,
            roi_callback=None,
            background_class=single_config.roi_config.background_class,
            save_format=single_config.roi_config.save_format,
            quality=single_config.roi_config.quality,
        )
        framework_data_manager.create_roi_format(
            dataset_name=single_config.dataset_name,
            coco_data=split_coco_data[single_config.split_name],
            split=single_config.split_name,
            roi_config=roi_config,
        )
        return (i, name, True, None)
    except ValidationError as e:
        msg = f"❌ Configuration validation error for '{name}':\n" + "\n".join(
            f"   {error['loc'][0]}: {error['msg']}" for error in e.errors()
        )
        return (i, name, False, msg)
    except Exception as e:
        msg = f"❌ Unexpected error for '{name}': {str(e)}"
        if verbose:
            msg += f"\n   Traceback: {traceback.format_exc()}"
        return (i, name, False, msg)
