"""
Example script demonstrating how to apply filtering pipeline to ROI data.

This example shows how to:
1. Convert ROI data to COCO-like format for filtering
2. Apply various filters (clustering, size, etc.)
3. Convert filtered results back to ROI format
4. Save the filtered ROI data
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from wildtrain.adapters.roi_adapter import ROIAdapter
from wildtrain.filters import (
    FilterConfig,
    FilterPipeline,
    ROIFilterPipeline,
    ROIToCOCOConverter
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_coco_data(coco_file: Path) -> Dict[str, Any]:
    """Load COCO annotation data."""
    with open(coco_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_roi_data_from_coco(coco_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Create ROI data from COCO annotations using ROIAdapter.
    
    Args:
        coco_data: COCO annotation data
        output_dir: Directory to save ROI data
        
    Returns:
        ROI data dictionary
    """
    # Initialize ROI adapter
    roi_adapter = ROIAdapter(
        coco_data=coco_data,
        roi_box_size=128,
        min_roi_size=32,
        background_class="background",
        save_format="jpg",
        quality=95,
        dark_threshold=0.5,
    )
    
    # Convert to ROI format
    roi_data = roi_adapter.convert()
    
    # Save ROI data
    labels_dir = output_dir / "labels"
    images_dir = output_dir / "images"
    roi_adapter.save(roi_data, labels_dir, images_dir)
    
    logger.info(f"Created {len(roi_data['roi_images'])} ROI images")
    return roi_data


def create_filter_config() -> FilterConfig:
    """Create a filter configuration for ROI filtering."""
    config = FilterConfig()
    
    # Enable clustering filter
    config.clustering.enabled = True
    config.clustering.x_percent = 0.5  # Keep 50% of data
    config.clustering.n_clusters = 10
    
    # Enable feature extractor
    config.feature_extractor.model_name = "dinov2_vitb14"
    config.feature_extractor.device = "cpu"  # or "cuda" if available
    
    # Enable size filter (optional)
    config.quality.size_filter_enabled = True
    config.quality.min_size = 64  # Minimum size for both width and height
    
    return config


def filter_roi_data_example():
    """Main example function demonstrating ROI filtering."""
    
    # Example paths (adjust these to your actual data)
    coco_file = Path("path/to/your/coco_annotations.json")
    output_dir = Path("output/roi_filtered")
    
    # Step 1: Load COCO data
    logger.info("Loading COCO data...")
    coco_data = load_coco_data(coco_file)
    
    # Step 2: Create ROI data from COCO
    logger.info("Creating ROI data...")
    roi_data = create_roi_data_from_coco(coco_data, output_dir)
    
    # Step 3: Create filter configuration
    logger.info("Setting up filters...")
    filter_config = create_filter_config()
    
    # Step 4: Create filter pipeline
    filter_pipeline = FilterPipeline.from_config(filter_config)
    
    # Step 5: Create ROI filter pipeline
    roi_filter_pipeline = ROIFilterPipeline(filter_pipeline)
    
    # Step 6: Filter ROI data
    logger.info("Applying filters to ROI data...")
    filtered_roi_data = roi_filter_pipeline.filter_roi_data(
        roi_data=roi_data,
        roi_images_dir=output_dir / "images"
    )
    
    # Step 7: Save filtered results
    logger.info("Saving filtered ROI data...")
    filtered_output_dir = output_dir / "filtered"
    roi_filter_pipeline.save_filtered_roi_data(
        filtered_roi_data=filtered_roi_data,
        output_labels_dir=filtered_output_dir / "labels",
        output_images_dir=filtered_output_dir / "images",
        copy_images=True
    )
    
    # Step 8: Print statistics
    original_count = len(roi_data["roi_images"])
    filtered_count = len(filtered_roi_data["roi_images"])
    reduction = (original_count - filtered_count) / original_count * 100
    
    logger.info(f"Filtering complete!")
    logger.info(f"Original ROI count: {original_count}")
    logger.info(f"Filtered ROI count: {filtered_count}")
    logger.info(f"Reduction: {reduction:.1f}%")
    
    # Print filter history
    filter_history = roi_filter_pipeline.get_filter_history()
    logger.info("Applied filters:")
    for i, filter_info in enumerate(filter_history):
        logger.info(f"  {i+1}. {filter_info}")


def filter_existing_roi_files_example():
    """
    Example of filtering existing ROI files without recreating them.
    """
    
    # Example paths (adjust these to your actual data)
    roi_labels_file = Path("output/roi_filtered/labels/roi_labels.json")
    roi_images_dir = Path("output/roi_filtered/images")
    output_dir = Path("output/roi_filtered_v2")
    
    # Step 1: Create filter configuration
    filter_config = create_filter_config()
    filter_pipeline = FilterPipeline.from_config(filter_config)
    roi_filter_pipeline = ROIFilterPipeline(filter_pipeline)
    
    # Step 2: Filter existing ROI files
    logger.info("Filtering existing ROI files...")
    filtered_roi_data = roi_filter_pipeline.filter_roi_files(
        roi_labels_file=roi_labels_file,
        roi_images_dir=roi_images_dir
    )
    
    # Step 3: Save filtered results
    roi_filter_pipeline.save_filtered_roi_data(
        filtered_roi_data=filtered_roi_data,
        output_labels_dir=output_dir / "labels",
        output_images_dir=output_dir / "images",
        copy_images=True
    )
    
    logger.info("Filtering of existing ROI files complete!")


if __name__ == "__main__":
    # Run the main example
    filter_roi_data_example()
    
    # Uncomment to run the existing files example
    # filter_existing_roi_files_example() 