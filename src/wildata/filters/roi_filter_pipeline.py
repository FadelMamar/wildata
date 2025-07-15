"""
Specialized filter pipeline for ROI data that automatically handles conversion to/from COCO format.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .filter_pipeline import FilterPipeline
from .roi_to_coco_converter import ROIToCOCOConverter

logger = logging.getLogger(__name__)


class ROIFilterPipeline:
    """
    Specialized filter pipeline for ROI data.

    This pipeline automatically converts ROI data to COCO-like format for filtering,
    applies the filters, and converts the results back to ROI format.
    """

    def __init__(self, filter_pipeline: FilterPipeline):
        """
        Initialize ROI filter pipeline.

        Args:
            filter_pipeline: Pre-configured FilterPipeline instance
        """
        self.filter_pipeline = filter_pipeline
        self.converter: Optional[ROIToCOCOConverter] = None

    def filter_roi_data(
        self, roi_data: Dict[str, Any], roi_images_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Filter ROI data using the underlying filter pipeline.

        Args:
            roi_data: ROI data dictionary with 'roi_images' and 'roi_labels'
            roi_images_dir: Directory containing ROI images (for getting dimensions)

        Returns:
            Filtered ROI data in the same format
        """
        # Convert ROI data to COCO-like format
        self.converter = ROIToCOCOConverter(roi_data)
        coco_like_data = self.converter.convert_to_coco_like(roi_images_dir)

        logger.info(
            f"Converted {len(roi_data['roi_images'])} ROI images to COCO-like format"
        )

        # Apply filters
        filtered_coco_data = self.filter_pipeline.filter(coco_like_data)

        logger.info(f"Filtered to {len(filtered_coco_data['images'])} images")

        # Convert back to ROI format
        filtered_roi_data = self.converter.convert_filtered_coco_to_roi(
            filtered_coco_data
        )

        return filtered_roi_data

    def filter_roi_files(
        self, roi_labels_file: Path, roi_images_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Filter ROI data from files.

        Args:
            roi_labels_file: Path to roi_labels.json file
            roi_images_dir: Directory containing ROI images

        Returns:
            Filtered ROI data
        """
        # Load ROI data from files
        self.converter = ROIToCOCOConverter.from_roi_files(
            roi_labels_file, roi_images_dir
        )

        # Convert to COCO-like format
        coco_like_data = self.converter.convert_to_coco_like(roi_images_dir)

        logger.info(f"Loaded and converted {len(coco_like_data['images'])} ROI images")

        # Apply filters
        filtered_coco_data = self.filter_pipeline.filter(coco_like_data)

        logger.info(f"Filtered to {len(filtered_coco_data['images'])} images")

        # Convert back to ROI format
        filtered_roi_data = self.converter.convert_filtered_coco_to_roi(
            filtered_coco_data
        )

        return filtered_roi_data

    def save_filtered_roi_data(
        self,
        filtered_roi_data: Dict[str, Any],
        output_labels_dir: Path,
        output_images_dir: Optional[Path] = None,
        copy_images: bool = False,
    ) -> None:
        """
        Save filtered ROI data to files.

        Args:
            filtered_roi_data: Filtered ROI data
            output_labels_dir: Directory to save labels and metadata
            output_images_dir: Directory to save ROI images (if copy_images=True)
            copy_images: Whether to copy ROI images to output directory
        """
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        # Save filtered ROI labels
        import json

        labels_file = output_labels_dir / "filtered_roi_labels.json"
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(filtered_roi_data["roi_labels"], f, indent=2, ensure_ascii=False)

        # Save filtered class mapping
        class_mapping_file = output_labels_dir / "filtered_class_mapping.json"
        with open(class_mapping_file, "w", encoding="utf-8") as f:
            json.dump(
                filtered_roi_data["class_mapping"], f, indent=2, ensure_ascii=False
            )

        # Save filtering statistics
        filter_history = self.filter_pipeline.get_filter_history()
        filter_stats_file = output_labels_dir / "filter_statistics.json"
        with open(filter_stats_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "filter_history": filter_history,
                    "original_count": len(self.converter.roi_images)
                    if self.converter
                    else 0,
                    "filtered_count": len(filtered_roi_data["roi_images"]),
                    "reduction_percentage": (
                        (
                            len(self.converter.roi_images)
                            - len(filtered_roi_data["roi_images"])
                        )
                        / len(self.converter.roi_images)
                        * 100
                    )
                    if self.converter and self.converter.roi_images
                    else 0,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"Saved filtered ROI data to {output_labels_dir}")

        # Copy ROI images if requested
        if copy_images and output_images_dir:
            self._copy_roi_images(filtered_roi_data["roi_images"], output_images_dir)

    def _copy_roi_images(self, roi_images: List[Dict], output_dir: Path) -> None:
        """
        Copy ROI images to output directory.

        Args:
            roi_images: List of ROI image info
            output_dir: Directory to copy images to
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        import shutil

        from tqdm import tqdm

        copied_count = 0
        for roi_img in tqdm(roi_images, desc="Copying ROI images"):
            try:
                # Assuming ROI images are in the same directory as the original images
                # You may need to adjust this path logic based on your setup
                source_path = (
                    Path(roi_img.get("original_image_path", "")).parent
                    / roi_img["roi_filename"]
                )
                if source_path.exists():
                    dest_path = output_dir / roi_img["roi_filename"]
                    shutil.copy2(source_path, dest_path)
                    copied_count += 1
                else:
                    logger.warning(f"ROI image not found: {source_path}")
            except Exception as e:
                logger.warning(f"Failed to copy {roi_img['roi_filename']}: {e}")

        logger.info(
            f"Copied {copied_count}/{len(roi_images)} ROI images to {output_dir}"
        )

    def get_filter_history(self) -> List[Dict[str, Any]]:
        """Get the history of applied filters."""
        return self.filter_pipeline.get_filter_history()
