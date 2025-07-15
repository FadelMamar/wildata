"""
Utility to convert ROI data to COCO-like format for filtering pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ROIToCOCOConverter:
    """
    Converts ROI data to COCO-like format for use with filtering pipeline.

    ROI data structure:
    - roi_images: List of ROI image info with roi_filename, original_image_path, bbox, etc.
    - roi_labels: List of ROI label info with roi_id, class_id, class_name, etc.

    COCO-like format:
    - images: List of image info with id, file_name, width, height
    - annotations: List of annotation info with id, image_id, category_id, bbox
    - categories: List of category info with id, name
    """

    def __init__(self, roi_data: Dict[str, Any]):
        """
        Initialize converter with ROI data.

        Args:
            roi_data: Dictionary containing 'roi_images' and 'roi_labels' lists
        """
        self.roi_data = roi_data
        self.roi_images = roi_data.get("roi_images", [])
        self.roi_labels = roi_data.get("roi_labels", [])
        self.class_mapping = roi_data.get("class_mapping", {})

    def convert_to_coco_like(
        self, roi_images_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Convert ROI data to COCO-like format.

        Args:
            roi_images_dir: Directory containing ROI images (for getting actual dimensions)

        Returns:
            Dictionary in COCO-like format with images, annotations, and categories
        """
        # Create categories from class mapping
        categories = []
        for class_id, class_name in self.class_mapping.items():
            categories.append(
                {"id": class_id, "name": class_name, "supercategory": "object"}
            )

        # Create images list
        images = []
        for roi_img in self.roi_images:
            # Use ROI dimensions or get from actual image if available
            width = roi_img.get("width", 128)  # Default ROI size
            height = roi_img.get("height", 128)

            # If roi_images_dir is provided, try to get actual dimensions
            if roi_images_dir:
                roi_path = roi_images_dir / roi_img["roi_filename"]
                if roi_path.exists():
                    try:
                        from PIL import Image

                        with Image.open(roi_path) as img:
                            width, height = img.size
                    except Exception as e:
                        logger.warning(f"Could not get dimensions for {roi_path}: {e}")

            image_info = {
                "id": roi_img["roi_id"],
                "file_name": roi_img["roi_filename"],
                "width": width,
                "height": height,
                "original_image_path": roi_img.get("original_image_path", ""),
                "original_image_id": roi_img.get("original_image_id", ""),
            }
            images.append(image_info)

        # Create annotations list
        annotations = []
        for roi_label in self.roi_labels:
            # Find corresponding ROI image
            roi_img = next(
                (
                    img
                    for img in self.roi_images
                    if img["roi_id"] == roi_label["roi_id"]
                ),
                None,
            )
            if not roi_img:
                continue

            # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height] format
            bbox = roi_img.get("bbox", [0, 0, 128, 128])
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                coco_bbox = [x1, y1, x2 - x1, y2 - y1]
            else:
                coco_bbox = [0, 0, 128, 128]  # Default

            annotation_info = {
                "id": roi_label["roi_id"],
                "image_id": roi_label["roi_id"],
                "category_id": roi_label.get("class_id", 0),
                "bbox": coco_bbox,
                "area": coco_bbox[2] * coco_bbox[3],
                "iscrowd": 0,
                "original_annotation_id": roi_label.get("original_annotation_id", ""),
            }
            annotations.append(annotation_info)

        return {
            "images": images,
            "annotations": annotations,
            "categories": categories,
            "info": {
                "description": "ROI data converted to COCO-like format for filtering",
                "version": "1.0",
                "year": 2024,
                "contributor": "WildTrain ROI Converter",
                "url": "",
                "date_created": "",
            },
        }

    def convert_filtered_coco_to_roi(
        self, filtered_coco: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert filtered COCO-like data back to ROI format.

        Args:
            filtered_coco: COCO-like data that has been filtered

        Returns:
            ROI data with filtered roi_images and roi_labels
        """
        filtered_image_ids = {img["id"] for img in filtered_coco.get("images", [])}

        # Filter ROI images
        filtered_roi_images = [
            roi_img
            for roi_img in self.roi_images
            if roi_img["roi_id"] in filtered_image_ids
        ]

        # Filter ROI labels
        filtered_roi_labels = [
            roi_label
            for roi_label in self.roi_labels
            if roi_label["roi_id"] in filtered_image_ids
        ]

        # Update class mapping to only include used categories
        used_category_ids = {cat["id"] for cat in filtered_coco.get("categories", [])}
        filtered_class_mapping = {
            class_id: class_name
            for class_id, class_name in self.class_mapping.items()
            if class_id in used_category_ids
        }

        return {
            "roi_images": filtered_roi_images,
            "roi_labels": filtered_roi_labels,
            "class_mapping": filtered_class_mapping,
            "statistics": self.roi_data.get("statistics", {}),
        }

    @classmethod
    def from_roi_files(
        cls, roi_labels_file: Path, roi_images_dir: Optional[Path] = None
    ) -> "ROIToCOCOConverter":
        """
        Create converter from ROI files.

        Args:
            roi_labels_file: Path to roi_labels.json file
            roi_images_dir: Directory containing ROI images

        Returns:
            ROIToCOCOConverter instance
        """
        with open(roi_labels_file, "r", encoding="utf-8") as f:
            roi_labels = json.load(f)

        # Load class mapping if available
        class_mapping_file = roi_labels_file.parent / "class_mapping.json"
        class_mapping = {}
        if class_mapping_file.exists():
            with open(class_mapping_file, "r", encoding="utf-8") as f:
                class_mapping = json.load(f)

        # Load statistics if available
        statistics_file = roi_labels_file.parent / "statistics.json"
        statistics = {}
        if statistics_file.exists():
            with open(statistics_file, "r", encoding="utf-8") as f:
                statistics = json.load(f)

        # Reconstruct ROI data structure
        roi_data = {
            "roi_images": roi_labels,  # Assuming roi_labels.json contains the full ROI data
            "roi_labels": roi_labels,
            "class_mapping": class_mapping,
            "statistics": statistics,
        }

        return cls(roi_data)
