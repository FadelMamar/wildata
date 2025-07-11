import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_adapter import BaseAdapter

logger = logging.getLogger(__name__)


class YOLOAdapter(BaseAdapter):
    """
    Adapter for converting COCO annotation format to YOLO format.
    """

    def convert(self, split: str) -> Dict[str, List[str]]:
        """
        Convert the loaded COCO annotation to YOLO format for the specified split.
        Args:
            split (str): The data split to convert (e.g., 'train', 'val', 'test').
        Returns:
            Dict[str, List[str]]: Mapping from image file names to lists of YOLO label lines.
        """
        images = self.coco_data.get("images", [])
        image_id_to_image = {img["id"]: img for img in images}
        image_labels: Dict[str, List[str]] = {img["file_name"]: [] for img in images}
        annotations = self.coco_data.get("annotations", [])

        for ann in annotations:
            if ann["image_id"] in image_id_to_image:
                img = image_id_to_image[ann["image_id"]]
                width, height = img["width"], img["height"]
                yolo_line = self._annotation_to_yolo_line(ann, width, height)
                if yolo_line:
                    image_labels[img["file_name"]].append(yolo_line)
        return image_labels

    def save(
        self,
        yolo_data: Dict[str, List[str]],
        output_path: Optional[str] = None,
    ) -> None:
        """
        Save the YOLO-formatted annotation files to the output directory.
        Args:
            yolo_data (Dict[str, List[str]]): Mapping from image file names to YOLO label lines.
            output_path (Optional[str]): Directory to save the YOLO label files.
        """
        if not yolo_data:
            logger.warning("No YOLO data to save")
            return

        # Determine output directory
        if output_path:
            labels_dir = Path(output_path)
        else:
            # Use the first image file path to determine labels directory
            first_image = list(yolo_data.keys())[0]
            labels_dir = Path(first_image).parent
            labels_dir = Path(str(labels_dir).replace("images", "labels"))

        labels_dir.mkdir(exist_ok=True, parents=True)

        for image_file, label_lines in yolo_data.items():
            label_file = Path(image_file).with_suffix(".txt").name
            label_path = labels_dir / label_file
            with open(label_path, "w", encoding="utf-8") as f:
                for line in label_lines:
                    f.write(line + "\n")

        logger.info(f"Saved {len(yolo_data)} label files to {labels_dir}")

    def save_data_yaml(
        self, class_names: List[str], split_image_dirs: Dict[str, str], output_path: str
    ) -> None:
        """
        Save the data.yaml file for Ultralytics YOLO.
        Args:
            class_names (List[str]): List of class names.
            split_image_dirs (Dict[str, str]): Mapping from split names to image directories.
            output_path (str): Path to save the data.yaml file.
        """
        data_yaml = {
            "train": split_image_dirs.get("train", ""),
            "val": split_image_dirs.get("val", ""),
            "test": split_image_dirs.get("test", ""),
            "names": class_names,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            yaml_str = self._dict_to_yaml(data_yaml)
            f.write(yaml_str)

    # --- Private utility methods ---
    def _annotation_to_yolo_line(
        self, ann: Dict[str, Any], width: int, height: int
    ) -> str:
        # YOLO format: class x_center y_center w h (all normalized)
        if "bbox" not in ann or not ann["bbox"]:
            return ""
        x, y, w, h = ann["bbox"]
        x_center = (x + w / 2) / width
        y_center = (y + h / 2) / height
        w_norm = w / width
        h_norm = h / height
        class_id = ann["category_id"]
        # Polygon segmentation (YOLOv8): append after bbox if present
        seg_str = ""
        if "segmentation" in ann and ann["segmentation"]:
            # Only handle polygon (list of points)
            seg = ann["segmentation"]
            if isinstance(seg, list) and seg and isinstance(seg[0], (list, float, int)):
                # Flatten if nested
                if isinstance(seg[0], list):
                    seg_points = [
                        str(float(pt) / width if i % 2 == 0 else float(pt) / height)
                        for poly in seg
                        for i, pt in enumerate(poly)
                    ]
                else:
                    seg_points = [
                        str(float(pt) / width if i % 2 == 0 else float(pt) / height)
                        for i, pt in enumerate(seg)
                    ]
                seg_str = " " + " ".join(seg_points)
        return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}{seg_str}"

    def _dict_to_yaml(self, d: Dict[str, Any], indent: int = 0) -> str:
        # Minimal YAML serializer for simple dicts/lists
        lines = []
        for k, v in d.items():
            if isinstance(v, list):
                lines.append(" " * indent + f"{k}:")
                for item in v:
                    lines.append(" " * (indent + 2) + f"- {item}")
            else:
                lines.append(" " * indent + f"{k}: {v}")
        return "\n".join(lines)
