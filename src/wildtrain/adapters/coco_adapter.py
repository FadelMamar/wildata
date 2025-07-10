from .base_adapter import BaseAdapter
import json
from typing import Any, Dict, List

class COCOAdapter(BaseAdapter):
    """
    Adapter for converting the master annotation format to COCO format.
    """
    def load_master_annotation(self) -> None:
        """
        Load the master annotation JSON file into memory.
        """
        self.master_data = self._load_json(self.master_annotation_path)

    def convert(self, split: str) -> Dict[str, Any]:
        """
        Convert the loaded master annotation to COCO format for the specified split.
        Args:
            split (str): The data split to convert (e.g., 'train', 'val', 'test').
        Returns:
            Dict[str, Any]: The COCO-formatted annotation dictionary.
        """
        images = self._filter_images_by_split(split)
        image_ids = {img['id'] for img in images}
        annotations = self._filter_annotations_by_image_ids(image_ids)
        categories = self._map_categories()
        coco_dict = {
            'images': images,
            'annotations': [self._map_annotation_to_coco(ann) for ann in annotations],
            'categories': categories
        }
        return coco_dict

    def save(self, coco_data: Dict[str, Any], output_path: str) -> None:
        """
        Save the COCO-formatted annotation dictionary to a JSON file.
        Args:
            coco_data (Dict[str, Any]): The COCO-formatted annotation data.
            output_path (str): Path to save the output JSON file.
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, indent=2, ensure_ascii=False)

    # --- Private utility methods ---

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _filter_images_by_split(self, split: str) -> List[Dict[str, Any]]:
        return [img for img in self.master_data.get('images', []) if img.get('split') == split]

    def _filter_annotations_by_image_ids(self, image_ids: set) -> List[Dict[str, Any]]:
        return [ann for ann in self.master_data.get('annotations', []) if ann.get('image_id') in image_ids]

    def _map_categories(self) -> List[Dict[str, Any]]:
        return [
            {
                'id': cat['id'],
                'name': cat['name'],
                'supercategory': cat.get('supercategory', '')
            }
            for cat in self.master_data.get('dataset_info', {}).get('classes', [])
        ]

    def _map_annotation_to_coco(self, ann: Dict[str, Any]) -> Dict[str, Any]:
        # COCO annotation fields: id, image_id, category_id, bbox, area, iscrowd, segmentation, keypoints, etc.
        mapped = {
            'id': ann['id'],
            'image_id': ann['image_id'],
            'category_id': ann['category_id'],
            'bbox': ann.get('bbox', []),
            'area': ann.get('area', 0),
            'iscrowd': ann.get('iscrowd', 0),
            'segmentation': ann.get('segmentation', []),
        }
        # Optionally add keypoints if present
        if 'keypoints' in ann and ann['keypoints']:
            mapped['keypoints'] = ann['keypoints']
        # Optionally add attributes if present
        if 'attributes' in ann and ann['attributes']:
            mapped['attributes'] = ann['attributes']
        return mapped 