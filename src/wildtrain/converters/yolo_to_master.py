import json
import os
from pathlib import Path
import yaml
from typing import Any, Dict, List, Tuple
from ..validators.yolo_validator import YOLOValidator
from ..validators.master_validator import MasterValidator
from PIL import Image

class YOLOToMasterConverter:
    """
    Converter from YOLO format to master annotation format.
    """
    def __init__(self, yolo_data_yaml_path: str):
        """
        Initialize the converter with the path to the YOLO data.yaml file.
        """
        self.yolo_data_yaml_path = yolo_data_yaml_path
        self.yolo_data: Dict[str, Any] = {}
        self.base_path = None

    def load_yolo_data(self, filter_invalid_annotations: bool = False) -> None:
        """
        Load the YOLO data.yaml file and parse the dataset structure.
        Args:
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        """
        # Validate before loading
        from ..validators.yolo_validator import YOLOValidator
        validator = YOLOValidator(self.yolo_data_yaml_path, filter_invalid_annotations=filter_invalid_annotations)
        is_valid, errors, warnings = validator.validate()
        
        if not is_valid:
            error_msg = f"YOLO validation failed for {self.yolo_data_yaml_path}:\n"
            error_msg += "\n".join(errors)
            if warnings:
                error_msg += f"\nWarnings:\n" + "\n".join(warnings)
            raise ValueError(error_msg)
        
        # Load the data
        with open(self.yolo_data_yaml_path, 'r', encoding='utf-8') as f:
            self.yolo_data = yaml.safe_load(f)
        self.base_path = self.yolo_data.get('path')
        if self.base_path and not os.path.isabs(self.base_path):
            self.base_path = os.path.abspath(os.path.join(os.path.dirname(self.yolo_data_yaml_path), self.base_path))
        if filter_invalid_annotations:
            skipped_count = validator.get_skipped_count()
            if skipped_count > 0:
                print(f"Warning: Skipped {skipped_count} invalid YOLO annotation lines during validation")

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        if self.base_path:
            return os.path.abspath(os.path.join(self.base_path, p))
        return os.path.abspath(os.path.join(os.path.dirname(self.yolo_data_yaml_path), p))

    def convert_to_master(self, dataset_name: str, version: str = "1.0", task_type: str = "detection", validate_output: bool = True, filter_invalid_annotations: bool = False) -> Dict[str, Any]:
        """
        Convert the loaded YOLO data to master format.
        Args:
            dataset_name (str): Name of the dataset.
            version (str): Version of the dataset.
            task_type (str): Type of task (detection, segmentation).
            validate_output (bool): Whether to validate the output master annotation.
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        Returns:
            Dict[str, Any]: The master annotation dictionary.
        """
        # Extract class names
        class_names_dict = self.yolo_data.get('names', {})
        classes = [
            {
                'id': int(class_id),
                'name': name,
                'supercategory': ''  # YOLO doesn't have supercategories
            }
            for class_id, name in class_names_dict.items()
        ]

        # Process each split
        all_images = []
        all_annotations = []
        annotation_id = 1

        for split in ['train', 'val', 'test']:
            split_paths = self.yolo_data.get(split, [])
            if isinstance(split_paths, list):
                # Handle list of paths
                for split_path in split_paths:
                    resolved = self._resolve_path(split_path)
                    if not split_path or not os.path.exists(resolved):
                        continue
                    self._process_split_directory(resolved, split, all_images, all_annotations, annotation_id)
                    annotation_id = len(all_annotations) + 1
            elif isinstance(split_paths, str):
                resolved = self._resolve_path(split_paths)
                if resolved and os.path.exists(resolved):
                    self._process_split_directory(resolved, split, all_images, all_annotations, annotation_id)
                    annotation_id = len(all_annotations) + 1

        # Create master annotation structure
        master_annotation = {
            'dataset_info': {
                'name': dataset_name,
                'version': version,
                'schema_version': '1.0',
                'task_type': task_type,
                'classes': classes
            },
            'images': all_images,
            'annotations': all_annotations
        }

        # Validate the output if requested
        if validate_output:
            self._validate_master_annotation(master_annotation, filter_invalid_annotations)

        return master_annotation

    def _process_split_directory(self, split_path: str, split_name: str, all_images: List, all_annotations: List, annotation_id: int):
        """Process a single split directory."""
        # Get image files
        image_files = self._get_image_files(split_path)
        
        # Process each image
        for img_idx, img_file in enumerate(image_files):
            img_id = len(all_images) + 1
            
            # Get image dimensions (you might need to actually load the image)
            width, height = self._get_image_dimensions(img_file)
            
            # Create master image entry
            master_image = {
                'id': img_id,
                'file_name': os.path.basename(img_file),
                'width': width,
                'height': height,
                'split': split_name,
                'path': img_file
            }
            all_images.append(master_image)

            # Process corresponding label file
            label_file = self._get_label_file_path(img_file)
            if os.path.exists(label_file):
                annotations = self._parse_yolo_label_file(label_file, img_id, width, height)
                for ann in annotations:
                    ann['id'] = annotation_id
                    annotation_id += 1
                    all_annotations.append(ann)

    def save_master_annotation(self, master_data: Dict[str, Any], output_path: str) -> None:
        """
        Save the master annotation dictionary to a JSON file.
        Args:
            master_data (Dict[str, Any]): The master annotation data.
            output_path (str): Path to save the output JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)

    def _get_image_files(self, images_dir: str) -> List[str]:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for file in os.listdir(images_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(images_dir, file))
        return sorted(image_files)

    def _get_label_file_path(self, image_file: str) -> str:
        path = Path(image_file).with_suffix('.txt')
        return str(path).replace('images', 'labels')

    def _get_image_dimensions(self, image_file: str) -> Tuple[int, int]:
        with Image.open(image_file) as image:
            width, height = image.size
            return width, height

    def _parse_yolo_label_file(self, label_file: str, image_id: int, width: int, height: int) -> List[Dict[str, Any]]:
        annotations = []
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    # Try to parse class_id, skip if invalid
                    try:
                        class_id = int(parts[0])
                    except ValueError:
                        continue  # Skip invalid class_id
                    
                    # Try to parse coordinates, skip if invalid
                    try:
                        x_center_norm = float(parts[1])
                        y_center_norm = float(parts[2])
                        w_norm = float(parts[3])
                        h_norm = float(parts[4])
                        
                        # Validate coordinate ranges
                        if not (0 <= x_center_norm <= 1 and 0 <= y_center_norm <= 1 and 
                                0 <= w_norm <= 1 and 0 <= h_norm <= 1):
                            continue  # Skip invalid coordinates
                    except ValueError:
                        continue  # Skip invalid coordinates
                    
                    x_center = x_center_norm * width
                    y_center = y_center_norm * height
                    w = w_norm * width
                    h = h_norm * height
                    x = x_center - w / 2
                    y = y_center - h / 2
                    area = w * h
                    segmentation = []
                    
                    # Parse segmentation points if present
                    if len(parts) > 5:
                        try:
                            seg_points = []
                            for i in range(5, len(parts), 2):
                                if i + 1 < len(parts):
                                    x_norm = float(parts[i])
                                    y_norm = float(parts[i + 1])
                                    # Validate segmentation coordinate ranges
                                    if not (0 <= x_norm <= 1 and 0 <= y_norm <= 1):
                                        continue  # Skip invalid segmentation points
                                    x_abs = x_norm * width
                                    y_abs = y_norm * height
                                    seg_points.extend([x_abs, y_abs])
                            if seg_points:
                                segmentation = [seg_points]
                        except (ValueError, IndexError):
                            # Skip invalid segmentation points
                            pass
                    
                    annotation = {
                        'image_id': image_id,
                        'category_id': class_id,
                        'bbox': [x, y, w, h],
                        'area': area,
                        'iscrowd': 0,
                        'segmentation': segmentation,
                        'keypoints': [],
                        'attributes': {}
                    }
                    annotations.append(annotation)
        except FileNotFoundError:
            pass  # Label file doesn't exist
        return annotations

    def _validate_master_annotation(self, master_annotation: Dict[str, Any], filter_invalid_annotations: bool = False) -> None:
        """
        Validate the master annotation using MasterValidator.
        Args:
            master_annotation (Dict[str, Any]): The master annotation to validate.
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        Raises:
            ValueError: If validation fails.
        """
        validator = MasterValidator(filter_invalid_annotations=filter_invalid_annotations)
        is_valid, errors, warnings = validator.validate_data(master_annotation)
        
        if not is_valid:
            error_msg = "Master annotation validation failed:\n"
            error_msg += "\n".join(errors)
            if warnings:
                error_msg += f"\nWarnings:\n" + "\n".join(warnings)
            raise ValueError(error_msg)
        
        if warnings:
            print(f"Master annotation validation warnings:\n" + "\n".join(warnings))
        
        # Report skipped annotations if any
        if filter_invalid_annotations:
            skipped_count = validator.get_skipped_count()
            if skipped_count > 0:
                print(f"Warning: Skipped {skipped_count} invalid annotations during master validation") 