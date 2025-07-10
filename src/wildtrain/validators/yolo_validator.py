import os
import yaml
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

class YOLOValidator:
    """
    Validator for YOLO format datasets.
    """
    def __init__(self, yolo_data_yaml_path: str):
        """
        Initialize the validator with the path to the YOLO data.yaml file.
        """
        self.yolo_data_yaml_path = yolo_data_yaml_path
        self.yolo_data: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.base_path: Optional[str] = None

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Perform comprehensive validation of the YOLO dataset.
        Returns:
            Tuple[bool, List[str], List[str]]: (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        # Load and validate data.yaml
        if not self._load_data_yaml():
            return False, self.errors, self.warnings

        # Set base path for relative resolution
        path_value = self.yolo_data.get('path')
        if path_value is None:
            self.errors.append("'path' field is required in data.yaml")
            return False, self.errors, self.warnings
        self.base_path = str(Path(path_value).resolve())
        
        # Validate data.yaml structure
        self._validate_data_yaml_structure()
        
        # Validate directories and files
        self._validate_directories()
        
        # Validate label files
        self._validate_label_files()

        return len(self.errors) == 0, self.errors, self.warnings

    def _load_data_yaml(self) -> bool:
        """Load the data.yaml file and validate basic structure."""
        try:
            if not os.path.exists(self.yolo_data_yaml_path):
                self.errors.append(f"data.yaml file does not exist: {self.yolo_data_yaml_path}")
                return False

            with open(self.yolo_data_yaml_path, 'r', encoding='utf-8') as f:
                self.yolo_data = yaml.safe_load(f)

            if not isinstance(self.yolo_data, dict):
                self.errors.append("data.yaml root element must be a dictionary")
                return False
            
            path_value = self.yolo_data.get('path')
            if path_value is None:
                self.errors.append("'path' field is required in data.yaml")
                return False
            if not os.path.exists(path_value):
                self.errors.append(f"'path' field in data.yaml must be an existing directory: {path_value}")
                return False
            
            names = self.yolo_data.get('names')
            if names is None:
                self.errors.append("'names' field is required in data.yaml")
                return False
            elif len(names) == 0:
                self.errors.append("'names' field in data.yaml must be a non-empty dictionary")
                return False
            
            if self.yolo_data.get('train') is None:
                self.errors.append("'train' field is required in data.yaml")
                return False

            return True

        except yaml.YAMLError as e:
            self.errors.append(f"Invalid YAML in data.yaml: {str(e)}")
            return False
        except Exception as e:
            self.errors.append(f"Error loading data.yaml: {str(e)}")
            return False

    def _resolve_path(self, p: str) -> str:
        return os.path.join(self.base_path, p)

    def _validate_data_yaml_structure(self):
        """Validate the structure of data.yaml."""
        required_fields = ['names', 'path', 'train']
        
        for field in required_fields:
            if field not in self.yolo_data:
                self.errors.append(f"data.yaml missing required field: {field}")
            elif field == 'names' and not isinstance(self.yolo_data[field], dict):
                self.errors.append("data.yaml 'names' field must be a dictionary")
            elif field == 'path' and not isinstance(self.yolo_data[field], str):
                self.errors.append("data.yaml 'path' field must be a string")
            elif field == 'train' and not isinstance(self.yolo_data[field], (str, list)):
                self.errors.append("data.yaml 'train' field must be a string or list of strings")

        # Validate split directories
        splits = ['train', 'val', 'test']
        for split in splits:
            if split in self.yolo_data:
                split_paths = self.yolo_data[split]
                if isinstance(split_paths, list):
                    # Handle list of paths
                    for i, split_path in enumerate(split_paths):
                        resolved = self._resolve_path(split_path)
                        if not isinstance(split_path, str):
                            self.errors.append(f"data.yaml '{split}'[{i}] must be a string")
                        elif split_path and not os.path.exists(resolved):
                            self.warnings.append(f"Split directory does not exist: {resolved}")
                elif isinstance(split_paths, str):
                    # Handle single string path
                    resolved = self._resolve_path(split_paths)
                    if split_paths and not os.path.exists(resolved):
                        self.warnings.append(f"Split directory does not exist: {resolved}")
                else:
                    self.errors.append(f"data.yaml '{split}' field must be a string or list of strings")

        # Validate class names
        if 'names' in self.yolo_data:
            names = self.yolo_data['names']
            if not names:
                self.errors.append("data.yaml 'names' dictionary cannot be empty")
            else:
                for class_id, name in names.items():
                    if not isinstance(name, str):
                        self.errors.append(f"Class name for ID {class_id} must be a string")
                    elif not name.strip():
                        self.errors.append(f"Class name for ID {class_id} cannot be empty")
                    if not isinstance(class_id, int):
                        self.errors.append(f"Class ID {class_id} must be an integer")

    def _validate_directories(self):
        """Validate image and label directories."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            if split not in self.yolo_data:
                continue
                
            split_paths = self.yolo_data[split]
            if isinstance(split_paths, list):
                # Handle list of paths
                for split_path in split_paths:
                    resolved = self._resolve_path(split_path)
                    if not split_path or not os.path.exists(resolved):
                        self.errors.append(f"Split directory does not exist: {resolved}")
                        continue
                    self._validate_single_split_directory(resolved, split)
            elif isinstance(split_paths, str):
                # Handle single string path
                resolved = self._resolve_path(split_paths)
                if not resolved or not os.path.exists(resolved):
                    self.errors.append(f"Split directory does not exist: {resolved}")
                    continue
                self._validate_single_split_directory(resolved, split)

    def _validate_single_split_directory(self, split_path: str, split_name: str):
        """Validate a single split directory."""
        # The split_path is the images directory specified in data.yaml (e.g., 'train/images')
        images_dir = split_path
        if not os.path.exists(images_dir):
            self.errors.append(f"Missing 'images' directory in {split_name} split: {images_dir}")
            return

        # Labels directory should be at the same level as images, replacing 'images' with 'labels'
        labels_dir = images_dir.replace('images', 'labels')
        if not os.path.exists(labels_dir):
            self.errors.append(f"Missing 'labels' directory in {split_name} split: {labels_dir}")
            return

        # Check for image files
        image_files = self._get_image_files(images_dir)
        if not image_files:
            self.warnings.append(f"No image files found in {split_name}/images directory: {images_dir}")
            return

        # Check for corresponding label files
        missing_labels = []
        for img_file in image_files:
            label_file = self._get_label_file_path(img_file, images_dir, labels_dir)
            if not os.path.exists(label_file):
                missing_labels.append(os.path.basename(img_file))

        if missing_labels:
            self.warnings.append(f"Missing label files for {len(missing_labels)} images in {split_name} split")

        # Validate directory structure for each file
        self._validate_file_directory_structure(image_files, labels_dir, split_name)

    def _validate_file_directory_structure(self, image_files: List[str], labels_dir: str, split_name: str):
        """Validate that files are in the correct directories (images/ and labels/)."""
        for img_file in image_files:
            # Check that image file is in an 'images' directory
            img_path_parts = Path(img_file).parts
            if 'images' not in img_path_parts:
                self.errors.append(f"Image file must be in 'images' directory: {img_file}")
            elif img_path_parts.count('images') > 1:
                self.errors.append(f"Image file path contains multiple 'images' directories: {img_file}")
            
            # Check that corresponding label file is in a 'labels' directory
            label_file = self._get_label_file_path(img_file, os.path.dirname(img_file), labels_dir)
            label_path_parts = Path(label_file).parts
            if 'labels' not in label_path_parts:
                self.errors.append(f"Label file must be in 'labels' directory: {label_file}")
            elif label_path_parts.count('labels') > 1:
                self.errors.append(f"Label file path contains multiple 'labels' directories: {label_file}")

    def _validate_label_files(self):
        """Validate YOLO label files."""
        splits = ['train', 'val', 'test']
        
        for split in splits:
            if split not in self.yolo_data:
                continue
                
            split_paths = self.yolo_data[split]
            if isinstance(split_paths, list):
                # Handle list of paths
                for split_path in split_paths:
                    resolved = self._resolve_path(split_path)
                    if split_path and os.path.exists(resolved):
                        self._validate_label_files_in_directory(resolved)
            elif isinstance(split_paths, str):
                # Handle single string path
                resolved = self._resolve_path(split_paths)
                if split_paths and os.path.exists(resolved):
                    self._validate_label_files_in_directory(resolved)

    def _validate_label_files_in_directory(self, split_path: str,):
        """Validate label files in a specific directory."""
        # Get class names for validation
        class_names = self.yolo_data.get('names', {})
        num_classes = len(class_names)

        # Check each label file
        labels_dir = str(split_path).replace('images', 'labels')
        label_files = self._get_label_files(labels_dir)
        for label_file in label_files:
            self._validate_single_label_file(label_file, num_classes)

    def _validate_single_label_file(self, label_file: str, num_classes: int):
        """Validate a single YOLO label file."""
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    self.errors.append(f"{label_file}:{line_num} - Invalid format, need at least 5 values")
                    continue

                # Validate class ID
                try:
                    class_id = int(parts[0])
                    if class_id < 0 or class_id >= num_classes:
                        self.errors.append(f"{label_file}:{line_num} - Invalid class ID: {class_id}")
                except ValueError:
                    self.errors.append(f"{label_file}:{line_num} - Invalid class ID: {parts[0]}")

                # Validate normalized coordinates
                for i in range(1, 5):
                    try:
                        coord = float(parts[i])
                        if coord < 0 or coord > 1:
                            self.errors.append(f"{label_file}:{line_num} - Coordinate {i} out of range [0,1]: {coord}")
                    except ValueError:
                        self.errors.append(f"{label_file}:{line_num} - Invalid coordinate {i}: {parts[i]}")

                # Validate segmentation points if present
                if len(parts) > 5:
                    if len(parts) % 2 != 1:  # Must be odd (class + pairs of coordinates)
                        self.errors.append(f"{label_file}:{line_num} - Invalid number of segmentation points")
                    else:
                        for i in range(5, len(parts), 2):
                            try:
                                x = float(parts[i])
                                y = float(parts[i + 1])
                                if x < 0 or x > 1 or y < 0 or y > 1:
                                    self.errors.append(f"{label_file}:{line_num} - Segmentation point out of range: ({x}, {y})")
                            except (ValueError, IndexError):
                                self.errors.append(f"{label_file}:{line_num} - Invalid segmentation point")

        except FileNotFoundError:
            self.errors.append(f"Label file not found: {label_file}")
        except Exception as e:
            self.errors.append(f"Error reading label file {label_file}: {str(e)}")

    def _get_image_files(self, directory: str) -> List[str]:
        """Get list of image files in the directory."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        try:
            for file in os.listdir(directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(directory, file))
        except OSError:
            self.errors.append(f"Cannot read directory: {directory}")
            
        return sorted(image_files)

    def _get_label_files(self, directory: str) -> List[str]:
        """Get list of label files in the directory."""
        label_files = []
        
        try:
            for file in os.listdir(directory):
                if file.endswith('.txt'):
                    label_files.append(os.path.join(directory, file))
        except OSError:
            self.errors.append(f"Cannot read directory: {directory}")
            
        return sorted(label_files)

    def _get_label_file_path(self, image_file: str, images_dir: str, labels_dir: str) -> str:
        """Get the corresponding label file path for an image, given images_dir and labels_dir."""
        relative_image_path = os.path.relpath(image_file, images_dir)
        label_file = str(Path(relative_image_path).with_suffix('.txt')).replace('images','labels')
        return os.path.join(labels_dir, label_file)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of validation results."""
        # Count files
        total_images = 0
        total_labels = 0
        
        for split in ['train', 'val', 'test']:
            if split in self.yolo_data:
                split_paths = self.yolo_data[split]
                if isinstance(split_paths, list):
                    for split_path in split_paths:
                        resolved = self._resolve_path(split_path)
                        if split_path and os.path.exists(resolved):
                            images_dir = resolved
                            labels_dir = images_dir.replace('images', 'labels')
                            if os.path.exists(images_dir):
                                total_images += len(self._get_image_files(images_dir))
                            if os.path.exists(labels_dir):
                                total_labels += len(self._get_label_files(labels_dir))
                elif isinstance(split_paths, str):
                    resolved = self._resolve_path(split_paths)
                    if split_paths and os.path.exists(resolved):
                        images_dir = resolved
                        labels_dir = images_dir.replace('images', 'labels')
                        if os.path.exists(images_dir):
                            total_images += len(self._get_image_files(images_dir))
                        if os.path.exists(labels_dir):
                            total_labels += len(self._get_label_files(labels_dir))

        return {
            'data_yaml_path': self.yolo_data_yaml_path,
            'is_valid': len(self.errors) == 0,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings),
            'class_count': len(self.yolo_data.get('names', {})),
            'total_images': total_images,
            'total_labels': total_labels,
            'errors': self.errors,
            'warnings': self.warnings
        } 