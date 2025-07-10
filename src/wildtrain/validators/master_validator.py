import json
import os
from typing import Tuple, List, Dict, Any
from jsonschema import validate, ValidationError
from jsonschema.exceptions import SchemaError


class MasterValidator:
    """
    Validator for the master annotation format.
    """
    
    def __init__(self, master_annotation_path: str | None = None, schema_path: str | None = None):
        """
        Initialize the validator.
        Args:
            master_annotation_path (str): Path to the master annotation file to validate.
            schema_path (str): Path to the JSON schema file. If None, uses default schema.
        """
        self.master_annotation_path = master_annotation_path
        self.schema_path = schema_path or os.path.join(os.path.dirname(__file__), '..', '..', '..', 'schema', 'annotation_schema.json')
        self.schema = None
        self.master_data = None
        
    def load_schema(self) -> None:
        """
        Load the JSON schema for validation.
        """
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema: {e}")
    
    def load_master_annotation(self) -> None:
        """
        Load the master annotation file.
        """
        if not self.master_annotation_path:
            raise ValueError("No master annotation path provided")
        
        try:
            with open(self.master_annotation_path, 'r', encoding='utf-8') as f:
                self.master_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Master annotation file not found: {self.master_annotation_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in master annotation file: {e}")
    
    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        Validate the master annotation against the schema.
        Returns:
            Tuple[bool, List[str], List[str]]: (is_valid, errors, warnings)
        """
        errors = []
        warnings = []
        
        try:
            # Load schema if not loaded
            if self.schema is None:
                self.load_schema()
            
            # Load master data if not loaded
            if self.master_data is None and self.master_annotation_path:
                self.load_master_annotation()
            
            # Validate against schema
            if self.master_data:
                validate(instance=self.master_data, schema=self.schema)
                
                # Additional custom validations
                custom_errors, custom_warnings = self._custom_validation()
                errors.extend(custom_errors)
                warnings.extend(custom_warnings)
            
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            if e.path:
                errors.append(f"Path: {' -> '.join(str(p) for p in e.path)}")
        except SchemaError as e:
            errors.append(f"Schema error: {e.message}")
        except Exception as e:
            errors.append(f"Unexpected error during validation: {e}")
        
        return len(errors) == 0, errors, warnings
    
    def validate_data(self, master_data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """
        Validate master data dictionary directly.
        Args:
            master_data (Dict[str, Any]): The master annotation data to validate.
        Returns:
            Tuple[bool, List[str], List[str]]: (is_valid, errors, warnings)
        """
        self.master_data = master_data
        return self.validate()
    
    def _custom_validation(self) -> Tuple[List[str], List[str]]:
        """
        Perform additional custom validations beyond schema validation.
        Returns:
            Tuple[List[str], List[str]]: (errors, warnings)
        """
        errors = []
        warnings = []
        
        if not self.master_data:
            return errors, warnings
        
        # Validate dataset_info
        dataset_info = self.master_data.get('dataset_info', {})
        if not dataset_info:
            errors.append("Missing dataset_info section")
            return errors, warnings
        
        # Validate classes
        classes = dataset_info.get('classes', [])
        if not classes:
            warnings.append("No classes defined in dataset_info")
        else:
            class_ids = set()
            class_names = set()
            for i, cls in enumerate(classes):
                if 'id' not in cls:
                    errors.append(f"Class {i} missing 'id' field")
                elif cls['id'] in class_ids:
                    errors.append(f"Duplicate class ID: {cls['id']}")
                else:
                    class_ids.add(cls['id'])
                
                if 'name' not in cls:
                    errors.append(f"Class {i} missing 'name' field")
                elif cls['name'] in class_names:
                    warnings.append(f"Duplicate class name: {cls['name']}")
                else:
                    class_names.add(cls['name'])
        
        # Validate images
        images = self.master_data.get('images', [])
        if not images:
            warnings.append("No images found in dataset")
        else:
            image_ids = set()
            for i, img in enumerate(images):
                if 'id' not in img:
                    errors.append(f"Image {i} missing 'id' field")
                elif img['id'] in image_ids:
                    errors.append(f"Duplicate image ID: {img['id']}")
                else:
                    image_ids.add(img['id'])
                
                if 'width' in img and img['width'] <= 0:
                    errors.append(f"Image {i} has invalid width: {img['width']}")
                if 'height' in img and img['height'] <= 0:
                    errors.append(f"Image {i} has invalid height: {img['height']}")
                
                split = img.get('split', '')
                if split not in ['train', 'val', 'test']:
                    warnings.append(f"Image {i} has unusual split: {split}")
        
        # Validate annotations
        annotations = self.master_data.get('annotations', [])
        if not annotations:
            warnings.append("No annotations found in dataset")
        else:
            annotation_ids = set()
            valid_image_ids = image_ids
            
            for i, ann in enumerate(annotations):
                if 'id' not in ann:
                    errors.append(f"Annotation {i} missing 'id' field")
                elif ann['id'] in annotation_ids:
                    errors.append(f"Duplicate annotation ID: {ann['id']}")
                else:
                    annotation_ids.add(ann['id'])
                
                if 'image_id' not in ann:
                    errors.append(f"Annotation {i} missing 'image_id' field")
                elif ann['image_id'] not in valid_image_ids:
                    errors.append(f"Annotation {i} references non-existent image_id: {ann['image_id']}")
                
                if 'category_id' not in ann:
                    errors.append(f"Annotation {i} missing 'category_id' field")
                elif ann['category_id'] not in class_ids:
                    errors.append(f"Annotation {i} references non-existent category_id: {ann['category_id']}")
                
                # Validate bbox if present
                bbox = ann.get('bbox', [])
                if bbox:
                    if len(bbox) != 4:
                        errors.append(f"Annotation {i} has invalid bbox length: {len(bbox)}")
                    else:
                        x, y, w, h = bbox
                        if w <= 0 or h <= 0:
                            errors.append(f"Annotation {i} has invalid bbox dimensions: {bbox}")
                
                # Validate area if present
                area = ann.get('area', 0)
                if area < 0:
                    errors.append(f"Annotation {i} has negative area: {area}")
        
        return errors, warnings
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a detailed validation summary.
        Returns:
            Dict[str, Any]: Validation summary with statistics.
        """
        if not self.master_data:
            return {"error": "No master data loaded"}
        
        summary = {
            "dataset_info": {},
            "statistics": {},
            "issues": {}
        }
        
        # Dataset info summary
        dataset_info = self.master_data.get('dataset_info', {})
        summary["dataset_info"] = {
            "name": dataset_info.get('name', 'Unknown'),
            "version": dataset_info.get('version', 'Unknown'),
            "task_type": dataset_info.get('task_type', 'Unknown'),
            "schema_version": dataset_info.get('schema_version', 'Unknown'),
            "num_classes": len(dataset_info.get('classes', []))
        }
        
        # Statistics
        images = self.master_data.get('images', [])
        annotations = self.master_data.get('annotations', [])
        
        split_counts = {}
        for img in images:
            split = img.get('split', 'unknown')
            split_counts[split] = split_counts.get(split, 0) + 1
        
        summary["statistics"] = {
            "total_images": len(images),
            "total_annotations": len(annotations),
            "images_per_split": split_counts,
            "avg_annotations_per_image": len(annotations) / len(images) if images else 0
        }
        
        # Issues summary
        is_valid, errors, warnings = self.validate()
        summary["issues"] = {
            "is_valid": is_valid,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings
        }
        
        return summary 