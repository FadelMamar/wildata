import json
import os
from typing import Any, Dict, List, Tuple

from jsonschema import ValidationError, validate
from jsonschema.exceptions import SchemaError


class MasterValidator:
    """
    Validator for the master annotation format.
    """

    def __init__(
        self,
        master_annotation_path: str | None = None,
        schema_path: str | None = None,
        filter_invalid_annotations: bool = False,
    ):
        """
        Initialize the validator.
        Args:
            master_annotation_path (str): Path to the master annotation file to validate.
            schema_path (str): Path to the JSON schema file. If None, uses default schema.
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        """
        self.master_annotation_path = master_annotation_path
        self.schema_path = schema_path or os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "schema",
            "annotation_schema.json",
        )
        self.schema = None
        self.master_data = None
        self.filter_invalid_annotations = filter_invalid_annotations
        self.skipped_annotations = []

    def load_schema(self) -> None:
        """
        Load the JSON schema for validation.
        """
        try:
            with open(self.schema_path, "r", encoding="utf-8") as f:
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
            with open(self.master_annotation_path, "r", encoding="utf-8") as f:
                self.master_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Master annotation file not found: {self.master_annotation_path}"
            )
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

            # Additional custom validations (including filtering) first
            custom_errors, custom_warnings = self._custom_validation()
            errors.extend(custom_errors)
            warnings.extend(custom_warnings)

            # Validate against schema (only if no critical errors from custom validation)
            if self.master_data and self.schema and len(errors) == 0:
                try:
                    validate(instance=self.master_data, schema=self.schema)
                except ValidationError as e:
                    if self.filter_invalid_annotations:
                        warnings.append(f"Schema validation warning: {e.message}")
                    else:
                        errors.append(f"Schema validation error: {e.message}")
                        if e.path:
                            errors.append(
                                f"Path: {' -> '.join(str(p) for p in e.path)}"
                            )

        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
            if e.path:
                errors.append(f"Path: {' -> '.join(str(p) for p in e.path)}")
        except SchemaError as e:
            errors.append(f"Schema error: {e.message}")
        except Exception as e:
            errors.append(f"Unexpected error during validation: {e}")

        return len(errors) == 0, errors, warnings

    def validate_data(
        self, master_data: Dict[str, Any]
    ) -> Tuple[bool, List[str], List[str]]:
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
        dataset_info = self.master_data.get("dataset_info", {})
        if not dataset_info:
            errors.append("Missing dataset_info section")
            return errors, warnings

        # Validate classes
        classes = dataset_info.get("classes", [])
        if not classes:
            warnings.append("No classes defined in dataset_info")
        else:
            class_ids = set()
            class_names = set()
            for i, cls in enumerate(classes):
                if "id" not in cls:
                    errors.append(f"Class {i} missing 'id' field")
                elif cls["id"] in class_ids:
                    errors.append(f"Duplicate class ID: {cls['id']}")
                else:
                    class_ids.add(cls["id"])

                if "name" not in cls:
                    errors.append(f"Class {i} missing 'name' field")
                elif cls["name"] in class_names:
                    warnings.append(f"Duplicate class name: {cls['name']}")
                else:
                    class_names.add(cls["name"])

        # Validate images
        images = self.master_data.get("images", [])
        if not images:
            warnings.append("No images found in dataset")
        else:
            image_ids = set()
            for i, img in enumerate(images):
                if "id" not in img:
                    errors.append(f"Image {i} missing 'id' field")
                elif img["id"] in image_ids:
                    errors.append(f"Duplicate image ID: {img['id']}")
                else:
                    image_ids.add(img["id"])

                if "width" in img and img["width"] <= 0:
                    errors.append(f"Image {i} has invalid width: {img['width']}")
                if "height" in img and img["height"] <= 0:
                    errors.append(f"Image {i} has invalid height: {img['height']}")

                split = img.get("split", "")
                if split not in ["train", "val", "test"]:
                    warnings.append(f"Image {i} has unusual split: {split}")

        # Validate annotations
        annotations = self.master_data.get("annotations", [])
        if not annotations:
            warnings.append("No annotations found in dataset")
        else:
            annotation_ids = set()
            valid_image_ids = image_ids
            valid_annotations = []
            skipped_count = 0

            for i, ann in enumerate(annotations):
                is_valid = True
                skip_reason = None

                # Check for missing required fields
                if "id" not in ann:
                    if self.filter_invalid_annotations:
                        skip_reason = "missing 'id' field"
                        is_valid = False
                    else:
                        errors.append(f"Annotation {i} missing 'id' field")
                        continue
                elif ann["id"] in annotation_ids:
                    if self.filter_invalid_annotations:
                        skip_reason = f"duplicate annotation ID: {ann['id']}"
                        is_valid = False
                    else:
                        errors.append(f"Duplicate annotation ID: {ann['id']}")
                        continue
                else:
                    annotation_ids.add(ann["id"])

                if "image_id" not in ann:
                    if self.filter_invalid_annotations:
                        skip_reason = "missing 'image_id' field"
                        is_valid = False
                    else:
                        errors.append(f"Annotation {i} missing 'image_id' field")
                        continue
                elif ann["image_id"] not in valid_image_ids:
                    if self.filter_invalid_annotations:
                        skip_reason = (
                            f"references non-existent image_id: {ann['image_id']}"
                        )
                        is_valid = False
                    else:
                        errors.append(
                            f"Annotation {i} references non-existent image_id: {ann['image_id']}"
                        )
                        continue

                if "category_id" not in ann:
                    if self.filter_invalid_annotations:
                        skip_reason = "missing 'category_id' field"
                        is_valid = False
                    else:
                        errors.append(f"Annotation {i} missing 'category_id' field")
                        continue
                elif ann["category_id"] not in class_ids:
                    if self.filter_invalid_annotations:
                        skip_reason = (
                            f"references non-existent category_id: {ann['category_id']}"
                        )
                        is_valid = False
                    else:
                        errors.append(
                            f"Annotation {i} references non-existent category_id: {ann['category_id']}"
                        )
                        continue

                # Validate bbox if present
                bbox = ann.get("bbox", [])
                if bbox:
                    if len(bbox) != 4:
                        if self.filter_invalid_annotations:
                            skip_reason = f"invalid bbox length: {len(bbox)}"
                            is_valid = False
                        else:
                            errors.append(
                                f"Annotation {i} has invalid bbox length: {len(bbox)}"
                            )
                            continue
                    else:
                        x, y, w, h = bbox
                        if w <= 0 or h <= 0:
                            if self.filter_invalid_annotations:
                                skip_reason = f"invalid bbox dimensions: {bbox}"
                                is_valid = False
                            else:
                                errors.append(
                                    f"Annotation {i} has invalid bbox dimensions: {bbox}"
                                )
                                continue

                # Validate area if present
                area = ann.get("area", 0)
                if area < 0:
                    if self.filter_invalid_annotations:
                        skip_reason = f"negative area: {area}"
                        is_valid = False
                    else:
                        errors.append(f"Annotation {i} has negative area: {area}")
                        continue

                # Add to valid annotations or skip
                if is_valid:
                    valid_annotations.append(ann)
                else:
                    skipped_count += 1
                    self.skipped_annotations.append(
                        {"index": i, "annotation": ann, "reason": skip_reason}
                    )

            # Update the master data with filtered annotations
            if self.filter_invalid_annotations and skipped_count > 0:
                self.master_data["annotations"] = valid_annotations
                warnings.append(f"Skipped {skipped_count} invalid annotations")

        return errors, warnings

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a detailed validation summary.
        Returns:
            Dict[str, Any]: Validation summary with statistics.
        """
        if not self.master_data:
            return {"error": "No master data loaded"}

        summary = {"dataset_info": {}, "statistics": {}, "issues": {}}

        # Dataset info summary
        dataset_info = self.master_data.get("dataset_info", {})
        summary["dataset_info"] = {
            "name": dataset_info.get("name", "Unknown"),
            "version": dataset_info.get("version", "Unknown"),
            "task_type": dataset_info.get("task_type", "Unknown"),
            "schema_version": dataset_info.get("schema_version", "Unknown"),
            "num_classes": len(dataset_info.get("classes", [])),
        }

        # Statistics
        images = self.master_data.get("images", [])
        annotations = self.master_data.get("annotations", [])

        split_counts = {}
        for img in images:
            split = img.get("split", "unknown")
            split_counts[split] = split_counts.get(split, 0) + 1

        summary["statistics"] = {
            "total_images": len(images),
            "total_annotations": len(annotations),
            "images_per_split": split_counts,
            "avg_annotations_per_image": len(annotations) / len(images)
            if images
            else 0,
        }

        # Issues summary
        is_valid, errors, warnings = self.validate()
        summary["issues"] = {
            "is_valid": is_valid,
            "error_count": len(errors),
            "warning_count": len(warnings),
            "errors": errors,
            "warnings": warnings,
        }

        return summary

    def get_skipped_annotations(self) -> List[Dict[str, Any]]:
        """
        Get information about skipped annotations when filter_invalid_annotations=True.
        Returns:
            List[Dict[str, Any]]: List of skipped annotation information.
        """
        return self.skipped_annotations

    def get_skipped_count(self) -> int:
        """
        Get the number of skipped annotations.
        Returns:
            int: Number of skipped annotations.
        """
        return len(self.skipped_annotations)
