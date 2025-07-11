import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..validators.master_validator import MasterValidator


class BaseConverter(ABC):
    """
    Abstract base class for annotation format adapters.
    Defines the interface for loading, converting, and saving annotations.
    """

    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def convert_to_master(self) -> None:
        pass

    def save_master_annotation(
        self, master_data: Dict[str, Any], output_path: str
    ) -> None:
        """
        Save the master annotation dictionary to a JSON file.
        Args:
            master_data (Dict[str, Any]): The master annotation data.
            output_path (str): Path to save the output JSON file.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(master_data, f, indent=2, ensure_ascii=False)

    def _validate_master_annotation(
        self,
        master_annotation: Dict[str, Any],
        filter_invalid_annotations: bool = False,
    ) -> None:
        """
        Validate the master annotation using MasterValidator.
        Args:
            master_annotation (Dict[str, Any]): The master annotation to validate.
            filter_invalid_annotations (bool): If True, filter out invalid annotations instead of raising errors.
        Raises:
            ValueError: If validation fails.
        """
        validator = MasterValidator(
            filter_invalid_annotations=filter_invalid_annotations
        )
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
                print(
                    f"Warning: Skipped {skipped_count} invalid annotations during master validation"
                )
