import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseAdapter(ABC):
    """
    Abstract base class for annotation format adapters.
    Defines the interface for loading, converting, and saving annotations.
    """

    def __init__(
        self,
        master_annotation_path: Optional[str] = None,
        master_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter with the path to the master annotation file.
        """
        self.master_annotation_path = master_annotation_path
        self.master_data: Dict[str, Any] = master_data or {}

        assert (
            self.master_annotation_path is not None or self.master_data is not None
        ), "Either master_annotation_path or master_data must be provided"

    def load_master_annotation(self) -> None:
        """
        Load the master annotation JSON file into memory.
        """
        if self.master_annotation_path is not None:
            self.master_data = self._load_json(self.master_annotation_path)
        else:
            self.master_data = self.master_data

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def convert(self, split: str) -> Any:
        """
        Convert the loaded master annotation to the target format for the specified split.
        Args:
            split (str): The data split to convert (e.g., 'train', 'val', 'test').
        Returns:
            Any: The target format annotation data.
        """
        pass

    @abstractmethod
    def save(self, data: Any, output_path: str) -> None:
        """
        Save the converted annotation data to the output path.
        Args:
            data (Any): The converted annotation data.
            output_path (str): Path to save the output file or directory.
        """
        pass
