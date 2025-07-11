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
        coco_annotation_path: Optional[str] = None,
        coco_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the adapter with the path to the COCO annotation file.
        """
        self.coco_annotation_path = coco_annotation_path
        self.coco_data: Dict[str, Any] = coco_data or {}

        assert (
            self.coco_annotation_path is not None or self.coco_data is not None
        ), "Either coco_annotation_path or coco_data must be provided"

    def load_coco_annotation(self) -> None:
        """
        Load the COCO annotation JSON file into memory.
        """
        if self.coco_annotation_path is not None:
            self.coco_data = self._load_json(self.coco_annotation_path)
        else:
            self.coco_data = self.coco_data

    def _load_json(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    def convert(self, split: str) -> Any:
        """
        Convert the loaded COCO annotation to the target format for the specified split.
        Args:
            split (str): The data split to convert (e.g., 'train', 'val', 'test').
        Returns:
            Any: The target format annotation data.
        """
        pass

    @abstractmethod
    def save(self, data: Any, output_path: Optional[str] = None) -> None:
        """
        Save the converted annotation data to the output path.
        Args:
            data (Any): The converted annotation data.
            output_path (Optional[str]): Path to save the output file or directory.
        """
        pass
