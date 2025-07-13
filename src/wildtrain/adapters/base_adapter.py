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
        coco_data: Dict[str, Any],
    ):
        """
        Initialize the adapter with the path to the COCO annotation file.
        """
        self.coco_data: Dict[str, Any] = coco_data

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
