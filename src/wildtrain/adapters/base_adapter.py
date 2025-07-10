from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseAdapter(ABC):
    """
    Abstract base class for annotation format adapters.
    Defines the interface for loading, converting, and saving annotations.
    """
    def __init__(self, master_annotation_path: str):
        """
        Initialize the adapter with the path to the master annotation file.
        """
        self.master_annotation_path = master_annotation_path
        self.master_data: Dict[str, Any] = {}

    @abstractmethod
    def load_master_annotation(self) -> None:
        """
        Load the master annotation JSON file into memory.
        """
        pass

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