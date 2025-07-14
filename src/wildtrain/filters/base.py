"""
Base classes for feature extraction in filtering algorithms.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    Defines the interface for extracting features from images for filtering algorithms.
    """

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """
        Return the dimension of the extracted feature vector.
        """
        pass

    @abstractmethod
    def extract_features(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Extract features from a list of images.
        Args:
            image_paths: List of image file paths
        Returns:
            Tuple of (features, valid_paths) where features is a numpy array
            and valid_paths contains paths of successfully processed images
        """
        pass


class BaseFilter(ABC):
    """
    Abstract base class for all filters.
    """

    @abstractmethod
    def filter(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the filter to the input COCO data and return filtered data.
        """
        pass

    def get_filter_info(self) -> Dict[str, Any]:
        """
        Return information about the filter (for logging/debugging).
        """
        return {"filter_type": self.__class__.__name__}
