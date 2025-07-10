"""
Base transformer class for data transformations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from pathlib import Path
import numpy as np
from wildtrain.config import AugmentationConfig, TilingConfig


class BaseTransformer(ABC):
    """
    Abstract base class for all data transformations.
    
    This class defines the interface that all transformers must implement.
    Transformers can be applied to both images and their corresponding annotations.
    """
    
    def __init__(self, config: Optional[Union[AugmentationConfig, TilingConfig]] = None):
        """
        Initialize the transformer.
        
        Args:
            config: Configuration dictionary or dataclass for the transformer
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def transform(self, inputs:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform both image and annotations together.
        
        Args:
            image: Input image as numpy array
            annotations: List of annotation dictionaries
            image_info: Metadata about the image
            
        Returns:
            Tuple of (transformed_image, transformed_annotations, updated_image_info)
        """
        pass
    
                
    def get_transformation_info(self) -> Dict[str, Any]:
        """
        Get information about the transformation that was applied.
        
        Returns:
            Dictionary with transformation metadata
        """
        return {
            'transformer_type': self.__class__.__name__,
            'config': self.config
        } 