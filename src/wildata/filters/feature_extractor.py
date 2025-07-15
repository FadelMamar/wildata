"""
Feature extraction for image filtering algorithms.

This module provides feature extraction capabilities for clustering
and filtering algorithms in object detection training data selection.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from ..logging_config import get_logger
from .base import FeatureExtractor


class Dinov2Extractor(FeatureExtractor):
    """
    Feature extractor using the DINOv2 model from HuggingFace (facebook/dinov2-with-registers-small).
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-with-registers-small",
        device: str = "auto",
    ):
        """
        Initialize the DINOv2 feature extractor.
        Args:
            model_name: HuggingFace model name (default: 'facebook/dinov2-with-registers-small')
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype="auto", device_map=device
        )
        self.device = self.model.device

    @property
    def feature_dim(self) -> int:
        """
        Return the dimension of the extracted feature vector.
        """
        # To be implemented: return the correct feature dimension
        return 384

    @torch.no_grad()
    def extract_features(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Extract features from a list of images using DINOv2.
        Args:
            image_paths: List of image file paths
        Returns:
            Features as a numpy array
        """
        images = [Image.open(image_path) for image_path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        features = outputs.pooler_output.cpu().reshape(len(images), -1).numpy()
        return features
