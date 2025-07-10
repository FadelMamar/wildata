from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

ROOT = Path(__file__).parents[2]

@dataclass
class AugmentationConfig:
    rotation_range: Tuple[float, float] = (-45, 45)
    probability: float = 0.5
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    noise_std: float = 0.01

    def __post_init__(self):
        self._validate_config()
        self.rotation_range = tuple(float(x) for x in self.rotation_range)
        self.brightness_range = tuple(float(x) for x in self.brightness_range)
        self.contrast_range = tuple(float(x) for x in self.contrast_range)
        self.noise_std = float(self.noise_std)
        self.probability = float(self.probability)

    def _validate_config(self):
        # Allow zero rotation range for deterministic tests
        if self.rotation_range[0] > self.rotation_range[1]:
            raise ValueError("Rotation range start must be less than or equal to end")
        if self.probability < 0 or self.probability > 1:
            raise ValueError("Probability must be between 0 and 1")
        if self.brightness_range[0] > self.brightness_range[1]:
            raise ValueError("Brightness range start must be less than or equal to end")
        if self.contrast_range[0] > self.contrast_range[1]:
            raise ValueError("Contrast range start must be less than or equal to end")
        if self.noise_std < 0:
            raise ValueError("Noise standard deviation must be non-negative")

@dataclass
class TilingConfig:
    tile_size: int = 512
    stride: int = 256
    min_visibility: float = 0.1
    max_negative_tiles_in_negative_image: int = 3
    negative_positive_ratio: float = 1.0

    def __post_init__(self):
        self._validate_config()
        self.tile_size = int(self.tile_size)
        self.stride = int(self.stride)
        self.min_visibility = float(self.min_visibility)
        self.max_negative_tiles_in_negative_image = int(self.max_negative_tiles_in_negative_image)
        self.negative_positive_ratio = float(self.negative_positive_ratio)
        self._filter_empty_tiles = self.negative_positive_ratio > 0

    def _validate_config(self):
        if self.tile_size <= 0:
            raise ValueError("Tile size must be greater than 0")
        if self.stride <= 0 or self.stride > self.tile_size:
            raise ValueError("Stride must be greater than 0 and less than or equal to tile size")
        if self.min_visibility < 0 or self.min_visibility > 1:
            raise ValueError("Minimum visibility must be between 0 and 1")
        if self.max_negative_tiles_in_negative_image <= 0:
            raise ValueError("Maximum negative tiles must be greater than 0")
        if self.negative_positive_ratio < 0:
            raise ValueError("Negative positive ratio must be non-negative")
       

@dataclass
class TransformationConfig:
    augmentation: Optional[AugmentationConfig] = None
    tiling: Optional[TilingConfig] = None
    # Add more transformation configs as needed
