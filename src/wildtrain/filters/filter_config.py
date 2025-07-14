"""
FilterConfig dataclass and loader for filter and feature extractor settings.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class FeatureExtractorConfig:
    model_name: str = "facebook/dinov2-with-registers-small"
    device: str = "auto"


@dataclass
class QualityFilterConfig:
    size_filter_enabled: bool = True
    min_size: int = 10
    max_size_ratio: float = 0.8
    aspect_ratio_filter_enabled: bool = True
    min_ratio: float = 0.1
    max_ratio: float = 10.0
    # Add more fields as needed


@dataclass
class ClusteringFilterConfig:
    enabled: bool = False
    n_clusters: int = 50
    samples_per_cluster: int = 5
    method: str = "kmeans"  # or "agglomerative", etc.
    x_percent: float = 0.3  # Fraction of data to keep after filtering
    # Add more fields as needed


@dataclass
class FilterConfig:
    feature_extractor: FeatureExtractorConfig = field(
        default_factory=FeatureExtractorConfig
    )
    quality: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    clustering: ClusteringFilterConfig = field(default_factory=ClusteringFilterConfig)
    # Add more filter groups as needed

    @classmethod
    def from_yaml(cls, path: str) -> "FilterConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "FilterConfig":
        return cls(
            feature_extractor=FeatureExtractorConfig(
                **data.get("feature_extractor", {})
            ),
            quality=QualityFilterConfig(**data.get("quality", {})),
            clustering=ClusteringFilterConfig(**data.get("clustering", {})),
        )
