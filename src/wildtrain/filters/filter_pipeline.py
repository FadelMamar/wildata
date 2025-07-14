"""
Filter pipeline for orchestrating multiple data filtering steps on COCO data.
"""

from typing import Any, Dict, List, Optional

from .algorithms import ClusteringFilter, SizeFilter
from .base import BaseFilter
from .feature_extractor import Dinov2Extractor
from .filter_config import FilterConfig
from .hard_sample_mining import (
    ConfidenceMining,
    HardSampleMiningFilter,
    ROIEmbeddingMining,
    SVMConfidenceMining,
)


class FilterPipeline:
    """
    Pipeline for orchestrating multiple data filtering steps on COCO data.
    """

    def __init__(self, filters: Optional[List[BaseFilter]] = None):
        self.filters = filters or []
        self.filter_history: List[Dict[str, Any]] = []

    def add_filter(self, filter_obj: BaseFilter) -> None:
        self.filters.append(filter_obj)

    def clear_filters(self) -> None:
        self.filters.clear()
        self.filter_history.clear()

    def filter(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        data = coco_data
        self.filter_history.clear()
        for i, filter_obj in enumerate(self.filters):
            data = filter_obj.filter(data)
            self.filter_history.append(filter_obj.get_filter_info())
        return data

    def get_filter_history(self) -> List[Dict[str, Any]]:
        return self.filter_history

    @classmethod
    def from_config(cls, config: FilterConfig) -> "FilterPipeline":
        filters: List[BaseFilter] = []

        # Quality/Size filter
        if getattr(config.quality, "size_filter_enabled", False):
            filters.append(SizeFilter(config.quality))

        # Clustering filter
        if getattr(config.clustering, "enabled", False):
            filters.append(
                ClusteringFilter(
                    config.clustering,
                    feature_extractor=Dinov2Extractor(
                        model_name=config.feature_extractor.model_name,
                        device=config.feature_extractor.device,
                    ),
                )
            )

        # Hard sample mining filter (example, user should extend config as needed)
        if hasattr(config, "hard_sample_mining") and getattr(
            config.hard_sample_mining, "enabled", False
        ):
            miner_type = getattr(config.hard_sample_mining, "miner_type", "confidence")
            miner = None
            if miner_type == "confidence":
                miner = ConfidenceMining()
            elif miner_type == "svm_confidence":
                miner = SVMConfidenceMining(
                    gt_annotations=getattr(
                        config.hard_sample_mining, "gt_annotations", []
                    ),
                    nms_thresholds=getattr(
                        config.hard_sample_mining, "nms_thresholds", None
                    ),
                    margin_band=getattr(config.hard_sample_mining, "margin_band", 0.1),
                )
            elif miner_type == "roi_embedding":
                miner = ROIEmbeddingMining(
                    gt_annotations=getattr(
                        config.hard_sample_mining, "gt_annotations", []
                    ),
                    feature_extractor=Dinov2Extractor(),
                    roi_box_size=getattr(
                        config.hard_sample_mining, "roi_box_size", 128
                    ),
                    min_roi_size=getattr(config.hard_sample_mining, "min_roi_size", 32),
                    nms_thresholds=getattr(
                        config.hard_sample_mining, "nms_thresholds", None
                    ),
                    margin_band=getattr(config.hard_sample_mining, "margin_band", 0.1),
                    batch_size=getattr(config.hard_sample_mining, "batch_size", 32),
                )
            if miner is not None:
                filters.append(
                    HardSampleMiningFilter(
                        miner,
                        top_k=getattr(config.hard_sample_mining, "top_k", None),
                        threshold=getattr(config.hard_sample_mining, "threshold", None),
                        batch_size=getattr(
                            config.hard_sample_mining, "batch_size", 128
                        ),
                    )
                )

        return cls(filters)
