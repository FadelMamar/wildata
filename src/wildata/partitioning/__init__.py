"""
Data partitioning module for handling train-val-test splits with spatial autocorrelation.

This module provides robust partitioning strategies for aerial imagery data with high
spatial autocorrelation, including camp-based grouping and GPS coordinate analysis.
"""

from .camp_partitioner import CampPartitioner
from .metadata_partitioner import MetadataPartitioner
from .partitioning_pipeline import PartitioningPipeline
from .spatial_partitioner import SpatialPartitioner
from .strategies import (
    CampBasedSplit,
    GroupShuffleSplit,
    MetadataBasedSplit,
    SpatialGroupShuffleSplit,
)

__all__ = [
    "SpatialPartitioner",
    "CampPartitioner",
    "MetadataPartitioner",
    "PartitioningPipeline",
    "GroupShuffleSplit",
    "SpatialGroupShuffleSplit",
    "CampBasedSplit",
    "MetadataBasedSplit",
]
