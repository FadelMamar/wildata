"""
Tiling transformer for extracting tiles/patches from images and annotations.
"""

import logging
import random
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import torch

from ..config import TilingConfig
from .base_transformer import BaseTransformer

logger = logging.getLogger(__name__)


class TileUtils:
    """Utility class for extracting tiles/patches from images."""

    @staticmethod
    def get_patches(
        image: torch.Tensor,
        patch_size: int,
        stride: int,
        channels: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract patches from an image tensor using unfolding.

        Args:
            image (torch.Tensor): Image tensor to extract patches from.
            patch_size (int): Size of each patch (square patches).
            stride (int): Stride between patches.
            channels (Optional[int]): Expected number of channels. If None, uses image channels.

        Returns:
            torch.Tensor: Tensor of image patches with shape (num_patches, channels, patch_size, patch_size).

        Raises:
            ValueError: If image dimensions are invalid or patch_size > image dimensions.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(image)}")

        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")

        if image.dim() == 2:
            image = image.unsqueeze(0)  # Add channel dimension
            squeeze_output = True
        else:
            squeeze_output = False

        C, H, W = image.shape

        # Validate channel count if specified
        if channels is not None and C != channels:
            raise ValueError(f"Expected {channels} channels, got {C}")

        # Check if image is large enough for patches
        if H < patch_size or W < patch_size:
            raise ValueError(
                f"Image size ({H}x{W}) is smaller than patch_size ({patch_size})"
            )

        # Use unfold to create tiles
        # First unfold along height dimension
        unfolded_h = image.unfold(1, patch_size, stride)

        # Then unfold along width dimension
        tiles = unfolded_h.unfold(2, patch_size, stride)

        # Reshape to get individual tiles
        tiles = tiles.contiguous().view(C, -1, patch_size, patch_size)
        tiles = tiles.permute(
            1, 0, 2, 3
        )  # (num_patches, channels, patch_size, patch_size)

        if squeeze_output:
            tiles = tiles.squeeze(1)

        return tiles

    @staticmethod
    def get_patches_and_offset_info(
        image: torch.Tensor,
        patch_size: int,
        stride: int,
        channels: Optional[int] = None,
        file_name: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract patches from an image and compute offset information.

        Args:
            image (torch.Tensor): Image tensor to extract patches from.
            patch_size (int): Size of each patch (square patches).
            stride (int): Stride between patches.
            channels (Optional[int]): Expected number of channels. If None, uses image channels.
            file_name (Optional[str]): File name for the offset info.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]:
                - Tensor of patches with shape (num_patches, channels, patch_size, patch_size)
                - Dictionary with offset information including x_offset, y_offset, x_end, y_end, file_name

        Raises:
            ValueError: If image dimensions are invalid or patch_size > image dimensions.
        """
        if not isinstance(image, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(image)}")

        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size and stride must be positive")

        C, H, W = image.shape

        # Validate channel count if specified
        if channels is not None and C != channels:
            raise ValueError(f"Expected {channels} channels, got {C}")

        # Handle case where image is too small for patches
        if H <= patch_size or W <= patch_size:
            logger.debug(
                f"Image size ({H}x{W}) is too small for patch extraction with size {patch_size}"
            )
            offset_info = {
                "y_offset": [0],
                "x_offset": [0],
                "y_end": [H],
                "x_end": [W],
                "file_name": file_name or "unknown",
            }
            return image.unsqueeze(0), offset_info

        # Extract patches
        tiles = TileUtils.get_patches(image, patch_size, stride, channels)

        # Calculate offset information
        offset_info = TileUtils._calculate_offset_info(
            H, W, patch_size, stride, file_name
        )

        # validate tiling
        TileUtils.validate_results(image, tiles, offset_info)

        return tiles, offset_info

    @staticmethod
    def validate_results(
        image: torch.Tensor, tiles: torch.Tensor, offset_info: Dict[str, Any]
    ) -> None:
        """Validate that tiles match the original image regions using advanced indexing."""
        # Convert offset info to tensors for vectorized operations
        x_offsets = torch.tensor(offset_info["x_offset"])
        y_offsets = torch.tensor(offset_info["y_offset"])
        x_ends = torch.tensor(offset_info["x_end"])
        y_ends = torch.tensor(offset_info["y_end"])

        # Extract all regions at once using advanced indexing
        # This avoids the loop and list comprehension
        extracted_regions = torch.stack(
            [
                image[:, y_offsets[i] : y_ends[i], x_offsets[i] : x_ends[i]]
                for i in range(tiles.shape[0])
            ]
        )

        # Single vectorized comparison for all tiles
        assert torch.allclose(tiles, extracted_regions, atol=1e-6), "error in tiling"

    @staticmethod
    def _calculate_offset_info(
        height: int,
        width: int,
        patch_size: int,
        stride: int,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate offset information for patches.

        Args:
            height (int): Image height.
            width (int): Image width.
            patch_size (int): Size of patches.
            stride (int): Stride between patches.
            file_name (Optional[str]): File name for the offset info.

        Returns:
            Dict[str, Any]: Offset information dictionary.
        """
        # Calculate number of patches in each dimension
        x = torch.arange(0, width).reshape((1, -1))
        y = torch.arange(0, height).reshape((-1, 1))

        ones = torch.ones((3, height, width))

        xx = TileUtils.get_patches(
            ones * x,
            patch_size,
            stride,
        )

        yy = TileUtils.get_patches(ones * y, patch_size, stride)
        x_offset = (
            xx.min(keepdim=True, dim=1)[0]
            .min(keepdim=True, dim=2)[0]
            .min(keepdim=True, dim=3)[0]
            .squeeze()
        ).int()

        y_offset = (
            yy.min(keepdim=True, dim=1)[0]
            .min(keepdim=True, dim=2)[0]
            .min(keepdim=True, dim=3)[0]
            .squeeze()
        ).int()

        return {
            "y_offset": y_offset.tolist(),
            "x_offset": x_offset.tolist(),
            "y_end": (y_offset + patch_size).tolist(),
            "x_end": (x_offset + patch_size).tolist(),
            "file_name": file_name or "unknown",
        }

    @staticmethod
    def validate_patch_parameters(
        image_shape: Tuple[int, ...], patch_size: int, stride: int
    ) -> bool:
        """
        Validate patch extraction parameters.

        Args:
            image_shape (Tuple[int, ...]): Shape of the image tensor.
            patch_size (int): Size of patches.
            stride (int): Stride between patches.

        Returns:
            bool: True if parameters are valid.

        Raises:
            ValueError: If parameters are invalid.
        """
        if len(image_shape) < 2:
            raise ValueError("Image must have at least 2 dimensions")

        if patch_size <= 0:
            raise ValueError("patch_size must be positive")

        if stride <= 0:
            raise ValueError("stride must be positive")

        if stride > patch_size:
            raise ValueError("stride cannot be larger than patch_size")

        return True

    @staticmethod
    def get_patch_count(height: int, width: int, patch_size: int, stride: int) -> int:
        """
        Calculate the number of patches that will be extracted.

        Args:
            height (int): Image height.
            width (int): Image width.
            patch_size (int): Size of patches.
            stride (int): Stride between patches.

        Returns:
            int: Number of patches.
        """
        dummy = torch.ones((3, width, height))
        patch_count = TileUtils.get_patches(dummy, patch_size, stride).shape[0]
        return patch_count


class TilingTransformer(BaseTransformer):
    """
    Transformer for extracting tiles/patches from images and their annotations.

    Uses TileUtils for efficient image tiling and provides annotation tiling functionality.

    Supports:
    - Regular grid tiling with configurable stride
    - Square patches (tile_size x tile_size)
    - Annotation-aware tiling (filter tiles based on annotation content)
    - Efficient PyTorch-based tile extraction
    """

    def __init__(self, config: Optional[TilingConfig] = None):
        """
        Initialize the tiling transformer.

        Args:
            config: TilingConfig dataclass or configuration dictionary
        """
        super().__init__()
        self.config = config or TilingConfig()

    def transform(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self._validate_input(inputs)
        outputs = []
        for data in inputs:
            outputs.extend(self._transform_once(data))

        return outputs

    def _validate_input(self, inputs: List[Dict[str, Any]]) -> None:
        """
        Validate the input of the tiling transformer.
        """
        for input in inputs:
            if "image" not in input:
                raise ValueError("Image not found in input")
            if "info" not in input:
                raise ValueError("Info not found in input")

    def _validate_output(
        self, outputs: List[Dict[str, Any]], annotations: List[Dict[str, Any]]
    ) -> None:
        """
        Validate the output of the tiling transformer.
        """
        num_annotations = 0
        for output in outputs:
            if "image" not in output:
                raise ValueError("Image not found in output")

            if "info" not in output:
                raise ValueError("Info not found in output")

            num_annotations += len(output.get("annotations", []))

        if len(annotations) != 0:
            if num_annotations == 0:
                raise ValueError("No annotations found in output of TilingTransformer")

    def _transform_once(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tiles from the image and annotations.
        """

        image = inputs["image"]
        annotations = inputs.get("annotations", [])
        image_info = inputs["info"]

        # Convert numpy image to torch tensor
        if len(image.shape) == 3:
            # HWC to CHW format
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()

        if image_info.get("file_name") is None:
            logger.warning("Image info does not contain file name, using 'unknown'")
            file_name = "unknown"
        else:
            file_name = image_info["file_name"]

        # Extract tiles using TileUtils
        tiles, offset_info = TileUtils.get_patches_and_offset_info(
            image_tensor, self.config.tile_size, self.config.stride, file_name=file_name
        )

        # Convert tiles back to numpy and process annotations
        empty_tiles = []
        non_empty_tiles = []

        for i in range(tiles.shape[0]):
            # Convert tile back to numpy (CHW to HWC)
            tile_image = tiles[i].permute(1, 2, 0).numpy().astype(np.uint8)

            # Get tile offset information
            x_offset = offset_info["x_offset"][i]
            y_offset = offset_info["y_offset"][i]
            x_end = offset_info["x_end"][i]
            y_end = offset_info["y_end"][i]

            # Extract annotations for this tile
            tile_annotation = self._extract_tile_annotations(
                annotations, x_offset, y_offset, x_end, y_end
            )

            # Create tile info
            tile_info = {
                "tile_id": f"{file_name}_tile_{i}_{x_offset}_{y_offset}",
                "tile_coords": {
                    "x_offset": x_offset,
                    "y_offset": y_offset,
                    "x_end": x_end,
                    "y_end": y_end,
                },
                "tile_size": {
                    "width": tile_image.shape[2],
                    "height": tile_image.shape[1],
                },
                "original_image_info": image_info,
                "tile_index": i,
            }

            # Check if tile meets criteria and categorize
            if tile_annotation:  # Non-empty tile
                non_empty_tiles.append(
                    {
                        "image": tile_image,
                        "annotations": tile_annotation,
                        "info": tile_info,
                    }
                )
            else:  # Empty tile
                empty_tiles.append(
                    {
                        "image": tile_image,
                        "annotations": tile_annotation,
                        "info": tile_info,
                    }
                )

        # Sample tiles based on the empty/non-empty ratio
        selected_tiles = self._sample_tiles_with_ratio(empty_tiles, non_empty_tiles)

        if len(annotations) > 0:
            pass

        self._validate_output(selected_tiles, annotations)

        return selected_tiles

    def _extract_tile_annotations(
        self,
        annotations: List[Dict[str, Any]],
        x_offset: int,
        y_offset: int,
        x_end: int,
        y_end: int,
    ) -> List[Dict[str, Any]]:
        """Extract annotations that fall within the tile bounds."""
        tile_annotations = []

        for annotation in annotations:
            # Handle bounding boxes
            if "bbox" in annotation:
                bbox = annotation["bbox"]
                if self._bbox_intersects_tile(bbox, x_offset, y_offset, x_end, y_end):
                    tile_bbox, ratio = self._clip_bbox_to_tile(
                        bbox, x_offset, y_offset, x_end, y_end
                    )
                    tile_annotation = annotation.copy()
                    tile_annotation["bbox"] = tile_bbox
                    if ratio > self.config.min_visibility:
                        tile_annotations.append(tile_annotation)
            else:
                raise ValueError(f"Annotation type {type(annotation)} not supported")

        return tile_annotations

    def _bbox_intersects_tile(
        self, bbox: List[float], x_offset: int, y_offset: int, x_end: int, y_end: int
    ) -> bool:
        """Check if bounding box intersects with tile."""
        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        bbox_x_end = bbox_x + bbox_w
        bbox_y_end = bbox_y + bbox_h

        # Check intersection
        return (
            bbox_x < x_end
            and bbox_x_end > x_offset
            and bbox_y < y_end
            and bbox_y_end > y_offset
        )

    def _clip_bbox_to_tile(
        self, bbox: List[float], x_offset: int, y_offset: int, x_end: int, y_end: int
    ) -> Tuple[List[float], float]:
        """Clip bounding box coordinates to tile bounds."""
        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        # Clip to tile bounds
        new_x = max(0, bbox_x - x_offset)
        new_y = max(0, bbox_y - y_offset)
        new_w = min(bbox_w, x_end - bbox_x)
        new_h = min(bbox_h, y_end - bbox_y)

        # Calculate ratio of clipped area to original area
        ratio = (new_w * new_h) / (bbox_w * bbox_h)

        # Return clipped bbox and ratio
        return [new_x, new_y, new_w, new_h], ratio

    def _sample_tiles_with_ratio(
        self, empty_tiles: List[Dict[str, Any]], non_empty_tiles: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sample tiles maintaining the specified ratio between empty and non-empty tiles.

        Args:
            empty_tiles: List of (image, annotations, info) tuples for empty tiles
            non_empty_tiles: List of (image, annotations, info) tuples for non-empty tiles

        Returns:
            List of selected tile tuples
        """
        ratio = self.config.negative_positive_ratio

        # Calculate how many tiles of each type to sample
        if non_empty_tiles:
            # If we have non-empty tiles, calculate based on ratio
            max_empty = min(len(empty_tiles), int(len(non_empty_tiles) * ratio))
        else:
            # If no non-empty tiles, just take empty tiles up to max
            max_empty = min(
                len(empty_tiles), self.config.max_negative_tiles_in_negative_image
            )

        # Sample tiles
        selected_tiles = []

        # Add non-empty tiles
        selected_tiles.extend(non_empty_tiles)

        # Add empty tiles
        random.shuffle(empty_tiles)
        selected_tiles.extend(empty_tiles[:max_empty])

        return selected_tiles
