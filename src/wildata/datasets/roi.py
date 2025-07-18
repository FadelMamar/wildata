import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision.io import decode_image

from ..logging_config import get_logger
from ..pipeline.path_manager import PathManager

logger = get_logger("ROI_DATASET")


class ROIDataset(Dataset):
    """
    PyTorch Dataset for loading ROI datasets (images and labels) for a given split.

    Args:
        dataset_name (str): Name of the dataset.
        split (str): One of 'train', 'val', or 'test'.
        path_manager (PathManager): Instance for path resolution.
        transform (callable, optional): Optional transform to be applied on a sample.

    Example:
        >>> ds = ROIDataset(dataset_name="demo-dataset", split="train", root_data_directory="/path/to/data")
        >>> img, label = ds[0]
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        root_data_directory: Path,
        transform: Optional[Callable] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.path_manager = PathManager(root_data_directory)
        self.transform = transform

        # Resolve directories
        self.images_dir = self.path_manager.get_framework_split_image_dir(
            dataset_name, framework="roi", split=split
        )
        self.labels_dir = self.path_manager.get_framework_split_annotations_dir(
            dataset_name, framework="roi", split=split
        )

        # Load class mapping
        class_mapping_path = self.labels_dir / "class_mapping.json"
        with open(class_mapping_path, "r", encoding="utf-8") as f:
            self.class_mapping = json.load(f)
        # Convert keys to int if needed
        self.class_mapping = {int(k): v for k, v in self.class_mapping.items()}

        # Load ROI labels
        roi_labels_path = self.labels_dir / "roi_labels.json"
        with open(roi_labels_path, "r", encoding="utf-8") as f:
            self.roi_labels = json.load(f)

    def __len__(self):
        return len(self.roi_labels)

    def __getitem__(self, idx: int):
        label_info = self.roi_labels[idx]
        img_path = self.images_dir / label_info["file_name"]
        image = decode_image(img_path)
        label = label_info["class_id"]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([label]).int()


def load_all_roi_datasets(
    root_data_directory: Path,
    split: str,
    transform: Optional[dict[str, Callable]] = None,
    concat: bool = False,
) -> dict[str, ROIDataset] | ConcatDataset:
    """
    Load all available ROI datasets for a given split.
    Returns a dict mapping dataset_name to ROIDataset.
    Skips datasets that do not have the requested split.
    The transform argument should be a dict with keys 'train' and 'val'.
    The 'train' transform is used for the train split, and the 'val' transform is used for both val and test splits.
    If concat=True, returns (ConcatDataset, class_mapping) instead, after checking all class_mappings are identical.
    """
    path_manager = PathManager(root_data_directory)
    all_datasets = path_manager.list_datasets()
    roi_datasets = {}
    for dataset_name in all_datasets:
        if not path_manager.framework_format_exists(dataset_name, "roi"):
            continue
        # Check if split exists (by checking for roi_labels.json in split dir)
        split_labels_dir = path_manager.get_framework_split_annotations_dir(
            dataset_name, framework="roi", split=split
        )
        roi_labels_path = split_labels_dir / "roi_labels.json"
        if not roi_labels_path.exists():
            continue
        # Select transform based on split
        split_transform = None
        if transform:
            if split == "train":
                split_transform = transform.get("train")
            else:
                split_transform = transform.get("val")
        try:
            ds = ROIDataset(
                dataset_name=dataset_name,
                split=split,
                root_data_directory=root_data_directory,
                transform=split_transform,
            )
            roi_datasets[dataset_name] = ds
        except Exception as e:
            logger.warning(f"Error loading dataset {dataset_name}: {e}")
            continue

    if concat and len(roi_datasets) > 1:
        if not roi_datasets:
            raise ValueError("No ROI datasets found to concatenate.")
        # Check all class_mappings are identical
        class_mappings = [ds.class_mapping for ds in roi_datasets.values()]
        first_mapping = class_mappings[0]
        for mapping in class_mappings[1:]:
            if mapping != first_mapping:
                raise ValueError(
                    "Class mappings are not identical across datasets. Cannot concatenate."
                )
        concat_dataset = ConcatDataset(list(roi_datasets.values()))
        return concat_dataset

    return roi_datasets


def load_all_splits_concatenated(
    root_data_directory: Path,
    splits: list[str] = ["train", "val", "test"],
    transform: Optional[dict[str, Callable]] = None,
) -> dict[str, ConcatDataset]:
    """
    Load and concatenate all available ROI datasets for each split.
    Returns a dictionary mapping split names to ConcatDataset objects.
    Only includes splits that have at least one dataset.
    Raises ValueError if class mappings are not identical for a split.
    """
    result = {}
    for split in splits:
        roi_datasets = load_all_roi_datasets(
            root_data_directory=root_data_directory,
            split=split,
            transform=transform,
            concat=False,
        )
        if not roi_datasets:
            continue
        # Check class mapping consistency
        class_mappings = [ds.class_mapping for ds in roi_datasets.values()]
        first_mapping = class_mappings[0]
        for mapping in class_mappings[1:]:
            if mapping != first_mapping:
                raise ValueError(
                    f"Class mappings are not identical for split '{split}'. Cannot concatenate."
                )
        concat_dataset = ConcatDataset(list(roi_datasets.values()))
        result[split] = concat_dataset
    return result
