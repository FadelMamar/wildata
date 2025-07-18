import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image

from ..pipeline.path_manager import PathManager


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
