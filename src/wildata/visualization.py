"""
FiftyOne integration for WildDetect.

This module handles dataset creation, visualization, and annotation collection
using FiftyOne for wildlife detection datasets.
"""

import json
import logging
import os
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Dict, ItemsView, List, Optional, Union

import fiftyone as fo
import numpy as np
import supervision as sv
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

from .datasets.detection import load_detection_dataset
from .datasets.roi import ConcatDataset, ROIDataset

logger = logging.getLogger(__name__)


class FiftyOneManager:
    """Manages FiftyOne datasets for wildlife detection."""

    def __init__(
        self,
        persistent: bool = True,
    ):
        """Initialize FiftyOne manager.

        Args:
            dataset_name: Name of the dataset to use
            config: Optional configuration override
        """
        self.dataset_name = "demo-dataset"
        self.dataset: Optional[fo.Dataset] = None
        self.persistent = persistent

        self.prediction_field = {
            "detection": "det_predictions",
            "classification": "class_predictions",
        }
        self.ground_truth_field = {
            "detection": "det_ground_truth",
            "classification": "class_ground_truth",
        }

    def _init_dataset(self):
        """Initialize or load the FiftyOne dataset."""
        try:
            self.dataset = fo.load_dataset(self.dataset_name)
            logger.info(f"Loaded existing dataset: {self.dataset_name}")
        except ValueError:
            self.dataset = fo.Dataset(self.dataset_name, persistent=self.persistent)
            logger.info(f"Created new dataset: {self.dataset_name}")

    def _ensure_dataset_initialized(self):
        """Ensure dataset is initialized before operations."""
        if self.dataset is None:
            self._init_dataset()

    def _create_classification_sample(
        self,
        image_path: str,
        label: int,
        class_mapping: dict[int, str],
        split: Optional[str] = None,
    ):
        """Create a FiftyOne sample for classification."""
        class_name = class_mapping[label] if label in class_mapping else str(label)
        sample = fo.Sample(
            filepath=str(image_path),
        )

        sample[self.ground_truth_field["classification"]] = fo.Classification(
            label=class_name
        )

        if split is not None:
            sample["split"] = split
        return sample

    def import_classification_data(
        self,
        root_data_directory: str,
        dataset_name: str,
        load_as_single_class: bool = False,
        background_class_name: str = "background",
        single_class_name: str = "wildlife",
        keep_classes: Optional[list[str]] = None,
        discard_classes: Optional[list[str]] = None,
        split: str = "train",
    ):
        """User-facing API: Load and import a classification dataset for visualization."""

        self.dataset_name = f"{dataset_name}-{split}"

        self._ensure_dataset_initialized()

        dataset = ROIDataset(
            dataset_name=dataset_name,
            split=split,
            root_data_directory=root_data_directory,
            load_as_single_class=load_as_single_class,
            background_class_name=background_class_name,
            single_class_name=single_class_name,
            keep_classes=keep_classes,
            discard_classes=discard_classes,
        )
        class_mapping = dataset.class_mapping
        samples = []

        for i in range(len(dataset)):
            image_path = dataset.get_image_path(i)
            label = dataset.get_label(i)
            sample = self._create_classification_sample(
                image_path, label, class_mapping, split
            )
            samples.append(sample)

        print(f"len(samples): {len(samples)}")

        if self.dataset is not None:
            self.dataset.add_samples(samples)
            self.save_dataset()
        else:
            logger.error("Dataset is not initialized")

    def _create_detection_sample(
        self,
        image_path: str,
        image_data: np.ndarray,
        detections: sv.Detections,
        class_mapping: dict[int, str],
        split: str,
    ):
        """Create a FiftyOne sample for detection."""
        sample = fo.Sample(
            filepath=Path(image_path).resolve().as_posix(),
        )
        if image_data is None:
            return sample

        assert image_data.shape[2] == 3, "Image must be RGB and HWC"

        def to_fo_detection(detections: sv.Detections):
            result = []
            for i, box in enumerate(detections.xyxy):
                box = box.copy().astype(np.float32)
                box[[0, 2]] /= image_data.shape[1]
                box[[1, 3]] /= image_data.shape[0]
                box[[2, 3]] -= box[[0, 1]]
                # print(f"box: {box}")
                result.append(
                    fo.Detection(
                        label=class_mapping[int(detections.class_id[i])],
                        bounding_box=box,
                    )
                )
            return result

        if split is not None:
            sample["split"] = split

        if len(detections.xyxy) > 0:
            sample[self.ground_truth_field["detection"]] = fo.Detections(
                detections=to_fo_detection(detections)
            )

        return sample

    def import_detection_data(
        self, root_data_directory: str, dataset_name: str, split: str = "train"
    ):
        """User-facing API: Load and import a detection dataset for visualization."""
        self.dataset_name = f"{dataset_name}-{split}"
        self._ensure_dataset_initialized()

        dataset, class_mapping = load_detection_dataset(
            root_data_directory, dataset_name, split
        )
        samples = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    self._create_detection_sample,
                    file_path,
                    image_data,
                    detections,
                    class_mapping,
                    split,
                )
                for file_path, image_data, detections in dataset
            ]

            for future in tqdm(futures, desc="Importing detection data"):
                sample = future.result()
                samples.append(sample)

        if self.dataset is not None:
            self.dataset.add_samples(samples)
            self.save_dataset()
        else:
            logger.error("Dataset is not initialized")

    def send_to_labelstudio(
        self,
        annot_key: str,
        label_map: dict,
        dotenv_path: Optional[str] = None,
        label_type="detections",
    ):
        """Launch the FiftyOne annotation app."""
        if dotenv_path is not None:
            load_dotenv(dotenv_path, override=True)

        classes = [label_map[i] for i in sorted(label_map.keys())]

        try:
            dataset = fo.load_dataset(self.dataset_name)
            dataset.annotate(
                annot_key,
                backend="labelstudio",
                label_field=self.prediction_field,
                label_type=label_type,
                classes=classes,
                api_key=os.environ["FIFTYONE_LABELSTUDIO_API_KEY"],
                url=os.environ["FIFTYONE_LABELSTUDIO_URL"],
            )
        except Exception:
            logger.error(f"Error exporting to LabelStudio: {traceback.format_exc()}")
            raise

    def save_dataset(self):
        """Save the dataset to disk."""

        if self.dataset is None:
            logger.error("Dataset is not initialized")
            return

        try:
            self.dataset.save()
            logger.info("Dataset saved successfully")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")

    def close(self):
        """Close the dataset."""
        if self.dataset:
            self.dataset.close()
            logger.info("Dataset closed")
