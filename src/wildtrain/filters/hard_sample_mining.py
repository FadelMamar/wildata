# Hard sample mining strategies and filters
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchmetrics.functional.detection as torchmetrics
import torchvision.ops as tvops
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from ..adapters.roi_adapter import extract_roi_from_image_bbox


# --- Utility: NMS ---
def nms(preds, nms_thresh):
    # preds: list of dicts with 'bbox' and 'score'
    if not preds:
        return []
    # Convert bboxes to (x1, y1, x2, y2) if in xywh
    boxes = torch.tensor([p["bbox"] for p in preds], dtype=torch.float32)
    if boxes.shape[1] == 4:
        # Assume input is xywh, convert to xyxy
        boxes_xyxy = boxes.clone()
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes_xyxy[:, 0] = boxes[:, 0]
        boxes_xyxy[:, 1] = boxes[:, 1]
    else:
        boxes_xyxy = boxes
    scores = torch.tensor([p["score"] for p in preds], dtype=torch.float32)
    keep_indices = tvops.nms(boxes_xyxy, scores, nms_thresh)
    return [preds[i] for i in keep_indices.tolist()]


# --- Miner Abstract Base ---
class Miner(ABC):
    def __init__(self):
        self.reset()

    @abstractmethod
    def ingest_batch(
        self, images: List[Dict[str, Any]], predictions: List[Dict[str, Any]]
    ):
        pass

    @abstractmethod
    def finalize(self) -> Dict[Any, float]:
        pass

    def reset(self):
        pass


# --- ConfidenceMining ---
class ConfidenceMining(Miner):
    def __init__(self):
        self.reset()

    def reset(self):
        self.image_confidences = {}  # image_id -> list of max confidences

    def ingest_batch(
        self, images: List[Dict[str, Any]], predictions: List[Dict[str, Any]]
    ):
        preds_by_image = {}
        for pred in predictions:
            img_id = pred.get("image_id")
            if img_id is not None:
                preds_by_image.setdefault(img_id, []).append(pred)
        for img in images:
            img_id = img["id"]
            preds = preds_by_image.get(img_id, [])
            max_confs = [pred.get("score", 0.0) for pred in preds]
            if max_confs:
                self.image_confidences.setdefault(img_id, []).extend(max_confs)
            else:
                self.image_confidences.setdefault(img_id, []).append(0.0)

    def finalize(self) -> Dict[Any, float]:
        hardness = {}
        for img_id, confs in self.image_confidences.items():
            max_conf = max(confs) if confs else 0.0
            hardness[img_id] = 1.0 - max_conf
        return hardness


# --- SVMConfidenceMining ---
class SVMConfidenceMining(Miner):
    def __init__(self, gt_annotations, nms_thresholds=None, margin_band=0.1):
        self.gt_annotations = gt_annotations
        self.nms_thresholds = nms_thresholds or [0.35, 0.5, 0.65, 0.75]
        self.margin_band = margin_band
        self.reset()

    def reset(self):
        self.features = []
        self.labels = []
        self.pred_info = []
        self.image_pred_indices = defaultdict(list)

    def ingest_batch(self, images, predictions):
        gt_by_image = defaultdict(list)
        for ann in self.gt_annotations:
            gt_by_image[ann["image_id"]].append(ann["bbox"])
        for nms_thresh in self.nms_thresholds:
            for img in images:
                img_id = img["id"]
                preds = [p for p in predictions if p["image_id"] == img_id]
                if not preds:
                    continue
                kept = nms(preds, nms_thresh)
                gt_boxes = gt_by_image.get(img_id, [])
                if gt_boxes:
                    gt_boxes_tensor = torch.tensor(gt_boxes)
                else:
                    gt_boxes_tensor = torch.empty((0, 4))
                for pred in kept:
                    pred_box_tensor = torch.tensor([pred["bbox"]])
                    if len(gt_boxes) > 0:
                        ious = torchmetrics.intersection_over_union(
                            pred_box_tensor, gt_boxes_tensor
                        )[0]
                        max_iou = float(torch.max(ious))
                    else:
                        max_iou = 0.0
                    if max_iou >= nms_thresh:
                        label = 1
                    else:
                        label = 0
                    self.features.append([pred["score"], nms_thresh])
                    self.labels.append(label)
                    self.pred_info.append((img_id, None))
                    self.image_pred_indices[img_id].append(len(self.pred_info) - 1)

    def finalize(self):
        if not self.features or not self.labels:
            return {}
        scaler = StandardScaler()
        X = scaler.fit_transform(self.features)
        y = self.labels
        svm = LinearSVC(class_weight="balanced", max_iter=1000)
        svm.fit(X, y)
        margins = svm.decision_function(X)
        for i, (img_id, _) in enumerate(self.pred_info):
            self.pred_info[i] = (img_id, margins[i])
        hardness = {}
        for img_id, indices in self.image_pred_indices.items():
            count = sum(abs(self.pred_info[i][1]) < self.margin_band for i in indices)
            hardness[img_id] = count
        return hardness


class ROIEmbeddingMining(Miner):
    def __init__(
        self,
        gt_annotations,
        feature_extractor,
        roi_box_size=128,
        min_roi_size=32,
        nms_thresholds=None,
        margin_band=0.1,
        batch_size=32,
        padding=None,
    ):
        self.gt_annotations = gt_annotations
        self.feature_extractor = feature_extractor
        self.roi_box_size = roi_box_size
        self.min_roi_size = min_roi_size
        self.nms_thresholds = nms_thresholds or [0.35, 0.5, 0.65, 0.75]
        self.margin_band = margin_band
        self.batch_size = batch_size
        self.padding = padding
        self.reset()

    def reset(self):
        self.features = []  # [embedding..., confidence, nms_thresh]
        self.labels = []  # 1=TP, 0=FP
        self.pred_info = []  # (image_id, margin placeholder)
        self.image_pred_indices = defaultdict(list)
        self.roi_crops = []  # (PIL.Image, image_id, confidence, nms_thresh, label)

    def _extract_roi(self, image_path, bbox):
        return extract_roi_from_image_bbox(
            image_path, bbox, self.roi_box_size, self.min_roi_size, self.padding
        )

    def ingest_batch(self, images, predictions):
        gt_by_image = defaultdict(list)
        for ann in self.gt_annotations:
            gt_by_image[ann["image_id"]].append(ann["bbox"])
        for nms_thresh in self.nms_thresholds:
            for img in images:
                img_id = img["id"]
                img_path = img["file_name"]
                preds = [p for p in predictions if p["image_id"] == img_id]
                if not preds:
                    continue
                kept = nms(preds, nms_thresh)
                gt_boxes = gt_by_image.get(img_id, [])
                if gt_boxes:
                    gt_boxes_tensor = torch.tensor(gt_boxes)
                else:
                    gt_boxes_tensor = torch.empty((0, 4))
                for pred in kept:
                    pred_box_tensor = torch.tensor([pred["bbox"]])
                    if len(gt_boxes) > 0:
                        ious = torchmetrics.intersection_over_union(
                            pred_box_tensor, gt_boxes_tensor
                        )[0]
                        max_iou = float(torch.max(ious))
                    else:
                        max_iou = 0.0
                    label = 1 if max_iou >= nms_thresh else 0
                    roi = self._extract_roi(img_path, pred["bbox"])
                    if roi is not None:
                        self.roi_crops.append(
                            (roi, img_id, pred["score"], nms_thresh, label)
                        )
                        self.pred_info.append((img_id, None))
                        self.image_pred_indices[img_id].append(len(self.pred_info) - 1)

    def finalize(self):
        if not self.roi_crops:
            return {}
        # Extract features in batches
        all_embeddings = []
        for i in range(0, len(self.roi_crops), self.batch_size):
            batch_rois = [
                roi for roi, _, _, _, _ in self.roi_crops[i : i + self.batch_size]
            ]
            feats = self.feature_extractor.extract_features(batch_rois)
            all_embeddings.append(feats)
        all_embeddings = np.vstack(all_embeddings)
        # Build feature matrix: [embedding..., confidence, nms_thresh]
        features = []
        labels = []
        for idx, (roi, img_id, conf, nms_thresh, label) in enumerate(self.roi_crops):
            emb = all_embeddings[idx]
            features.append(np.concatenate([emb, [conf, nms_thresh]]))
            labels.append(label)
        scaler = StandardScaler()
        X = scaler.fit_transform(features)
        y = labels
        svm = LinearSVC(class_weight="balanced", max_iter=1000)
        svm.fit(X, y)
        margins = svm.decision_function(X)
        # Fill in margins in pred_info
        for i, (img_id, _) in enumerate(self.pred_info):
            self.pred_info[i] = (img_id, margins[i])
        # For each image, count ROIs with |margin| < margin_band
        hardness = {}
        for img_id, indices in self.image_pred_indices.items():
            count = sum(abs(self.pred_info[i][1]) < self.margin_band for i in indices)
            hardness[img_id] = count
        return hardness


# --- HardSampleMiningFilter ---
from .base import BaseFilter


class HardSampleMiningFilter(BaseFilter):
    def __init__(
        self,
        miner: Miner,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        batch_size: int = 128,
    ):
        self.miner = miner
        self.top_k = top_k
        self.threshold = threshold
        self.batch_size = batch_size

    def filter(self, coco_data: Dict[str, Any]) -> Dict[str, Any]:
        images = coco_data.get("images", [])
        predictions = coco_data.get("predictions", [])
        if not images or not predictions:
            from ..logging_config import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "No images or predictions found in coco_data for hard sample mining."
            )
            return coco_data
        self.miner.reset()
        for i in range(0, len(images), self.batch_size):
            batch_imgs = images[i : i + self.batch_size]
            batch_img_ids = {img["id"] for img in batch_imgs}
            batch_preds = [
                pred for pred in predictions if pred.get("image_id") in batch_img_ids
            ]
            self.miner.ingest_batch(batch_imgs, batch_preds)
        hardness = self.miner.finalize()
        if self.top_k is not None:
            sorted_ids = sorted(hardness, key=hardness.get, reverse=True)[: self.top_k]
            selected_ids = set(sorted_ids)
        elif self.threshold is not None:
            selected_ids = {
                img_id for img_id, score in hardness.items() if score >= self.threshold
            }
        else:
            from ..logging_config import get_logger

            logger = get_logger(__name__)
            logger.warning(
                "No selection rule (top_k or threshold) specified for HardSampleMiningFilter. Returning all data."
            )
            return coco_data
        filtered_images = [img for img in images if img["id"] in selected_ids]
        filtered_annotations = [
            ann
            for ann in coco_data.get("annotations", [])
            if ann["image_id"] in selected_ids
        ]
        filtered_coco = dict(coco_data)
        filtered_coco["images"] = filtered_images
        filtered_coco["annotations"] = filtered_annotations
        return filtered_coco

    def get_filter_info(self) -> Dict[str, Any]:
        return {
            "filter_type": self.__class__.__name__,
            "miner": self.miner.__class__.__name__,
            "top_k": self.top_k,
            "threshold": self.threshold,
            "batch_size": self.batch_size,
        }
