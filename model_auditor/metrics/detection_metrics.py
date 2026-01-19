"""
Detection metrics for object detection model auditing.

These metrics evaluate bounding box predictions against ground truth annotations
using Intersection over Union (IoU) and standard COCO-style metrics.
"""

from typing import List, Dict, Any, Tuple, Union
import torch
import numpy as np
from ..metric import Metric

# =============================================================================
# Helper Functions
# =============================================================================

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    
    Args:
        box1: (N, 4) tensor of boxes [x1, y1, x2, y2]
        box2: (M, 4) tensor of boxes [x1, y1, x2, y2]
        
    Returns:
        (N, M) tensor containing IoU values
    """
    # Calculate intersection areas
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]
    
    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    
    union = area1[:, None] + area2 - inter
    
    return inter / union

def match_predictions_to_gt(
    pred_boxes: torch.Tensor, 
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor, 
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Match predictions to ground truth boxes.
    
    Args:
        pred_boxes: (N, 4) predicted boxes
        pred_scores: (N,) predicted scores
        gt_boxes: (M, 4) ground truth boxes
        iou_threshold: IoU threshold for matching
        
    Returns:
        tp: (N,) boolean array, True if prediction is a true positive
        fp: (N,) boolean array, True if prediction is a false positive
        n_gt: Number of ground truth boxes
    """
    n_pred = pred_boxes.shape[0]
    n_gt = gt_boxes.shape[0]
    
    # Sort predictions by score
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    
    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)
    
    if n_gt == 0:
        fp[:] = 1
        return tp, fp, n_gt
        
    if n_pred == 0:
        return tp, fp, n_gt

    # Compute IoU matrix
    ious = compute_iou(pred_boxes, gt_boxes)  # [N, M]
    
    # Keep track of matched GT boxes
    gt_matched = np.zeros(n_gt, dtype=bool)
    
    # Iterate through predictions (highest score first)
    for i in range(n_pred):
        # Find best matching GT box
        # ious[i] is (M,) tensor of IoUs with all GT boxes
        max_iou_val, max_iou_idx = torch.max(ious[i], dim=0)
        
        if max_iou_val >= iou_threshold:
            if not gt_matched[max_iou_idx]:
                tp[i] = 1
                gt_matched[max_iou_idx] = True
            else:
                fp[i] = 1  # Duplicate detection
        else:
            fp[i] = 1  # Localization error
            
    return tp, fp, n_gt

def compute_interpolated_ap(precisions: np.ndarray, recalls: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation or area under curve.
    Here we use the all-points interpolation standard (COCO style).
    """
    # Append sentinel values at both ends
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To calculate area under PR curve, look for points where X axis (recall) changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta Recall) * Precision
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])


# =============================================================================
# Metric Classes
# =============================================================================

class DetectionMetric(Metric):
    """Base class for detection metrics."""
    
    def __init__(self, name: str):
        super().__init__(name=name)

    def _get_matches(self, predictions: List[Dict], dataset, iou_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Aggregate matches across the entire dataset.
        
        Returns:
            all_tp: (Total_Preds,) array
            all_fp: (Total_Preds,) array
            total_gt: int
        """
        all_tp = []
        all_fp = []
        total_gt = 0
        
        dataset_iter = iter(dataset)
        
        for i, pred_dict in enumerate(predictions):
            # Get GT for this sample
            # Dataset yields (img, target) or potentially batches. 
            # Assuming dataset order matches predictions order.
            # We access the dataset by index if possible or iterate
            try:
                if hasattr(dataset, '__getitem__'):
                    _, target = dataset[i]
                else:
                    _, target = next(dataset_iter)
            except StopIteration:
                break
                
            pred_boxes = pred_dict['boxes'].cpu()
            pred_scores = pred_dict['scores'].cpu()
            # pred_labels = pred_dict['labels'].cpu() # Currently ignoring class labels for AP, assuming single class or micro-average
            
            gt_boxes = target['boxes'].cpu()
            # gt_labels = target['labels'].cpu() 
            
            tp, fp, n_gt = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold)
            
            all_tp.append(tp)
            all_fp.append(fp)
            total_gt += n_gt
            
        if not all_tp:
            return np.array([]), np.array([]), 0
            
        return np.concatenate(all_tp), np.concatenate(all_fp), total_gt

class DetectionAP(DetectionMetric):
    """
    Average Precision (AP) at IoU=0.5.
    """
    def __init__(self, iou_threshold: float = 0.5):
        super().__init__(f"DetectionAP@{iou_threshold}")
        self.iou_threshold = iou_threshold
        
    def calculate(self, predictions: List[Dict], dataset) -> float:
        tp, fp, total_gt = self._get_matches(predictions, dataset, self.iou_threshold)
        
        if total_gt == 0:
            return 0.0
        if len(tp) == 0:
            return 0.0
            
        # Accumulate TP/FP across all images
        # But wait, `_get_matches` already returns concatenated tp/fp sorted by score? 
        # Actually `match_predictions_to_gt` sorts *per image*. 
        # For full AP, we need to sort *all* predictions by score globally.
        
        # Refactoring to collect all preds first for global sorting
        all_preds = []
        total_gt = 0
        dataset_iter = iter(dataset)

        for i, pred_dict in enumerate(predictions):
            try:
                if hasattr(dataset, '__getitem__'):
                    _, target = dataset[i]
                else:
                    _, target = next(dataset_iter)
            except StopIteration:
                break

            boxes = pred_dict['boxes'].cpu()
            scores = pred_dict['scores'].cpu()
            gt_boxes = target['boxes'].cpu()
            
            # For each prediction, we need to know if it matches a GT.
            # This is complex because matching is greedy per-image.
            # Correct approach:
            # 1. Match per-image using local scores to determine TP/FP status for each pred
            # 2. Collect all preds with their scores and TP/FP status
            # 3. Sort globally by score
            # 4. Compute cumsum
            
            tp, fp, n_gt = match_predictions_to_gt(boxes, scores, gt_boxes, self.iou_threshold)
            
            # match_predictions_to_gt returns arrays sorted by score for that image.
            # We need to store the scores too to resort globally.
            # Sort local predictions to match match_predictions_to_gt output order
            sorted_indices = torch.argsort(scores, descending=True)
            sorted_scores = scores[sorted_indices]
            
            for j in range(len(tp)):
                all_preds.append({
                    'score': float(sorted_scores[j]),
                    'tp': tp[j],
                    'fp': fp[j]
                })
            
            total_gt += n_gt
            
        if total_gt == 0:
            return 0.0
        if not all_preds:
            return 0.0
            
        # Sort globally by score
        all_preds.sort(key=lambda x: x['score'], reverse=True)
        
        tp_cum = np.cumsum([x['tp'] for x in all_preds])
        fp_cum = np.cumsum([x['fp'] for x in all_preds])
        
        recalls = tp_cum / total_gt
        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
        
        return float(compute_interpolated_ap(precisions, recalls))

class DetectionAP50_95(DetectionMetric):
    """
    Mean Average Precision (mAP) averaged over IoU thresholds 0.5:0.95:0.05.
    """
    def __init__(self):
        super().__init__("DetectionAP@[0.5:0.95]")
        
    def calculate(self, predictions: List[Dict], dataset) -> float:
        thresholds = np.arange(0.5, 1.0, 0.05)
        aps = []
        
        for iou in thresholds:
            metric = DetectionAP(iou_threshold=iou)
            aps.append(metric.calculate(predictions, dataset))
            
        return float(np.mean(aps))

class DetectionRecall(DetectionMetric):
    """
    Recall at a fixed IoU threshold.
    Returns the maximum recall achieved (at lowest confidence threshold).
    """
    def __init__(self, iou_threshold: float = 0.5):
        super().__init__(f"DetectionRecall@{iou_threshold}")
        self.ap_metric = DetectionAP(iou_threshold)
        
    def calculate(self, predictions: List[Dict], dataset) -> float:
        # Reuse logic from AP but just look at final recall
        # This is slightly inefficient but ensures consistency
        
        # Helper to just get global TP/FP
        all_preds = []
        total_gt = 0
        dataset_iter = iter(dataset)

        for i, pred_dict in enumerate(predictions):
            try:
                if hasattr(dataset, '__getitem__'):
                    _, target = dataset[i]
                else:
                    _, target = next(dataset_iter)
            except StopIteration:
                break

            boxes = pred_dict['boxes'].cpu()
            scores = pred_dict['scores'].cpu()
            gt_boxes = target['boxes'].cpu()
            
            tp, fp, n_gt = match_predictions_to_gt(boxes, scores, gt_boxes, self.ap_metric.iou_threshold)
            total_gt += n_gt
            
            # We count total TPs found regardless of score threshold (implicitly score > 0 or whatever was kept)
            all_preds.extend(tp)

        if total_gt == 0:
            return 0.0
            
        total_tp = sum(all_preds)
        return float(total_tp / total_gt)

class DetectionPrecision(DetectionMetric):
    """
    Precision at a fixed IoU threshold.
    Note: Precision depends heavily on the confidence threshold. 
    This returns the precision considering ALL predictions returned by the model.
    """
    def __init__(self, iou_threshold: float = 0.5):
        super().__init__(f"DetectionPrecision@{iou_threshold}")
        self.ap_metric = DetectionAP(iou_threshold)
        
    def calculate(self, predictions: List[Dict], dataset) -> float:
        all_tp = 0
        all_fp = 0
        dataset_iter = iter(dataset)

        for i, pred_dict in enumerate(predictions):
            try:
                if hasattr(dataset, '__getitem__'):
                    _, target = dataset[i]
                else:
                    _, target = next(dataset_iter)
            except StopIteration:
                break

            boxes = pred_dict['boxes'].cpu()
            scores = pred_dict['scores'].cpu()
            gt_boxes = target['boxes'].cpu()
            
            tp, fp, _ = match_predictions_to_gt(boxes, scores, gt_boxes, self.ap_metric.iou_threshold)
            
            all_tp += tp.sum()
            all_fp += fp.sum()
            
        total_preds = all_tp + all_fp
        if total_preds == 0:
            return 0.0
            
        return float(all_tp / total_preds)

class DetectionF1(DetectionMetric):
    """
    F1 Score at a fixed IoU threshold.
    Harmonic mean of Recall and Precision (considering all predictions).
    """
    def __init__(self, iou_threshold: float = 0.5):
        super().__init__(f"DetectionF1@{iou_threshold}")
        self.rec_metric = DetectionRecall(iou_threshold)
        self.prec_metric = DetectionPrecision(iou_threshold)
        
    def calculate(self, predictions: List[Dict], dataset) -> float:
        recall = self.rec_metric.calculate(predictions, dataset)
        precision = self.prec_metric.calculate(predictions, dataset)
        
        if recall + precision == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
