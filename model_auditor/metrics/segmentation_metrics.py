"""
Segmentation metrics for semantic segmentation model auditing.

These metrics evaluate pixel-level overlap, distance, and boundary accuracy
between predicted segmentation masks and ground truth annotations.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from ..metric import Metric

try:
    from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures
except ImportError:
    raise ImportError(
        "MetricsReloaded is required for segmentation metrics. "
        "Install with: pip install MetricsReloaded"
    )


class SegmentationMetric(Metric):
    """
    Base class for segmentation metrics.
    
    Handles segmentation-specific input processing:
    - Model outputs: (N, C, H, W) logits/softmax where C is number of classes
    - Ground truth: (H, W) integer masks from dataset
    
    Metrics are computed per-sample and averaged across the dataset.
    For multi-class segmentation, metrics are computed per-class and averaged.
    """
    
    def __init__(self, name: str, metric_key: str, dict_args: Optional[Dict[str, Any]] = None):
        super().__init__(name=name)
        self.metric_key = metric_key
        self.dict_args = dict_args or {}
    
    def _prepare_inputs(self, logits: torch.Tensor, dataset) -> tuple:
        """
        Convert model outputs and dataset labels to prediction and reference masks.
        
        Args:
            logits: Model outputs of shape (N, C, H, W) where C >= 1
            dataset: Dataset yielding (image, mask) tuples
            
        Returns:
            Tuple of (predictions, labels) as numpy arrays of shape (N, H, W)
        """
        logits = logits.cpu()
        num_classes = logits.shape[1]
        
        if num_classes == 1:
            # Single-channel binary segmentation: apply sigmoid + threshold
            # Model outputs (N, 1, H, W) logits for binary classification
            probs = torch.sigmoid(logits).squeeze(1)  # (N, H, W)
            predictions = (probs > 0.5).long().numpy()  # (N, H, W)
        else:
            # Multi-class segmentation: use argmax over class dimension
            predictions = torch.argmax(logits, dim=1).numpy()  # (N, H, W)
        
        # Collect ground truth masks from dataset
        labels = []
        for _, mask in dataset:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = np.array(mask)
            # Squeeze to handle (1, H, W) -> (H, W)
            labels.append(np.squeeze(mask_np))
        labels = np.stack(labels, axis=0)  # (N, H, W)
        
        return predictions, labels
    
    def _compute_binary_metric(self, pred: np.ndarray, ref: np.ndarray) -> float:
        """
        Compute metric for a single binary mask pair.
        
        Args:
            pred: Predicted binary mask (H, W)
            ref: Reference binary mask (H, W)
            
        Returns:
            Metric value
        """
        # Flatten to 1D for MetricsReloaded
        pred_flat = pred.flatten().astype(np.int32)
        ref_flat = ref.flatten().astype(np.int32)
        
        measures = BinaryPairwiseMeasures(
            pred=pred_flat,
            ref=ref_flat,
            measures=[self.metric_key],
            dict_args=self.dict_args
        )
        return measures.to_dict_meas()[self.metric_key]
    
    def calculate(self, logits: torch.Tensor, dataset) -> float:
        """
        Calculate the segmentation metric.
        
        For multi-class segmentation, computes metric per class (excluding background 0)
        and returns the mean. For binary segmentation, computes directly.
        
        Args:
            logits: Model outputs of shape (N, C, H, W)
            dataset: Dataset containing (image, mask) pairs
            
        Returns:
            Mean metric value across samples and classes
        """
        predictions, labels = self._prepare_inputs(logits, dataset)
        num_classes = logits.shape[1]
        
        # Binary segmentation: C=1 (single channel with sigmoid) or C=2 (two-class softmax)
        if num_classes <= 2:
            # Binary segmentation: compute directly on class 1 (foreground)
            scores = []
            for i in range(len(predictions)):
                pred_binary = (predictions[i] == 1).astype(np.int32)
                ref_binary = (labels[i] == 1).astype(np.int32)
                
                # Skip if both masks are empty (undefined metric)
                if pred_binary.sum() == 0 and ref_binary.sum() == 0:
                    continue
                    
                try:
                    score = self._compute_binary_metric(pred_binary, ref_binary)
                    if not np.isnan(score) and not np.isinf(score):
                        scores.append(score)
                except Exception:
                    continue
            
            return float(np.mean(scores)) if scores else 0.0
        else:
            # Multi-class: compute per class, then average
            class_scores = []
            for c in range(1, num_classes):  # Skip background class 0
                sample_scores = []
                for i in range(len(predictions)):
                    pred_binary = (predictions[i] == c).astype(np.int32)
                    ref_binary = (labels[i] == c).astype(np.int32)
                    
                    # Skip if both masks are empty
                    if pred_binary.sum() == 0 and ref_binary.sum() == 0:
                        continue
                    
                    try:
                        score = self._compute_binary_metric(pred_binary, ref_binary)
                        if not np.isnan(score) and not np.isinf(score):
                            sample_scores.append(score)
                    except Exception:
                        continue
                
                if sample_scores:
                    class_scores.append(np.mean(sample_scores))
            
            return float(np.mean(class_scores)) if class_scores else 0.0


# =============================================================================
# Overlap Metrics (Pixel Counting)
# =============================================================================

class SegmentationDice(SegmentationMetric):
    """
    Dice Similarity Coefficient (DSC) / F1 Score for segmentation.
    
    DSC = 2 * |pred ∩ ref| / (|pred| + |ref|)
    
    Range: [0, 1], higher is better.
    """
    def __init__(self):
        super().__init__("segmentation_dice", "dsc")


class SegmentationIoU(SegmentationMetric):
    """
    Intersection over Union (IoU) / Jaccard Index for segmentation.
    
    IoU = |pred ∩ ref| / |pred ∪ ref|
    
    Range: [0, 1], higher is better.
    """
    def __init__(self):
        super().__init__("segmentation_iou", "iou")


class SegmentationAccuracy(Metric):
    """
    Pixel-wise accuracy for segmentation.
    
    Accuracy = (correctly classified pixels) / (total pixels)
    
    Range: [0, 1], higher is better.
    Note: Can be misleading for imbalanced classes.
    """
    def __init__(self):
        super().__init__("segmentation_accuracy")
    
    def calculate(self, logits: torch.Tensor, dataset) -> float:
        logits = logits.cpu()
        num_classes = logits.shape[1]
        
        if num_classes == 1:
            # Single-channel binary segmentation: apply sigmoid + threshold
            probs = torch.sigmoid(logits).squeeze(1)  # (N, H, W)
            predictions = (probs > 0.5).long().numpy()  # (N, H, W)
        else:
            # Multi-class segmentation: use argmax
            predictions = torch.argmax(logits, dim=1).numpy()  # (N, H, W)
        
        labels = []
        for _, mask in dataset:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.numpy()
            else:
                mask_np = np.array(mask)
            # Squeeze to handle (1, H, W) -> (H, W)
            labels.append(np.squeeze(mask_np))
        labels = np.stack(labels, axis=0)  # (N, H, W)
        
        correct = (predictions == labels).sum()
        total = predictions.size
        return float(correct / total)


# =============================================================================
# Distance Metrics (Boundary-based)
# =============================================================================

class HausdorffDistance(SegmentationMetric):
    """
    Hausdorff Distance - maximum surface distance between boundaries.
    
    HD = max(max_a d(a, B), max_b d(b, A))
    
    Range: [0, ∞), lower is better.
    Sensitive to outliers.
    """
    def __init__(self):
        super().__init__("hausdorff_distance", "hd")


class HausdorffDistance95(SegmentationMetric):
    """
    95th percentile Hausdorff Distance.
    
    More robust than HD as it ignores the worst 5% of distances.
    
    Range: [0, ∞), lower is better.
    """
    def __init__(self, percentile: float = 95.0):
        super().__init__(
            "hausdorff_distance_95", 
            "hd_perc",
            dict_args={"hd_perc": percentile}
        )


class AverageSurfaceDistance(SegmentationMetric):
    """
    Mean Average Surface Distance (MASD).
    
    Average of all distances from boundary points in pred to nearest
    boundary point in ref, and vice versa.
    
    Range: [0, ∞), lower is better.
    """
    def __init__(self):
        super().__init__("average_surface_distance", "masd")


class NormalizedSurfaceDistance(SegmentationMetric):
    """
    Normalized Surface Distance (NSD).
    
    Fraction of boundary points within a tolerance distance τ of the
    corresponding surface.
    
    Range: [0, 1], higher is better.
    
    Args:
        tolerance: Distance tolerance in pixels (default: 2.0)
    """
    def __init__(self, tolerance: float = 2.0):
        super().__init__(
            "normalized_surface_distance",
            "nsd", 
            dict_args={"nsd": tolerance}
        )


# =============================================================================
# Boundary Metrics
# =============================================================================

class BoundaryIoU(SegmentationMetric):
    """
    Boundary IoU - IoU computed only on boundary pixels.
    
    Evaluates boundary accuracy without being affected by
    interior pixel correctness.
    
    Range: [0, 1], higher is better.
    """
    def __init__(self):
        super().__init__("boundary_iou", "boundary_iou")
