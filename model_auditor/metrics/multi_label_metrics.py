import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, hamming_loss, precision_score, recall_score, f1_score, jaccard_score, roc_auc_score
)
# Assuming the Metric base class is in the parent directory
# Adjust the import path if necessary
try:
    from ..metric import Metric
except ImportError:
    # Fallback if running script directly or structure differs
    from model_auditor.metric import Metric


class MultiLabelMetric(Metric):
    """Base wrapper for multi-label classification metrics using scikit-learn."""
    def __init__(self, name, metric_func=None, average=None, requires_proba=False, **kwargs):
        super().__init__(name=name)
        self.metric_func = metric_func
        self.average = average # For micro, macro, weighted, samples
        self.requires_proba = requires_proba # If metric needs probabilities instead of binary predictions
        self.kwargs = kwargs # Additional arguments for the metric function

    def _prepare_inputs(self, logits: torch.Tensor, dataset) -> tuple:
        """
        Prepares predictions and labels for scikit-learn multi-label metrics.

        Args:
            logits: Raw output scores from the model (N, C).
            dataset: An iterable yielding (input, labels), where labels
                     is expected to be an iterable of positive label indices for that sample.

        Returns:
            A tuple containing:
            - predictions (np.ndarray): Binary predictions (N, C).
            - labels_binary (np.ndarray): True binary labels (N, C).
            - probs (np.ndarray): Sigmoid probabilities (N, C).
        """
        num_classes = logits.shape[1]
        if num_classes <= 1:
            raise ValueError("Multi-label metrics require more than one class.")

        # Apply sigmoid to get independent probabilities per class
        probs = torch.sigmoid(logits).cpu().numpy()
        # Use a standard threshold of 0.5 for binary predictions
        predictions = (probs > 0.5).astype(np.int32)

        # Extract labels from the dataset
        # Assumes dataset yields (input, label_list)
        labels_list = [y for _, y in dataset]
        num_samples = len(labels_list)

        # Convert list of positive label indices to multi-label binary format (N, C)
        labels_binary = np.zeros((num_samples, num_classes), dtype=np.int32)
        for i, positive_labels in enumerate(labels_list):
            # Ensure positive_labels is iterable and contains valid indices
            if isinstance(positive_labels, (list, tuple, np.ndarray)):
                # Filter out any potential out-of-bounds indices gracefully
                valid_labels = [int(lbl) for lbl in positive_labels if 0 <= lbl < num_classes]
                if valid_labels:
                    labels_binary[i, valid_labels] = 1
            elif isinstance(positive_labels, (int, np.integer)): # Handle single integer label case if needed
                 if 0 <= positive_labels < num_classes:
                     labels_binary[i, positive_labels] = 1
            # Else: row remains all zeros if positive_labels is None, empty, or invalid type

        return predictions, labels_binary, probs

    def calculate(self, logits: torch.Tensor, dataset) -> float:
        """
        Calculates the specified multi-label metric.

        Args:
            logits: Raw output scores from the model.
            dataset: The dataset containing inputs and true labels.

        Returns:
            The calculated metric value.
        """
        predictions, labels, probs = self._prepare_inputs(logits, dataset)

        if self.metric_func is None:
             raise NotImplementedError("Metric function not specified for this class.")

        # Prepare arguments for sklearn metric functions
        metric_args = {}
        if self.average:
            # Subset accuracy (accuracy_score) and hamming_loss don't use 'average'
            if self.metric_func not in [accuracy_score, hamming_loss]:
                metric_args['average'] = self.average
        
        # Add zero_division parameter for relevant metrics to handle undefined cases
        if self.metric_func in [precision_score, recall_score, f1_score, jaccard_score]:
             metric_args['zero_division'] = self.kwargs.get('zero_division', 0)

        # Add multi_class parameter for roc_auc_score if needed (though defaults are usually fine for multi-label)
        if self.metric_func == roc_auc_score:
            # sklearn's roc_auc_score handles multi-label automatically when y_true is binary format
            # We might want to expose 'multi_class' or other roc_auc_score specific args if needed
            pass

        # Add any other specific kwargs passed during initialization
        metric_args.update({k: v for k, v in self.kwargs.items() if k not in ['zero_division']})

        if self.requires_proba:
             # Metrics like AUROC use probabilities
             # Ensure labels are passed as y_true and probs as y_score
             return self.metric_func(labels, probs, **metric_args)
        else:
             # accuracy_score is used for subset accuracy in multi-label context
             # hamming_loss doesn't need averaging args
             if self.metric_func == accuracy_score:
                 # Ensure normalize=True for standard subset accuracy proportion
                 return self.metric_func(labels, predictions, normalize=True)
             elif self.metric_func == hamming_loss:
                 return self.metric_func(labels, predictions)
             else:
                 # Other metrics like precision, recall, f1, jaccard use binary predictions
                 return self.metric_func(labels, predictions, **metric_args)


# --- Specific Metric Implementations ---

# Exact Match Ratio (Subset Accuracy)
class MultiLabelSubsetAccuracy(MultiLabelMetric):
    """Multi-label: Exact match ratio (predictions must match labels exactly)."""
    def __init__(self):
        super().__init__(name="subset_accuracy", metric_func=accuracy_score)

# Hamming Loss
class MultiLabelHammingLoss(MultiLabelMetric):
    """Multi-label: Fraction of labels that are incorrectly predicted (lower is better)."""
    def __init__(self):
        super().__init__(name="hamming_loss", metric_func=hamming_loss)

# --- Precision ---
class MultiLabelPrecision(MultiLabelMetric):
    """Multi-label: Weighted-averaged precision (weighted by support per label)."""
    def __init__(self, zero_division=0):
        super().__init__(name="precision", metric_func=precision_score, average="weighted", zero_division=zero_division)

# --- Recall ---
class MultiLabelRecall(MultiLabelMetric):
    """Multi-label: Weighted-averaged recall (weighted by support per label)."""
    def __init__(self, zero_division=0):
        super().__init__(name="recall", metric_func=recall_score, average="weighted", zero_division=zero_division)

# --- F1 Score ---
class MultiLabelF1Score(MultiLabelMetric):
    """Multi-label: Weighted-averaged F1 score (weighted by support per label)."""
    def __init__(self, zero_division=0):
        super().__init__(name="f1_score", metric_func=f1_score, average="weighted", zero_division=zero_division)

# --- Jaccard Score (Intersection over Union) ---
class MultiLabelJaccardScore(MultiLabelMetric):
    """Multi-label: Weighted-averaged Jaccard similarity coefficient (weighted by support per label)."""
    def __init__(self, zero_division=0):
        super().__init__(name="jaccard_score", metric_func=jaccard_score, average="weighted", zero_division=zero_division)

# --- AUROC (Area Under the ROC Curve) ---
class MultiLabelAUROC(MultiLabelMetric):
    """Multi-label: Weighted-averaged Area Under the ROC Curve (weighted by support per label)."""
    def __init__(self, **kwargs):
        super().__init__(name="auroc", metric_func=roc_auc_score, average="weighted", requires_proba=True, **kwargs)
