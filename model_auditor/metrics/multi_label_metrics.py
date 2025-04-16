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
                     can be either an iterable of positive label indices
                     or a binary vector/tensor [0, 1, 0, ...].

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

        # --- Revised Label Handling ---
        labels_list_from_dataset = [y for _, y in dataset]
        num_samples = len(labels_list_from_dataset)

        if num_samples == 0:
            # Handle empty dataset case
            return predictions, np.zeros((0, num_classes), dtype=np.int32), probs

        # Check the format of the first label to determine how to process
        first_label = labels_list_from_dataset[0]

        # Determine if labels are provided as binary vectors or lists of indices
        is_binary_vector = False
        if isinstance(first_label, (torch.Tensor, np.ndarray)):
            if len(first_label.shape) == 1 and first_label.shape[0] == num_classes:
                 is_binary_vector = True
            # Add check for (N, 1) shape tensor/array if needed
            elif len(first_label.shape) == 2 and first_label.shape[0] == num_classes and first_label.shape[1] == 1:
                 # Reshape to (N,) if necessary, then treat as binary vector
                 # This might need adjustment based on exact format
                 pass # Add specific handling if this case is expected

        if is_binary_vector:
            # Labels are already in binary vector format (N, C)
            try:
                # Stack potentially mixed list of tensors/arrays
                processed_labels = []
                for lbl in labels_list_from_dataset:
                    if isinstance(lbl, torch.Tensor):
                        processed_labels.append(lbl.cpu().numpy()) # Ensure numpy and on CPU
                    elif isinstance(lbl, np.ndarray):
                        processed_labels.append(lbl)
                    else:
                         raise TypeError(f"Inconsistent label types in dataset when expecting binary vectors. Got {type(lbl)}")
                
                labels_binary = np.stack(processed_labels).astype(np.int32)

                # Basic validation
                if labels_binary.shape != (num_samples, num_classes):
                     raise ValueError(f"Label format detection failed: Shape mismatch. Expected ({num_samples}, {num_classes}), Got {labels_binary.shape}")
                if not ((labels_binary == 0) | (labels_binary == 1)).all():
                     # Allow -1 if necessary, depending on dataset conventions? For now, strictly binary.
                     raise ValueError("Label format detection failed: Non-binary (0/1) values found in label vectors.")

            except Exception as e:
                 raise ValueError(f"Error processing assumed binary label vectors: {e}")

        else:
             # Assume labels are lists/tuples/arrays of positive indices
             labels_binary = np.zeros((num_samples, num_classes), dtype=np.int32)
             for i, positive_labels in enumerate(labels_list_from_dataset):
                 try:
                     # Ensure positive_labels is iterable and contains valid indices
                     # Check if item is number-like before casting to int
                     valid_indices = [int(lbl) for lbl in positive_labels 
                                      if isinstance(lbl, (int, float, np.number)) and 0 <= int(lbl) < num_classes]
                     if valid_indices:
                         labels_binary[i, valid_indices] = 1
                 except TypeError:
                     # Handle cases where positive_labels itself is not iterable or contains non-numeric types
                     print(f"Warning: Skipping non-iterable or invalid label format for sample {i}: {positive_labels}")
                     continue # Keep row as zeros

        # Final check on labels_binary shape
        if labels_binary.shape[0] != num_samples or labels_binary.shape[1] != num_classes:
            raise ValueError(f"Final labels_binary shape mismatch. Expected ({num_samples}, {num_classes}), Got {labels_binary.shape}")
            
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
