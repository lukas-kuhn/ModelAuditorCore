import torch
import numpy as np
from ..metric import Metric
from MetricsReloaded.metrics.pairwise_measures import BinaryPairwiseMeasures, MultiClassPairwiseMeasures
from MetricsReloaded.metrics.prob_pairwise_measures import ProbabilityPairwiseMeasures
from MetricsReloaded.metrics.calibration_measures import CalibrationMeasures

class MetricsReloadedWrapper(Metric):
    """Base wrapper for MetricsReloaded metrics"""
    def __init__(self, name, metric_key):
        super().__init__(name=name)
        self.metric_key = metric_key
    
    def _prepare_inputs(self, logits: torch.Tensor, dataset) -> tuple:
        logits = logits.cpu()
        probs = torch.softmax(logits, dim=1).numpy()
        
        # Convert list to numpy array first to avoid the warning
        labels_list = [y for _, y in dataset]
        labels_np = np.array(labels_list, dtype=np.int64)
        labels = torch.tensor(labels_np, dtype=torch.long)
        
        return probs, labels

    def _to_one_hot(self, labels: torch.Tensor, num_classes: int) -> np.ndarray:
        labels_np = labels.numpy()
        return np.eye(num_classes)[labels_np]

class BinaryMetric(MetricsReloadedWrapper):
    """Wrapper for binary metrics. Only works with binary classification tasks."""
    def calculate(self, logits: torch.Tensor, dataset) -> float:
        if logits.shape[1] != 2:
            raise ValueError(f"{self.name} only works with binary classification (2 classes)")
        probs, labels = self._prepare_inputs(logits, dataset)
        predictions = (probs[:, 1] > 0.5).astype(np.int32)
        labels = labels.numpy().astype(np.int32)
        measures = BinaryPairwiseMeasures(
            pred=predictions, 
            ref=labels, 
            measures=[self.metric_key],
            dict_args=getattr(self, 'dict_args', {})
        )
        return measures.to_dict_meas()[self.metric_key]

class MultiClassMetric(MetricsReloadedWrapper):
    """Wrapper for multiclass metrics. Works with any number of classes."""
    def calculate(self, logits: torch.Tensor, dataset) -> float:
        probs, labels = self._prepare_inputs(logits, dataset)
        predictions = np.argmax(probs, axis=1).astype(np.int32)
        labels = labels.numpy().astype(np.int32)
        
        # Ensure labels is 1D
        if len(labels.shape) > 1:
            labels = labels.squeeze()
            
        # Get unique classes from both predictions and labels
        unique_labels = sorted(np.unique(np.concatenate([predictions, labels])))
        
        measures = MultiClassPairwiseMeasures(
            pred=predictions, 
            ref=labels, 
            list_values=unique_labels, 
            measures=[self.metric_key],
            dict_args=getattr(self, 'dict_args', {})
        )
        return measures.to_dict_meas()[self.metric_key]

class ProbabilisticMetric(MetricsReloadedWrapper):
    """Wrapper for probabilistic metrics. Works with any number of classes."""
    def calculate(self, logits: torch.Tensor, dataset) -> float:
        probs, labels = self._prepare_inputs(logits, dataset)
        num_samples = probs.shape[0]
        num_classes = probs.shape[1]
        
        # Create one-hot encoded labels with the same shape as probs
        labels_one_hot = np.zeros((num_samples, num_classes), dtype=np.float32)
        for i, label in enumerate(labels.numpy()):
            if label < num_classes:  # Ensure label is valid
                labels_one_hot[i, label] = 1.0
        
        measures = ProbabilityPairwiseMeasures(
            pred_proba=probs,
            ref_proba=labels_one_hot,
            measures=[self.metric_key],
            dict_args=getattr(self, 'dict_args', {})
        )
        return measures.to_dict_meas()[self.metric_key]

class CalibrationMetric(MetricsReloadedWrapper):
    """Wrapper for calibration metrics. Works with any number of classes."""
    def calculate(self, logits: torch.Tensor, dataset) -> float:
        probs, labels = self._prepare_inputs(logits, dataset)
        labels = labels.numpy().astype(np.int32)
        measures = CalibrationMeasures(
            pred_proba=probs,
            ref=labels,
            measures=[self.metric_key],
            dict_args=getattr(self, 'dict_args', {})
        )
        return measures.to_dict_meas()[self.metric_key]

# Binary Only Metrics
class Sensitivity(BinaryMetric):
    """Binary only: True positive rate."""
    def __init__(self):
        super().__init__("sensitivity", "sensitivity")

class Specificity(BinaryMetric):
    """Binary only: True negative rate."""
    def __init__(self):
        super().__init__("specificity", "specificity")

class PositivePredictiveValue(BinaryMetric):
    """Binary only: Precision."""
    def __init__(self):
        super().__init__("positive_predictive_value", "ppv")

class NegativePredictiveValue(BinaryMetric):
    """Binary only: Negative predictive value."""
    def __init__(self):
        super().__init__("negative_predictive_value", "npv")

class PositiveLikelihoodRatio(BinaryMetric):
    """Binary only: Positive likelihood ratio."""
    def __init__(self):
        super().__init__("positive_likelihood_ratio", "lr+")

class DiceSimilarityCoefficient(BinaryMetric):
    """Binary only: F1 score / Dice coefficient."""
    def __init__(self):
        super().__init__("dice_similarity_coefficient", "dsc")

class FBetaScore(BinaryMetric):
    """Binary only: F-beta score."""
    def __init__(self, beta=1):
        super().__init__("fbeta_score", "fbeta")
        self.dict_args = {"beta": beta}

class NetBenefit(BinaryMetric):
    """Binary only: Net benefit."""
    def __init__(self, exchange_rate=1):
        super().__init__("net_benefit", "nb")
        self.dict_args = {"exchange_rate": exchange_rate}

# Multiclass Metrics
class Accuracy(MetricsReloadedWrapper):
    """Overall accuracy. Works with both binary and multiclass."""
    def __init__(self):
        super().__init__("accuracy", "accuracy")

    def calculate(self, logits: torch.Tensor, dataset) -> float:
        probs, labels = self._prepare_inputs(logits, dataset)
        predictions = np.argmax(probs, axis=1)
        labels = labels.numpy()
        return float(np.mean(predictions == labels))

class BalancedAccuracy(MultiClassMetric):
    """Multiclass: Average recall across all classes."""
    def __init__(self):
        super().__init__("balanced_accuracy", "ba")

class MatthewsCorrelationCoefficient(MultiClassMetric):
    """Multiclass: MCC, correlation coefficient between predictions and ground truth."""
    def __init__(self):
        super().__init__("matthews_correlation_coefficient", "mcc")

class WeightedCohensKappa(MultiClassMetric):
    """Multiclass: Agreement between predictions and ground truth."""
    def __init__(self):
        super().__init__("weighted_cohens_kappa", "wck")

class ExpectedCost(MultiClassMetric):
    """Multiclass: Expected cost of predictions."""
    def __init__(self):
        super().__init__("expected_cost", "ec")

# Probabilistic Metrics (work with both binary and multiclass)
class AUROC(ProbabilisticMetric):
    """Probabilistic: Area under ROC curve."""
    def __init__(self):
        super().__init__("auroc", "auroc")

class AveragePrecision(ProbabilisticMetric):
    """Probabilistic: Area under precision-recall curve."""
    def __init__(self):
        super().__init__("average_precision", "ap")

# Calibration Metrics (work with both binary and multiclass)
class BrierScore(CalibrationMetric):
    """Calibration: Mean squared error of probabilities."""
    def __init__(self):
        super().__init__("brier_score", "bs")

class RootBrierScore(CalibrationMetric):
    """Calibration: Root mean squared error of probabilities."""
    def __init__(self):
        super().__init__("root_brier_score", "rbs")

class ExpectedCalibrationError(CalibrationMetric):
    """Calibration: Expected calibration error."""
    def __init__(self):
        super().__init__("expected_calibration_error", "ece")

class ClassWiseECE(CalibrationMetric):
    """Calibration: Class-wise expected calibration error."""
    def __init__(self):
        super().__init__("class_wise_ece", "cwece")

class ECEKernelDensity(CalibrationMetric):
    """Calibration: Kernel density based ECE."""
    def __init__(self):
        super().__init__("ece_kernel_density", "ece_kde")

class KernelCalibrationError(CalibrationMetric):
    """Calibration: Kernel-based calibration error."""
    def __init__(self):
        super().__init__("kernel_calibration_error", "kce")

class NegativeLogLikelihood(CalibrationMetric):
    """Calibration: Negative log likelihood."""
    def __init__(self):
        super().__init__("negative_log_likelihood", "nll") 