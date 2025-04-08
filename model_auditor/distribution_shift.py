from abc import ABC, abstractmethod
from typing import List
from torch.utils.data import Dataset

class DistributionShift(ABC):
    def __init__(self, name: str, severity_scale: List[float]):
        self.name = name
        self.severity_scale = severity_scale
        self.severity = severity_scale[0]  # Start with lowest severity
    
    @abstractmethod
    def apply(self, dataset: Dataset) -> Dataset:
        """Apply the distribution shift to the dataset"""
        pass 