from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset

class Metric(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, logits: torch.Tensor, dataset: Dataset) -> float:
        """Calculate metric given model logits and dataset"""
        pass 