from typing import List, Dict, Any, Union, Tuple
import torch
import numpy as np
import onnxruntime as ort
from torch.utils.data import Dataset
from rich.console import Console
from rich.table import Table
from .distribution_shift import DistributionShift
from .metric import Metric

class ModelAuditor:
    def __init__(self, model: Union[torch.nn.Module, str], dataset: Dataset):
        """
        Args:
            model: Either a PyTorch model or path to ONNX model
            dataset: Dataset to audit
        """
        self.is_onnx = isinstance(model, str)
        if self.is_onnx:
            self.model = ort.InferenceSession(model)
        else:
            self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.model.to(self.device)
        self.dataset = dataset
        self.shifts: List[DistributionShift] = []
        self.metrics: List[Metric] = []
        self.console = Console()
        
    def add_shift(self, shift: DistributionShift) -> None:
        self.shifts.append(shift)
        
    def add_metric(self, metric: Metric) -> None:
        self.metrics.append(metric)
        
    def _get_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Helper method to get predictions from either PyTorch or ONNX model"""
        if self.is_onnx:
            # Convert to numpy for ONNX runtime
            x_numpy = x.numpy()
            input_name = self.model.get_inputs()[0].name
            pred = self.model.run(None, {input_name: x_numpy})[0]
            return torch.from_numpy(pred)
        else:
            return self.model(x.to(self.device))

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the audit across predefined severity levels for each shift.
        
        Returns:
            Dict with structure: {shift_name: {severity: {metric_name: score}}}
        """
        results = {}
        
        for shift in self.shifts:
            shift_results = {}
            
            for severity in shift.severity_scale:
                # Apply shift with current severity
                shift.severity = severity
                shifted_dataset = shift.apply(self.dataset)
                severity_results = {}
                
                # Get model predictions on shifted data
                if not self.is_onnx:
                    self.model.eval()
                with torch.no_grad():
                    logits = []
                    for x, _ in shifted_dataset:
                        x = x.unsqueeze(0)
                        logits.append(self._get_predictions(x))
                    logits = torch.cat(logits, dim=0)
                
                # Calculate metrics
                for metric in self.metrics:
                    score = metric.calculate(logits, shifted_dataset)
                    severity_results[metric.name] = score
                    
                shift_results[severity] = severity_results
                
            results[shift.name] = shift_results
            
        return results 

    def display_results(self, results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """
        Display results in a pretty format using rich.
        
        Args:
            results: Dict with structure {shift_name: {severity: {metric_name: score}}}
        """
        self.console.print("\n[bold blue]Model Audit Results[/bold blue]\n")
        
        for shift_name, shift_results in results.items():
            table = Table(
                title=f"[bold]{shift_name}[/bold]",
                show_header=True,
                header_style="bold magenta"
            )
            
            # Add severity columns
            severities = list(shift_results.keys())
            table.add_column("Metric", style="cyan", no_wrap=True)
            for severity in severities:
                table.add_column(f"Severity {severity}", justify="right")
            
            # Add metric rows
            metrics = list(next(iter(shift_results.values())).keys())
            for metric in metrics:
                row = [metric]
                for severity in severities:
                    value = shift_results[severity][metric]
                    formatted_value = f"{value:.4f}"
                    row.append(formatted_value)
                table.add_row(*row)
            
            self.console.print(table)
            self.console.print() 
            
            return table # Add a blank line between tables 