# Model Auditor

A library for auditing ML models under distribution shifts.

## Installation

Install directly from GitHub using pip:

```bash
pip install git+https://github.com/lukaskuhn/ModelAuditorCore.git
```

Or using uv:

```bash
uv pip install git+https://github.com/lukaskuhn/ModelAuditorCore.git
```

## Usage

```python
from model_auditor import ModelAuditor
from model_auditor.metrics import Accuracy, AUROC
from model_auditor.shifts import GaussianNoise, BrightnessShift

# Create auditor with your model
auditor = ModelAuditor(model)

# Add distribution shifts
auditor.add_shift(GaussianNoise(std=0.1))
auditor.add_shift(BrightnessShift(factor=0.2))

# Add metrics
auditor.add_metric(Accuracy())
auditor.add_metric(AUROC())

# Run audit
results = auditor.audit(dataset)
```

### Segmentation

```python
from model_auditor.metrics import (
    SegmentationDice, SegmentationIoU, HausdorffDistance
)

# For segmentation models, output shape should be (N, C, H, W)
# where C is the number of classes
auditor.add_metric(SegmentationDice())
auditor.add_metric(SegmentationIoU())
auditor.add_metric(HausdorffDistance())
```

## License

Apache-2.0