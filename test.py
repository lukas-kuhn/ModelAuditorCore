import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision.models import resnet18, ResNet18_Weights
from medmnist import PathMNIST

from model_auditor import ModelAuditor
from model_auditor.shifts.image_shifts import (
    GaussianNoise,
    BrightnessShift,
    Rotation,
    HorizontalFlip,
    ZoomOut,
    ZoomIn,
    GaussianBlur,
    ContrastShift
)
from model_auditor.metrics.metrics_reloaded import (
    Accuracy,
    AUROC,
    BalancedAccuracy,
    BrierScore,
    ExpectedCalibrationError,
    MatthewsCorrelationCoefficient,
    WeightedCohensKappa,
    FBetaScore
)

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pretrained ResNet18
    model = resnet18(weights=None)  # No ImageNet weights
    model.fc = torch.nn.Linear(model.fc.in_features, 9)  # PathMNIST has 9 classes
    model.load_state_dict(torch.load('resnet18_pathmnist.pth'))
    model = model.to(device)
    model.eval()

    # Define transforms for PathMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                          std=[0.5, 0.5, 0.5])
    ])

    # Load PathMNIST test set
    test_dataset = PathMNIST(
        root='./data', 
        split='test',
        download=True, 
        transform=transform
    )
    
    # Use a subset for faster testing
    test_dataset = Subset(test_dataset, torch.randperm(len(test_dataset))[:50])

    # Create auditor
    auditor = ModelAuditor(model, test_dataset)

    # Add shifts with their predefined severity scales
    auditor.add_shift(ZoomIn())
    auditor.add_shift(ZoomOut())
    
    # Add metrics
    auditor.add_metric(Accuracy())
    auditor.add_metric(AUROC())
    auditor.add_metric(BalancedAccuracy())
    auditor.add_metric(BrierScore())
    auditor.add_metric(ExpectedCalibrationError())
    auditor.add_metric(MatthewsCorrelationCoefficient())
    auditor.add_metric(WeightedCohensKappa())

    # Run audit
    results = auditor.run()
    
    # Display results in a pretty format
    auditor.display_results(results)

if __name__ == "__main__":
    main() 