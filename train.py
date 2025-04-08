import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console

def train_model():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    console = Console()
    console.print(f"[bold green]Using device: {device}[/bold green]")

    # Data augmentation and normalization for training
    # Just normalization for validation
    transform_train = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224 images
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Load CIFAR10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )

    # Load pretrained ResNet18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify final layer for CIFAR10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training loop
    num_epochs = 10
    best_acc = 0.0
    
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress: 
        
        epoch_task = progress.add_task("[red]Epochs...", total=num_epochs)
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            train_task = progress.add_task(f"[green]Training Epoch {epoch+1}...", total=len(trainloader))
            
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                progress.advance(train_task)
            
            # Validation phase
            model.eval()
            correct = 0
            total = 0
            val_task = progress.add_task(f"[cyan]Validating Epoch {epoch+1}...", total=len(testloader))
            
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    progress.advance(val_task)
            
            acc = 100 * correct / total
            progress.advance(epoch_task)
            
            console.print(f"Epoch {epoch+1}: Loss = {running_loss/len(trainloader):.4f}, "
                         f"Accuracy = {acc:.2f}%")
            
            # Save best model
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'cifar10_resnet18.pth')
                console.print(f"[bold green]Saved new best model with accuracy: {acc:.2f}%[/bold green]")
            
            scheduler.step()

    console.print("[bold green]Training completed![/bold green]")
    console.print(f"[bold blue]Best accuracy: {best_acc:.2f}%[/bold blue]")

if __name__ == "__main__":
    train_model() 