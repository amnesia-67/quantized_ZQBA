import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm

from wideresnet import WideResNet  # Ensure your patched wideresnet.py is in the same directory

def get_baseline_model(model_name, num_classes=10):
    """
    Dynamically loads and modifies architectures for CIFAR-10 (32x32 resolution).
    """
    if model_name == "resnet34":
        model = models.resnet34()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "resnet56":
        # PyTorch doesn't have a native resnet56, using resnet50 as the standard high-capacity proxy
        model = models.resnet50()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "vgg16":
        model = models.vgg16_bn() # Swapped to the Batch Norm version
        model.features[0] = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(4096, num_classes)
        
    elif model_name == "wideresnet":
        # Using standard parameters for Defensive Quantization bounds
        model = WideResNet(depth=34, num_classes=num_classes, widen_factor=10)
        
    else:
        raise ValueError(f"Model {model_name} not supported.")
        
    return model

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train Baselines for ZQBA Phase 1")
    parser.add_argument('--arch', type=str, default='resnet34', 
                        choices=['resnet34', 'resnet56', 'vgg16', 'wideresnet'], 
                        help='Target architecture to train')
    args = parser.parse_args()

    # Dynamic save directory based on architecture
    save_dir = f"models/{args.arch}_CIFAR10"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Initializing baseline architecture: {args.arch}")

    # No normalization (matching original protocol configuration)
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_test
    )

    # Strictly enforced batch_size=128 per research protocol
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = get_baseline_model(args.arch, num_classes=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # VGG explodes with 0.1, throttle it to 0.01
    initial_lr = 0.01 if args.arch == "vgg16" else 0.1
    
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1
    )

    epochs = 200
    best_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(trainloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix({
                "Batch Loss": f"{loss.item():.4f}",
                "LR": optimizer.param_groups[0]['lr']
            })

        epoch_loss = running_loss / len(trainloader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss:.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Test Accuracy: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_path = os.path.join(save_dir, f"{args.arch}_cifar10_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with accuracy: {best_acc:.2f}% to {save_path}")

        scheduler.step()

    print("Training finished.")
    print(f"Best Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()