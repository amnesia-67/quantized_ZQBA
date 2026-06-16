import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.quantization import resnet18 as qresnet18
from tqdm import tqdm

def main():
    # Setup argument parsing for the automation loop
    parser = argparse.ArgumentParser(description="Ablation Study Training Script")
    parser.add_argument('--role', type=str, choices=['target', 'surrogate'], required=True, help="Train the FP32 Target or QAT Surrogate")
    parser.add_argument('--split_file', type=str, required=True, help="Path to the PyTorch index list (.pt file)")
    parser.add_argument('--save_name', type=str, required=True, help="Output filename for the model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "models/ablation_study"
    os.makedirs(save_dir, exist_ok=True)

    # Standard CIFAR-10 Augmentations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    
    # Load the full dataset, then slice it using the deterministic split indices
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    
    if not os.path.exists(args.split_file):
        raise FileNotFoundError(f"Split file not found at {args.split_file}. Run the split generator first.")
        
    indices = torch.load(args.split_file)
    trainset = torch.utils.data.Subset(full_trainset, indices)
    
    # Keep batch_size at 128 to match your previous Phase 1 parameters
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    print(f"[{args.role.upper()}] Initializing training on {len(indices)} images. Target Device: {device}")

    # ==========================================
    # Target Architecture (FP32 ResNet-34)
    # ==========================================
    if args.role == 'target':
        model = models.resnet34()
        # Modify for CIFAR-10 32x32 resolution
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        epochs = 200 
        
    # ==========================================
    # Surrogate Architecture (QAT INT8 ResNet-18)
    # ==========================================
    else: 
        model = qresnet18(weights=None, num_classes=10)
        # Modify for CIFAR-10
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
        # Prepare QAT graph
        model.train()
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        model.to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = None
        epochs = 10 

    criterion = nn.CrossEntropyLoss()

    # ==========================================
    # Training Loop
    # ==========================================
    for epoch in range(epochs):
        model.train()
        
        # Freeze observer stats late in QAT to stabilize
        if args.role == 'surrogate' and epoch > 8:
            model.apply(torch.quantization.disable_observer)
        if args.role == 'surrogate' and epoch > 9:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [{args.role}]", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        if scheduler:
            scheduler.step()

    # Move to CPU before saving to prevent device mapping errors later
    model.cpu()
    
    # Save the model state (For surrogates, this saves the fake-quant state required for PGD)
    save_path = os.path.join(save_dir, args.save_name)
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] {args.role.capitalize()} model saved to {save_path}\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()