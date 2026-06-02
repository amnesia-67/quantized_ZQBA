import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.quantization import resnet18 as qresnet18
from tqdm import tqdm


'''
Author  : Anish Subash
Date    : June 1st, 2026
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str, choices=['target', 'surrogate'], required=True)
    parser.add_argument('--split_file', type=str, required=True, help="Path to the .pt index file")
    parser.add_argument('--save_name', type=str, required=True, help="Name of the output model file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "models/ablation_study"
    os.makedirs(save_dir, exist_ok=True)

    # Data setup
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    
    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    indices = torch.load(args.split_file)
    trainset = torch.utils.data.Subset(full_trainset, indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Training {args.role} on {len(indices)} images...")

    if args.role == 'target':
        model = models.resnet34()
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, 10)
        model.to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        epochs = 200 # Target requires full convergence
        
    else: # surrogate
        model = qresnet18(weights=None, num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.train()
        model.fuse_model()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        model.to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        scheduler = None
        epochs = 10 # QAT fine-tuning

    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        if scheduler:
            scheduler.step()

    model.cpu()
    torch.save(model.state_dict(), os.path.join(save_dir, args.save_name))
    print(f"Saved to {os.path.join(save_dir, args.save_name)}\n")

if __name__ == "__main__":
    main()