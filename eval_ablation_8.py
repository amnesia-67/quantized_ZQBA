import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.quantization import resnet18 as qresnet18
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def projected_gradient_descent(model, images, labels, eps=8/255, alpha=2/255, steps=7):
    images = images.clone().detach().to(images.device)
    labels = labels.clone().detach().to(images.device)
    adv_images = images.clone().detach()
    
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            eta = torch.clamp(adv_images - images, min=-eps, max=eps)
            adv_images = torch.clamp(images + eta, min=0, max=1)
            adv_images = adv_images.detach()
    return adv_images

def load_target(path, device):
    model = models.resnet34()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)

def load_surrogate(path, device):
    model = qresnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.train() 
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    overlaps = [100, 85, 70, 55, 40, 25, 10, 0]
    target_path = "models/ablation_study/Target_Fixed.pth"
    target_model = load_target(target_path, device)
    
    asr_results = []

    for overlap in overlaps:
        print(f"\n--- Attacking with {overlap}% Overlap Surrogate ---")
        s_path = f"models/ablation_study/Surrogate_{overlap}.pth"
        
        if not os.path.exists(s_path):
            print(f"Skipping {overlap}%, model not found.")
            asr_results.append(0)
            continue

        surrogate_model = load_surrogate(s_path, device)
        correct = 0
        total = 0

        for images, labels in tqdm(testloader, desc="PGD Batches"):
            images, labels = images.to(device), labels.to(device)
            adv_images = projected_gradient_descent(surrogate_model, images, labels)
            
            with torch.no_grad():
                outputs = target_model(adv_images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        asr = 100.0 - (100.0 * correct / total)
        print(f"ASR: {asr:.2f}%")
        asr_results.append(asr)

    # Visualization
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [f"{x}%" for x in overlaps]
    
    ax.plot(x_labels, asr_results, marker='o', markersize=8, linestyle='-', linewidth=2.5, color='#c44e52')
    ax.set_xlabel('Surrogate Data Overlap with Target', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Attack Success Rate (ASR %)', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylim(0, 100)
    ax.invert_xaxis() # Read naturally from 100% down to 0%
    ax.set_title('Impact of Dataset Overlap on Transferability', fontsize=12, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('overlap_ablation_8.pdf', format='pdf', dpi=300)
    print("\nPlot saved as overlap_ablation_8.pdf")

if __name__ == "__main__":
    main()