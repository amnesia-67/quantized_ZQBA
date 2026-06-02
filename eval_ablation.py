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

'''
Author  : Anish Subash
Date    : June 1st, 2026
'''

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

    scenarios = [
        {"name": "90/20 (10% Overlap)", "target": "Target_A_90.pth", "surrogate": "Surrogate_A_20.pth"},
        {"name": "80/30 (Small Overlap)", "target": "Target_B_80.pth", "surrogate": "Surrogate_B_30_Small.pth"},
        {"name": "80/30 (Large Overlap)", "target": "Target_C_80.pth", "surrogate": "Surrogate_C_30_Large.pth"},
        {"name": "50/50 (Disjoint)", "target": "Target_D_50.pth", "surrogate": "Surrogate_D_50.pth"}
    ]

    results = []

    for s in scenarios:
        print(f"\n--- Evaluating {s['name']} ---")
        t_path = os.path.join("models/ablation_study", s["target"])
        s_path = os.path.join("models/ablation_study", s["surrogate"])
        
        if not os.path.exists(t_path) or not os.path.exists(s_path):
            print(f"Skipping {s['name']}, models not found.")
            continue

        target_model = load_target(t_path, device)
        surrogate_model = load_surrogate(s_path, device)

        correct = 0
        total = 0

        for images, labels in tqdm(testloader, desc="Attacking"):
            images, labels = images.to(device), labels.to(device)
            adv_images = projected_gradient_descent(surrogate_model, images, labels)
            with torch.no_grad():
                outputs = target_model(adv_images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        asr = 100.0 - (100.0 * correct / total)
        print(f"Attack Success Rate: {asr:.2f}%")
        results.append(asr)

    # Plotting
    if len(results) == 4:
        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = ['90/20\n(10% Overlap)', '80/30\n(Small Overlap)', '80/30\n(Large Overlap)', '50/50\n(Disjoint)']
        
        ax.plot(labels, results, marker='o', markersize=8, linestyle='-', linewidth=2.5, color='#c44e52')
        ax.set_ylabel('Attack Success Rate (ASR %)', fontsize=11, fontweight='bold', labelpad=10)
        ax.set_ylim(0, 100)
        ax.set_title('Impact of Dataset Overlap on PGD Transferability', fontsize=12, fontweight='bold', pad=15)

        for i, asr in enumerate(results):
            ax.annotate(f'{asr:.1f}%', (labels[i], results[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig('ablation_transferability_plot.pdf', format='pdf', dpi=300)
        print("\nPlot saved as ablation_transferability_plot.pdf")

if __name__ == "__main__":
    main()