import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from torchvision.models.quantization import resnet18 as qresnet18
from train_baselines import get_baseline_model

def projected_gradient_descent(model, images, labels, eps=8/255, alpha=2/255, steps=7):
    """
    Generates adversarial examples using PGD on the Surrogate.
    """
    images = images.clone().detach().to(images.device)
    labels = labels.clone().detach().to(images.device)
    adv_images = images.clone().detach()
    
    # Standard PGD loop
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        
        # Calculate gradients on the Fake-Quantized model
        model.zero_grad()
        loss.backward()

        with torch.no_grad():
            # Step in the direction of the gradient sign
            adv_images = adv_images + alpha * adv_images.grad.sign()
            # Project back to the epsilon-ball around the original image
            eta = torch.clamp(adv_images - images, min=-eps, max=eps)
            # Ensure the pixel values remain valid for an image [0, 1]
            adv_images = torch.clamp(images + eta, min=0, max=1)
            adv_images = adv_images.detach()

    return adv_images

def load_fake_quant_surrogate(path, device):
    """
    Loads the QAT model BEFORE it was converted to INT8.
    This preserves the Straight-Through Estimator (STE) for backpropagation.
    """
    model = qresnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # MUST be in train mode for prepare_qat to attach observers
    model.train() 
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    
    # Load the float fake-quant state, DO NOT convert to int8
    model.load_state_dict(torch.load(path, map_location=device))
    
    # Switch to eval mode AFTER loading for the actual attack
    model.eval()
    return model.to(device)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing PGD attack on device: {device}")

    # Paths
    target_path = "models/resnet34_CIFAR10/resnet34_cifar10_best.pth"
    surrogate_path = "models/resnet18_CIFAR10_QAT/resnet18_qat_before_convert.pth"
    
    if not os.path.exists(target_path):
        print(f"Error: Target missing at {target_path}")
        return
    if not os.path.exists(surrogate_path):
        print(f"Error: Surrogate missing at {surrogate_path}. Did you re-run the QAT script?")
        return

    # Load Models
    print("Loading FP32 Target (ResNet-34)...")
    target_model = get_baseline_model("resnet34", num_classes=10).to(device)
    target_model.load_state_dict(torch.load(target_path, map_location=device))
    target_model.eval()

    print("Loading Fake-Quant Surrogate (ResNet-18)...")
    surrogate_model = load_fake_quant_surrogate(surrogate_path, device)
    surrogate_model.eval()

    # Data setup
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    # Metrics
    clean_correct = 0
    adv_correct = 0
    total = 0

    print("Initiating PGD Transfer Evaluation...")
    progress_bar = tqdm(testloader, desc="Attacking Batches")
    
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # 1. Evaluate clean accuracy on Target
        with torch.no_grad():
            clean_outputs = target_model(images)
            _, clean_preds = clean_outputs.max(1)
            clean_correct += clean_preds.eq(labels).sum().item()
            total += labels.size(0)

        # 2. Generate Adversarial Examples using the Surrogate
        adv_images = projected_gradient_descent(surrogate_model, images, labels, eps=8/255, alpha=2/255, steps=7)

        # 3. Evaluate adversarial examples on the Target
        with torch.no_grad():
            adv_outputs = target_model(adv_images)
            _, adv_preds = adv_outputs.max(1)
            adv_correct += adv_preds.eq(labels).sum().item()

    clean_acc = 100. * clean_correct / total
    adv_acc = 100. * adv_correct / total
    asr = 100. - adv_acc  # Attack Success Rate (assuming targeted misclassification)

    print("\n" + "="*40)
    print(" PGD Transfer Attack Results (Method A)")
    print("="*40)
    print(f" Target Model        : ResNet-34 (FP32)")
    print(f" Surrogate Model     : ResNet-18 (Fake-Quant)")
    print(f" Clean Target Acc    : {clean_acc:.2f}%")
    print(f" Attacked Target Acc : {adv_acc:.2f}%")
    print(f" Attack Success Rate : {asr:.2f}%")
    print("="*40)

if __name__ == "__main__":
    main()