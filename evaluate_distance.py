import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Import the metric hooks we created
from eval_metrics import calc_kl_divergence, calc_l2_distance
from train_baselines import get_baseline_model

def load_fp32_target(arch, path, device):
    model = get_baseline_model(arch, num_classes=10)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)

def load_qat_surrogate(path, device):
    # The surrogate used by Changjun is a quantized resnet18 structure
    from torchvision.models.quantization import resnet18 as qresnet18
    model = qresnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    
    # MUST be in train mode for prepare_qat
    model.train()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    torch.quantization.prepare_qat(model, inplace=True)
    int8_model = torch.quantization.convert(model, inplace=False)
    
    int8_model.load_state_dict(torch.load(path, map_location=device))
    
    # Switch to eval mode AFTER loading for safe inference
    int8_model.eval()
    return int8_model.to(device)

def main():
    # PyTorch official quantization inference runs strictly on CPU
    device = torch.device("cpu")
    
    # Paths to your freshly trained models
    target_path = "models/resnet34_CIFAR10/resnet34_cifar10_best.pth"
    surrogate_path = "models/resnet18_CIFAR10_QAT/resnet18_int8_final.pth"
    
    if not os.path.exists(target_path) or not os.path.exists(surrogate_path):
        print("Error: Ensure both the target weights and surrogate weights exist.")
        return

    print("Loading models onto CPU for quantization-safe inference...")
    target_model = load_fp32_target("resnet34", target_path, device)
    surrogate_model = load_qat_surrogate(surrogate_path, device)

    # Dataset Loader (No normalization, matching baseline training)
    transform_test = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    total_kl = 0.0
    batches = 0

    print("\nEvaluating behavioral distance (KL-Divergence) over the test set...")
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            
            # Extract logits from both models
            target_logits = target_model(images)
            surrogate_logits = surrogate_model(images)
            
            # Calculate batch KL-Divergence
            batch_kl = calc_kl_divergence(target_logits, surrogate_logits)
            total_kl += batch_kl
            batches += 1

    avg_kl = total_kl / batches
    
    # Parametric Distance (Note: shapes differ between RN34 and RN18, 
    # so calc_l2_distance will safely compare identical matching layers)
    l2_dist = calc_l2_distance(target_model, surrogate_model)

    print("\n" + "="*40)
    print(" Target-Surrogate Distance Report (Week 2)")
    print("="*40)
    print(f" Target Model    : ResNet-34 (FP32)")
    print(f" Surrogate Model : ResNet-18 (INT8 QAT)")
    print(f" Behavioral Distance (Avg KL-Div) : {avg_kl:.6f}")
    print(f" Parametric Distance (Shared L2)  : {l2_dist:.6f}")
    print("="*40)

if __name__ == "__main__":
    main()