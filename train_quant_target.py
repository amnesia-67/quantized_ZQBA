import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.ao.quantization import get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from tqdm import tqdm

try:
    from torch.ao.quantization import disable_observer
except Exception:
    from torch.quantization import disable_observer

'''
Author  : Anish Subash
Purpose : QAT INT8 ResNet-34 target for the quantized-vs-quantized transfer study.
          torchvision has no eager quantizable resnet34, so the FP32 target is
          quantized via FX graph-mode QAT (auto conv-bn-relu fusion + fake-quant).
          Saved checkpoint is the PREPARED (fake-quant / STE) state_dict, so the
          same file feeds both GPU fake-quant eval and convert_fx INT8 eval.
'''

CIFAR_EXAMPLE = (torch.randn(1, 3, 32, 32),)


def build_resnet34(num_classes=10):
    m = models.resnet34()
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_qat_resnet34(backend="fbgemm", num_classes=10, base_weight=None):
    m = build_resnet34(num_classes)
    if base_weight:
        # load FP32 weights BEFORE prepare so they carry into the fused graph
        m.load_state_dict(torch.load(base_weight, map_location="cpu"))
    m.train()
    mapping = get_default_qat_qconfig_mapping(backend)
    return prepare_qat_fx(m, mapping, CIFAR_EXAMPLE)


def main():
    parser = argparse.ArgumentParser(description="QAT INT8 ResNet-34 target trainer (FX graph mode)")
    parser.add_argument('--split_file', required=True, help="PyTorch index list (use the target's fixed 40k split)")
    parser.add_argument('--save_name', required=True, help="Output filename")
    parser.add_argument('--base_weight', default=None, help="FP32 ResNet-34 target weights to QAT fine-tune from")
    parser.add_argument('--backend', default='fbgemm', choices=['fbgemm', 'x86', 'qnnpack'])
    parser.add_argument('--epochs', type=int, default=None, help="Default: 10 if fine-tuning, 60 if from scratch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "models/ablation_study"
    os.makedirs(save_dir, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])

    full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    indices = torch.load(args.split_file)
    trainset = torch.utils.data.Subset(full_trainset, indices)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    from_scratch = args.base_weight is None
    epochs = args.epochs if args.epochs is not None else (60 if from_scratch else 10)
    print(f"[TARGET-QAT] {len(indices)} imgs | backend={args.backend} | "
          f"base={'scratch' if from_scratch else args.base_weight} | epochs={epochs} | {device}")

    model = build_qat_resnet34(backend=args.backend, base_weight=args.base_weight)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.85)], gamma=0.1)
        if from_scratch else None)
    criterion = nn.CrossEntropyLoss()

    # QAT observer/BN freeze schedule over the final epochs
    disable_obs_at = max(epochs - 3, 0)
    freeze_bn_at = max(epochs - 2, 0)

    for epoch in range(epochs):
        model.train()
        if epoch >= disable_obs_at:
            model.apply(disable_observer)
        if epoch >= freeze_bn_at:
            model.apply(lambda mod: mod.freeze_bn_stats() if hasattr(mod, "freeze_bn_stats") else None)

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [target_qat]", leave=False)
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

    model.cpu()
    save_path = os.path.join(save_dir, args.save_name)
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] QAT target (fake-quant state_dict) saved to {save_path}\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()