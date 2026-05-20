import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models.quantization import resnet18 as qresnet18
from tqdm import tqdm

# Configuration and path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
save_dir = "models/resnet18_CIFAR10_QAT"
os.makedirs(save_dir, exist_ok=True)
load_path = "models/resnet18_CIFAR10/resnet18_cifar10_best.pth" # Points to the pre-trained FP32 model

# Data process
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# Updated batch_size to 128 to strictly match the Week 1 protocol constraints
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

# Pretrained quantized resnet18
# qresnet18 with QuantStub/DeQuantStub and FloatFunctional
model = qresnet18(weights=None, num_classes=10)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

try:
    # Load weights onto the target device
    state_dict = torch.load(load_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    print("Pre-trained FP32 weights loaded.")
except:
    print("Warning: Direct load failed, ensure weight keys match.")

# QAT preparation
model.train()
# Conv+BN+ReLU
model.fuse_model()
# fbgemm for x86 (used for inference backend)
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# Prepare QAT (Inserts fake quantization nodes)
torch.quantization.prepare_qat(model, inplace=True)

# Move model to GPU for training
model.to(device)

# QAT optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

epochs_qat = 10
for epoch in range(epochs_qat):
    model.train()
    # Freeze observer statistics in the last few epochs to stabilize convergence
    if epoch > 8:
        model.apply(torch.quantization.disable_observer)
    if epoch > 9:
        model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    for images, labels in tqdm(trainloader, desc=f"QAT Epoch {epoch+1}"):
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Move model back to CPU before saving fake-quant state and converting
model.cpu()

# Save the fake quantized model before conversion, for the use of guided backprop
torch.save(model.state_dict(), os.path.join(save_dir, "resnet18_qat_before_convert.pth"))
print("QAT (float, fake quant) model saved.")

# Convert to int8
model.eval()
int8_model = torch.quantization.convert(model, inplace=False)

torch.save(int8_model.state_dict(), os.path.join(save_dir, "resnet18_int8_final.pth"))
print("INT8 model saved. Ready for ZQBA attack analysis.")