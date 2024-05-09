import torch
import torchvision

device = torch.device("cuda")
model = torchvision.models.resnet18(pretrained=False).to(device)
model = model.to(device)

print("Model loaded successfully")
print("Device: ", device)