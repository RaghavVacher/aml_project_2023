import torch

# load a trained .pt file for analysis
model = torch.load('resnet50epochs.pt', map_location=torch.device('cpu'))

print(model)

