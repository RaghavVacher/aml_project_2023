import torch
import torch.nn as nn
import torchvision.models as models

def get_pretrained_regression_model(output_size):
    # Load the pretrained ResNet18 model
    pretrained_model = models.resnet18(weights='DEFAULT')
    
    # Modify the last fully connected layer for regression with custom output size
    in_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(in_features, output_size)
    
    return pretrained_model
