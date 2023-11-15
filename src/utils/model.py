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

class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None

    def compile(self, model, optimizer, learning_rate, loss_fn):
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn

    def fit(self, num_epochs, train_loader, val_loader = None):
        self.train_loader = train_loader
        self.val_loader = val_loader

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss}")

            if self.val_loader is not None:
                val_loss = self.evaluate(self.val_loader, "Validation")
                print(f"Validation Loss: {val_loss}")

    def evaluate(self, data_loader, mode="Test"):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"{mode} Loss: {avg_loss}")
        return avg_loss