import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm
import torch.nn.functional as F
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_pretrained_regression_model(output_size):
    # Load the pretrained ResNet18 model
    # pretrained_model = models.resnet18(weights='DEFAULT')
    pretrained_model = torch.hub.load('.', 'resnet18', source='local')
    
    # Modify the last fully connected layer for regression with custom output size
    in_features = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(in_features, output_size)
    
    return pretrained_model

class ResNet2HeadModel(nn.Module):
    def __init__(self, output_size):
        super(ResNet2HeadModel, self).__init__()
        
        # Load the pretrained ResNet18 model
        self.pretrained_model = models.resnet18(weights='DEFAULT')
        # in_features = self.pretrained_model.avgpool.in_features
        in_features = 512
        self.pretrained_model = torch.nn.Sequential(*(list(self.pretrained_model.children())[:-1]))
        
        # Remove the last fully connected layer
        # self.pretrained_model.fc = nn.Identity()
        
        # Add two new linear layers for regression with custom output size
        # self.Adj_layer = nn.Linear(1, 512)
        self.fc1 = nn.Linear(in_features, output_size)
        self.fc2 = nn.Linear(in_features, output_size)

    def forward(self, x):
        # Forward pass through the pretrained ResNet18 model
        x = self.pretrained_model(x)
        # x = self.Adj_layer(x)
        x = torch.flatten(x, 1)
        # Forward pass through the first linear layer
        output1 = self.fc1(x)
    
        # Forward pass through the second linear layer
        output2 = self.fc2(x)
        
        return output1, output2

class LinearSequentialModel(nn.Module):
    def __init__(self, input_size=512, hidden_size=256):
        super(LinearSequentialModel, self).__init__()

        # Sequential model
        self.sequntial_model = nn.Sequential(
            # Block 1
            nn.Linear(input_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            # Block 2
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            # Block 3
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            # Block 4
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),

            # dropout
            nn.Dropout(0.25)
        )

    def forward(self, x):
        # Forward pass through the sequential model
        output = self.sequntial_model(x)

        return output

class ResNet1HeadID(nn.Module):
    def __init__(self, output_size):
        super(ResNet1HeadID, self).__init__()
        
        # Load the pretrained ResNet18 model
        # self.pretrained_model = models.resnet18(weights='DEFAULT')
        self.pretrained_model = torch.hub.load('utils', 'resnet18', source='local')
        in_features = 512 # in_features for first layer after CNN

        # Remove the last fully connected layer
        self.pretrained_model = torch.nn.Sequential(*(list(self.pretrained_model.children())[:-1]))
        
        # Add shared layer
        self.shared = LinearSequentialModel(input_size=in_features, hidden_size=256)

        # Add subject-specific layers
        self.sub1 = LinearSequentialModel(input_size=in_features, hidden_size=256)
        self.sub2 = LinearSequentialModel(input_size=in_features, hidden_size=256)
        self.sub3 = LinearSequentialModel(input_size=in_features, hidden_size=256)
        self.sub4 = LinearSequentialModel(input_size=in_features, hidden_size=256)
        self.sub5 = LinearSequentialModel(input_size=in_features, hidden_size=256)
        self.sub6 = LinearSequentialModel(input_size=in_features, hidden_size=256)
        self.sub7 = LinearSequentialModel(input_size=in_features, hidden_size=256)
        self.sub8 = LinearSequentialModel(input_size=in_features, hidden_size=256)

        # Combine shared and subject-specific layers
        self.head = nn.Linear(256, output_size)

    def forward(self, x):
        # Extract image and subject ID from the input
        if isinstance(x, tuple):
            images, ids = x
        else:
            images = x

        # Forward pass through the pretrained ResNet18 model
        features = self.pretrained_model(images)

        # Flatten the features
        flat_features = torch.flatten(features, 1)

        # Forward pass through the shared layer
        shared = self.shared(flat_features)

        # Forward pass through the subject-specific layers if subject ID is given
        if ids != None:
            if ids[0] == 1:
                subject = self.sub1(flat_features)
            elif ids[0] == 2:
                subject = self.sub2(flat_features)
            elif ids[0] == 3:
                subject = self.sub3(flat_features)
            elif ids[0] == 4:
                subject = self.sub4(flat_features)
            elif ids[0] == 5:
                subject = self.sub5(flat_features)
            elif ids[0] == 6:
                subject = self.sub6(flat_features)
            elif ids[0] == 7:
                subject = self.sub7(flat_features)
            elif ids[0] == 8:
                subject = self.sub8(flat_features)

            # Average the shared and subject-specific layers
            combined = (shared + subject) / 2
        
        else:
            combined = shared
        
        # Forward pass through the first linear layer
        output = self.head(combined)

        return output

class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.history = {'train_loss': [], 'val_loss': []}

    def compile(self, model, optimizer, learning_rate, loss_fn):
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn

    def fit(self, num_epochs, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for inputs, targets in self.train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                    # Update the progress bar
                    pbar.update(1)

            avg_loss = total_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_loss)
            print(f"> Training Loss: {avg_loss}")

            if self.val_loader is not None:
                val_loss = self.evaluate(self.val_loader, "Validation")
                self.history['val_loss'].append(val_loss)

    def evaluate(self, data_loader, mode="Test"):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"> {mode} Loss: {avg_loss}")
        return avg_loss
    
    def fit_dual_head(self, num_epochs, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for (inputs, targets_head1, targets_head2) in self.train_loader:
                print('batch')
                self.optimizer.zero_grad()
                outputs_head1, outputs_head2 = self.model(inputs)
                loss_head1 = self.loss_fn(outputs_head1, targets_head1)
                loss_head2 = self.loss_fn(outputs_head2, targets_head2)
                total_loss += (loss_head1 + loss_head2).item()

                # You can choose to backpropagate on each head separately or sum the losses and backpropagate once.
                (loss_head1 + loss_head2).backward()

                self.optimizer.step()

            avg_loss = total_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss}")

            if self.val_loader is not None:
                val_loss = self.evaluate_dual_head(self.val_loader, "Validation")
                self.history['val_loss'].append(val_loss)


    def evaluate_dual_head(self, data_loader, mode="Test"):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets_head1, targets_head2 in data_loader:
                outputs_head1, outputs_head2 = self.model(inputs)
                loss_head1 = self.loss_fn(outputs_head1, targets_head1)
                loss_head2 = self.loss_fn(outputs_head2, targets_head2)
                total_loss += (loss_head1 + loss_head2).item()

        avg_loss = total_loss / len(data_loader)
        print(f"{mode} Loss: {avg_loss}")
        return avg_loss
    
    def fitID(self, num_epochs, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for images, subject_ids, targets in self.train_loader:
                    images = images.to(device)
                    subject_ids = subject_ids.to(device)
                    targets = targets.to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model((images, subject_ids))
                    loss = self.loss_fn(outputs, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                    # Update the progress bar
                    pbar.update(1)
            avg_loss = total_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_loss)
            print(f"Training Loss: {avg_loss}")

            if self.val_loader is not None:
                val_loss = self.evaluateID(self.val_loader, "Validation")
                self.history['val_loss'].append(val_loss)

    def evaluateID(self, data_loader, mode="Test"):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, ids, targets in data_loader:
                outputs = self.model((images, ids))
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"{mode} Loss: {avg_loss}")
        return avg_loss
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model and training history saved to {filepath}")

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model and training history loaded from {filepath}")
