#this script holds the code to build our model
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet, AlexNet
from tqdm import tqdm
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from datetime import datetime

# Check if GPU is available and if not, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#sub-model used for custom layers in final model below
class LinearSequentialModel(nn.Module):
    def __init__(self, input_size = 512, hidden_size=256):
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
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # Forward pass through the sequential model
        output = self.sequntial_model(x)

        return output

class ResNet1HeadID(nn.Module):
    def __init__(self, output_size, feature_extractor=None, simple_head=False):
        super(ResNet1HeadID, self).__init__()
        
        if feature_extractor is None:
            # Load the pretrained ResNet18 model
            self.feature_extractor = torch.hub.load('utils', 'resnet18', source='local')
        else:
            # Load specified model
            self.feature_extractor = feature_extractor

        # Print model structure
        print(self.feature_extractor)

        # Get input size of head before removing it
        if isinstance(self.feature_extractor, ResNet):
            in_features = self.feature_extractor.fc.in_features
        elif isinstance(self.feature_extractor, AlexNet):
            in_features = self.feature_extractor.classifier[6].in_features
        else:
            raise TypeError('Invalid feature_extractor type')

        # Remove the last fully connected layer (head)
        self.pretrained_model = torch.nn.Sequential(*(list(self.feature_extractor.children())[:-1]))

        # Print model structure after removing the head
        print(self.pretrained_model)

        ### FREEZE PRETRAINED
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Calculate output shape of pretrained model
        dummy_input = torch.randn(1, 3, 224, 224)
        output = self.pretrained_model(dummy_input)
        output_shape = output.shape[1] * output.shape[2] * output.shape[3]
        print(f'Output shape of the model after removing head: {output_shape}') 
        
        if not simple_head:
            # Shared model
            self.shared = LinearSequentialModel(input_size = output_shape, hidden_size=256)
            
            # Subject-specific models
            self.sub1 = LinearSequentialModel(input_size = output_shape, hidden_size=256)
            self.sub2 = LinearSequentialModel(input_size = output_shape, hidden_size=256)
            self.sub3 = LinearSequentialModel(input_size = output_shape, hidden_size=256)
            self.sub4 = LinearSequentialModel(input_size = output_shape, hidden_size=256)
            self.sub5 = LinearSequentialModel(input_size = output_shape, hidden_size=256)
            self.sub6 = LinearSequentialModel(input_size = output_shape, hidden_size=256)
            self.sub7 = LinearSequentialModel(input_size = output_shape, hidden_size=256)
            self.sub8 = LinearSequentialModel(input_size = output_shape, hidden_size=256)
        else:
            # Shared layer
            self.shared = nn.Linear(in_features=output_shape, out_features=256)
            
            # Subject-specific layers
            self.sub1 = nn.Linear(in_features=output_shape, out_features=256)
            self.sub2 = nn.Linear(in_features=output_shape, out_features=256)
            self.sub3 = nn.Linear(in_features=output_shape, out_features=256)
            self.sub4 = nn.Linear(in_features=output_shape, out_features=256)
            self.sub5 = nn.Linear(in_features=output_shape, out_features=256)
            self.sub6 = nn.Linear(in_features=output_shape, out_features=256)
            self.sub7 = nn.Linear(in_features=output_shape, out_features=256)
            self.sub8 = nn.Linear(in_features=output_shape, out_features=256)

        # Combine shared and subject-specific layers
        self.head = nn.Linear(256, output_size)

    def forward(self, x):
        # Extract image and subject ID from the input
        if isinstance(x, tuple):
            images, ids = x
        else:
            images = x
            ids = None

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
        
        # Forward pass through the final linear layer
        output = self.head(combined)

        return output
    
class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.source = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_loss_batch': [], 'val_loss_batch': []}

    def compile(self, model, optimizer, learning_rate, loss_fn):
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay = 0.01)
        self.loss_fn = loss_fn
 
    def fitID(self, num_epochs, train_loader, val_loader=None, patience=5, min_delta=0.0001):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience
        self.best_val_loss = float('inf')
        self.current_patience = 0
        self.epochs_without_improvement = 0
        self.min_delta = min_delta
        figure_num = 0

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

                # Plot the loss
                plt.plot(self.history['train_loss'], label='train_loss')
                plt.plot(self.history['val_loss'], label='val_loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Loss over epochs')
                plt.legend()
                plt.savefig(f'plots/loss_plot{str(figure_num)}.png')
                plt.clf()

                # Check for early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.current_patience = 0
                else:
                    self.current_patience += 1
                    if self.current_patience >= self.patience:
                        print(f"Early stopping after {epoch + 1} epochs.")
                        break

            ### save checkpoint of model and plot every 3 epochs

            if (epoch + 1) % 3 == 0:
                ### MODEL
                try:
                    os.makedirs('../trained_models', exist_ok=True)
                    model_name = f"checkpoint_PCA_{len(outputs)}_SIMPLEHEAD_{self.simple_head}_SAMPLES_{len(self.data_loader.dataset)}_EPOCHS{num_epochs}_BATCHSIZE_{self.data_loader.batch_size}.pt"
                    torch.save(self.model.state_dict(), '../trained_models/' + model_name)
                    print(f'Checkpoint after epoch {epoch+1} saved successfully')
                except Exception as e:
                    print(f"Error saving checkpoint with name: {e}")
                    model_name = f"checkpoint_TIME_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
                    torch.save(self.model.state_dict(), '../trained_models/' + model_name)
                    print('Unnamed checkpoint saved successfully')

                
                try:
                    ### PLOT
                    history = self.history
                    # Plot the loss history
                    plt.plot(history['train_loss'], label='Train Loss')
                    plt.plot(history['val_loss'], label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training History')
                    plt.legend()
                    plt.savefig(f"../trained_models/checkpoint_PCA_{len(outputs)}_SIMPLEHEAD_{self.simple_head}_SAMPLES_{len(self.data_loader.dataset)}_EPOCHS{num_epochs}_BATCHSIZE_{self.data_loader.batch_size}.png")
                except:
                    print('Error generating plot')

    def evaluateID(self, data_loader, mode="Test"):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for images, ids, targets in data_loader:
                # Move the data to the GPU
                images = images.to(device)
                ids = ids.to(device)
                targets = targets.to(device) 

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

    def load(self, filepath, source):
        if source == "cpu":
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model and training history loaded from {filepath}")