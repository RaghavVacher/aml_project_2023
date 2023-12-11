import sys
import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import data_loading, final_model as model
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torchvision.models as models
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from sklearn.decomposition import PCA
import random
import argparse
from datetime import datetime

###### Take in bash arguments

# Create the parser
parser = argparse.ArgumentParser(description='Training script')

# Add arguments
parser.add_argument('epochs', type=int, help='Number of epochs')
parser.add_argument('model', type=str, help='Chosen model')
parser.add_argument('learning_rate', type=float, help='Learning rate')
parser.add_argument('samples', type=str, help='Learning rate')
parser.add_argument('batch_size', type=int, help='Batch size')
parser.add_argument('patience', type=int, help='Patience')
parser.add_argument('pca_components', type=int, help='PCA Components')
parser.add_argument('simple_head', type=bool, help='Simple head')

# Parse the arguments
args = parser.parse_args()

# Now you can use args.gpu and args.epochs in your script
print(f"Training for {args.epochs} epochs.")
print(f"Using {args.model} as the model.")
print(f"Using {args.learning_rate} as the learning rate.")
print(f"Using {args.samples} samples per subject.")
print(f"Using {args.batch_size} as the batch size.")
print(f"Using {args.patience} as the patience.")
print(f"Using {args.pca_components} as the number of PCA components.")
print(f"Using simple head: {args.simple_head}")

#############

# Check if GPU is available and if not, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
n_components = args.pca_components # PCA components
num_epochs = args.epochs # Number of epochs to train
if (args.samples == 'all'):
    num_samples = None # Number of samples per subject (None - all samples)
else:
    num_samples = int(args.samples) # Number of samples per subject (None - all samples)
n_subjects = 8 # Number of subjects to train on
batch_size = args.batch_size # Batch size
learning_rate = args.learning_rate # Learning rate
if(args.model == 'resnet18'):
    feature_extractor = torch.hub.load('utils', args.model, source='local') # CNN to use for feature extraction
else:
    feature_extractor = torch.load('utils/pretrained_'+ args.model +'.pt')
optimizer = torch.optim.Adam
loss = torch.nn.MSELoss()
patience = args.patience # Number of epochs without improvement to wait before early stopping
min_delta = 0.001 # Minimum change in loss to be considered an improvement

# Create concatenated lists including X samples * 8 subjects
brain_concat = []
images_concat = []
ids_concat = []

for subj in range(1,n_subjects+1):
    pca_brain, images, id_list  = data_loading.load_subject_data(subj, 0, num_samples, include_subject_id=True, pca_components=n_components)
    ### TODO
    # lh = [fmri[:18978] for fmri in lh]
    # rh = [fmri[:20220] for fmri in rh]
    
    brain_concat.extend(pca_brain) ### investigate whether concat of lh and rh results in what we want
    images_concat += images
    ids_concat += id_list
#Data Aug
transforms_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),
    transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])
])
# Create dataset with concatenated hemispheres
dataset = data_loading.CustomDataset(images_list = images_concat, outputs_list = brain_concat, id_list = ids_concat, transform=transforms_image, PCA = None)
print('\nDataset made up of ', len(dataset), 'truples? of data\n--------')
print('Shape of 1st element:', dataset[0][0].shape)
print('Type of 2nd element:', type(dataset[0][1]))
print('Shape of 3rd element:', dataset[0][2].shape, '\n\n')

# Create a list of indices from 0 to the length of the dataset
indices = list(range(len(dataset)))

# Shuffle the indices
np.random.shuffle(indices)

# Create a train and validation subset of variable dataset with torch
train_size = int(0.89 * len(dataset))
val_size = len(dataset) - train_size

# Split the indices into train and validation sets
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Use the CustomSubset class for the train and validation subsets
ids_concat = np.array(ids_concat)
train_dataset = data_loading.CustomSubset(dataset, train_indices, ids_concat[train_indices])
val_dataset = data_loading.CustomSubset(dataset, val_indices, ids_concat[val_indices])

# Put train dataset into a loader with 2 batches and put test data in val loader
train_sampler = data_loading.SubjectSampler(train_dataset)
val_sampler = data_loading.SubjectSampler(val_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

# Initialize model and trainer
if n_components == 0:
    n_components = 39198
reg_model = model.ResNet1HeadID(n_components, feature_extractor, simple_head=args.simple_head)
reg_model.to(device)
trainer = model.Trainer()
trainer.compile(reg_model, optimizer, learning_rate=learning_rate, loss_fn=loss)

# Train model and save
trainer.fitID(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, patience=patience, min_delta=min_delta)
# define the name for trained model based on set parameters and date
try:
    os.makedirs('trained_models', exist_ok=True)
    model_name = f"{args.model}_LR{args.learning_rate}_SAMPLES_{args.samples}_EPOCHS{args.epochs}_BATCHSIZE_{args.batch_size}_TIME_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    trainer.save('trained_models/'+model_name)
except:
    model_name = f"trained_model_{args.model}_LR{args.learning_rate}_SAMPLES_{args.samples}_EPOCHS{args.epochs}_TIME_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    trainer.save(model_name)

# Access the history
train_loss = trainer.history['train_loss']
val_loss = trainer.history['val_loss']

# plot the loss over epochs
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Loss ov. epochs | {args.model} | LR = {args.learning_rate} \n {args.samples}/subj | PCA = {args.pca_components} | Simple head={args.simple_head}')
plt.legend()

# Ensure the directory exists
os.makedirs('plots', exist_ok=True)

# Save the plot as image
plt.savefig(f'plots/loss_{args.model}_LR{args.learning_rate}_SAMPLES_{args.samples}_EPOCHS{args.epochs}_TIME_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.png')