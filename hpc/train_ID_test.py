import sys
import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import data_loading, model_test as model
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

# Parse the arguments
args = parser.parse_args()

# Now you can use args.gpu and args.epochs in your script
print(f"Training for {args.epochs} epochs.")
print(f"Using {args.model} as the model.")
print(f"Using {args.learning_rate} as the learning rate.")
print(f"Using {args.samples} samples per subject.")

#############

# Check if GPU is available and if not, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
n_components = 100 # PCA components
num_epochs = args.epochs # Number of epochs to train
if (args.samples == 'all'):
    num_samples = None # Number of samples per subject (None - all samples)
else:
    num_samples = int(args.samples) # Number of samples per subject (None - all samples)
n_subjects = 8 # Number of subjects to train on
batch_size = 16 # Batch size
learning_rate = args.learning_rate # Learning rate
feature_extractor = torch.hub.load('utils', args.model, source='local') # CNN to use for feature extraction
optimizer = torch.optim.Adam
loss = torch.nn.MSELoss()

# Create concatenated lists including X samples * 8 subjects
brain_concat = []
images_concat = []
ids_concat = []

for subj in range(1,n_subjects+1):
    lh, rh, images, id_list  = data_loading.load_subject_data(subj, 0, num_samples, include_subject_id=True)
    ### TODO
    lh = [fmri[:18978] for fmri in lh]
    rh = [fmri[:20220] for fmri in rh]
    
    brain_concat.extend(np.concatenate((lh, rh), axis=1)) ### investigate whether concat of lh and rh results in what we want
    images_concat += images
    ids_concat += id_list
#Data Aug
transforms_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224,224), antialias=True),
    transforms.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])
])
# Create dataset with concatenated hemispheres
dataset = data_loading.CustomDataset(images_list = images_concat, outputs_list = brain_concat, id_list = ids_concat, transform=transforms_image, PCA = PCA(n_components = n_components))
print('\nDataset made up of ', len(dataset), 'truples? of data\n--------')
print('Shape of 1st element:', dataset[0][0].shape)
print('Type of 2nd element:', type(dataset[0][1]))
print('Shape of 3rd element:', dataset[0][2].shape, '\n\n')

# Create a train and validation subset of variable dataset with torch
train_size = int(0.89 * len(dataset))
val_size = len(dataset) - train_size

# Use the CustomSubset class for the train and validation subsets
train_dataset = data_loading.CustomSubset(dataset, range(0, train_size), ids_concat[:train_size])
val_dataset = data_loading.CustomSubset(dataset, range(train_size, len(dataset)), ids_concat[train_size:])

# Put train dataset into a loader with 2 batches and put test data in val loader
train_sampler = data_loading.SubjectSampler(train_dataset)
val_sampler = data_loading.SubjectSampler(val_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

# Initialize model and trainer
reg_model = model.ResNet1HeadID(n_components, feature_extractor)
reg_model.to(device)
trainer = model.Trainer()
trainer.compile(reg_model, optimizer, learning_rate=learning_rate, loss_fn=loss)

# Train model and save
trainer.fitID(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader)
# define the name for trained model based on set parameters and date
try:
    model_name = f"trained_model_{args.model}_{args.learning_rate}_{args.samples}_{args.epochs}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    trainer.save(model_name)
except:
    trainer.save('trained_model.pt')