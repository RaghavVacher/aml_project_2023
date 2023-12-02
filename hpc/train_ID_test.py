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

# Check if GPU is available and if not, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
n_components = 10 # PCA components
num_epochs = 2 # Number of epochs to train
batch_size = 16 # Batch size
learning_rate = 0.0001 # Learning rate
feature_extractor = models.resnet18(weights='DEFAULT') # CNN to use for feature extraction
# feature_extractor.to(device) # Move CNN to GPU if available

optimizer = torch.optim.Adam
loss = torch.nn.MSELoss()

# Create concatenated lists including X samples * 8 subjects
brain_concat = []
images_concat = []
ids_concat = []

for subj in range(1,2):
    lh, rh, images, id_list  = data_loading.load_subject_data(subj, 0, 100, include_subject_id=True)
    brain_concat.extend(np.concatenate((lh, rh), axis=1)) ### investigate whether concat of lh and rh results in what we want
    images_concat += images
    ids_concat += id_list

# Create dataset with concatenated hemispheres
dataset = data_loading.CustomDataset(images_list = images_concat, outputs_list = brain_concat, id_list = ids_concat, transform=transforms.ToTensor(), PCA = PCA(n_components = n_components))
print('\nDataset made up of ', len(dataset), 'truples? of data\n--------')
print('Shape of 1st element:', dataset[0][0].shape)
print('Type of 2nd element:', type(dataset[0][1]))
print('Shape of 3rd element:', dataset[0][2].shape, '\n\n')

# Create a train and validation subset of variable dataset with torch
train_size = int(len(dataset))
# val_size = len(dataset) - train_size

# Use the CustomSubset class for the train and validation subsets
train_dataset = data_loading.CustomSubset(dataset, range(0, train_size), ids_concat[:train_size])
# val_dataset = data_loading.CustomSubset(dataset, range(train_size, len(dataset)), ids_concat[train_size:])

# Put train dataset into a loader with 2 batches and put test data in val loader
train_sampler = data_loading.SubjectSampler(train_dataset)
# val_sampler = data_loading.SubjectSampler(val_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

# Initialize model, trainer, optimizer and loss function
reg_model = model.ResNet1HeadID(n_components, feature_extractor)
trainer = model.Trainer()
trainer.compile(reg_model, optimizer, learning_rate=learning_rate, loss_fn=loss)

trainer.fitID(num_epochs = num_epochs, train_loader=train_loader)

trainer.save('trained_model.pt')