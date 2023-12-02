import sys
import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import data_loading, model
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
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from sklearn.decomposition import PCA
import random

# Create concatenated lists including X samples * 8 subjects
brain_concat = []
images_concat = []
ids_concat = []

for subj in range(1,8+1):
    lh, rh, images, id_list  = data_loading.load_subject_data(subj, None, 100, include_subject_id=True)
    ### TODO
    lh = [fmri[:18978] for fmri in lh]
    rh = [fmri[:20220] for fmri in rh]
    brain_concat.extend(np.concatenate((lh, rh), axis=1)) ### investigate whether concat of lh and rh results in what we want
    images_concat += images
    ids_concat += id_list

# Create dataset with concatenated hemispheres
dataset = data_loading.CustomDataset(images_list = images_concat, outputs_list = brain_concat, id_list = ids_concat, transform=transforms.ToTensor(), PCA = PCA(n_components = 100))
print('\nDataset made up of ', len(dataset), 'truples? of data\n--------')
print('Shape of 1st element:', dataset[0][0].shape)
print('Type of 2nd element:', type(dataset[0][1]))
print('Shape of 3rd element:', dataset[0][2].shape, '\n\n')

# Create a train and validation subset of variable dataset with torch
train_size = int(0.89*len(dataset))
val_size = len(dataset) - train_size

# Use the CustomSubset class for the train and validation subsets
train_dataset = data_loading.CustomSubset(dataset, range(0, train_size), ids_concat[:train_size])
val_dataset = data_loading.CustomSubset(dataset, range(train_size, len(dataset)), ids_concat[train_size:])

# Put train dataset into a loader with 2 batches and put test data in val loader
train_sampler = data_loading.SubjectSampler(train_dataset)
val_sampler = data_loading.SubjectSampler(val_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, sampler=val_sampler)

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
print("Device", device)

# Initialize model, trainer, optimizer and loss function
reg_model = model.ResNet1HeadID(100)
reg_model.to(device)
trainer = model.Trainer()
optimizer = torch.optim.Adam
loss = torch.nn.MSELoss()
trainer.compile(reg_model, optimizer, learning_rate=0.0001, loss_fn=loss)

trainer.fitID(num_epochs = 1, train_loader=train_loader)
# create string with current date and time

trainer.save('trained_model.py')