from utils import data_loading, model
import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr as corr
from sklearn.decomposition import PCA

lh, rh, images  = data_loading.load_subject_data(1, 0, 1000)
brain = np.concatenate((lh, rh), axis=1)

# Create dataset with concatenated hemispheres
dataset = data_loading.CustomDataset(images_list= images, outputs_list = brain, transform=transforms.ToTensor(), PCA = PCA(n_components = 100))
print('\nDataset made up of ', len(dataset), 'pairs of data\n--------')
print('Shape of 1st element in pair:', dataset[0][0].shape)
print('Shape of 2nd element in pair:', dataset[0][1].shape)

# Create a train and validation subset of variable dataset with torch
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Put train dataset into a loader with 2 batches and put test data in val loader
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)

# Initialize model, trainer, optimizer and loss function
reg_model = model.get_pretrained_regression_model(100)
trainer = model.Trainer()
optimizer = torch.optim.Adam
loss = torch.nn.MSELoss()
trainer.compile(reg_model, optimizer, learning_rate=0.0001, loss_fn=loss)

trainer.fit(num_epochs = 3, train_loader=train_loader, val_loader=val_loader)