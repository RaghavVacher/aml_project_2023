#script to load and preprocess data so that it can be passed to train the model
from torch.utils.data import Dataset, Sampler, Subset
from sklearn.decomposition import PCA 
import numpy as np
from PIL import Image
import torch
import os

def load_images_from_folder(folder_path, start=None, end=None):
    image_paths = []
    for filename in os.listdir(folder_path)[start:end]:
        img_path = os.path.join(folder_path, filename)
        image_paths.append(img_path)
    return image_paths

def load_subject_data(subject, index_start=None, index_end=None, return_dict=False, include_subject_id = False, pca_components=100):
    if(include_subject_id):
        print('\n\n----------------\nLoading subject data with subject ID', str(subject) + '...')
    else:
        print('\n\n----------------\nLoading subject data...')
    current_proj_dir = os.getcwd().split('hpc')[0] + 'hpc'
    print('Current project directory: %s' % current_proj_dir)
    path = current_proj_dir + '/data/training_split/subj0' + str(subject)
    if pca_components == 0: #if no PCA, truncate longer fmri arrays
        data_lh = np.load(path + '/training_fmri/lh_train_fmri.npy')[index_start : index_end]
        data_rh = np.load(path + '/training_fmri/rh_train_fmri.npy')[index_start : index_end]
        data_lh = [fmri[:18978] for fmri in data_lh]
        data_rh = [fmri[:20220] for fmri in data_rh]
        pca_brain = np.concatenate((data_lh, data_rh), axis = 1)
        print('Shape of pca_brain: ', pca_brain.shape)
    else:
        if pca_components == 100:
            pca_suffix = ""
        else:
            pca_suffix = "_" + str(pca_components)
        pca_brain = np.load(path + f'/training_fmri/pca_brain{pca_suffix}.npy')[index_start : index_end]
    folder_path = path+"/training_images/"
    image_paths = load_images_from_folder(folder_path, index_start, index_end)
    id_list = [subject for i in range(len(image_paths))]
    
    if include_subject_id:
        return pca_brain, image_paths, id_list
    else:
        return pca_brain, image_paths

class CustomDataset(Dataset):
    def __init__(self, images_list, outputs_list, transform=None, PCA=None, id_list=None):
        self.num_samples = len(images_list)
        print('\n----------------\nInitialize CustomDataset\n')
        print('Number of samples: ', self.num_samples)
        self.transform = transform
        print('Transform: ', self.transform)
        self.PCA = PCA
        print('PCA: ', self.PCA)
        self.id_list = id_list
        self.data, self.output = self.load_data(images_list, outputs_list, id_list)

    def load_data(self, images_list, outputs_list, id_list=None):
        data = []
        output_concat = []

        for i in range(self.num_samples):
            # Load image from the given list
            image = images_list[i]

            output = outputs_list[i]
           
            if id_list:
                id = id_list[i]
                data.append(([image, id], output))
            else:
                data.append((image, output))
                
            output_concat.append(output)
        
        if self.PCA:
            self.PCA.fit(output_concat)
            
        print('-------\nData loaded\n')
        print('Data: ', len(data), '* ', data[0])
        print('Output_concat: ', len(output_concat), '*', len(output_concat[0]), ': ', output_concat[0])
        return data, outputs_list

    def give_output(self):
        return self.output

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.id_list:
            (image_path, subject_id), output = self.data[idx]
            image = np.load(image_path)
        else:
            image_path, output = self.data[idx]
            image = np.load(image_path)

        if self.transform:
            image = self.transform(image)

        if self.PCA:
            output = self.PCA.transform(output.reshape(1, -1))
        else:
            output = output.reshape(1, -1)

        if self.id_list:
            return image, subject_id, torch.FloatTensor(output[0])
        else:
            return image, torch.FloatTensor(output[0])
        
# Custom subset class to preserve .id_list attribute when splitting into train and val
class CustomSubset(Subset):
    def __init__(self, dataset, indices, id_list):
        super(CustomSubset, self).__init__(dataset, indices)
        self.id_list = id_list
        
class SubjectSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.num_samples = len(dataset)

    def __iter__(self):
        np.random.shuffle(self.indices)  # ensures epochs do not have the same order
        for subject_id in set(self.dataset.id_list):
            subject_indices = [i for i in self.indices if self.dataset.id_list[i] == subject_id]
            for id in subject_indices:
                yield id

    def __len__(self):
        return self.num_samples