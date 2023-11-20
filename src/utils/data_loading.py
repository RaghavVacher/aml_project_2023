from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA 
from torchvision import transforms
import numpy as np
from PIL import Image
import torch
import os

def load_images_from_folder(folder_path, start=None, end=None):
    images = []
    for filename in os.listdir(folder_path)[start:end]:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)
        img_array = np.array(img)
        # .transpose((2, 0, 1))
        images.append(img_array)
    return images

def load_subject_data(subject, index_start=None, index_end=None, return_dict=False):
    path = '../../data/algonauts/subj0' + str(subject)
    data_lh = np.load(path + '/training_split/training_fmri/lh_training_fmri.npy')[index_start : index_end]
    data_rh = np.load(path + '/training_split/training_fmri/rh_training_fmri.npy')[index_start : index_end]
    folder_path = path+"/training_split/training_images/"
    image_data = load_images_from_folder(folder_path, index_start, index_end)
    
    if return_dict:
        subject_data = {'subject_id': subject,
                        'data_lh': data_lh,
                        'data_rh': data_rh,
                        'image_data': image_data}
        return subject_data
    else:
        return data_lh, data_rh, image_data

class CustomDataset(Dataset):
    def __init__(self, images_list, outputs_list, transform=None, PCA=None):
        self.num_samples = len(images_list)
        print('       \nInitialize CustomDataset \n--------')
        print('Number of samples: ', self.num_samples)
        self.transform = transform
        print('Transform: ', self.transform)
        self.PCA = PCA
        print('PCA: ', self.PCA)
        self.data, self.output = self.load_data(images_list, outputs_list)

    def load_data(self, images_list, outputs_list):
        data = []
        output_concat = []

        for i in range(self.num_samples):
            # Load image from the given list
            image = Image.fromarray(images_list[i])

            output = outputs_list[i]

            data.append((image, output))
            output_concat.append(output)
        
        if self.PCA:
            self.PCA.fit(output_concat)
            
        print('\nData loaded\n-------')
        print('Data: ', len(data), '* ', data[0])
        print('Output_concat: ', len(output_concat), '*', len(output_concat[0]), ': ', output_concat[0])
        return data, output_concat

    def give_output(self):
        return self.output

    def internal_PCA(self):
        if self.PCA:
            return self.PCA
        else:
            print(f'PCA is {self.PCA}')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, output = self.data[idx]

        if self.transform:
            image = self.transform(image)
        if self.PCA:
            output = self.PCA.transform(output.reshape(1, -1))

        return image, torch.FloatTensor(output[0])