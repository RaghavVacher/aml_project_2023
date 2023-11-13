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
        img_array = np.array(img).transpose((2, 0, 1))
        images.append(img_array)
    return images

def load(subject, index_start=None, index_end=None):
    path = '../../data/algonauts/subj0' + str(subject)
    data_lh = np.load(path + '/training_split/training_fmri/lh_training_fmri.npy')[index_start : index_end]
    data_rh = np.load(path + '/training_split/training_fmri/lh_training_fmri.npy')[index_start : index_end]
    folder_path = path+"/training_split/training_images/"
    image_data = load_images_from_folder(folder_path, index_start, index_end)
    return data_lh, data_rh, image_data

class CustomDataset(Dataset):
    def __init__(self, images_list, outputs_list, transform=None, PCA=None):
        self.num_samples = len(images_list)
        self.transform = transform
        self.PCA = PCA
        self.data, self.output = self.load_data(images_list, outputs_list)

    def load_data(self, images_list, outputs_list):
        data = []
        output_concat = []

        for i in range(self.num_samples):
            # Load image from the given list
            image = Image.fromarray(images_list[i])

            # Load output array from the given list
            output = outputs_list[i]

            data.append((image, output))
            output_concat.append(output)

        if self.PCA:
            self.PCA.fit(output_concat)

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