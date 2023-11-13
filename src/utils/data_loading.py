import numpy as np
from PIL import Image
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