#%%
import sys
sys.path.append('../')
#%%
import torch
import torch.nn as nn
import torchvision.models as models
from utils import data_loading
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA

#%% 
class SubjID2Head(nn.Module):
    def __init__(self, output_size):
        super(SubjID2Head, self).__init__()
        
        # Load the pretrained ResNet18 model
        self.pretrained_model = models.resnet18(pretrained=True)
        in_features = self.pretrained_model.fc.in_features
        
        # Remove the last fully connected layer
        self.pretrained_model.fc = nn.Identity()

        # Embedding layer
        self.embedding = nn.Embedding(8, 1) # 8 subjects, 1 dimension??
        
        # Fully connected layers after concatenating the embedding with the flattened features
        self.fc = nn.Sequential(
            nn.Linear(in_features + 1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Add two new linear layers for regression with custom output size
        self.head1 = nn.Linear(256, output_size)
        self.head2 = nn.Linear(256, output_size)

    def forward(self, x):
        if type(x) == dict:
            subject_id = x['subject_id']
            x = x['image_data']
        
        # Forward pass through the pretrained ResNet18 model
        x = self.pretrained_model(x)
        x = torch.flatten(x, 1)
        

        # Forward pass through the embedding layer
        subject_id = self.embedding(subject_id)

        # Concatenate the subject ID embedding with the flattened features
        x = torch.cat((x, subject_id), dim=1)
                
        # Forward pass through the fully connected layers
        x = self.fc(x)

        # Forward pass through head 1
        output1 = self.head1(x)
    
        # Forward pass through head 2
        output2 = self.head2(x)
        
        return output1, output2

#%%
class CustomDataset2(Dataset):
    def __init__(self, subject_data_list, transform=None, PCA1=None, PCA2=None):
        self.num_samples = len(subject_data_list)
        print('Number of samples:', self.num_samples)
        self.transform = transform
        self.PCA1 = PCA1
        self.PCA2 = PCA2
        self.data, self.output1, self.output2 = self.load_data(subject_data_list)

    def load_data(self, subject_data_list):
        data = []
        output_concat1 = []
        output_concat2 = []

    for i in range(self.num_samples):
        subject_data = subject_data_list[i]

        # For each subject in the list, extract the image data from the image_data list in the dictionary
        for image_data in subject_data['image_data']:
            # Load image from the image_data
            image = Image.fromarray(image_data)

            # Load output arrays from the subject_data dictionary
            output1 = subject_data['data_lh']
            output2 = subject_data['data_rh']

            data.append({'images': image, 'subject_ids': subject_data['subject_id'], 'outputs1': output1, 'outputs2': output2})
            output_concat1.append(output1)
            output_concat2.append(output2)

        if self.PCA1:
            self.PCA1.fit(output_concat1)

        if self.PCA2:
            self.PCA2.fit(output_concat2)

        print('Output_concat1:', len(output_concat1[0]))
        print('Output_concat2:', len(output_concat2[0]))
        
        return data, output_concat1, output_concat2

    def give_output(self):
        return self.output1, self.output2

    def get_PCA1(self):
        return self.PCA1

    def get_PCA2(self):
        return self.PCA2

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample['images'] = self.transform(sample['images'])

        if self.PCA1:
            sample['outputs1'] = self.PCA1.transform(sample['outputs1'].reshape(1, -1))

        if self.PCA2:
            sample['outputs2'] = self.PCA2.transform(sample['outputs2'].reshape(1, -1))

        return sample


#%%
subj1 = data_loading.load_subject_data(1, index_start=0, index_end=10, return_dict=True)
subj2 = data_loading.load_subject_data(2, index_start=0, index_end=10, return_dict=True)
#%%
data = [subj1, subj2]
# %%
model = SubjID2Head(100)
# %%
dataset = CustomDataset2(data, transform=transform.ToTensor(), PCA1=PCA(100), PCA2=PCA(100))
# %%
