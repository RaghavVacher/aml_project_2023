import torch
import joblib
import numpy as np
from utils.final_model import ResNet1HeadID
from torchvision.models.feature_extraction import create_feature_extractor
from utils.evaluation import MNNPC
import torchvision.transforms as T
import os

def load_images_from_folder(folder_path, start=None, end=None):
    image_paths = []
    for filename in os.listdir(folder_path)[start:end]:
        img_path = os.path.join(folder_path, filename)

        image_paths.append(img_path)
    return image_paths

#slightly adjusted load_subject_data func
def load_subject_data(subject, index_start=None, index_end=None):
    current_proj_dir = os.getcwd().split('hpc')[0] + 'hpc'
    path = '/Users/emilykruger/Documents/GitHub/aml_project_2023/data/training_split/subj0' + str(subject)
    data_lh = np.load(path + '/training_fmri/lh_train_fmri.npy')[index_start : index_end]
    data_rh = np.load(path + '/training_fmri/rh_train_fmri.npy')[index_start : index_end]
    brain = np.concatenate((data_lh, data_rh), axis = 1)
    print('Shape of pca_brain: ', brain.shape)
    folder_path = path+"/training_images/"
    image_paths = load_images_from_folder(folder_path, index_start, index_end)
    
    return brain, image_paths

#For ResNet 
def get_block_names(model):
    layer_names = []
    for layer_name, _ in model.named_children():
        layer_names.append(layer_name)
    return layer_names

#For Aleknet
def get_module_names(model):
    layer_names = []
    for layer_name, _ in model.named_modules():
        layer_names.append(layer_name)
    return layer_names

def flatten_features(outputs):
    for key, item in outputs.items():
        outputs[key] = item.flatten()
    return outputs

def make_prediction(model, flattened_dict, in_feat_model):
    preds = {}
    for key, item in flattened_dict.items():
        adj_layer = torch.nn.Linear(len(item), in_feat_model)
        adj_layer.requires_grad = False
        adj_output = adj_layer(item)
        pred = model.head(adj_output)
        preds[key] = pred
    return preds

# def make_prediction(model, flattened_dict, in_feat_model, subject):
#     preds = {}
#     for key, item in flattened_dict.items():
#         adj_layer = torch.nn.Linear(len(item), in_feat_model)
#         print('3. in_size of adj_layer', len(item))
#         print('4. out_size of adj_layer', in_feat_model)
#         print('5. in_size of shared layer', model.shared.in_features)
#         adj_layer.requires_grad = False
#         adj_output = adj_layer(item)
#         shared = model.shared(adj_output)
#         if subject == 1:
#             subject = model.sub1(adj_output)
#         elif subject == 2:
#             subject = model.sub2(adj_output)
#         elif subject == 3:
#             subject = model.sub3(adj_output)
#         elif subject == 4:
#             subject = model.sub4(adj_output)
#         elif subject == 5:
#             subject = model.sub5(adj_output)
#         elif subject == 6:
#             subject = model.sub6(adj_output)
#         elif subject == 7:
#             subject = model.sub7(adj_output)
#         elif subject == 8:
#             subject = model.sub8(adj_output)

#         # Average the shared and subject-specific layers
#         combined = (shared + subject) / 2
        
#         pred = model.head(combined)
#         preds[key] = pred
#     return preds

def get_pca_model(subject):
    sub = str(subject)
    pca = joblib.load(f'/Users/emilykruger/Documents/GitHub/aml_project_2023/hpc/utils/pca_models/pca_model_subj01.joblib')
    return pca

def preprocess(img, size=224):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((size, size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])
    return transform(img)

def process_subject_data(data,subject, split=True):
    subject = f'subj0{subject}'
    # Define the split dictionary
    split_dict = {"subj01": 19004, "subj02": 19004, "subj03": 19004, "subj04": 19004,
                  "subj05": 19004, "subj06": 18978, "subj07": 19004, "subj08": 18981}


    if subject not in split_dict.keys():
        print("Invalid subject")
        return None, None

    # Split data based on the split dictionary
    if split:
        lh_data = data[:split_dict[subject]]
        rh_data = data[split_dict[subject]:]
    else:
        lh_data = data
        rh_data = None

    # Read ROI directories
    roi_dir_lh = np.load(f'/Users/emilykruger/Documents/GitHub/aml_project_2023/data/training_split/{subject}/roi_masks/lh.all-vertices_fsaverage_space.npy')
    if rh_data is not None:
        roi_dir_rh = np.load(f'/Users/emilykruger/Documents/GitHub/aml_project_2023/data/training_split/{subject}/roi_masks/rh.all-vertices_fsaverage_space.npy')
    else:
        roi_dir_rh = None

    # Create responses
    fsaverage_response_lh = np.zeros(len(roi_dir_lh))
    fsaverage_response_lh[np.where(roi_dir_lh)[0]] = lh_data

    if rh_data is not None:
        fsaverage_response_rh = np.zeros(len(roi_dir_rh))
        fsaverage_response_rh[np.where(roi_dir_rh)[0]] = rh_data
    else:
        fsaverage_response_rh = None

    return fsaverage_response_lh, fsaverage_response_rh