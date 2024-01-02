import torch
import joblib
import numpy as np
from utils.final_model import ResNet1HeadID
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn.functional as F
from utils.evaluation import MNNPC
import torchvision.transforms as T
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr

def load_images_from_folder(folder_path, start=None, end=None):
    image_paths = []
    for filename in os.listdir(folder_path)[start:end]:
        img_path = os.path.join(folder_path, filename)

        image_paths.append(img_path)
    return image_paths

#slightly adjusted load_subject_data func
def load_subject_data(subject, index_start=None, index_end=None):
    current_proj_dir = os.getcwd().split('hpc')[0] + 'hpc'
    path = '/Users/emilykruger/Documents/GitHub/aml_project_2023/data/test_split/subj0' + str(subject)
    data_lh = np.load(path + '/test_fmri/lh_test_fmri.npy')[index_start : index_end]
    data_rh = np.load(path + '/test_fmri/rh_test_fmri.npy')[index_start : index_end]
    brain = np.concatenate((data_lh, data_rh), axis = 1)
    print('Shape of pca_brain: ', brain.shape)
    folder_path = path+"/test_images/"
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

# def make_prediction(model, flattened_dict, in_feat_model):
#     preds = {}
#     for key, item in flattened_dict.items():
#         adj_layer = torch.nn.Linear(len(item), in_feat_model)
#         adj_layer.requires_grad = False
#         adj_output = adj_layer(item)
#         pred = model.head(adj_output)
#         preds[key] = pred
#     return preds

def make_prediction(model, flattened_dict, in_feat_model):
    preds = {}
    # Create the linear layer outside the loop
    adj_layer = torch.nn.Linear(len(flattened_dict[next(iter(flattened_dict))]), in_feat_model)
    adj_layer.requires_grad = False
    for key, item in flattened_dict.items():
        # Assuming item is a NumPy array or a PyTorch tensor
        item_tensor = torch.from_numpy(item) if not torch.is_tensor(item) else item     
        # Assuming adj_layer is a linear layer with in_feat_model as the output size
        adj_output = adj_layer(item_tensor)
        # Assuming model.head is the final layer of your model
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
    pca = joblib.load(f'/Users/emilykruger/Documents/GitHub/aml_project_2023/hpc/utils/pca_models/pca_model_subj0{subject}.joblib')
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

def corr_roi_plot(gt_activation, pred_activation, subject, split = True):
    subject = f'subj0{subject}'
    working_dir = rf'/Users/emilykruger/Documents/GitHub/aml_project_2023/data/test_split/{subject}/' #Change as needed

    # Define the split dictionary
    split_dict = {"subj01": 19004, "subj02": 19004, "subj03": 19004, "subj04": 19004,
                  "subj05": 19004, "subj06": 18978, "subj07": 19004, "subj08": 18981}


    if subject not in split_dict.keys():
        print("Invalid subject")
        return None, None

    # Split data based on the split dictionary
    if split:
        lh_data_pred = pred_activation[:,:split_dict[subject]].detach().numpy()
        rh_data_pred = pred_activation[:,split_dict[subject]:].detach().numpy()

    else:
        lh_data_pred = pred_activation
        rh_dataPred = None

    # Split gt based on the split dictionary
    lh = gt_activation[:,:split_dict[subject]]
    rh = gt_activation[:,split_dict[subject]:]

    # Empty correlated array
    lh_correlation = np.zeros(lh.shape[1])
    rh_correlation = np.zeros(rh.shape[1])

    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    for v in tqdm(range(lh.shape[1])):
        lh_correlation[v] = pearsonr(lh_data_pred[:,v], lh[:,v])[0]

    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh.shape[1])):
        rh_correlation[v] = pearsonr(rh_data_pred[:,v], rh[:,v])[0]

    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(working_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(working_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(working_dir, 'roi_masks',
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)

    # Create the plot
    lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r])
        for r in range(len(lh_roi_correlation))]
    rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r])
        for r in range(len(rh_roi_correlation))]
    plt.figure(figsize=(18,6))
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width/2, lh_mean_roi_correlation, width, label='Left Hemisphere')
    plt.bar(x + width/2, rh_mean_roi_correlation, width,
        label='Right Hemishpere')
    plt.xlim(left=min(x)-.5, right=max(x)+.5)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('ROIs')
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.ylabel('Mean Pearson\'s $r$')
    plt.title(f'Encoding Accuracy of Individual ROIs for {subject}')
    plt.legend(frameon=True, loc=1)
    plt.show()

def corr_roi_plot_allsub(subject_list, model, split = True):
    correlations_rh = []
    correlations_lh = []
    #calculate correlations for each subject
    for subject in subject_list:
        gt_activation, image_paths = load_subject_data(subject)
        subject_id = f'subj0{subject}'
        working_dir = rf'/Users/emilykruger/Documents/GitHub/aml_project_2023/data/test_split/{subject_id}/' #Change as needed

        split_dict = {"subj01": 19004, "subj02": 19004, "subj03": 19004, "subj04": 19004,
                    "subj05": 19004, "subj06": 18978, "subj07": 19004, "subj08": 18981}
        

        if subject_id not in split_dict.keys():
            print("Invalid subject")
            return None, None
        
        lh = gt_activation[:,:split_dict[subject_id]]
        rh = gt_activation[:,split_dict[subject_id]:]

            #pick image and activation
        images = []
        for i in range(len(image_paths)):
            image = np.load(image_paths[i])
            image = preprocess(image)
            images.append(image)

        pca = get_pca_model(subject)

        #use loaded model from beginning and pass image through
        preds = np.zeros(gt_activation.shape)
        for i, img in enumerate(images):
            pred = model(img.unsqueeze(0))
            inversed_pred = pca.inverse_transform(pred.detach().numpy()).flatten()
            preds[i,:] = inversed_pred

        pred_activation = torch.Tensor(preds)
        print('Inversed Predictions Sucessfull')

        # Split data based on the split dictionary
        if split:
            lh_data_pred = pred_activation[:,:split_dict[subject_id]]
            rh_data_pred = pred_activation[:,split_dict[subject_id]:]

        else:
            lh_data_pred = pred_activation
            rh_dataPred = None

        # Empty correlated array
        lh_correlation = np.zeros(lh.shape[1])
        rh_correlation = np.zeros(rh.shape[1])

        # Correlate each predicted LH vertex with the corresponding ground truth vertex
        for v in tqdm(range(lh.shape[1])):
            lh_correlation[v] = pearsonr(lh_data_pred[:,v], lh[:,v])[0]
        correlations_lh.append(lh_correlation)

        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh.shape[1])):
            rh_correlation[v] = pearsonr(rh_data_pred[:,v], rh[:,v])[0]
        correlations_rh.append(rh_correlation)

    lh_correlation = np.mean(np.stack(correlations_lh), axis = 0)
    rh_correlation = np.mean(np.stack(correlations_rh), axis = 0)

    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(working_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(working_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(working_dir, 'roi_masks',
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])
    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)

    # Create the plot
    lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r])
        for r in range(len(lh_roi_correlation))]
    rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r])
        for r in range(len(rh_roi_correlation))]

    plt.figure(figsize=(18,6))
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width/2, lh_mean_roi_correlation, width, label='Left Hemisphere')
    plt.bar(x + width/2, rh_mean_roi_correlation, width,
        label='Right Hemishpere')
    plt.xlim(left=min(x)-.5, right=max(x)+.5)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('ROIs')
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.ylabel('Mean Pearson\'s $r$')
    plt.title(f'Encoding Accuracy of Individual ROIs for All Subjects')
    plt.legend(frameon=True, loc=1)
    plt.show()