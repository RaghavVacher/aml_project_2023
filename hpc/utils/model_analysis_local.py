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

def make_prediction(model, flattened_dict, in_feat_model):
    preds = {}
    for key, item in flattened_dict.items():
        adj_layer = torch.nn.Linear(len(item), in_feat_model)
        adj_layer.requires_grad = False
        adj_output = adj_layer(item)
        pred = model.head(adj_output)
        preds[key] = pred
    return preds

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
    roi_dir_lh = np.load(f'/Users/emilykruger/Documents/GitHub/aml_project_2023/data/test_split/{subject}/roi_masks/lh.all-vertices_fsaverage_space.npy')
    if rh_data is not None:
        roi_dir_rh = np.load(f'/Users/emilykruger/Documents/GitHub/aml_project_2023/data/test_split/{subject}/roi_masks/rh.all-vertices_fsaverage_space.npy')
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

def norm_pearson(gt_activation, pred_activation):
    # Correlate each predicted LH vertex with the corresponding ground truth vertex
    correlation = np.zeros(gt_activation.shape[1])
    for v in tqdm(range(gt_activation.shape[1])):
        correlation[v] = pearsonr(pred_activation[:,v], gt_activation[:,v])[0]
    correlation = 1 - (correlation + 1) / 2
    return np.mean(correlation)


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
    lh_correlation = 1 - (lh_correlation + 1) / 2

    # Correlate each predicted RH vertex with the corresponding ground truth vertex
    for v in tqdm(range(rh.shape[1])):
        rh_correlation[v] = pearsonr(rh_data_pred[:,v], rh[:,v])[0]
    rh_correlation = 1 - (rh_correlation + 1) / 2

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
    plt.ylabel('Mean Normalized Pearson\'s $r$')
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
        lh_correlation = 1 - (lh_correlation + 1) / 2

        # Correlate each predicted RH vertex with the corresponding ground truth vertex
        for v in tqdm(range(rh.shape[1])):
            rh_correlation[v] = pearsonr(rh_data_pred[:,v], rh[:,v])[0]
        rh_correlation = 1 - (rh_correlation + 1) / 2

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
        
        #append subject-specific ROI corr to list
        correlations_lh.append(lh_mean_roi_correlation)
        correlations_rh.append(rh_mean_roi_correlation)
        print(f'Subject {subject} done.')
    
    lh_correlation = np.mean(np.array(correlations_lh), axis = 0)
    rh_correlation = np.mean(np.array(correlations_rh), axis = 0)


    # Create the plot
    lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r])
        for r in range(len(lh_roi_correlation))]
    rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r])
        for r in range(len(rh_roi_correlation))]


    plt.figure(figsize=(18,6))
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width/2, lh_correlation, width, label='Left Hemisphere')
    plt.bar(x + width/2, rh_correlation, width,
        label='Right Hemishpere')
    plt.xlim(left=min(x)-.5, right=max(x)+.5)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('ROIs')
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.ylabel('Mean Normalized Pearson\'s $r$')
    plt.title(f'Encoding Accuracy of Individual ROIs for All Subjects')
    plt.legend(frameon=True, loc=1)
    plt.show()


def corr_roi_plot_allsub_layers(subject_list, first_image_id, last_image_id, model, feature_extractor, split = True):
    # Subject 1
    dict_1_lh = {}
    dict_1_rh = {}
    # Subject 2
    dict_2_lh = {}
    dict_2_rh = {}
    # Subject 3
    dict_3_lh = {}
    dict_3_rh = {}
    # Subject 4
    dict_4_lh = {}
    dict_4_rh = {}
    # Subject 5
    dict_5_lh = {}
    dict_5_rh = {}
    # Subject 6
    dict_6_lh = {}
    dict_6_rh = {}
    # Subject 7
    dict_7_lh = {}
    dict_7_rh = {}
    # Subject 8
    dict_8_lh = {}
    dict_8_rh = {}

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
        
        gt_activation = gt_activation[first_image_id:last_image_id,:]
        lh = gt_activation[:,:split_dict[subject_id]]
        rh = gt_activation[:,split_dict[subject_id]:]

        all_predictions = {}
        #pick image and activation
        for img in image_paths[42:46]:
            image = np.load(img)
            image = preprocess(image)
            #Extract features & predict fMRI data 
            output = feature_extractor(image.unsqueeze(0))
            flat_outputs = flatten_features(output)
            pred_activation = make_prediction(model, flat_outputs, model.head.in_features)
            for key, value in pred_activation.items():
                if key not in all_predictions:
                    all_predictions[key] = [value]
                else:
                    all_predictions[key].append(value)

        for key in all_predictions:
            all_predictions[key] = torch.stack(all_predictions[key], dim=0)

        pca = get_pca_model(1)
        inversed_predictions = all_predictions.copy()
        for key in inversed_predictions:
            #convert prediction tensors to np arrays to make it compatible for inverse pca
            preds = inversed_predictions[key].detach().numpy()
            preds = torch.Tensor(pca.inverse_transform(preds))
            # inverse-pca and store in new dict
            inversed_predictions[key] = preds


        predictions = inversed_predictions.values()
        layers = inversed_predictions.keys()

        for layer, pred in zip(layers,predictions):
        # Split data based on the split dictionary

            lh_data_pred = pred[:,:split_dict[subject_id]]
            rh_data_pred = pred[:,split_dict[subject_id]:]

            # Empty correlated array
            lh_correlation = np.zeros(lh.shape[1])
            rh_correlation = np.zeros(rh.shape[1])

            # Correlate each predicted LH vertex with the corresponding ground truth vertex
            for v in tqdm(range(lh.shape[1])):
                lh_correlation[v] = pearsonr(lh_data_pred[:,v], lh[:,v])[0]
            lh_correlation = 1 - (lh_correlation + 1) / 2

            # Correlate each predicted RH vertex with the corresponding ground truth vertex
            for v in tqdm(range(rh.shape[1])):
                rh_correlation[v] = pearsonr(rh_data_pred[:,v], rh[:,v])[0]
            rh_correlation = 1 - (rh_correlation + 1) / 2

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
            lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r]) for r in range(len(lh_roi_correlation))]
            rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r]) for r in range(len(rh_roi_correlation))]

            # Assuming layer, lh_mean_roi_correlation, and rh_mean_roi_correlation are defined variables
            if subject == 1:
                dict_1_lh[layer] = lh_mean_roi_correlation
                dict_1_rh[layer] = rh_mean_roi_correlation
            elif subject == 2:
                dict_2_lh[layer] = lh_mean_roi_correlation
                dict_2_rh[layer] = rh_mean_roi_correlation
            elif subject == 3:
                dict_3_lh[layer] = lh_mean_roi_correlation
                dict_3_rh[layer] = rh_mean_roi_correlation
            elif subject == 4:
                dict_4_lh[layer] = lh_mean_roi_correlation
                dict_4_rh[layer] = rh_mean_roi_correlation
            elif subject == 5:
                dict_5_lh[layer] = lh_mean_roi_correlation
                dict_5_rh[layer] = rh_mean_roi_correlation
            elif subject == 6:
                dict_6_lh[layer] = lh_mean_roi_correlation
                dict_6_rh[layer] = rh_mean_roi_correlation
            elif subject == 7:
                dict_7_lh[layer] = lh_mean_roi_correlation
                dict_7_rh[layer] = rh_mean_roi_correlation
            elif subject == 8:
                dict_8_lh[layer] = lh_mean_roi_correlation
                dict_8_rh[layer] = rh_mean_roi_correlation

    final_dict_lh = {}
    final_dict_rh = {}

    #getting average values for each layer and ROI across subjects
    dicts_lh = [dict_1_lh, dict_2_lh, dict_3_lh, dict_4_lh, dict_5_lh, dict_6_lh, dict_7_lh, dict_8_lh]
    dicts_rh = [dict_1_rh, dict_2_rh, dict_3_rh, dict_4_rh, dict_5_rh, dict_6_rh, dict_7_rh, dict_8_rh]


    # Iterate over keys
    for key in dict_1_lh.keys():
        # Element-wise average for each key
        avg_lh = [sum(d[key][i] for d in dicts_lh) / len(dicts_lh) for i in range(len(dicts_lh[0][key]))]
        final_dict_lh[key] = avg_lh

        avg_rh = [sum(d[key][i] for d in dicts_rh) / len(dicts_rh) for i in range(len(dicts_rh[0][key]))]
        final_dict_rh[key] = avg_rh

    for key in final_dict_lh.keys():
        plt.figure(figsize=(18,6))
        x = np.arange(len(roi_names))
        width = 0.30
        plt.bar(x - width/2, final_dict_lh[key], width, label='Left Hemisphere')
        plt.bar(x + width/2, final_dict_rh[key], width,
            label='Right Hemishpere')
        plt.xlim(left=min(x)-.5, right=max(x)+.5)
        plt.ylim(bottom=0, top=1)
        plt.xlabel('ROIs')
        plt.xticks(ticks=x, labels=roi_names, rotation=60)
        plt.ylabel('Mean Normalized Pearson\'s $r$')
        plt.title(f'Encoding Accuracy of Layer {key} for Individual ROIs for All Subjects')
        plt.legend(frameon=True, loc=1)
        plt.show()