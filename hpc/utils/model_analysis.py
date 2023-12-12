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



def make_prediction(model, flattened_dict, in_feat_model, subject):
    preds = {}
    for key, item in flattened_dict.items():
        adj_layer = torch.nn.Linear(len(item), in_feat_model)
        print('3. in_size of adj_layer', len(item))
        print('4. out_size of adj_layer', in_feat_model)
        print('5. in_size of shared layer', model.shared.in_features)
        adj_layer.requires_grad = False
        adj_output = adj_layer(item)
        shared = model.shared(adj_output)
        if subject == 1:
            subject = model.sub1(adj_output)
        elif subject == 2:
            subject = model.sub2(adj_output)
        elif subject == 3:
            subject = model.sub3(adj_output)
        elif subject == 4:
            subject = model.sub4(adj_output)
        elif subject == 5:
            subject = model.sub5(adj_output)
        elif subject == 6:
            subject = model.sub6(adj_output)
        elif subject == 7:
            subject = model.sub7(adj_output)
        elif subject == 8:
            subject = model.sub8(adj_output)

        # Average the shared and subject-specific layers
        combined = (shared + subject) / 2
        
        pred = model.head(combined)
        preds[key] = pred
    return preds

def get_pca_model(subject):
    sub = str(subject)
    pca = joblib.load(f'hpc/utils/pca_models/pca_model_subj0{sub}.joblib')
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
    roi_dir_lh = np.load(fr"C:\Users\rvacher\Downloads\algonauts_2023_tutorial_data\{subject}\roi_masks\lh.all-vertices_fsaverage_space.npy")
    if rh_data is not None:
        roi_dir_rh = np.load(fr"C:\Users\rvacher\Downloads\algonauts_2023_tutorial_data\{subject}\roi_masks\rh.all-vertices_fsaverage_space.npy")
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


if __name__ == "__main__":
    #Change as needed
    checkpoint_path = 'hpc/utils/trained_models/resnet101_LR0.001_SAMPLES_all_EPOCHS20_BATCHSIZE_64_TIME_2023-12-05_18:59:14.pt'
    output_size = 100
    backbone = 'resnet18'
    #load images and original activations
    subject_id = 1
    brain, image_paths = load_subject_data(1, index_start=0, index_end=5)

    #pick random image and activation
    image = np.load(image_paths[0])
    image = preprocess(image)

    activation = torch.Tensor(brain[0,:])
    print('activation type:', type(activation))

    scores = {}

    # Check if GPU is available and if not, use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Load in model 
    trained_model_state_dict = torch.load(checkpoint_path, map_location = device)
    feature_extractor = torch.hub.load('src/utils', backbone, source = 'local')

    trained_model = ResNet1HeadID(output_size = output_size, feature_extractor= feature_extractor).eval()
    trained_model.load_state_dict(trained_model_state_dict["model_state_dict"])
    trained_model_backbone = trained_model.feature_extractor

    #Create feature extractor
    if backbone == "alexnet":
        output_layer_names = get_module_names(trained_model_backbone)[1:-1]
    else:
        output_layer_names = get_block_names(trained_model_backbone)[:-1]
    feature_extractor = create_feature_extractor(trained_model_backbone, output_layer_names)

    #Extract features & predict fMRI data 
    outputs = feature_extractor(image.unsqueeze(0))
    print('1. features extracted from image:', len(outputs), 'of type:', type(outputs))
    flat_outputs = flatten_features(outputs)
    print('2. flattened features:', len(flat_outputs), 'of type:', type(flat_outputs))
    predictions = make_prediction(trained_model, flat_outputs, trained_model.head.in_features, subject = subject_id)
    print(predictions)

    #Inverse transform of preds with frozen PCA models
    pca = get_pca_model(subject_id)
    inversed_predictions = predictions.copy()
    for key in inversed_predictions:
        #convert prediction tensors to np arrays to make it compatible for inverse pca
        preds = inversed_predictions[key].detach().numpy()
        preds = torch.Tensor(pca.inverse_transform(preds))
        print('preds type:', type(preds))
        # inverse-pca and store in new dict
        inversed_predictions[key] = preds

    #Caculating MNNPC on Preds
    # for key in inversed_predictions:
    #     preds = inversed_predictions[key]
    #     mnnpc = MNNPC()
    #     score = mnnpc(pred = preds, gt=activation)
    #     scores[key] = score

    # print(scores)