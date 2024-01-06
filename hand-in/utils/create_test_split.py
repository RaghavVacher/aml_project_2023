#since we do not have access to the fmri activation to the test-split images form the official challenge
#we need to create our own test split
import os
import numpy as np
import random
from data_loading import load_subject_data
from PIL import Image

### CREATE TEST SPLIT ###

# For each subject
for subject in range(1, 9):  # Assuming subjects are numbered from 1 to 8
    # Load the data
    data_lh, data_rh, image_data, id_list = load_subject_data(subject, None, None, return_dict=False, include_subject_id=True)

    # Randomly select 10% of the IDs for test and the remaining IDs for train
    ids_to_sample = range(len(image_data))
    random.seed(subject)
    num_samples = len(image_data)
    num_test_samples = num_samples // 10
    test_ids = random.sample(ids_to_sample, num_test_samples)
    train_ids = [id for id in ids_to_sample if id not in test_ids]
    print(len(test_ids))
    print(len(train_ids))

    
    current_proj_dir = os.getcwd().split('aml_project_2023')[0] + 'aml_project_2023'
    
    # Create a test folder if it doesn't exist
    test_folder = f"subj0{subject}"
    test_dir = os.path.join(current_proj_dir, 'data', 'test_split', test_folder)
    test_dir_fmri = os.path.join(test_dir, 'test_fmri')
    test_dir_images = os.path.join(test_dir, 'test_images')
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(test_dir_fmri, exist_ok=True)
    os.makedirs(test_dir_images, exist_ok=True)

    # Create a train folder if it doesn't exist
    train_folder = f"subj0{subject}"
    train_dir = os.path.join(current_proj_dir, 'data', 'training_split', train_folder)
    train_dir_fmri = os.path.join(train_dir, 'training_fmri')
    train_dir_images = os.path.join(train_dir, 'training_images')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(train_dir_fmri, exist_ok=True)
    os.makedirs(train_dir_images, exist_ok=True)
    
    lh_test_fmri = []
    rh_test_fmri = []

    # For each selected ID, save the corresponding image and fMRI data in the test folder
    for id in test_ids:
        print('subject: ', subject, 'id: ', id)
        index = ids_to_sample.index(id)
        image = image_data[index]

        path = current_proj_dir + '/data/algonauts/subj0' + str(subject)
        folder_path = path+"/training_split/training_images/"
        image_name = os.listdir(folder_path)[index].split('.png')[0]
        image_path = os.path.join(folder_path, image_name + '.png')

        fmri_lh = data_lh[index]
        fmri_rh = data_rh[index]

        # Save the image and fMRI data
        np.save(os.path.join(test_dir_images, f"{image_name}.npy"), image)

        # Compare and delete original
        original_image = np.array(Image.open(image_path))
        saved_image = np.load(os.path.join(test_dir_images, f"{image_name}.npy"))

        # Compare the images
        if np.array_equal(original_image, saved_image):
            pass
        else:
            print("Images not equal")
            break
        
        lh_test_fmri.append(fmri_lh)
        rh_test_fmri.append(fmri_rh)

    np.save(os.path.join(test_dir_fmri, f"lh_test_fmri.npy"), lh_test_fmri)
    np.save(os.path.join(test_dir_fmri, f"rh_test_fmri.npy"), rh_test_fmri)

    # Compare saved and original FMRI records and save matching ID's
    folder_path_fmri = path+"/training_split/training_fmri/"
    original_lh_fmri = np.load(os.path.join(folder_path_fmri, f"lh_training_fmri.npy"))
    original_rh_fmri = np.load(os.path.join(folder_path_fmri, f"rh_training_fmri.npy"))

    # select only test_ids from original_lh_fmri
    original_lh_fmri_sample = original_lh_fmri[test_ids]
    original_rh_fmri_sample = original_rh_fmri[test_ids]
    # append all files starting with "lh_test_fmri_subj" to a list
    copied_lh_fmri = np.load(os.path.join(test_dir_fmri, f"lh_test_fmri.npy"))
    copied_rh_fmri = np.load(os.path.join(test_dir_fmri, f"rh_test_fmri.npy"))

    # Compare the FMRI records
    if np.array_equal(original_lh_fmri_sample, copied_lh_fmri) and np.array_equal(original_rh_fmri_sample, copied_rh_fmri):
        print("All FMRI records for the subject are equal")
    else:
        print("FMRI records not equal")
        print('shape original_lh_fmri_sample: ', original_lh_fmri_sample.shape)
        print('shape copied_lh_fmri: ', len(copied_lh_fmri))
        break


    ### CREATE TRAIN SPLIT ###

    # For each selected ID, save the corresponding image and fMRI data in the test folder
    lh_train_fmri = []
    rh_train_fmri = []

    for id in train_ids:
        print('subject: ', subject, 'id: ', id)
        index = ids_to_sample.index(id)
        image = image_data[index]

        path = current_proj_dir + '/data/algonauts/subj0' + str(subject)
        folder_path = path+"/training_split/training_images/"
        image_name = os.listdir(folder_path)[index].split('.png')[0]
        image_path = os.path.join(folder_path, image_name + '.png')

        fmri_lh = data_lh[index]
        fmri_rh = data_rh[index]

        # Save the image and fMRI data
        np.save(os.path.join(train_dir_images, f"{image_name}.npy"), image)

        # Compare and delete original
        original_image = np.array(Image.open(image_path))
        saved_image = np.load(os.path.join(train_dir_images, f"{image_name}.npy"))

        # Compare the images
        if np.array_equal(original_image, saved_image):
            pass
        else:
            print("Images not equal")
            break
        
        lh_train_fmri.append(fmri_lh)
        rh_train_fmri.append(fmri_rh)

    np.save(os.path.join(train_dir_fmri, f"lh_train_fmri.npy"), lh_train_fmri)
    np.save(os.path.join(train_dir_fmri, f"rh_train_fmri.npy"), rh_train_fmri)

    # Compare saved and original FMRI records and save matching ID's
    folder_path_fmri = path+"/training_split/training_fmri/"
    original_lh_fmri = np.load(os.path.join(folder_path_fmri, f"lh_training_fmri.npy"))
    original_rh_fmri = np.load(os.path.join(folder_path_fmri, f"rh_training_fmri.npy"))

    # select only test_ids from original_lh_fmri
    original_lh_fmri_sample = original_lh_fmri[train_ids]
    original_rh_fmri_sample = original_rh_fmri[train_ids]
    # append all files starting with "lh_test_fmri_subj" to a list
    copied_lh_fmri = np.load(os.path.join(train_dir_fmri, f"lh_train_fmri.npy"))
    copied_rh_fmri = np.load(os.path.join(train_dir_fmri, f"rh_train_fmri.npy"))

    # Compare the FMRI records
    if np.array_equal(original_lh_fmri_sample, copied_lh_fmri) and np.array_equal(original_rh_fmri_sample, copied_rh_fmri):
        print("All FMRI records for the subject are equal")
    else:
        print("FMRI records not equal")
        print('shape original_lh_fmri_sample: ', original_lh_fmri_sample.shape)
        print('shape copied_lh_fmri: ', len(copied_lh_fmri))
        break

train_1 = os.listdir(train_dir_images)
test_1 = os.listdir(test_dir_images)

for test in test_1:
    if test in train_1:
        print("Duplicate!")
        break
print("No duplicates!")

