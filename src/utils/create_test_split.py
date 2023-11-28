import os
import numpy as np
import shutil
import random
from data_loading import load_subject_data
from PIL import Image

datapoints_per_subject = 100


### CREATE TEST SPLIT ###


# For each subject
for subject in range(1, 9):  # Assuming subjects are numbered from 1 to 8
    # Load the data
    data_lh, data_rh, image_data, id_list = load_subject_data(subject, 0, datapoints_per_subject, return_dict=False, include_subject_id=True)

    # Randomly select 10% of the IDs
    ids_to_sample = range(len(image_data))
    random.seed(subject)
    num_samples = len(image_data)
    num_test_samples = num_samples // 10
    test_ids = random.sample(ids_to_sample, num_test_samples)
    print(len(test_ids))

    # Create a test folder if it doesn't exist
    current_proj_dir = os.getcwd().split('aml_project_2023')[0] + 'aml_project_2023'
    test_folder = f"test_subj0{subject}"
    test_dir = os.path.join(current_proj_dir, 'data', 'test', test_folder)
    
    os.makedirs(test_dir, exist_ok=True)
    lh_test_fmri = []
    rh_test_fmri = []
    # For each selected ID, save the corresponding image and fMRI data in the test folder
    for id in test_ids:
        print('subject: ', subject, 'id: ', id)
        index = ids_to_sample.index(id)
        image = image_data[index]

        print('Current project directory: %s' % current_proj_dir)
        path = current_proj_dir + '/data/algonauts/subj0' + str(subject)
        folder_path = path+"/training_split/training_images/"
        image_name = os.listdir(folder_path)[index].split('.png')[0]
        image_path = os.path.join(folder_path, image_name + '.png')

        fmri_lh = data_lh[index]
        fmri_rh = data_rh[index]

        # Save the image and fMRI data
        np.save(os.path.join(test_dir, f"{image_name}.npy"), image)

        # Compare and delete original
        original_image = np.array(Image.open(image_path))
        saved_image = np.load(os.path.join(test_dir, f"{image_name}.npy"))

        # Compare the images
        if np.array_equal(original_image, saved_image):
            pass
        else:
            print("Images not equal")
            break
        
        lh_test_fmri.append(fmri_lh)
        rh_test_fmri.append(fmri_rh)

    np.save(os.path.join(test_dir, f"lh_test_fmri_subj0{subject}.npy"), lh_test_fmri)
    np.save(os.path.join(test_dir, f"rh_test_fmri_subj0{subject}.npy"), rh_test_fmri)

    # Compare saved and original FMRI records and save matching ID's
    folder_path_fmri = path+"/training_split/training_fmri/"
    original_lh_fmri = np.load(os.path.join(folder_path_fmri, f"lh_training_fmri.npy"))
    original_rh_fmri = np.load(os.path.join(folder_path_fmri, f"rh_training_fmri.npy"))

    # select only test_ids from original_lh_fmri
    original_lh_fmri_sample = original_lh_fmri[test_ids]
    original_rh_fmri_sample = original_rh_fmri[test_ids]
    # append all files starting with "lh_test_fmri_subj" to a list
    copied_lh_fmri = np.load(os.path.join(test_dir, f"lh_test_fmri_subj0{subject}.npy"))
    copied_rh_fmri = np.load(os.path.join(test_dir, f"rh_test_fmri_subj0{subject}.npy"))

    # Compare the FMRI records
    if np.array_equal(original_lh_fmri_sample, copied_lh_fmri) and np.array_equal(original_rh_fmri_sample, copied_rh_fmri):
        print("All FMRI records for the subject are equal")
    else:
        print("FMRI records not equal")
        print('shape original_lh_fmri_sample: ', original_lh_fmri_sample.shape)
        print('shape copied_lh_fmri: ', len(copied_lh_fmri))
        break


    # np.save(os.path.join(test_folder, f"fmri_rh_{id}.npy"), fmri_rh)
    # Move the test folder to a separate directory
    # shutil.move(test_dir, f"test_data/{test_folder}")

    # Compare the saved image and fMRI data with the original data


### CREATE TRAIN SPLIT ###

# For each subject
for subject in range(1, 9):  # Assuming subjects are numbered from 1 to 8
    # Load the data
    data_lh, data_rh, image_data, id_list = load_subject_data(subject, 0, datapoints_per_subject, return_dict=False, include_subject_id=True)

    # Randomly select 10% of the IDs
    ids_to_sample = range(len(image_data))
    random.seed(subject)
    num_samples = len(image_data)
    num_test_samples = num_samples // 10
    test_ids = random.sample(ids_to_sample, num_test_samples)
    print(len(test_ids))
    train_ids = [id for id in ids_to_sample if id not in test_ids]

    # Create a test folder if it doesn't exist
    current_proj_dir = os.getcwd().split('aml_project_2023')[0] + 'aml_project_2023'
    train_folder = f"train_subj0{subject}"
    train_dir = os.path.join(current_proj_dir, 'data', 'train', train_folder)
    
    os.makedirs(train_dir, exist_ok=True)
    lh_train_fmri = []
    rh_train_fmri = []
    # For each selected ID, save the corresponding image and fMRI data in the test folder
    for id in train_ids:
        print('subject: ', subject, 'id: ', id)
        index = ids_to_sample.index(id)
        image = image_data[index]

        print('Current project directory: %s' % current_proj_dir)
        path = current_proj_dir + '/data/algonauts/subj0' + str(subject)
        folder_path = path+"/training_split/training_images/"
        image_name = os.listdir(folder_path)[index].split('.png')[0]
        image_path = os.path.join(folder_path, image_name + '.png')

        fmri_lh = data_lh[index]
        fmri_rh = data_rh[index]

        # Save the image and fMRI data
        np.save(os.path.join(train_dir, f"{image_name}.npy"), image)

        # Compare and delete original
        original_image = np.array(Image.open(image_path))
        saved_image = np.load(os.path.join(train_dir, f"{image_name}.npy"))

        # Compare the images
        if np.array_equal(original_image, saved_image):
            pass
        else:
            print("Images not equal")
            break
        
        lh_train_fmri.append(fmri_lh)
        rh_train_fmri.append(fmri_rh)

    np.save(os.path.join(train_dir, f"lh_train_fmri_subj0{subject}.npy"), lh_train_fmri)
    np.save(os.path.join(train_dir, f"rh_train_fmri_subj0{subject}.npy"), rh_train_fmri)

    # Compare saved and original FMRI records and save matching ID's
    folder_path_fmri = path+"/training_split/training_fmri/"
    original_lh_fmri = np.load(os.path.join(folder_path_fmri, f"lh_training_fmri.npy"))
    original_rh_fmri = np.load(os.path.join(folder_path_fmri, f"rh_training_fmri.npy"))

    # select only test_ids from original_lh_fmri
    original_lh_fmri_sample = original_lh_fmri[train_ids]
    original_rh_fmri_sample = original_rh_fmri[train_ids]
    # append all files starting with "lh_test_fmri_subj" to a list
    copied_lh_fmri = np.load(os.path.join(train_dir, f"lh_train_fmri_subj0{subject}.npy"))
    copied_rh_fmri = np.load(os.path.join(train_dir, f"rh_train_fmri_subj0{subject}.npy"))

    # Compare the FMRI records
    if np.array_equal(original_lh_fmri_sample, copied_lh_fmri) and np.array_equal(original_rh_fmri_sample, copied_rh_fmri):
        print("All FMRI records for the subject are equal")
    else:
        print("FMRI records not equal")
        print('shape original_lh_fmri_sample: ', original_lh_fmri_sample.shape)
        print('shape copied_lh_fmri: ', len(copied_lh_fmri))
        break


    # np.save(os.path.join(test_folder, f"fmri_rh_{id}.npy"), fmri_rh)
    # Move the test folder to a separate directory
    # shutil.move(test_dir, f"test_data/{test_folder}")

    # Compare the saved image and fMRI data with the original data
