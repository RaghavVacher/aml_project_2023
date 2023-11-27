import os
import numpy as np
import shutil
import random
from data_loading import load_subject_data

# For each subject
for subject in range(1, 9):  # Assuming subjects are numbered from 1 to 8
    # Load the data
    data_lh, data_rh, image_data, id_list = load_subject_data(subject, 0, 100, return_dict=False, include_subject_id=True)

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

        fmri_lh = data_lh[index]
        fmri_rh = data_rh[index]

        # Save the image and fMRI data
        np.save(os.path.join(test_dir, f"{image_name}.npy"), image)
        lh_test_fmri.append(fmri_lh)
        rh_test_fmri.append(fmri_rh)

    np.save(os.path.join(test_dir, f"lh_test_fmri_subj0{subject}.npy"), lh_test_fmri)
    np.save(os.path.join(test_dir, f"rh_test_fmri_subj0{subject}.npy"), rh_test_fmri)
    # np.save(os.path.join(test_folder, f"fmri_rh_{id}.npy"), fmri_rh)
    # Move the test folder to a separate directory
    # shutil.move(test_dir, f"test_data/{test_folder}")
