import joblib
from sklearn.decomposition import PCA
import numpy as np
import os

#loop thorugh each subject to perform PCA on both hemipsheres combined
for subject in range(1,8+1):
    #load data
    current_proj_dir = os.getcwd().split('hpc')[0] + 'hpc'
    path = current_proj_dir + '/data/training_split/subj0' + str(subject) + '/training_fmri'
    data_lh = np.load(path + '/lh_train_fmri.npy')
    data_rh = np.load(path + '/rh_train_fmri.npy')
    #concat both hemispheres
    brain = np.concatenate((data_lh, data_rh), axis = 1)
    #delete data_lh & data_rh from memory
    del data_lh, data_rh

    #PCA
    pca_brain = PCA(n_components=100)
    pca_brain.fit(brain)
    brain_transform = pca_brain.transform(brain)

    #save the PCA model
    os.makedirs('pca_models', exist_ok=True)
    save_path = f'pca_models/pca_model_subj0{subject}.joblib'
    joblib.dump(pca_brain, save_path)
    print('saved PCA model to', save_path)

    #save transformed data
    save_path = path + '/pca_brain.npy'
    # check if brain_transform is a numpy array
    if type(brain_transform) is not np.ndarray:
        print('transformed data is not a numpy array, converting it to one')
        brain_transform = brain_transform.toarray()
    np.save(save_path, brain_transform)
    print('saved transformed data to', save_path)