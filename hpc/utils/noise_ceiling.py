import numpy as np
from scipy.stats import pearsonr
from data_loading import load_subject_data

def split_data_into_halves(data):
    # Shuffle the trials to randomize the split
    np.random.shuffle(data)
    
    # Split the data into two halves
    half = data.shape[0] // 2
    first_half_data = data[:half, :]
    second_half_data = data[half:2*half, :]
    
    return first_half_data, second_half_data

# Assuming beta_1 and beta_2 are the responses of the same stimuli from two disjoint sets
# beta_1 and beta_2 should be numpy arrays of shape [voxels, stimuli]

def calculate_split_half_noise_ceiling(beta_1, beta_2):
    covariances = np.cov(beta_1, beta_2, rowvar=False)
    variances_1 = np.var(beta_1, axis=1, ddof=1)  # ddof=1 provides an unbiased estimator
    variances_2 = np.var(beta_2, axis=1, ddof=1)

    # Diagonal of the covariance matrix gives the covariance between each pair of beta_1 and beta_2 for each voxel
    Q_values = covariances.diagonal() / np.sqrt(variances_1 * variances_2)

    # Calculate the split-half noise ceiling for each voxel
    Q_SHnc_values = np.sqrt(2 * Q_values / (Q_values + 1))

    # Adjust for any negative correlations by setting the noise ceiling to 0 for those cases
    Q_SHnc_values[Q_values <= 0] = 0

    return Q_SHnc_values

# Example usage:
# beta_1 = np.random.randn(10000, 50)  # Replace with actual data
# beta_2 = np.random.randn(10000, 50)  # Replace with actual data
# noise_ceilings = calculate_split_half_noise_ceiling(beta_1, beta_2)
# print(noise_ceilings)


subj = 6

# def split_half_correlations(data):
#     n_trials = data.shape[0]
#     indices = np.arange(n_trials)
#     np.random.shuffle(indices)
    
#     half = n_trials // 2
#     first_half_indices = indices[:half]
#     second_half_indices = indices[half:2*half]
    
#     first_half_data = data[first_half_indices, :]
#     second_half_data = data[second_half_indices, :]
    
#     correlations = np.array([
#         pearsonr(first_half_data[:, voxel], second_half_data[:, voxel])[0]
#         for voxel in range(data.shape[1])
#     ])
    
#     # Handle any NaN values that may arise due to constant voxel time series
#     correlations = np.nan_to_num(correlations)
    
#     return correlations

# def spearman_brown_correction(r):
#     # Avoid division by zero in case r is exactly -1
#     corrected_r = (2 * r) / (1 + r) if r != -1 else -1
#     return corrected_r

# # Load your fMRI data here
# fMRI_data = np.empty((0, 39198))


print(f"Subject {subj}")
subj_data = load_subject_data(subj, index_start=None, index_end=None, return_dict=False, include_subject_id = False, pca_components=0)[0]
print('subj data shape:', subj_data.shape)
# fMRI_data = np.append(fMRI_data, subj_data, axis=0)
    
# try:
#     print('shape:', fMRI_data.shape)
# except:
#     print('couldnt print shape')

# # Calculate split-half correlations for each voxel
# split_half_corrs = split_half_correlations(fMRI_data)
beta_1, beta_2 = split_data_into_halves(subj_data)

vals = calculate_split_half_noise_ceiling(beta_1, beta_2)

# # Apply the Spearman-Brown prophecy formula to each voxel's correlation
# noise_ceilings = np.array([spearman_brown_correction(corr) for corr in split_half_corrs])

# print(f"The split-half noise ceiling estimate is: {noise_ceilings}")
np.save('noise_ceilings.npy', vals)
