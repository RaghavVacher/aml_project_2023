#script to run descriptive stats on our data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up paths
data_dir = './data/algonauts'

# Create 'figs' directory
figs_dir = os.path.join(os.getcwd(), 'figs')
os.makedirs(figs_dir, exist_ok=True)

# Loop over subjects to calculate descriptive statistics and plot histograms
subjects = [1, 2, 3, 4, 5, 6, 7, 8]
brain_stats = {}

for subj in subjects:
    subj = "0" + str(subj)
    subj_path = os.path.join(data_dir, 'subj'+subj)
    fmri_dir = os.path.join(subj_path, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    brain = np.concatenate((lh_fmri, rh_fmri), axis=1)

    # Delete hemispheres to save memory
    del lh_fmri, rh_fmri

    # Calculate descriptive statistics
    mean = np.mean(brain)
    median = np.median(brain)
    std = np.std(brain)
    min_value = np.min(brain)
    max_value = np.max(brain)

    # Create dictionary for subject
    subj_dict = {'mean' : mean, 'median': median, 'std': std, 'min_value': min_value, 'max_value': max_value}

    # Add to overall dict
    brain_stats[subj] = subj_dict

    # Plot
    sns.histplot(brain, stat='density')
    sns.kdeplot(brain, color='red')
    plt.title(f'Subject {subj} Histogram')
    plt.xlabel('BOLD Signal')
    plt.ylabel('Frequency')
    plt.savefig(f'/figs/subj{subj}_histogram.png')
    plt.close() # Close figure to save memory

# Convert to dataframe
desc_df = pd.DataFrame(brain_stats)

# Save to csv
desc_df.to_csv('desc_stats.csv', index=False)
