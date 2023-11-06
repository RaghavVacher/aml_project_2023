# Intro meeting

### Problem statement
Creating/training a neural network that will predict brain responses to images (stimuli) while remaining bio-inspired regarding the hierarchy.

##### Main goals:
- Do a comprehensive pre-processing and some visualisation of the fMRI data. Can you see patterns in the data for either input category?
- Train a voxel-wise encoding model (deep CNN network) to predict fMRI responses in different brain regions from the stimulus data as input. You might consider vanilla versus pre-trained image processing networks. Can you identify an architecture (and meta-parameter settings) that predicts well and remains bio-inspired regarding the hierarchy?
- Compare predictions of the individual layers (model) with activations of different regions (imaging data), e.g. through heatmaps. How does the cortex hierarchy compare to the model's hierarchy? Do you observe any patterns?
