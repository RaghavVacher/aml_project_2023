# Algonauts submissions review - 11.11.2023

### 1. Memory Encoding Model
- simple implementation of memory (they used transformers, maybe we can do LSTM, RNN, GRNN)
- layer to flatten the 3D brain into a flat ugly 2D brain #squish
- "We use a pretrained image backbone model DiNOv2-ViT-B [21] to build a straight-forward model"
- avgmax pooling?
- maybe get rid of ROIs

### 2. Transformers
- vision transformer DINOv2
	- 1x cross-attentions
	- 1x self-attention
	- using ROIs in some way
- loss metric - MSE between predicted activity and measured activity
- separate model for each participant
- "Prediction: Linear mapping of decoder output tokens to fMRI activity (via PCA)"

### 3.  ALBANY
- cosine learning rate scheduler
- loss functions provided - experiment with others?
- left. and  right predicted separately
- not sure if #squish 
- AdamW and early stopping
- pre-trained on shared images with embedded subject ID, then fine-tuned on the individual images
- DNN (CNN) to extract features from images
- predicts for individual ROIs and the entire thing - then averages
	- multihead
- final pred made through MLP using linear/dense layers and batch normalisation

### 4. PMMPfA
- feature projection from intermediate layers
- different models tested, chose EVA02 with CLIP teaching thingy
- PCA fMRI activity embedding
- shared linear + subject linears with subject ID 

---

# Plan
- [ ] starting simple
	- [ ] Basic CNN - pretrained model like AlexNet or ResNet?
	- [ ] PCA on brain data
	- [ ] MSE for loss - later with regularization
	- [ ] single subject!
- [ ] practilatis
	- [ ] hpc
	- [ ] colab
	- [ ] 