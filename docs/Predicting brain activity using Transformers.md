**Vision transformer**
- Encoder: self-supervised vision transformer (DINOv2)
- Decoder: ROIs as queries to select relevant information from encoder output for predicting neural activity in each ROI
- Prediction: Linear mapping of decoder output tokens to fMRI activity (via PCA)
---
![[Pasted image 20231103130235.png]]

---
**Encoding (feature extraction)**
- Patching: 31x31 (14x14 pixels)
- 12-layer pre-trained vision transformer (token size 768)
- Encode input embeddings -->  send queries to decoder
---
**Decoding**
- Learn query embeddings corresponding to ROIs
	- Different from encoded linear feature mapped queries
- 1x cross-attentions
- 1x self-attention
---
**Predicting**
- Linearly map attention-weighted ROI queries to fMRI response
- Each ROI query is mapped to a vector of size = n vertices in hemisphere
- Apply ROI mask
	- This ensures that gradient signal form loss only trains linear mappings to the vertices of the queried ROI
- Combine predictions for each ROI using mask to produce prediction of each hemisphere
---
**Training**
- MSE between predicted activity and measured acitivty
- Train one model per participant
---
**Ensemble model**
- Combine 22 different models
	- Different combinations of encoder output layer and decoder ROIs
	- 10-fold CV with different seeds --> preds for all training images
- Post-processing
	- PCA of predictions (100 pcs per hemisphere)
	- Behavioural features of stimuli (not available to us)
- linear regression to map predictions + behavioural features to 100 principal components of fMRI signal --> reconstruct fMRI
	- This captures dependencies between vertices
---
**Ensemble predictions**
- Predict with all models for each vertex
- Combine predictions for each vertex using softmax weights
- Weights are based on model performance on that vertex
---
![[Pasted image 20231103130235.png]]

---
![[Pasted image 20231103133637.png]]