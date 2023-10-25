# Intro meeting

### Problem statement
Creating/training a neural network that will predict brain responses to images (stimuli) while remaining bio-inspired regarding the hierarchy.

##### Main goals:
- Do a comprehensive pre-processing and some visualisation of the fMRI data. Can you see patterns in the data for either input category?
- Train a voxel-wise encoding model (deep CNN network) to predict fMRI responses in different brain regions from the stimulus data as input. You might consider vanilla versus pre-trained image processing networks. Can you identify an architecture (and meta-parameter settings) that predicts well and remains bio-inspired regarding the hierarchy?
- Compare predictions of the individual layers (model) with activations of different regions (imaging data), e.g. through heatmaps. How does the cortex hierarchy compare to the model's hierarchy? Do you observe any patterns?

### To-do:
- [x] Todo list
- [x] Timeline
- [ ] Understand data
	 - [ ] look into existing documentation
	 - [ ] structure of data
	 - [ ] visual analysis
	 - [ ] strategy for preprocessing
	 - [ ] data specific libraries
- [ ] Models research
	- [ ] understanding voxel-wise encoder CNNs (VECNN)
	- [ ] take subset of data to test on models
	- [ ] look into + experiment with VECNN variations:
		- [ ] 1. initial pre-processing
		- [ ] 2. test variation + regularisation and hyperparameters
	- [ ] decide on architecture
	- [ ] decide on evaluation metrics
- [ ] Preprocessing
	- [ ] combine:
		- [ ] strategy of preprocessing
		- [ ] model initial preprocessing
		- [ ] experimental preprocessing
- [ ] Model analysis
	- [ ] saliency maps
	- [ ] feature maps
	- [ ] recent literature on that
	- [ ] draw conclusions

# Timeline

06.11.2023 Last lecture + exercises
(08.11.2023 Project Kick-Off (after the lectures))

25.10.2023 Devise the plan
- understand data individually
|
|
|
01.11.2023 Understand data together
|
|
|
08.11.2023 Models research meeting
- training
|
|
|
**22.11.2023 Supervision meeting (mandatory)**
25.11.2023 Finalise Model
27.11.2023 Start Model Analysis
|
|
|
04.12.2023 Finishing Model Analysis
**06.12.2023 Project sharing round - presentation (mandatory)**
08.12.2023 Last changes before writing report
**09.12.2023 Report writing kick-off**
|
|
|
**16.12.2023 Internal Deadline for Project Report**
xxx
xxx
xxx
08.01.2024 Project report hand-in (mandatory)

**23.-24.01.2024 Oral exams (includes your project)**
