# aml_project_2023
Group Project Assignment AML(ITU 2023)

## Learning Goals
In parallel to working towards the general learning objectives of the Advanced Machine Learning Course, you are intended to

- Solve an interesting problem,
- Implement and apply a recent machine learning method,
- Evaluate and report the method performance.

### Link to Google Drive
https://drive.google.com/drive/u/1/folders/1r4ZxT6RB416Ts4IAPNL4KrleKJfJU3FC

### To-do:
- [x] Todo list
- [x] Timeline
- [x] Understand data
	 - [x] look into existing documentation
	 - [x] structure of data
	 - [ ] visual analysis
		 - [ ] see how to do the flattened image with ROIs - for later
	 - [ ] strategy for preprocessing
		 - [ ] de-noising / PCA
		 - [ ] finding outliers
		- [x]  data specific libraries
			- [x] nilearn (datasets and plotting)
			- [x] Image from PIL
- [ ] **look at [original paper](https://soeg.kb.dk/permalink/45KBDK_KGL/1f0go08/cdi_unpaywall_primary_10_1038_s41593_021_00962_x)** 
	- [ ] the way they normalize data
	- [ ] the way they project data onto universal brain
- [x] **algonauts submissions**
	- [x] Emily 1 huze
	- [x] Frederik 2 hosseinadeli
	- [x] Raghav 3 UARK-SUNY-Albany
	- [x] Alek 6 blobGPT
- [ ] Models research
	- [ ] understanding voxel-wise encoder CNNs (VoEC)
	- [ ] take subset of data to test on models
	- [ ] look into + experiment with VoEC variations:
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
|  
| understand data individually  
|  
|  
|  
01.11.2023 Understand data together  
|  
| initial models research   
|  
11.11.2023 14:00 Models research meeting  
13.11.2023 10:00 Extra models meeting
| training  
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
| xxx  
| xxx  
| xxx  
08.01.2024 Project report hand-in (mandatory)  

**23.-24.01.2024 Oral exams (includes your project)**

## Problem Statement
To reach our learning goals, your task as a group is to:

* understand a chosen problem, select appropriate (existing and recent) dataset(s), and understand feasible existing solutions (the *state of the art*), 
* select and implement (at least) one machine learning method - as a general rule, you may take code pieces from existing repositories/exercises/papers as long as you understand and can explain that code, and cite it, 
* evaluate and report your solution for appropriate measure w.r.t. your chosen problem (e.g. accuracy/performance, edge cases, model behaviour) - and discuss/reflect peculiar strengths and weaknesses.

As optional additional steps, we highly encourage you to (and/or):
* expand a solution with your own original ideas, 
* transfer your solution onto different datasets (potentially from different domains),
* experiment with novel approaches to add insight/explainability about the models' decisions.

For this, you can either select one of the prepared projects below or create your own.
