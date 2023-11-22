# Supervision Meeting with Sami - 22/11-2023

## To present to Sami:
- starting point: algonauts submission --> implement and combine their idea
- starting with simple model, then adding more complexity by adding more info (subj ID / annottions)
- considered double-head but PCA exploration showed that combined brains is better
	- seems to catch more dependencies between verteces
- show PCA exploration


ideas for model analysis:
- compare metrics for different models (convergence, loss etc)
- look into tensorboard
- maybe some hyperparameter tuning
- pick cherries and lemons: look at images with good predictions vs. images with bad predictions --> can we detect patterns?
	- loss for each COCO annotation?
	- maybe the same for ROI? i.e. is drastically better or worse in some ROIs?

if need to be more advanced:
- sort-of-heat maps: applying ROI masks: which parts of the input image lead to high activation in this specific ROI (pick 2 or 3)


## Questions for Sami:
- is it enough state of the art? the methods we found from other papers
- how to determine the number of PCA components? and whether to do PCA at all?
	- is our methodology valid?
- calculation of loss before or after inverse-PCA-ing?
- what is meant by "bioinspired hierarchy"?

## Feedback from Sami:

- more data exploration
	- outliers? preprocessing?
	- what preprocessing is already done?
	- get inspiration from submisisons

- double vs. single head: check submissions what the effect is
- be careful not to overfit to subjects when adding subj ID etc
- using/ reproducing ideas from top-performing submission is state-of-the-art enough
- compare 95%-PCA model to elob-component PCA (ca 30)
- bioinspired:
	- possibly ask Stefan for clarification
	- look at best-performing submission
	- google relation between CNN and brain / vision 
	- possibly use pretrained model that has been shown to be bio-inspired
- loss before or after inverse-PCA: do some sort of math?

- model analysis:
	- look at intermediate layer responses, do they correlate to the regions
	- what is being learned in the network
	- e.g. compare ROI activation in different intermediate layers
	- correlation between layer response and regions --> is there a region that spiked in correlation scores for different layers?
	- Do intermediate layers correlate with regions
	- Visualise correlations between layer activations and ROI activations
	- Do some layers 'define' specific regions?

# Todos:

today:
- [x] new model structure: shared layer + subject specific layer
	- as substitute for embedding subject ID
- [ ] run PCA exploration elbow method for combined brainn for all subjects
- [ ] look into error calculation on inversed PCA brain activity

later:
- [ ] look into different pretrained models (e.g. for ones that have been shown to be bio-inspired / good for brain activations etc. to make informed choice)
- [ ] connect COCO annotations to images
- [ ] set up HPC
- [ ] LR scheduler + early stopper
- [ ] complete/look into training script
	- [x] single-head
	- [ ] double-head


For next time:

- model woth shared and subject specific layers is built
- dataloading funcs and CustomDataset Class is adjusted, but the output doesn*t fulfill the requirements for the DataLoader yet --> need to fix next time