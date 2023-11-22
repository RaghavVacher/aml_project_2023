# Supervision Meeting with Sami - 22/11-2023

# To present to Sami:
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


# Questions for Sami:
- is it enough state of the art? the methods we found from other papers
- how to determine the number of PCA components? and whether to do PCA at all?
	- is our methodology valid?
- calculation of loss before or after inverse-PCA-ing?
- what is meant by "bioinspired hierarchy"?

# Todos:
- [ ] set up HPC
- [ ] connect COCO annotations to images
- [ ] look at embedding subject ID (Raghav's paper)
- [ ] complete/look into training script
	- [x] single-head
	- [ ] double-head
- [ ] look into error calculation on inversed PCA brain activity

later:
- [ ] LR scheduler + early stopper