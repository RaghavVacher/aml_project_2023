
### **Ideas:**
- split hemispheres
- subject ID
- training on shared images
- RNN for "memory"
- add annotations from COCO
- test different models (ResNet, Inception, AlexNet)

**Training:**
- LR scheduler
- early stop

**Encoding:**
- subject
- annotations
- finetuning/pre-training

XXX Memory - a nono because too complicated
- RNNs
- LSTMs
- ViT

### Todos:
- read up on CNN and PCA
- read existing code and classes and understand it
- connect COCO annotations to images
- fix the Trainer and CustomDataset classes to double hemisphere approach
- look at embedding subject ID (Raghav's paper)
- LR scheduler + early stopper
- complete/look into training script
- look at PCA components number discussion in the submission papers (done - EK)
- experiment with different PCA component vals (done - EK)
- look into error calculation on inversed PCA brain activity (RV)

### Questions for Sami:
- is it enough state of the art? the methods we found from other papers
- how to determine the number of PCA components? and whether to do PCA at all?
- 

![[sketch_embedding_class_ID_idea.jpeg]]