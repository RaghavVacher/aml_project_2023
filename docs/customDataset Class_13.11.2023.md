# CustomDataset Class

## Description
The `CustomDataset` class is a PyTorch Dataset implementation designed for custom datasets. It takes a list of images and their corresponding outputs, allows optional transformations, and supports Principal Component Analysis (PCA) for output data.

## Constructor
```python
def __init__(self, images_list, outputs_list, transform=None, PCA=None)
Parameters
images_list: List of image data.
outputs_list: List of corresponding output data.
transform: Optional image transformation (default is None).
PCA: Optional Principal Component Analysis object (default is None).
Methods
load_data(images_list, outputs_list)
Description
Loads image and output data from input lists.

Parameters
images_list: List of image data.
outputs_list: List of corresponding output data.
give_output()
Description
Returns the concatenated output data.

internal_PCA()
Description
Returns the PCA object if available, else prints a message.

__len__()
Description
Returns the number of samples in the dataset.

__getitem__(idx)
Description
Returns a tuple containing the image and output data at the given index.

Parameters
idx: Index of the sample.
get_pretrained_regression_model Function
Description
The get_pretrained_regression_model function retrieves a pre-trained ResNet18 model with a modified fully connected layer for regression tasks.

Parameters
output_size: Desired size of the output layer.
Returns
Pre-trained ResNet18 model with a modified output layer for regression.