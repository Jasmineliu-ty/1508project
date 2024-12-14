# Code for ECE1508 project: Model Pruning on Large Convolution Neural Network
Team member: Tongyu Liu, Shiming Zhang

## Running on dog-vs-cat dataset
[VGG16_prune_dogs_cats.ipynb](VGG16_prune_dogs_cats.ipynb) is the jupyter notebook of pruning VGG16 on dog-vs-cat dataset. 

[dog_vs_cat](dog_vs_cat) contains the source code for running training, pruning and visualization for dog-vs-cat dataset.

In [dog_vs_cat](dog_vs_cat) folder, [process_dataset](dog_vs_cat/process_dataset) contains the code to organize the original [dog-vs-cat](https://www.kaggle.com/c/dogs-vs-cats) dataset. [visualization](dog_vs_cat/visualization) contains the code to generate the plot that visualize the models.

### Command to run training and pruning
To train the VGG16 model, run:
`python finetune.py --train`

To prune the pre-trained model, run:
`python finetune.py --prune`

## Running on CIFAR10 dataset
[VGG16_prune_cifar10.ipynb](VGG16_prune_cifar10.ipynb) is the jupyter notebook of pruning VGG16 on CIFAR10 dataset.

[cifar10](cifar10) contains the source code for running training, pruning and visualization for CIFAR10 dataset.


## Reference
[pytorch-pruning](https://github.com/jacobgil/pytorch-pruning), [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440), [VGG16](https://arxiv.org/abs/1409.1556)
