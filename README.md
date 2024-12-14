# Code for ECE1508 project
Team member: Tongyu Liu, Shiming Zhang

## Running on dog-vs-cat dataset
[VGG16_prune_dogs_cats.ipynb](VGG16_prune_dogs_cats.ipynb) is the jupyter notebook of pruning VGG16 on dog-vs-cat dataset. 

[dog_vs_cat](dog_vs_cat) contains the source code for running training, pruning and visualization for dog-vs-cat dataset.

### Command to run
To train the VGG16 model, run:
`python finetune.py --train`

To prune the pre-trained model, run:
`python finetune.py --prune`

## Running on CIFAR10 dataset
[VGG16_prune_cifar10.ipynb](VGG16_prune_cifar10.ipynb) is the jupyter notebook of pruning VGG16 on CIFAR10 dataset.

[cifar10](cifar10) contains the source code for running training, pruning and visualization for CIFAR10 dataset.
