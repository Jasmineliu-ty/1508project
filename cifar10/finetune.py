import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import *
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from visualization import *

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        # Freeze the feature extractor
        # for param in self.features.parameters():
        #     param.requires_grad = False

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  # Add adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)     # Pass through the classifier
        return x

class FilterPrunner:
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        x = self.model.adaptive_pool(x)
        x = x.view(x.size(0), -1)

        return self.model.classifier(x)


    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter,
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        self.grad_index += 1

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return nsmallest(num, data, itemgetter(2))

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            v = v / torch.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune

class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):
        self.train_data_loader = loader(train_path)
        self.test_data_loader = test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self):
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            if args.use_cuda:
                batch = batch.cuda()
            output = model(Variable(batch))
            pred = output.data.max(1)[1]
            correct += pred.cpu().eq(label).sum()
            total += label.size(0)

        print("Accuracy :", float(correct) / total)

        self.model.train()

    def train(self, optimizer = None, epoches=10):
        if optimizer is None:
            optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")


    def train_batch(self, optimizer, batch, label, rank_filters):

        if args.use_cuda:
            batch = batch.cuda()
            label = label.cuda()

        self.model.zero_grad()
        input = Variable(batch)

        if rank_filters:
            output = self.prunner.forward(input)
            self.criterion(output, Variable(label)).backward()
        else:
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer = None, rank_filters = False):
        for i, (batch, label) in enumerate(self.train_data_loader):
            self.train_batch(optimizer, batch, label, rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()
        self.train_epoch(rank_filters = True)
        self.prunner.normalize_ranks_per_layer()
        return self.prunner.get_prunning_plan(num_filters_to_prune)

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self,save_dir):
        #Get the accuracy before prunning
        self.test()
        self.model.train()

        #Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = 512
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)

        print("Number of prunning iterations to reduce 67% filters", iterations)

        for i in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()

            torch.save(model, save_dir + "/model_bf_"+str(i) + ".pth")

            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(model, layer_index, filter_index, use_cuda=args.use_cuda)

            torch.save(model, save_dir + "/model_at_" + str(i) + ".pth")

            self.model = model
            if args.use_cuda:
                self.model = self.model.cuda()

            message = str(100*float(self.total_num_filters()) / number_of_filters) + "%"
            print("Filters Remian ", str(message))

            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
            self.train(optimizer, epoches = 10)


        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=15)

        torch.save(model, "model_prunned")


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Load the image
def load_sample_image():
    dataset = train_dataset()
    image, label = dataset[0]
    print(f"sample data label:{label}")
    imshow(image)
    return image.unsqueeze(0)

# # Visualize feature maps
# def visualize_feature_maps(model, image, layer_indices):
#     model.eval()
#     x = image
#     feature_maps = []
#     with torch.no_grad():
#         for idx, layer in enumerate(model.features):
#             x = layer(x)
#             if idx in layer_indices:
#                 feature_maps.append(x)
#
#     for i, fmap in enumerate(feature_maps):
#         fmap_grid = make_grid(fmap[0].unsqueeze(1), nrow=8, normalize=True, scale_each=True)
#         plt.figure(figsize=(15, 15))
#         plt.title(f"Feature Maps from Layer {layer_indices[i]}")
#         plt.imshow(fmap_grid.permute(1, 2, 0))
#         plt.axis('off')
#         plt.show()
#
# def compare_visualizations(PRE_PRUNE_MODEL_PATH, POST_PRUNE_MODEL_PATH ):
#     # Load models
#
#     pre_prune_model = torch.load(PRE_PRUNE_MODEL_PATH, map_location=torch.device('cpu'))
#     post_prune_model = torch.load(POST_PRUNE_MODEL_PATH, map_location=torch.device('cpu'))
#
#     # Load sample image
#     image = load_sample_image()
#
#     # Select layers to visualize
#     layers_to_visualize = [0, 5, 10]
#
#     print("Pre-Pruning Visualizations")
#     visualize_feature_maps(pre_prune_model, image, layers_to_visualize)
#
#     print("Post-Pruning Visualizations")
#     visualize_feature_maps(post_prune_model, image, layers_to_visualize)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true",default=False)
    parser.add_argument("--prune", dest="prune", action="store_true",default=False)
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--visualize', action="store_true", default=False)
    parser.add_argument("--model_save_dir", default='./saved_model_prune')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()

    return args

def compare_model_visual(model,model_prunned,layer,num_filter):

    feature_map = get_feature_map(model, layer_name=layer, input_tensor=input_tensor)
    print(f"num_filter in layer {layer} before prune: {feature_map.squeeze(0).shape[0]}")
    visualize_feature_maps(feature_map, num_maps=num_filter, layer_name=f"Conv Layer {layer}")

    feature_map = get_feature_map(model_prunned, layer_name=layer, input_tensor=input_tensor)
    print(f"num_filter in layer {layer} after prune: {feature_map.squeeze(0).shape[0]}")
    visualize_feature_maps(feature_map, num_maps=num_filter, layer_name=f"Conv Layer {layer}")
    return


if __name__ == '__main__':
    args = get_args()
    accuracy_list = []

    if args.visualize:
        pre_model_path = "saved_model_prune/model_bf_0.pth"
        after_model_path = "saved_model_prune/model_at_4.pth"

        model = torch.load(pre_model_path, map_location=torch.device('cpu'))
        model_pruned = torch.load(after_model_path, map_location=torch.device('cpu')).eval()

        input_tensor = load_sample_image()

        compare_model_visual(model, model_pruned, "2", 16)
        compare_model_visual(model, model_pruned, "24", 16)
        plot_num_feater_change(model, model_pruned)

        # feature_map = get_feature_map(pre_prune_model, layer_name="12", input_tensor=input_tensor)  # 修改目标层
        # visualize_feature_maps(feature_map, num_maps=16, layer_name="Conv Layer 24")
        #
        # feature_map = get_feature_map(post_prune_model, layer_name="11", input_tensor=input_tensor)  # 修改目标层
        # visualize_feature_maps(feature_map, num_maps=16, layer_name="Conv Layer 24")

        # feature_map = get_feature_map(post_prune_model, layer_name="2", input_tensor=input_tensor)  # 修改目标层
        # visualize_feature_maps_combined(feature_map, num_maps=16, layer_name="Conv Layer 2")

    else:
        if args.train:
            model = ModifiedVGG16Model()
        elif args.prune:
            model = torch.load("model", map_location=lambda storage, loc: storage)

        if args.use_cuda:
            model = model.cuda()

        fine_tuner = PrunningFineTuner_VGG16(args.train_path, args.test_path, model)

        if args.train:
            fine_tuner.train(epoches=50)
            torch.save(model, "model")

        elif args.prune:
            fine_tuner.prune(save_dir=args.model_save_dir)

