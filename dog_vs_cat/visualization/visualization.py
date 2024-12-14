import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
import numpy as np

class ModifiedVGG16Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 加载模型
model = torch.load("model", map_location=torch.device('cpu')).eval()
model_pruned = torch.load("model_prunned", map_location=torch.device('cpu')).eval()

# 定义预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def visualize_model_output(image_path, models, model_names, transform):
    """
    显示输入图片及多个模型的输出。
    Args:
        image_path (str): 输入图片路径。
        models (list): 要对比的模型列表。
        model_names (list): 对应模型名称列表。
        transform (callable): 图像预处理方法。
    """
    # 加载并预处理图片
    original_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0)

    # 显示输入图片
    plt.figure(figsize=(10, len(models) * 4))
    plt.subplot(len(models) + 1, 1, 1)
    plt.imshow(original_image)
    plt.title("Input Image")
    plt.axis("off")

    # 获取每个模型的输出
    for i, (model, name) in enumerate(zip(models, model_names)):
        with torch.no_grad():
            output = model(input_tensor)
            # 如果是分类模型，取输出的分类分数
            output_image = output[0].cpu().numpy()
            output_image = output_image - output_image.min()  # 归一化
            output_image = output_image / output_image.max()

        # 显示输出图片
        print(output_image.shape)
        plt.subplot(len(models) + 1, 1, i + 2)
        plt.imshow(output_image.transpose(1, 2, 0))
        plt.title(f"Output from {name}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def get_feature_map(model, layer_name, input_tensor):
    """
    获取指定层的特征图。
    Args:
        model (torch.nn.Module): 模型对象。
        layer_name (str): 层的名称。
        input_tensor (torch.Tensor): 输入张量。
    Returns:
        feature_map (torch.Tensor): 特征图。
    """
    activation = {}

    def hook_fn(module, input, output):
        activation['feature_map'] = output

    # 注册钩子
    layer = dict(model.features._modules)[layer_name]
    hook = layer.register_forward_hook(hook_fn)

    # 前向传播
    with torch.no_grad():
        _ = model(input_tensor)

    hook.remove()
    return activation['feature_map']

# 可视化特定层的特征图 
def visualize_feature_maps(feature_maps, num_maps=16, layer_name=""):
    """
    可视化部分特征图并显示对应的通道索引。
    Args:
        feature_maps (torch.Tensor): 特征图张量，形状为 (C, H, W)。
        num_maps (int): 显示的特征图数量。
        layer_name (str): 层的名称，用于标题。
    """
    feature_maps = feature_maps.squeeze(0).cpu()  # 去掉 batch 维度
    num_maps = min(num_maps, feature_maps.shape[0])  # 确保不超出通道数量

    plt.figure(figsize=(12, 12))
    for i in range(num_maps):
        plt.subplot(4, 4, i + 1)
        plt.imshow(feature_maps[i].numpy(), cmap="viridis")
        plt.title(f"{layer_name}: Channel {i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# 示例：可视化特定层的特征图
# print(model)
# input_tensor = transform(Image.open("test\dogs\dog.662.jpg").convert("RGB")).unsqueeze(0)
# feature_map = get_feature_map(model_pruned, layer_name="24", input_tensor=input_tensor)  # 修改目标层
# visualize_feature_maps(feature_map, num_maps=16, layer_name="Conv Layer 24")


def visualize_feature_maps_combined(feature_maps, num_maps=16, layer_name=""):
    """
    可视化部分特征图，将多个通道合并成一张图片显示，并标注层名称。
    Args:
        feature_maps (torch.Tensor): 特征图张量，形状为 (C, H, W)。
        num_maps (int): 显示的特征图数量。
        layer_name (str): 层的名称，用于标题。
    """
    feature_maps = feature_maps.squeeze(0).cpu()  # 去掉 batch 维度
    num_maps = min(num_maps, feature_maps.shape[0])  # 确保不超出通道数量
    
    # 计算每个通道的归一化图像 (0-255 范围)
    normalized_maps = []
    for i in range(num_maps):
        feature = feature_maps[i].numpy()
        feature_min, feature_max = feature.min(), feature.max()
        normalized = ((feature - feature_min) / (feature_max - feature_min) * 255).astype(np.uint8)
        normalized_maps.append(normalized)
    
    # 将特征图组合成一个 4x4 网格
    grid_size = int(np.ceil(np.sqrt(num_maps)))
    height, width = normalized_maps[0].shape
    combined_image = np.zeros((grid_size * height, grid_size * width), dtype=np.uint8)

    for idx, feature in enumerate(normalized_maps):
        row = idx // grid_size
        col = idx % grid_size
        combined_image[row * height: (row + 1) * height, col * width: (col + 1) * width] = feature
    
    # 显示合并后的图像
    plt.figure(figsize=(10, 10))
    plt.imshow(combined_image, cmap="viridis")
    # plt.title(f"{layer_name}: First {num_maps} Channels")
    plt.axis("off")
    plt.show()

# 示例：可视化特定层的特征图
input_tensor = transform(Image.open("test\cats\cat.126.jpg").convert("RGB")).unsqueeze(0)
feature_map = get_feature_map(model, layer_name="0", input_tensor=input_tensor)  # 修改目标层
visualize_feature_maps_combined(feature_map, num_maps=4, layer_name="Conv Layer 0")
