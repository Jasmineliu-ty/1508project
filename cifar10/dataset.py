import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def train_dataset():
    return datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)


def loader(path, batch_size=32, num_workers=4, pin_memory=True):
    return DataLoader(train_dataset(), batch_size=64, shuffle=True, num_workers=2)


def test_dataset():
    return datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


def test_loader(path, batch_size=32, num_workers=4, pin_memory=True):
    return DataLoader(test_dataset(), batch_size=64, shuffle=False, num_workers=2)




