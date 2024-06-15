import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx

from model import MultiOutputModel
from dataset import FashionDataset, AttributesDataset, mean, std
import torchvision.transforms as transforms

start_epoch = 1
N_epochs = 50
batch_size = 8
num_workers = 4  # number of processes to handle dataset loading
device = 0
device = torch.device("cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu")

attributes = AttributesDataset(args.attributes_file)

# specify image transforms for augmentation during training
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.breedJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
    transforms.ToTensor(),
    transforms.Resize((640, 640)),
    transforms.Normalize(mean, std)
])

# during validation we use only tensor and normalization transforms
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((640, 640)),
    transforms.Normalize(mean, std)
])

multioutput_model = MultiOutputModel(n_breed_classes=attributes.num_breed,
                         n_hair_classes=attributes.num_hair).to(device)