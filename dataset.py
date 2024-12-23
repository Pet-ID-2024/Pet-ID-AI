import csv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
mean= [0.485]
std = [0.229]

class AttributesDataset():
    def __init__(self, annotation_path):
        breed_labels = []
        hair_labels = []
        weight_labels = []
        color_labels = []

        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                breed_labels.append(row['breed'])
                hair_labels.append(row['classes'])

                row['weight'] = int(float(row['weight']))
                if row['weight'] >= 30:
                    row['weight'] = 30
                weight_labels.append(row['weight'])
                color_labels.append(row['color'])

        self.breed_labels = np.unique(breed_labels)
        self.hair_labels = np.unique(hair_labels)
        self.weight_labels = np.unique(weight_labels)
        self.color_labels = np.unique(color_labels)

        self.num_breed = len(self.breed_labels)
        self.num_hair = len(self.hair_labels)
        self.num_weight = len(self.weight_labels)
        self.num_color = len(self.color_labels)

        self.breed_id_to_name = dict(zip(range(len(self.breed_labels)), self.breed_labels))
        self.breed_name_to_id = dict(zip(self.breed_labels, range(len(self.breed_labels))))

        print(self.breed_id_to_name)
        self.hair_id_to_name = dict(zip(range(len(self.hair_labels)), self.hair_labels))
        self.hair_name_to_id = dict(zip(self.hair_labels, range(len(self.hair_labels))))

        print(self.hair_id_to_name)
        self.weight_id_to_name = dict(zip(range(len(self.weight_labels)), self.weight_labels))

        print(self.weight_id_to_name)
        self.weight_name_to_id = dict(zip(self.weight_labels, range(len(self.weight_labels))))

        self.color_id_to_name = dict(zip(range(len(self.color_labels)), self.color_labels))
        print(self.color_id_to_name)
        self.color_name_to_id = dict(zip(self.color_labels, range(len(self.color_labels))))

class FashionDataset(Dataset):
    def __init__(self, annotation_path, attributes, transform=None):
        super().__init__()

        self.transform = transform
        self.attr = attributes

        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.weight_labels = []
        self.breed_labels = []
        self.hair_labels = []
        self.color_labels = []

        # read the annotations from the CSV file
        with open(annotation_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['cropped_image_id'])
                # row['weight'] = int(float(row['weight']))
                row['weight'] = float(row['weight'])

                if row['weight'] >= 30:
                    row['weight'] = 30

                # self.weight_labels.append(self.attr.weight_name_to_id[row['weight']])
                self.weight_labels.append(row['weight'])
                self.breed_labels.append(self.attr.breed_name_to_id[row['breed']])
                self.hair_labels.append(self.attr.hair_name_to_id[row['classes']])
                self.color_labels.append(self.attr.color_name_to_id[row['color']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]

        # Check if the image file exists
        if not os.path.exists(img_path):
            print(f"File does not exist: {img_path}")
            return None  # Or handle this case as needed

        try:
            # read image
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None  # Or handle this case as needed

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)

        # return the image and all the associated labels
        dict_data = {
            'img': img,
            'labels': {
                'breed_labels': self.breed_labels[idx],
                'hair_labels': self.hair_labels[idx],
                'weight_labels': self.weight_labels[idx],
                'color_labels': self.color_labels[idx]
            }
        }
        return dict_data
