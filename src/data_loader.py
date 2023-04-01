import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from PIL import Image
import numpy as np

# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html used as help

header = ["Filename", "baby", "bird", "car", "clouds", "dog", "female", "flower", "male", "night", "people", "portrait", "river", "sea", "tree"]


class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, root_dir="../data/images/", transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.df.iloc[idx,0])
        image = Image.open(img_name).convert("RGB")
        target_labels = self.df.iloc[idx, 1:]
        
        sample = {"image_name": img_name, "image": image, "target_labels": target_labels}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class TestDataset(Dataset):
    def __init__(self, root_dir="../data/test/images/", transform=None):
        self.image_names = os.listdir(root_dir)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.image_names[idx])
        image = Image.open(img_name).convert("RGB")
        
        sample = {"image_name": self.image_names[idx], "image": image}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class Preprocess(object):
    def __call__(self, sample):
        image_name, image = sample["image_name"], sample['image']
        target_labels = sample.get('target_labels', None)

        image = torchvision.transforms.functional.resize(image, 256)
        image = torchvision.transforms.functional.center_crop(image, 224)
        image = torchvision.transforms.functional.to_tensor(image)
        image = torchvision.transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if target_labels is not None:
            return {
                'image_name': image_name,
                'image': image,
                'target_labels': torch.tensor(target_labels.values.astype(np.float32))
            }
        else:
            return {'image_name': image_name, 'image': image}


def image_has_label(image_name, label):
    image_number = re.sub('\D', '', image_name)
    with open("../data/annotations/{}.txt".format(label)) as file:
        for line in file:
            if line.strip() == image_number:
                return 1
    return 0
    

def load_data(batch_size):
    if not os.path.isfile("../data/images_encoded.csv"):
        data = []
        for image_name in os.listdir("../data/images/"):
            row = []
            row.append(image_name)
            for label in header[1:]:
                row.append(image_has_label(image_name, label))
            data.append(row)


        df = pd.DataFrame(data, columns=header)
        df.head()
        df.to_csv("../data/images_encoded.csv", index=False)

    transform = torchvision.transforms.Compose([Preprocess()])
    image_dataset = MultiLabelDataset(csv_file="../data/images_encoded.csv", transform=transform)
    split_dataset = random_split(image_dataset, [0.85, 0.15])
    image_dataset_train = split_dataset[0]
    image_dataset_validation = split_dataset[1]
    image_loader_train = DataLoader(image_dataset_train, batch_size=batch_size, shuffle=True)
    image_loader_validation = DataLoader(image_dataset_validation, batch_size=batch_size, shuffle=True)
    return image_loader_train, image_loader_validation

def load_test_data(batch_size):
    transform = torchvision.transforms.Compose([Preprocess()])
    image_dataset_test = TestDataset(transform=transform)
    image_dataset_test.image_names.sort()
    image_loader_test = DataLoader(image_dataset_test, batch_size=batch_size, shuffle=False)
    return image_loader_test