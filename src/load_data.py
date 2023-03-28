import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import numpy as np


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
        image = Image.open(img_name)
        target_labels = self.df.iloc[idx, 1:]
        
        sample = {"image_name": img_name, "image": image, "target_labels": target_labels}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image_name, image, target_labels = sample["image_name"], sample['image'], sample['target_labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = torchvision.transforms.functional.to_tensor(image)
        return {'image_name': image_name,
                'image': image,
                'target_labels': torch.tensor(target_labels.values.astype(np.int32))}


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

    transform = torchvision.transforms.Compose([ToTensor()])
    image_dataset = MultiLabelDataset(csv_file="images_encoded.csv", transform=transform)
    image_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True)
    return image_loader

