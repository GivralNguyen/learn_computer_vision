"""
Shows a small example of how to use transformations (perhaps unecessarily many)

"""

# Imports
import torch
import torchvision.transforms as transforms
from torchvision.transforms.transforms import ToPILImage  # Transformations we can perform on our dataset
from torchvision.utils import save_image
import sys
from torch.utils.data import (
    Dataset,
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import os
import pandas as pd
from skimage import io
# Simple CNN

class CatsAndDogsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        #print(img_path)
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))
        #print(y_label)

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


# Load Data
my_transforms = transforms.Compose(
    [  # Compose makes it possible to have many transforms
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Resizes (32,32) to (36,36)
        transforms.RandomCrop((224, 224)),  # Takes a random (32,32) crop
        transforms.ColorJitter(brightness=0.5),  # Change brightness of image
        transforms.RandomRotation(
            degrees=45
        ),  # Perhaps a random rotation from -45 to 45 degrees
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Flips the image horizontally with probability 0.5
        transforms.RandomVerticalFlip(
            p=0.05
        ),  # Flips image vertically with probability 0.05
        transforms.RandomGrayscale(p=0.2),  # Converts to grayscale with probability 0.2
        transforms.ToTensor(),  # Finally converts PIL image to tensor so we can train w. pytorch
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),  # Note: these values aren't optimal / (value -mean)/std
    ]
)

dataset = CatsAndDogsDataset(
    csv_file="../5_custom_dataset/train_cats_dogs.csv",
    root_dir="../5_custom_dataset/cats_dogs_resized",
    transform=my_transforms
)

img_num = 0

for _ in range (10):
    for img,label in dataset:
        save_image(img,'img'+str(img_num)+'.png')