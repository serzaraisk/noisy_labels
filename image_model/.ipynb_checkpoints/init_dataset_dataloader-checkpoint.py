# define my custom dataset for the task
import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision import transforms
import torch
import re
import pandas as pd

class IMagesDataset(Dataset):
    def __init__(self, data_root, lookup_table, transform=None):
        
        self.samples = []
        self.lookup_table = pd.read_csv(lookup_table)

        for label in os.listdir(data_root):
            label_folder = os.path.join(data_root, label)

            for pic in os.listdir(label_folder):
                pic_filepath = os.path.join(label_folder, pic)
                self.samples.append((label, pic_filepath))
    
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label, image = sample
        index = int(re.search('\d+.jpg', image)[0][:-4])
        image = Image.open(image)
        
        x, y = image.size
        maxside = max(x, y)
        delta_w = maxside - x
        delta_h = maxside - y
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2),
        )
        image = ImageOps.expand(image, padding)

        if self.transform:
            image = self.transform(image)
        return image, int(label), self.lookup_table.loc[index, 'answer']

    
def create_datasets(folder_name, image_size):
    
    train_transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(degrees=(-15, 15)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                     ])

    valid_test_transform = transforms.Compose([transforms.Resize(image_size),
                                transforms.CenterCrop(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                          ])
    
    
    
    train_dataset = IMagesDataset(folder_name + 'train','lookup_tables/train.csv', transform=train_transform)
    val_dataset = IMagesDataset(folder_name + 'val','lookup_tables/valid.csv', transform=valid_test_transform)
    test_dataset = IMagesDataset(folder_name + 'test','lookup_tables/test.csv', transform=valid_test_transform)
    return train_dataset, val_dataset, test_dataset

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers):
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader