#%%
from config import *

# PREPARE THE DATASET   

# Load the dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ds = load_dataset("scene_parse_150", cache_dir= cache_dir)

# Remove test split
dataset = DatasetDict()
dataset = concatenate_datasets([ds['train'], ds['validation']])

# Inspect the dataset and counting the number of occurrences of each label 'name'
from collections import Counter
import numpy as np

names = dataset.features['scene_category'].names
id2names = dict(zip(range(len(names)), names))

# Count the occurrences of each label
tot_labs = dataset['scene_category']
counter = Counter(tot_labs)

# Get the labels
labels = list(counter.keys())

names2id_filtered = dict()
for label in labels:
    if counter[label] >= 10:
        if id2names[label] == 'misc':
            continue
        else:
            names2id_filtered[id2names[label]] = label

filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())

# make dicts
new_names2id = dict()
for i, name in enumerate(names2id_filtered.keys()):
    new_names2id[name] = i
# reverse dict
id2names = {v: k for k, v in new_names2id.items()}

old_2_new_map = dict()
for name, old_id in names2id_filtered.items():
    new_id = new_names2id[name]
    old_2_new_map[old_id] = new_id

# map old labels to new labels
new_labels= [old_2_new_map[x] for x in filter_dataset['scene_category']]
final_dataset = filter_dataset.remove_columns('scene_category').add_column('scene_category', new_labels)

# Redefine class labels
class_labels = ClassLabel(names=list(names2id_filtered.keys()), num_classes=len(names2id_filtered.keys()))
final_dataset =  final_dataset.cast_column('scene_category', class_labels)

# Define Collection Dataset

from PIL import Image
import torch
from torch.utils.data import Dataset
class CollectionsDataset(Dataset):
    def __init__(self, 
                 hf_dataset, 
                 transform=None):
        self.data = hf_dataset
        self.transform = transform
        self.num_classes = len(self.data.features['scene_category'].names)
        print(self.num_classes)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['scene_category']
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1

        if self.transform:
            image = self.transform(images=image, return_tensors='pt')

        return {'image': image,
                'labels': label_tensor
                }
    
# Define Training loop 

def train_model(model, 
                data_loader, 
                dataset_size, 
                optimizer, 
                scheduler, 
                num_epochs):
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        scheduler.step()
        model.train()

        running_loss = 0.0
        # Iterate over data.
        for bi, d in enumerate(data_loader):
            inputs = d["image"]
            labels = d["labels"]
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.float)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / dataset_size
        print('Loss: {:.4f}'.format(epoch_loss))
    return model

# Init model
from model import ClipModelWithClassifier

model_ft = ClipModelWithClassifier(num_labels=len(final_dataset.features['scene_category'].names))

import torch
from torchvision import transforms

# define some re-usable stuff
BATCH_SIZE = 32
device = torch.device("cuda:0")

# use the collections dataset class we created earlier
# Init dataset
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained(model_checkpoint, cache_dir=cache_dir)

train_dataset = CollectionsDataset(final_dataset, transform=processor)

# create the pytorch data loader
train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=4)
# push model to device
model_ft = model_ft.to(device)


import torch.optim as optim
from torch.optim import lr_scheduler

plist = [
        {'params': model_ft.classifier_head.parameters(), 'lr': 5e-4}
        ]
optimizer_ft = optim.Adam(plist, lr=0.001)
lr_sch = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

model_ft = train_model(model_ft,
                       train_dataset_loader,
                       len(train_dataset),
                       optimizer_ft,
                       lr_sch,
                       num_epochs=4)

torch.save(model_ft.state_dict(), "model.bin")