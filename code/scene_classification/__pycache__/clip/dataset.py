import torch
from torch.utils.data import Dataset

from config import *

### PREPARE THE DATASET   

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
final_dataset = final_dataset.train_test_split(test_size=0.1)

# Split the dataset
### Define Collection Dataset
import torch
from torch.utils.data import Dataset

class CollectionsDataset(Dataset):
    def __init__(self, 
                 hf_dataset, 
                 transform=None):
        self.data = hf_dataset
        self.transform = transform
        self.num_classes = len(self.data.features['scene_category'].names)
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

# define preprocess
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(model_checkpoint, cache_dir=cache_dir)



