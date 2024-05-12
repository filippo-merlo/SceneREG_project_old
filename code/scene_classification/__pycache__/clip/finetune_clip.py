#%%
from config import *

# PREPARE THE DATASET   

# Load the dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ds = load_dataset("scene_parse_150", cache_dir= cache_dir)

# Remove test split
dataset = DatasetDict()
dataset['train'] = ds['train']
dataset['validation'] = ds['validation']

# Inspect the dataset and counting the number of occurrences of each label 'name'
from collections import Counter
import numpy as np

names = dataset['train'].features['scene_category'].names
names2id = dict(zip(names, range(len(names))))
id2names = dict(zip(range(len(names)), names))

# Count the occurrences of each label
tot_labs = dataset['train']['scene_category'] + dataset['validation']['scene_category']
counter = Counter(tot_labs)

# Get the labels
labels = list(counter.keys())

names2id_filtered = dict()
for label in labels:
    if counter[label] >= 10:
        names2id_filtered[id2names[label]] = label

filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())
ds =  concatenate_datasets([filter_dataset['train'], filter_dataset['validation']])
splitted_dataset = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
final_dataset = DatasetDict()
final_dataset['train'] = splitted_dataset['train']
final_dataset['validation'] = splitted_dataset['test']

cl_lab = ClassLabel(names=list(names2id_filtered.keys()), num_classes=len(names2id_filtered.keys()))
final_dataset['train'] =  final_dataset['train'].cast_column('scene_category', cl_lab)
final_dataset['validation'] = final_dataset['validation'].cast_column('scene_category', cl_lab)

new_names2id = dict()
for i, name in enumerate(names2id_filtered.keys()):
    new_names2id[name] = i

old_2_new_map = dict()
for name, old_id in names2id_filtered.items():
    new_id = new_names2id[name]
    old_2_new_map[old_id] = new_id

new_train_l= [old_2_new_map[x] for x in final_dataset['train']['scene_category']]
new_valid_l = [old_2_new_map[x] for x in final_dataset['validation']['scene_category']]
final_dataset['train'] = final_dataset['train'].remove_columns('scene_category').add_column('scene_category', new_train_l)
final_dataset['validation'] = final_dataset['validation'].remove_columns('scene_category').add_column('scene_category', new_valid_l)

# 'translate' dataset 
ds = concatenate_datasets([final_dataset['train'], final_dataset['validation']])
# %%

#%%

# Define Collection Dataset

from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor

class CollectionsDataset(Dataset):
    def __init__(self, 
                 csv_file, 
                 root_dir, 
                 num_classes, 
                 transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 
                                self.data.loc[idx, 'id'] + '.png')
        image = Image.open(img_name)
        labels = self.data.loc[idx, 'attribute_ids']
        labels = labels.split()

        label_tensor = torch.zeros(self.num_classes)
        for i in labels:
            label_tensor[int(i)] = 1

        if self.transform:
            image = self.transform(image)

        return {'image': image,
                'labels': label_tensor
                }