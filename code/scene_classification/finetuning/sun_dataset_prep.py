#%%
### PREPARE THE DATASET   
from config import *
import random
import torchvision
import torch
from torch.utils.data import DataLoader

sun_data = torchvision.datasets.SUN397(root = cache_dir, download = True)
sun_classes = list(sun_data.class_to_idx.keys())

from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ade_data = load_dataset("scene_parse_150", cache_dir=cache_dir)
ade_classes = list(ade_data['train'].features['scene_category'].names)

from pprint import pprint
print('SUN classes:', len(sun_classes))
pprint(sun_classes[0:20])
print('ADE classes:', len(ade_classes))
print(ade_classes[0:20])
print('Classes in SUN but not in ADE:')
pprint(set(sun_classes)-set(ade_classes))