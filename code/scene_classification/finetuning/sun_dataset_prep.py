#%%
### PREPARE THE DATASET   
from config import *
import random
import torchvision
import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

checkpoint = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(checkpoint, cache_dir= cache_dir)
sun_data = torchvision.datasets.SUN397(root = cache_dir, download = True)
sun_classes = sun_data.class_to_idx

from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ade_data = load_dataset("scene_parse_150", cache_dir=cache_dir)
ade_classes = ade_data['train'].features['scene_category'].names
