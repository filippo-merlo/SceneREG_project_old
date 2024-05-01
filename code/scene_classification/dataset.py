#%% LOAD ADE20K INDEX
import pickle as pkl
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
from PIL import Image

# Load index with global information about ADE20K
DATASET_PATH = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

def parse_category_name(name):
    if name.split('/')[0] != '':
        name = name.split('/')[0]
    else:
        name = name.split('/')[1]
    where_specifiers = ['_indoor','_outdoor']
    for specifier in where_specifiers:
        if specifier in name:
            name = name.replace(specifier, '')
    return name
    
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, clip = False):
        self.annotation_file = annotations_file
        self.img_labels = [parse_category_name(x) for x in annotations_file['scene']]
        self.img_dir = img_dir
        self.clip = clip

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        relative_img_path = '{}/{}'.format(self.annotation_file['folder'][idx], self.annotation_file['filename'][idx])
        img_path = os.path.join(self.img_dir.replace('/ADE20K_2021_17_01',''), relative_img_path)
        if self.clip:
            image = Image.open(img_path)
        else:
            image = read_image(img_path)
        label = self.img_labels[idx]
        return image, label


#%%
from datasets import load_dataset
ds = load_dataset("sezer12138/ade20k_image_classification")
import json
with open('/Users/filippomerlo/Documents/GitHub/SceneReg_project/code/scene_classification/data_label2id.json', 'r') as f:
    data_label2id = json.load(f)
#%%
from pprint import pprint 
name_list = list(set([parse_category_name(x) for x in index_ade20k['scene']]))
name_list2 = list(set([parse_category_name(x) for x in list(data_label2id.keys())]))
print(len(name_list))
print(len(name_list2))
#%%
unique_1 = list(set(name_list) - set(name_list2))
print(len(unique_1))
unique_2 = list(set(name_list2) - set(name_list))
print(len(unique_2))

disjunction = list(set(name_list) ^ set(name_list2))
print(len(disjunction))
conjunction = list(set(name_list) & set(name_list2))
print(len(conjunction))

