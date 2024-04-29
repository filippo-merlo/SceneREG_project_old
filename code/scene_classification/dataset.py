#%% LOAD ADE20K INDEX
import pickle as pkl
from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import torch
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image

if torch.backends.mps.is_available():
   device = torch.device("mps")
   print('CUDA Ok')
else:
   print ("MPS device not found.")

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
    