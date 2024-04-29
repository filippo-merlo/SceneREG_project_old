#%%
from dataset import *
from models import *

#Evaluate CLIP 

dataset = CustomImageDataset(index_ade20k, DATASET_PATH, True)
model = clipModel()
