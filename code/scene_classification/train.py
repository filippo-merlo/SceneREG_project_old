#%%
from dataset import *

dataset = CustomImageDataset(index_ade20k, DATASET_PATH, True)
print(dataset.__len__())