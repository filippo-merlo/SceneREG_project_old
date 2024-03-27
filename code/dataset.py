#%%
from config import *
from utils import *

class Dataset:
    def __init__(self, coco_ann_path, coco_search_ann_path, images_path):
        coco_ann_paths = get_files(coco_ann_path)
        coco_search_ann_paths = get_files(coco_search_ann_path)
        images = get_files(images_path)
        
        
    
    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return self.image[index], self.targets[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f'Dataset({len(self)})'

    def __str__(self):
        return self.__repr__()

dataset = Dataset(coco_ann_path, coco_search_ann_path, images_path)