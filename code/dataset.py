#%%
from config import *
from utils import *
from pprint import pprint
import json
from tqdm import tqdm

class Dataset:
    def __init__(self):
        self.data = dict()

    def make_dataset(self, coco_ann_path, coco_search_ann_path, images_path):
            image_names = list()

            # 1
            images_paths = get_files(images_path)
            for image in images_paths:
                image_names.append(image.split('/')[-1])

            coco_search_ann_paths = get_files(coco_search_ann_path)
            complete_fixation_data = []
            
            # 2
            for path in coco_search_ann_paths:
                with open(path) as f:
                    fixation_data = json.load(f)
                    complete_fixation_data += fixation_data
                
            # 3
            coco_ann_paths = get_files(coco_ann_path)
            for path in coco_ann_paths:
                # name of the annotation file
                ann_name = path.split('/')[-1] + '_annotations'
                
                # load the annotation file 
                with open(path) as f:
                    coco_ann = json.load(f)

                    # iterate over the images in the annotation file
                    for image in tqdm(coco_ann['images']):
                        image_id = image['id']
                        filename = image['file_name']
                        # check if the image is in the images folder
                        if filename in image_names:

                            if filename not in self.data.keys():
                                self.data[filename] = dict()
                                self.data[filename]['fixations'] = list()

                            if ann_name not in self.data[filename].keys():
                                self.data[filename][ann_name] = list()
                            
                            for fix in complete_fixation_data:
                                if fix["name"] == filename:
                                    self.data[filename]['fixations'].append(fix)

                            for ann in coco_ann['annotations']:
                                if ann['image_id'] == image_id:
                                    self.data[filename][ann_name].append(ann)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f)
    
dataset = Dataset()
dataset.make_dataset(coco_ann_path, coco_search_ann_path, images_path)
dataset.save('/Users/filippomerlo/Desktop/Datasets/data/coco_search18_annotated.json')
#%%
