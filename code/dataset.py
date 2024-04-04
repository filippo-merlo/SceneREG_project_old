#%%
from config import *
from utils import *
from pprint import pprint
import json
from tqdm import tqdm
import random as rn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class Dataset:

    def __init__(self, dataset_path = None):
        if dataset_path:
            with open(dataset_path) as f:
                self.data = json.load(f)
        else:
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

    def save_dataset(self, path):
        with open(path, 'w') as f:
            json.dump(self.data, f, indent=4)
    
    def visualize_img(self, img_name = None):
        
        if img_name != None:
            image = self.data[img_name]
        else:
            while img_name == None:
                img_name = rn.choice(list(self.data.keys()))
                image = self.data[img_name]
                for fix in image['fixations']:
                    if fix['condition'] == 'absent':
                        target = None
                        break
                    if 'task' in fix.keys():
                        target = fix['task']
                        break
                    else:
                        target = None
                if target == None:
                    img_name = None
                
        pprint(image)
        print('*',target)
        images_paths = get_files(images_path)
        image_picture = None
        for image_path in images_paths:
            if img_name in image_path:
                image_picture = Image.open(image_path)
                break

        # Convert PIL image to OpenCV format
        image_cv2 = cv2.cvtColor(np.array(image_picture), cv2.COLOR_RGB2BGR)

        # Draw the box on the image
        
        #for fix in image['fixations']:
        #    if 'task' in fix.keys():
        #        target = fix['task']
        #    else:
        #        target = None
        
        for ann in image['instances_train2017_annotations']:
            id = ann['category_id']
            color = (255, 0, 0)  # Red color
            for cat in coco_categories:
                if cat['id'] == id:
                    cat_name = cat['name']
                    print(cat_name)
            if target == cat_name:
                print(target)
                color = (0, 0, 255)
            

            x, y, width, height = ann['bbox']
            thickness = 2
            cv2.rectangle(image_cv2, (int(x), int(y)), (int(x + width), int(y + height)), color, thickness)

        # Convert back to PIL format for displaying
        image_with_box = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

        # Display the image with the box
        plt.imshow(image_with_box)
        plt.axis('off')  # Turn off axis
        plt.show()

dataset = Dataset(dataset_path = '/Users/filippomerlo/Desktop/Datasets/data/coco_search18_annotated.json')
#%%
dataset.visualize_img()
#%%
