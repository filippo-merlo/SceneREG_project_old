#%% Get COCO-Search-18 image names

### FUNCTIONS ###
import os
import json
from pprint import pprint

# import json files from a folder
def import_json_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Filter only JSON files
    json_files = [f for f in files if f.endswith('.json')]
    
    # Initialize an empty list to store loaded JSON data
    json_data = []
    
    # Iterate over each JSON file
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            try:
                # Load JSON data from the file
                data = json.load(file)
                json_data.append(data)
            except Exception as e:
                print(f"Error loading JSON from {file_name}: {e}")
    
    return json_data
#%%
### EXECUTION ###

# import all JSON files from the folder
folder_path = '../data/coco_search18_TP'
all_sets = import_json_files(folder_path)
pprint(all_sets[0])
image_ids = []
for set in all_sets:
    for trial in set:
        if trial['name'] not in image_ids:
            image_ids.append(trial['name'])

print(len(image_ids))


#%%
from pycocotools.coco import COCO
import requests
from PIL import Image

dataType=['train2017','val2017']
annFile='/Users/filippomerlo/Documents/GitHub/MakingAScene/data/coco_annotations/annotations/instances_val2017.json'
# initialize COCO api for instance annotations
coco=COCO(annFile)

#%%
i = 0
for filename in image_ids:
    try:
        imgIds = coco.getImgIds(imgIds=coco.getImgIds(filename=filename))
        print(imgIds)
        image = coco.loadImgs(imgIds)
    except:
        i += 1
        continue
    # Define the path where you want to save the image
    folder_path = "/Users/filippomerlo/Desktop/Datasets/coco-search18/"

    # Save the image to the specified folder with a new name if desired
    image.save(folder_path + filename)