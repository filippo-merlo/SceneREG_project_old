#%%
# IMPORT LIBRARIES
import pandas as pd
import json
from config import *
from pprint import pprint
import pickle
import os

# OBJECT LIST FROM DATASET
dataset_path = '/Users/filippomerlo/Desktop/Datasets/data/coco_search18_annotated.json'
with open(dataset_path) as f:
    data = json.load(f)

objects_list = list()
for img in data.values():
    try:
        ann_list = img['instances_train2017_annotations']
    except:
        ann_list = img['instances_val2017_annotations']
    for ann in ann_list:
        id = ann['category_id']
        for cat in coco_categories:
            if cat['id'] == id:
                object_name = cat['name']
                if object_name not in objects_list:
                    objects_list.append(object_name)
print('Target Objects')
pprint(objects_list)
print(len(objects_list))

#%%
# FIRST TRY OF COUNTING COOCCURRENCIES: Looking into images json files INDEX ADE20K
# Load index with global information about ADE20K
index_ade20k_path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
with open(index_ade20k_path, 'rb') as f:
    index_ade20k = pickle.load(f)

# Get all JSON files
filenames = index_ade20k['filename']
filepaths = index_ade20k['folder']
ade20k_prepath = '/Users/filippomerlo/Desktop/Datasets/'
all_json_files = []
for folder_path in filepaths:
    # List all files in the directory
    annotation_files = os.listdir(os.path.join(ade20k_prepath, folder_path))
    # Filter JSON files
    all_json_files += [os.path.join(ade20k_prepath, folder_path, file) for file in annotation_files if file.endswith('.json')]

# viuslaize
print(all_json_files[0])
print(len(all_json_files))

# Count cooccurrences
cooccurencies = dict()
for jsonfile in all_json_files:
    try:
        with open(jsonfile, "r") as f:
            annotation = json.load(f)
            scene = annotation['annotation']['scene']
            scene = list(scene)
            scene = str(scene)
            if scene not in cooccurencies.keys():
                cooccurencies[scene] = dict()

            for object in annotation['annotation']['object']:
                if object['raw_name'] not in cooccurencies[scene].keys():
                    cooccurencies[scene][object['raw_name']] = 1
                else:
                    cooccurencies[scene][object['raw_name']] += 1
    except:
        continue

# Save cooccurencies
#with open('cooccurencies.json', 'w') as f:
#    json.dump(cooccurencies, f, indent=4)
#%%
# SECOND TRY OF COUNTING COOCCURRENCIES: Looking just into images json files INDEX ADE20K
import pickle as pkl
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm
#%%
# Load index with global information about ADE20K
DATASET_PATH = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

#%% Visualize information about ADE20K
print("File loaded, description of the attributes:")
print('--------------------------------------------')
for attribute_name, desc in index_ade20k['description'].items():
    print('* {}: {}'.format(attribute_name, desc))
print('--------------------------------------------\n')
# Get information about a specific image
i = 100
nfiles = len(index_ade20k['filename'])
file_name = index_ade20k['filename'][i]
num_obj = index_ade20k['objectPresence'][:, i].sum()
num_parts = index_ade20k['objectIsPart'][:, i].sum()

count_obj = index_ade20k['objectPresence'][:, i].max()
obj_id = np.where(index_ade20k['objectPresence'][:, i] == count_obj)[0][0]
obj_name = index_ade20k['objectnames'][obj_id]
full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])
print("The dataset has {} images".format(nfiles))
print("The image at index {} is {}".format(i, file_name))
print("It is located at {}".format(full_file_name))
print("It happens in a {}".format(index_ade20k['scene'][i]))
print("It has {} objects, of which {} are parts".format(num_obj, num_parts))
print("The most common object is object {} ({}), which appears {} times".format(obj_name, obj_id, count_obj))

#%%
# COMPUTE RELATIVE QUANTITY OF OBJECTS IN SCENES

def parse_category_name(name):
    if name.split('/')[0] != '':
        name = name.split('/')[0]
    else:
        name = name.split('/')[1]
    return name

def compute_n_objs_per_scene(index_ade20k):
    n_objs_per_scene = dict()
    n_obj_tot = dict()
    nfiles = len(index_ade20k['filename'])
    for i in range(0,nfiles):
        scene = parse_category_name(index_ade20k['scene'][i])
        if scene not in n_objs_per_scene.keys():
            n_objs_per_scene[scene] = dict()

        file_name = index_ade20k['filename'][i]
        count_obj = index_ade20k['objectPresence'][:, i]
        present_objects_idx = np.where(count_obj > 0)[0]
        # add objects with number of occurrences
        for idx in present_objects_idx:
            obj_name = index_ade20k['objectnames'][idx]
            if obj_name not in n_obj_tot.keys():
                n_obj_tot[obj_name] = count_obj[idx]
            else:
                n_obj_tot[obj_name] += count_obj[idx]

            if obj_name not in n_objs_per_scene[scene].keys():
                n_objs_per_scene[scene][obj_name] = count_obj[idx]
            else:
                n_objs_per_scene[scene][obj_name] += count_obj[idx]
    return n_objs_per_scene, n_obj_tot

# COMPUTE COOCCURRENCY OF OBJECTS IN SCENES
def compute_obj_scene_cooccurrency_matrix(index_ade20k):
    nfiles = len(index_ade20k['filename'])
    scenes_categories = [parse_category_name(scene) for scene in index_ade20k['scene']]
    scenes_categories = list(set(scenes_categories))
    cooccurencies_df = pd.DataFrame(columns=scenes_categories, index=range(len(index_ade20k['objectnames'])))
    cooccurencies_df = cooccurencies_df.fillna(0)

    for i in tqdm(range(0,nfiles)):
        scene = parse_category_name(index_ade20k['scene'][i])
        count_obj = index_ade20k['objectPresence'][:, i]
        present_objects_idx = np.where(count_obj > 0)[0]
        cooccurencies_df.loc[present_objects_idx,scene] += 1
    
    return cooccurencies_df

def compute_obj_scene_cooccurrency(os_cooccurrency_df, index = 0):

    if index == 0:
        # compute how many time an object appearn in a scene with respect to the total number of times it appears
        # how much is an object characteristic of a scene with respect to all the other scenes
        # Normalize by sum of rows
        object_occurrence_ratio_mat = os_cooccurrency_df.div(os_cooccurrency_df.sum(axis=1), axis=0)
        return object_occurrence_ratio_mat
    else:
        # compute how many time an object appears in a scene with respect of all the other objects in the same scene
        # how much is an object characteristic of a scene with respect to all the other objects in the scene
        # Normalize by sum of columns
        scene_specific_object_presence_mat = os_cooccurrency_df.div(os_cooccurrency_df.sum(axis=0), axis=1)
        return scene_specific_object_presence_mat
   
os_cooccurrency_df = compute_obj_scene_cooccurrency_matrix(index_ade20k)
#%%
co_mat = compute_obj_scene_cooccurrency(os_cooccurrency_df, 0)
object_name = 'wall'
target_index = index_ade20k['objectnames'].index(object_name)
scene = 'bathroom'
co_mat.iloc[target_index][scene]

#%%
# CHECK IF OBJECTS IN OBJECTS_LIST ARE PRESENT IN ADE20K
i = 0
tot = len(objects_list)
objects_found = list()
objects_not_found = list()

for obj in objects_list:
    for obj_data in list(index_ade20k['objectnames']):
        obj_data = obj_data.split(', ')
        if obj in obj_data:
            objects_found.append((obj,obj_data))
            i += 1
            break
        
for obj in objects_list:
    if obj not in [x[0] for x in objects_found]:
        objects_not_found.append(obj)

print(len(objects_not_found))
pprint(objects_not_found)
pprint(list(index_ade20k['objectnames']))