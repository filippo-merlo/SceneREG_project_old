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


# OPEN ADA20K
ada20k_path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/objects.txt'
encodings = ['latin1', 'ISO-8859-1']

for encoding in encodings:
    try:
        ada20k_objects = pd.read_csv(ada20k_path, sep='\t', encoding=encoding, index_col=False )
        print("File read successfully with encoding:", encoding)
        break
    except UnicodeDecodeError:
        print("Failed to read with encoding:", encoding)
#%%
print(ada20k_objects.head())
#%%
# OPEN INDEX ADE20K
index_ade20k_path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
with open(index_ade20k_path, 'rb') as f:
    index_ade20k = pickle.load(f)

#%%
filenames = index_ade20k['filename']
filepaths = index_ade20k['folder']
ade20k_prepath = '/Users/filippomerlo/Desktop/Datasets/'
all_json_files = []
for folder_path in filepaths:
    # List all files in the directory
    annotation_files = os.listdir(os.path.join(ade20k_prepath, folder_path))
    # Filter JSON files
    all_json_files += [os.path.join(ade20k_prepath, folder_path, file) for file in annotation_files if file.endswith('.json')]
print(all_json_files[0])
print(len(all_json_files))
#%%
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

#with open('cooccurencies.json', 'w') as f:
#    json.dump(cooccurencies, f, indent=4)
#%%
import pickle as pkl
import numpy as np

# Load index with global information about ADE20K
DATASET_PATH = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)
#%%
print("File loaded, description of the attributes:")
print('--------------------------------------------')
for attribute_name, desc in index_ade20k['description'].items():
    print('* {}: {}'.format(attribute_name, desc))
print('--------------------------------------------\n')
#%%
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
cooccurencies = dict()
for i in range(0,nfiles):
    scene = index_ade20k['scene'][i]
    if scene.split('/')[0] != '':
        scene = scene.split('/')[0]
    else:
        scene = scene.split('/')[1]
    scene = scene.split('_')[0]

    if scene not in cooccurencies.keys():
        cooccurencies[scene] = dict()

    file_name = index_ade20k['filename'][i]
    count_obj = index_ade20k['objectPresence'][:, i]
    present_objects_idx = np.where(count_obj > 0)[0]
    # add objects with number of occurrences
    for idx in present_objects_idx:
        obj_name = index_ade20k['objectnames'][idx]
        if obj_name not in cooccurencies[scene].keys():
            cooccurencies[scene][obj_name] = count_obj[idx]
        else:
            cooccurencies[scene][obj_name] += count_obj[idx]

pprint(len(cooccurencies.keys()))
pprint(cooccurencies)