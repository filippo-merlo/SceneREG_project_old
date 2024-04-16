#%%
import pandas as pd
import json
from config import *
from pprint import pprint
import pickle
import os

#%%
# Measure semnatic relatedness between objects and scenes
# get object list from dataset
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
print(len(objects_list))

#%%
# using cococcurencies in ADA20K dataset
ada20k_path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/objects.txt'

encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']

for encoding in encodings:
    try:
        ada20k_objects = pd.read_csv(ada20k_path, sep='\t', encoding=encoding, index_col=False )
        print("File read successfully with encoding:", encoding)
        break
    except UnicodeDecodeError:
        print("Failed to read with encoding:", encoding)
# fucntion to find the most similar object according to these properties

ada20k_objects.columns

#%%
ade20k_prepath = '/Users/filippomerlo/Desktop/Datasets/'
index_ade20k_path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'

with open(index_ade20k_path, 'rb') as f:
    index_ade20k = pickle.load(f)
pprint(index_ade20k.keys())
#%%
'''
'filename': 'array of length N=27574 with the image file '
                             'names',
                 'folder': 'array of length N with the image folder names.',
                 'objectIsPart': 'array of size [C, N] counting how many times '
                                 'an object is a part in each image. '
                                 'objectIsPart[c,i]=m if in image i object '
                                 'class c is a part of another object m times. '
                                 'For objects, objectIsPart[c,i]=0, and for '
                                 'parts we will find: objectIsPart[c,i] = '
                                 'objectPresence(c,i)',
                 'objectPresence': 'array of size [C, N] with the object '
                                   'counts per image. objectPresence(c,i)=n if '
                                   'in image i there are n instances of object '
                                   'class c.',
                 'objectcounts': 'array of length C with the number of '
                                 'instances for each object class.',
                 'objectnames': 'array of length C with the object class '
                                'names.',
                 'proportionClassIsPart': 'array of length C with the '
                                          'proportion of times that class c '
                                          'behaves as a part. If '
                                          'proportionClassIsPart[c]=0 then it '
                                          'means that this is a main object '
                                          '(e.g., car, chair, ...). See bellow '
                                          'for a discussion on the utility of '
                                          'this variable.',
                 'scene': 'array of length N providing the scene name (same '
                          'classes as the Places database) for each image.',
                 'wordnet_found': 'array of length C. It indicates if the '
                                  'objectname was found in Wordnet.',
                 'wordnet_frequency': 'array of length C. How many times each '
                                      'wordnet appears',
                 'wordnet_gloss': 'list of length C. WordNet definition.',
                 'wordnet_hypernym': 'list of length C. WordNet hypernyms for '
                                     'each object name.',
                 'wordnet_level1': 'list of length C. WordNet associated.',
                 'wordnet_synonyms': 'list of length C. Synonyms for the '
                                     'WordNet definition.',
                 'wordnet_synset': 'list of length C. WordNet synset for each '
                                   'object name. Shows the full hierarchy '
                                   'separated by .'
'''
filenames = index_ade20k['filename']
filepaths = index_ade20k['folder']

for folder_path in filepaths:
    # List all files in the directory
    annotation_files = os.listdir(os.path.join(ade20k_prepath, folder_path))
    # Filter JSON files
    json_files = [file for file in annotation_files if file.endswith('.json')]
    for ann_file in json_files:
        with open(os.path.join(ade20k_prepath, folder_path, ann_file)) as f:
            ann = json.load(f)
            pprint(ann)
        break







#%%
scene_labels = index_ade20k['scene']
objectnames = index_ade20k['objectnames']
objectclasses = list(ada20k_objects[' ADE names '].values)
i = 0
final_objects = list()
discarded_objects = list()
print(len(objectclasses))
for cls in objectclasses:
    clss = cls.split(',')
    for c in clss:
        for obj in objectnames:
            objs = obj.split(',')
            for o in objs:
                if c == o:
                    final_objects.append((cls,obj))

pprint(final_objects)
#%%
for ada_name in objectnames:
    if ada_name == ' ':
        continue
    if ada_name in objects_list:
        final_objects.append(ada_name)
    else:
        wordnetnames = ada20k_objects[ada20k_objects[' ADE names '].str.contains(ada_name, na=False)]['Wordnet name '].values
        for wn in wordnetnames:
            for data_name in objects_list:
                if data_name == wn:
                    final_objects.append((ada_name,data_name,wn))

pprint(final_objects)

