#%% IMPORTS
import json
from config import *
from pprint import pprint
import requests

# TARGET OBJECT LIST COCO-SEARCH18
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
print('Target Objects, N:',len(objects_list))

#%% LOAD ADE20K INDEX
import pickle as pkl
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm

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

#%% FUNCTIONS
# COMPUTE RELATIVE QUANTITY OF OBJECTS IN SCENES

def parse_category_name(name):
    if name.split('/')[0] != '':
        name = name.split('/')[0]
    else:
        name = name.split('/')[1]
    return name

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
    
    return cooccurencies_df.fillna(0)

def compute_obj_scene_cooccurrency(os_cooccurrency_df, index = 0):

    if index == 0:
        # compute how many time an object appearn in a scene with respect to the total number of times it appears
        # how much is an object characteristic of a scene with respect to all the other scenes
        # Normalize by sum of rows
        object_occurrence_ratio_mat = os_cooccurrency_df.div(os_cooccurrency_df.sum(axis=1), axis=0)
        return object_occurrence_ratio_mat.fillna(0)
    else:
        # compute how many time an object appears in a scene with respect of all the other objects in the same scene
        # how much is an object characteristic of a scene with respect to all the other objects in the scene
        # Normalize by sum of columns
        scene_specific_object_presence_mat = os_cooccurrency_df.div(os_cooccurrency_df.sum(axis=0), axis=1)
        return scene_specific_object_presence_mat.fillna(0)


def compute_tf_idf(cooccurencies_df):
    # Compute tf-idf of the co-occurrences matrix
    tf_idf_scores_mat = cooccurencies_df.copy()
    tf_idf_scores_mat.fillna(0, inplace=True)

    # compute tf
    ldl = cooccurencies_df.sum(axis=0) # number of objects in each scene

    # freq object in scene / total number of objects in scene
    def compute_tf(row):
        tf = row / ldl
        return tf
    tf_scores_mat = tf_idf_scores_mat.apply(compute_tf, axis=1)

    # compute idf
    N = len(cooccurencies_df.columns)
    n = cooccurencies_df.astype(bool).sum(axis=1)
    idf_values = np.log(N / n)

    # compute tf_idf
    tf_idf_scores_mat = tf_scores_mat.mul(idf_values, axis=0)

    return tf_idf_scores_mat.fillna(0)


def get_at_location_relations_for_scenes(object_name):
    # Base URL for ConceptNet API
    base_url = "http://api.conceptnet.io/"

    # Endpoint for querying relations
    endpoint = "query"

    # Parameters for the query
    params = {
        "rel": "/r/AtLocation",
        "node": f"/c/en/{object_name}",  # Specific object name
        "limit": 1000  # Limiting the number of results to 1000, you can adjust as needed
    }

    # Make the request
    response = requests.get(base_url + endpoint, params=params)

    # Check if request was successful
    if response.status_code == 200:
        data = response.json()

        # Extracting edges from the response
        edges = data['edges']

        # Filtering only the edge information
        at_location_edges = [edge for edge in edges if edge['rel']['@id'] == '/r/AtLocation']

        return at_location_edges
    else:
        # If the request fails, print the status code
        print("Request failed with status code:", response.status_code)
        return None

#%%
# Get 
#scene_categories = [parse_category_name(scene) for scene in index_ade20k['scene']]
#scene_categories = list(set(scene_categories))
#conceptnet_scene_object = dict()
#for scene in scene_categories:
#    conceptnet_scene_object[scene] = []
#    print('***',scene,'***')
#    at_location_relations = get_at_location_relations_for_scenes(scene)
#    if at_location_relations:
#        for relation in at_location_relations:
#            obj = relation['start']['label']
#            conceptnet_scene_object[scene].append(obj)
#
#pprint(conceptnet_scene_object)

# Saving the dictionary as a pickle file
#file_path = "conceptnet_scene_object.pkl"
#with open(file_path, "wb") as f:
#    pkl.dump(conceptnet_scene_object, f)

#%% COMPUTE STATISTICS
os_cooccurrency_df = compute_obj_scene_cooccurrency_matrix(index_ade20k)
tf_idf_scores_mat = compute_tf_idf(os_cooccurrency_df)
object_occurrence_ratio_mat = compute_obj_scene_cooccurrency(os_cooccurrency_df, 0)
scene_specific_object_presence_mat = compute_obj_scene_cooccurrency(os_cooccurrency_df, 1)
# Save matrices
#os_cooccurrency_df.to_pickle("object_scenes_cooccurrency.pkl")
#tf_idf_scores_mat.to_pickle("tf_idf_scores.pkl")
#%% INITIALIZE CUDA DEVICE
import torch

def name2idx(name, name_list):
    return name_list.index(name)

def idx2name(idx, name_list):
    return name_list[idx]

if torch.backends.mps.is_available():
   device = torch.device("mps")
   print('CUDA Ok')
else:
   print ("MPS device not found.")

#%% COUMPUTE SIMILARITY WITH BERT
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-nli-mean-tokens').to(device)

object_names = list(index_ade20k['objectnames'])
scene_names = list(set([parse_category_name(scene) for scene in index_ade20k['scene']]))
#Encoding:
with torch.no_grad():
    objects_name_embeddings = model.encode(object_names)
    scene_name_embeddings = model.encode(scene_names)

# cosine similarity 
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity_matrix = cosine_similarity(objects_name_embeddings, scene_name_embeddings)

# Uniform with cooccurrencies matrices Matrices
nfiles = len(index_ade20k['filename'])
scenes_categories = [parse_category_name(scene) for scene in index_ade20k['scene']]
scenes_categories = list(set(scenes_categories))
object_indexes = range(len(index_ade20k['objectnames']))
bert_similarities_mat = pd.DataFrame(columns=scenes_categories, index=object_indexes)

for scene_name in scenes_categories:
    scene_idx = name2idx(scene_name, scenes_categories)
    for obj_idx in object_indexes:
        bert_similarities_mat.loc[obj_idx,scene_name] = cosine_similarity_matrix[obj_idx,scene_idx]

#%% COMPUTE SIMILARITY WITH CLIP
from transformers import AutoTokenizer, AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Classify scene categories
object_names = list(index_ade20k['objectnames'])
scene_names = list(set([parse_category_name(scene) for scene in index_ade20k['scene']]))

#Encoding:
scenes_name_input = tokenizer(scene_names, padding=True, return_tensors="pt").to(device)
objects_name_input = tokenizer(object_names, padding=True, return_tensors="pt").to(device)

with torch.no_grad():
    # Get the image and text features
    scene_categories_features = model.get_text_features(**scenes_name_input).to('cpu')
    objects_name_features = model.get_text_features(**objects_name_input).to('cpu')

# Normalize the features
scene_categories_features = scene_categories_features / scene_categories_features.norm(dim=-1, keepdim=True)
objects_name_features = objects_name_features / objects_name_features.norm(dim=-1, keepdim=True)

# Compute similarity
def similarity_score(features1, features2):
    return (features1 @ features2.T).mean(dim=1)

# Compute similarity scores
clip_similarities = similarity_score(objects_name_features, scene_categories_features)

# Uniform with cooccurrencies matrices Matrices
scenes_categories = [parse_category_name(scene) for scene in index_ade20k['scene']]
object_indexes = range(len(index_ade20k['objectnames']))
clip_similarities_mat = pd.DataFrame(columns=scenes_categories, index=object_indexes)

print(clip_similarities_mat.shape)
clip_similarities_mat.head()

#%% 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def mat_correlation(df1, df2):
    # Calculate correlation matrix
    correlation_matrix = df1.corrwith(df2, axis=1)
    
    # Convert correlation matrix to DataFrame
    correlation_df = pd.DataFrame(correlation_matrix, columns=['Correlation'])
    
    # Handle missing values (if any)
    correlation_df.fillna(0, inplace=True)  # Fill missing values with 0
    
    # Ensure data types are numeric
    correlation_df['Correlation'] = pd.to_numeric(correlation_df['Correlation'])
    
    return correlation_df

def array_corr(df1,df2):
        # Concatenate all rows of each dataset separately
    concatenated_df1 = [df1.iloc[i].values for i in range(len(df1))]
    concatenated_df2 = [df2.iloc[i].values for i in range(len(df2))]

    array_1 = list()
    for i in concatenated_df1:
        array_1.extend(i)
    
    array_2 = list()
    for i in concatenated_df2:
        array_2.extend(i)

    # Compute correlation between the arrays
    correlation = np.corrcoef(array_1, array_2)
    return correlation

# tf_idf_scores_mat
# object_occurrence_ratio_mat
# scene_specific_object_presence_mat
# bert_similarities_mat

relatedness_metrics = {
    'tf-idf scores': tf_idf_scores_mat,
    't-scene vs other scenes': object_occurrence_ratio_mat,
    't-object vs other objects': scene_specific_object_presence_mat,
    'BERT similarities': bert_similarities_mat
}

# Set to keep track of printed pairs
printed_pairs = set()

# Iterate through each pair of metrics
for metric_name, metric_df in relatedness_metrics.items():
    for metric_name2, metric_df2 in relatedness_metrics.items():
        if metric_name != metric_name2 and (metric_name, metric_name2) not in printed_pairs and (metric_name2, metric_name) not in printed_pairs:
            # Print correlation between the two metrics
            print('Corr: \n* {}\n* {}'.format(metric_name, metric_name2))
            correlation_matrix = mat_correlation(metric_df, metric_df2)
            correlation_matrix = correlation_matrix.sort_values(by='Correlation', ascending=False)
            # Plot heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix')
            plt.show()
            print(array_corr(metric_df, metric_df2))
            print('\n\n')
            # Add the pair to printed_pairs set
            printed_pairs.add((metric_name, metric_name2))

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
