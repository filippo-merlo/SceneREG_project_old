#%%
### PREPARE THE DATASET   
from config import *

# Load the dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ds = load_dataset("scene_parse_150", cache_dir= cache_dir)

# Remove test split
dataset = DatasetDict()
dataset = concatenate_datasets([ds['train'], ds['validation']])

### CLUSTER LABELS
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModel, CLIPTextModel
import torch 

# cuda 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
v_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir).to(device)
txt_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir).to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir)

#%%
from tqdm import tqdm
data_points = []
captions = dict()

# preprocess and embed imgs and labels
for i in tqdm(range(len(dataset))[0:100]):
    v_inputs = processor(images=dataset[i]['image'], return_tensors="pt").to(device)
    if dataset[i]['scene_description'] not in captions:
        txt_inputs = tokenizer(f'the picture of a {dataset[i]['scene_description'].replace('_', ' ')}', return_tensors="pt").to(device)
        txt_outputs = txt_model(**txt_inputs)
        captions[dataset[i]['scene_description']] = txt_outputs.pooler_output.to('cpu')
    v_outputs = v_model(**v_inputs)
    pooled_output = v_outputs.pooler_output.to('cpu')

    data_points.append(pooled_output)

data_points = torch.stack(data_points).squeeze().detach().numpy()


from sklearn import cluster
# ---------- K-Mean clustering simplified ----------
clusters = cluster.KMeans(n_clusters=10).fit(data_points)
print(clusters.cluster_centers_.shape) # here there are the centroids (k, 768)
print(clusters.labels_)
scene_labels = list(captions.keys())
labels_emb = torch.stack(list(captions.values())).squeeze().detach().numpy()
print('labels_emb:',labels_emb.shape)
# find the labels most similar to the centroids
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(clusters.cluster_centers_, labels_emb)
print(cosine_sim.shape)

#%%
'''
### FILTER LABELS

# Inspect the dataset and counting the number of occurrences of each label 'name'
from collections import Counter

names = dataset.features['scene_category'].names
id2names = dict(zip(range(len(names)), names))

# Count the occurrences of each label
tot_labs = dataset['scene_category']
counter = Counter(tot_labs)

# Get the labels
labels = list(counter.keys())
names2id_filtered = dict()

for label in labels:
    if counter[label] >= 10:
        if id2names[label] == 'misc':
            continue
        else:
            names2id_filtered[id2names[label]] = label
filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())

# make dicts
new_names2id = dict()

for i, name in enumerate(names2id_filtered.keys()):
    new_names2id[name] = i

# reverse dict
id2names = {v: k for k, v in new_names2id.items()}
old_2_new_map = dict()

for name, old_id in names2id_filtered.items():
    new_id = new_names2id[name]
    old_2_new_map[old_id] = new_id

### ADJUST LABELS

# map old labels to new labels
new_labels= [old_2_new_map[x] for x in filter_dataset['scene_category']]
final_dataset = filter_dataset.remove_columns('scene_category').add_column('scene_category', new_labels)

# Redefine class labels
class_labels = ClassLabel(names=list(names2id_filtered.keys()), num_classes=len(names2id_filtered.keys()))
final_dataset =  final_dataset.cast_column('scene_category', class_labels)
final_dataset = final_dataset.train_test_split(test_size=0.1)
'''