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
from transformers import AutoProcessor, AutoTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection
import torch 

# cuda 
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
   
v_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir).to(device)
txt_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir).to(device)
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32", cache_dir= cache_dir)

#%%
# Remove misc
scene_names = list(dataset.features['scene_category'].names)
names2id = dict(zip(scene_names, range(len(scene_names))))
names2id_filtered = dict()
for label in scene_names:
    if label == 'misc':
        continue
    else:
        names2id_filtered[label] = names2id[label]
filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())


from tqdm import tqdm
import numpy as np 
data_points = []
captions = dict()

for c_l in scene_names:
    txt_inputs = tokenizer(f'the picture of a {c_l.replace('_', ' ')}', return_tensors="pt").to(device)
    txt_outputs = txt_model(**txt_inputs)
    captions[c_l] = txt_outputs.text_embeds.to('cpu')

# preprocess and embed imgs and labels
for i in tqdm(range(len(filter_dataset))):
    v_inputs = processor(images=filter_dataset[i]['image'], return_tensors="pt").to(device)
    v_outputs = v_model(**v_inputs)
    image_embeds = v_outputs.image_embeds.to('cpu')

    data_points.append(image_embeds)

data_points = torch.stack(data_points).squeeze().detach().numpy()


from sklearn import cluster
# ---------- K-Mean clustering simplified ----------
clusters = cluster.KMeans(n_clusters=100).fit(data_points)
print(clusters.cluster_centers_.shape) # here there are the centroids (k, 768)
scene_labels = list(captions.keys())
labels_emb = torch.stack(list(captions.values())).squeeze().detach().numpy()
# find the labels most similar to the centroids
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(clusters.cluster_centers_, labels_emb)
print(cosine_sim.shape)
idxs = np.argmax(cosine_sim, axis=1)
for idx in idxs:
    print(scene_labels[idx])

# save the labels
new_labels = {
    'scene_labels' : scene_labels,
    'scene_ids' : idxs,
    'img_label_ass' : labels_emb
}

# save new_labels dict in json format
import json 
with open('new_labels.json', 'w') as f:
    json.dump(new_labels, f)

#%%
'''
### FILTER LABELS

# Inspect the dataset and counting the number of occurrences of each label 'name'
from collections import Counter
import json

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