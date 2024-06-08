#%%
### PREPARE THE DATASET   
from config import *
import random
import torchvision
import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

checkpoint = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(checkpoint, cache_dir= cache_dir)

def transform(image):
    # Take a list of PIL images and turn them to pixel values
    image = processor(image.convert('RGB'), return_tensors='pt')
    # Don't forget to include the labels!
    return image



sun_data = torchvision.datasets.SUN397(root = cache_dir, transform=transform,  download = True)
dt = DataLoader(sun_data, batch_size = 2)
for batch in dt:
    print(len(batch))        
'''
#%%
# Inspect the dataset and counting the number of occurrences of each label 'name'
names = dataset.features['scene_category'].names
id2names = dict(zip(range(len(names)), names))

#%%
# Count the occurrences of each label
tot_labs = dataset['scene_category']
counter = Counter(tot_labs)

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