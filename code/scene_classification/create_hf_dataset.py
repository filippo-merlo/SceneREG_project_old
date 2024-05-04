#%%
from dataset import Dataset
from pprint import pprint
# load ade20k
import pickle as pkl 
ade_path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
# load index
with open(ade_path, 'rb') as f:
    index_ade20k = pkl.load(f)

scene_list = set(index_ade20k['scene'])

from datasets import load_dataset
from PIL import Image

ds = load_dataset("scene_parse_150")
scene_category = ds['train'].features['scene_category'].names
#%%
# Map scene list to scene category 
import itertools


def map_scene_to_category(scene_list, scene_category):
    category_to_scene = {}
    for scene in scene_list:
        scene_l = scene.split('/')
        scene_l = [s for s in scene_l if s]  # Remove empty strings
        scene_l = remove_duplicates(scene_l)
        combinations = []
        for r in range(1, len(scene_l) + 1):
            combinations.extend(list(itertools.combinations(scene_l, r)))

        all_permutations_of_combinations = []
        for combo in combinations:
            permutations_of_combo = list(itertools.permutations(combo))
            all_permutations_of_combinations.extend(permutations_of_combo)
        for scene_cat in scene_category:
            for perm in all_permutations_of_combinations:
                perm_str = '_'.join(perm)
                if perm_str == scene_cat:  # Changed from 'scene_category'
                    if scene_cat in category_to_scene.keys():
                        category_to_scene[scene_cat].append(scene)
                    else:
                        category_to_scene[scene_cat] = [scene]
    return category_to_scene

def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

category_to_scene = map_scene_to_category(scene_list, scene_category)
pprint(len(category_to_scene.keys()))
pprint(category_to_scene)

import json 
with open('category_to_scene.json', 'w') as f:
    json.dump(category_to_scene, f)