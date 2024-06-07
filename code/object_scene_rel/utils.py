#%%
import os 
from config import *
from pprint import pprint
import torch

# Is MPS even available? macOS 12.3+
device = torch.device("mps")

def get_files(directory):
    """
    Get all files in a directory with specified extensions.

    Args:
    - directory (str): The directory path.
    - extensions (list): A list of extensions to filter files by.

    Returns:
    - files (list): A list of file paths.
    """
    files = []
    for file in os.listdir(directory):
        if file.endswith(tuple([".json",".jpg"])):
            files.append(os.path.join(directory, file))
    return files

def print_dict_structure(dictionary, ind = ''):
    for key, value in dictionary.items():
        print(f"{ind}Key: [{key}], Type of Value: [{type(value).__name__}]")
        if isinstance(value, dict):
            ind2 = ind + '  '
            print_dict_structure(value, ind2)
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], dict):
                ind2 = ind + '  '
                print_dict_structure(value[0], ind2)


# vit model 
import wandb
from transformers import ViTForImageClassification, AutoImageProcessor
#%%
scene_labels_vit = ['natural', 'street', 'river', 'bathroom', 'highway', 'misc', 'staircase', 'museum_indoor', 'building_facade', 'home_office', 'creek', 'house', 'skyscraper', 'kitchen', 'attic', 'living_room', 'reception', 'bedroom', 'dinette_home', 'shoe_shop', 'corridor', 'exterior', 'art_gallery', 'garage_indoor', 'alley', 'apartment_building_outdoor', 'parking_lot', 'hotel_room', 'wild', 'game_room', 'mountain', 'office', 'vehicle', 'beach', 'conference_room', 'broadleaf', 'jacuzzi_indoor', 'dining_room', 'waiting_room', 'pasture', 'warehouse_indoor', 'cultivated', 'childs_room', 'airport_terminal', 'castle', 'coast', 'lighthouse', 'nursery', 'window_seat', 'shop', 'parlor', 'bridge', 'art_studio', 'lobby', 'classroom', 'mountain_snowy', 'poolroom_home', 'dorm_room', 'cockpit', 'youth_hostel', 'closet', 'bar', 'needleleaf', 'roundabout', 'playroom', 'casino_indoor', 'valley', 'park', 'amusement_park']
print(len(scene_labels_vit))
#%%
# Create the label to ID mapping
label2id = {label: idx for idx, label in enumerate(scene_labels_vit)}

# Reverse the mapping to create ID to label mapping
id2label = {idx: label for label, idx in label2id.items()}
# Create a new run
with wandb.init(project="vit-base-patch16-224") as run:
    # Pass the name and version of Artifact
    my_model_name = "model-6evzp0q6:v0"
    my_model_artifact = run.use_artifact(my_model_name)

    # Download model weights to a folder and return the path
    model_dir = my_model_artifact.download()

    # Load your Hugging Face model from that folder
    #  using the same model class
    vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    vit_model = ViTForImageClassification.from_pretrained(
        model_dir,
        num_labels=len(scene_labels_vit),
        id2label=id2label,
        label2id=label2id
    ).to(device)

def classify_scene_vit(image_picture):
    inputs = vit_processor(image_picture, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = vit_model(**inputs).logits

    # Get the top 5 predictions
    top5_prob, top5_indices = torch.topk(logits, 5)

    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(top5_prob, dim=-1)

    # Get the labels for the top 5 indices
    top5_labels = [vit_model.config.id2label[idx.item()] for idx in top5_indices[0]]

    # Print the top 5 labels and their corresponding probabilities
    #for label, prob in zip(top5_labels, probabilities[0]):
    #    print(f"{label}: {prob:.4f}")
    probabilities = probabilities[0].to('cpu').numpy()
    if top5_labels[0] == 'misc' and probabilities[0] - probabilities[1] < 0.2:
        return top5_labels[1]
    return top5_labels[0]


# clip model
from transformers import AutoTokenizer, AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Classify scene
def classify_scene_clip_llava(image_picture, image_captions=None):
        # Generate LLaVa caption
        llava_prompt = "Where is the picture taken?"
        llava_output = generate_llava_caption(image_picture, llava_prompt)
        print(llava_output)
        # Get CLIP caption and image features
        llava_text_input = tokenizer(llava_output, padding=True, return_tensors="pt").to(device)
        if image_captions:
            text_inputs = tokenizer(image_captions, padding=True, return_tensors="pt").to(device)
        cat_inputs = processor(text=scene_labels_context, return_tensors="pt", padding=True).to(device)
        img_inputs = processor(images=image_picture, return_tensors="pt").to(device)
        with torch.no_grad():
            # Get the image and text features
            llava_features = model.get_text_features(**llava_text_input).to('cpu')
            if image_captions:
                text_features = model.get_text_features(**text_inputs).to('cpu')
            cat_features = model.get_text_features(**cat_inputs).to('cpu')
            image_features = model.get_image_features(**img_inputs).to('cpu')

        # Normalize the features
        llava_features = llava_features / llava_features.norm(dim=-1, keepdim=True)
        if image_captions:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cat_features = cat_features / cat_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        caption_similarities = []
        if image_captions:
            for caption in text_features:
                caption_similarities.append(similarity_score(caption, cat_features))
        
        img_similarities = similarity_score(image_features, cat_features)

        llava_similarities = similarity_score(llava_features, cat_features)

        # Apply softmax along the specified dimension
        # img
        img_prob = torch.nn.functional.softmax(img_similarities, dim=0)
        # caption
        if image_captions:
            caption_prob = torch.nn.functional.softmax(sum(caption_similarities), dim=0)
        # llava
        llava_prob = torch.nn.functional.softmax(llava_similarities, dim=0)
        # llava_image
        llava_image_prob = torch.nn.functional.softmax(img_similarities + llava_similarities, dim=0)
        # txt_image
        if image_captions:
            txt_image_prob = torch.nn.functional.softmax(img_similarities*5 + sum(caption_similarities), dim=0)
        # llava_txt
        if image_captions:
            llava_txt_prob = torch.nn.functional.softmax(sum(caption_similarities) + llava_similarities*5, dim=0)
        # mix
        if image_captions:
            mix_prob = torch.nn.functional.softmax(img_similarities*5 + sum(caption_similarities)+ llava_similarities*5)

        k = 3
        if image_captions:
            _ , txt_indices = torch.topk(caption_prob, k)
            txt_indices = txt_indices.tolist()

        _ , img_indices = torch.topk(img_prob, k)
        img_indices = img_indices.tolist()

        _, llava_indices = torch.topk(llava_prob, k)
        llava_indices = llava_indices.tolist()

        _, llava_image_indices = torch.topk(llava_image_prob, k)
        llava_image_indices = llava_image_indices.tolist()

        if image_captions:
            _, txt_image_indices = torch.topk(txt_image_prob, k)
            txt_image_indices = txt_image_indices.tolist()

        if image_captions:
            _, llava_txt_indices = torch.topk(llava_txt_prob, k)
            llava_txt_indices = llava_txt_indices.tolist()

            _ , mix_indices = torch.topk(mix_prob, k)
            mix_indices = mix_indices.tolist()

        print('llava: ',[scene_labels[i] for i in llava_indices])
        if image_captions:
            print('txt: ',[scene_labels[i] for i in txt_indices])
        print('img: ',[scene_labels[i] for i in img_indices])
        print('llava_image: ',[scene_labels[i] for i in llava_image_indices])
        if image_captions:
            print('txt_image: ',[scene_labels[i] for i in txt_image_indices])
            print('llava_txt: ',[scene_labels[i] for i in llava_txt_indices])
            print('mix: ',[scene_labels[i] for i in mix_indices])

        return scene_labels[llava_image_indices[0]]

def similarity_score(tensor, tensor_list):
    similarities = []
    for c in tensor_list:
        similarities.append(torch.matmul(tensor, c.T).item())
    return torch.tensor(similarities)

def generate_llava_caption(image_picture, prompt):
    # LLaVa with Ollama 
    from langchain_community.llms import Ollama
    llava = Ollama(model="llava_short")
    llava_prompt = "Where is the picture taken?"
    image_b64 = convert_to_base64(image_picture)
    llm_with_image_context = llava.bind(images=[image_b64])
    return llm_with_image_context.invoke(prompt)


import base64
from io import BytesIO

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str
   
def subtract_in_bounds(x, y):
    if x - y > 0:
        return int(x - y) 
    else:
        return 0
    
def add_in_bounds(x, y, max):
    if x + y < max:
        return int(x + y)
    else:
        return int(max)

#%%
# FIND OBJECT TO REPLACE 

import pickle as pkl
import json
import pandas as pd
from pprint import pprint

# Load the object_scene_rel_matrix file
with open("/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/object_scene_rel/tf_idf_scores.pkl", "rb") as file:
    object_scene_rel_matrix = pkl.load(file)

# Load the size_mean_matrix file
tp_size_mean_path = '/Users/filippomerlo/Desktop/Datasets/osfstorage-archive/THINGSplus/Metadata/Concept-specific/size_meanRatings.tsv'
size_mean_matrix = pd.read_csv(tp_size_mean_path, sep='\t')
things_words = list(size_mean_matrix['Word'])

#%%
# Load the map_coco2ade file
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/object_scene_rel/object_map_coco2ade.json', "r") as file:
    map_coco2ade = json.load(file)

#%%
# Load the object_map_ade2things file
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/object_scene_rel/object_map_ade2things.json', "r") as file:
    map_ade2things = json.load(file)

# Load ade names
path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
with open(path, 'rb') as f:
    ade20k_index = pkl.load(f)
ade20k_object_names = ade20k_index['objectnames']

#%%

def find_object_to_replace(target_object_name, scene_name):
    if scene_name == 'misc':
        print('Scene not recognized')
        return None
    # get the more similar in size with the less semantic relatedness to the scene
    scores = []
    for ade_name in map_ade2things.keys():
        # ade object --> scene relatedness
        scene_relatedness_score = object_scene_rel_matrix.at[ade20k_object_names.index(ade_name), scene_name]
        if scene_relatedness_score != 0:
            scene_relatedness_score = 100
        # target, coco object --> ade object --> emb
        target_distr = object_scene_rel_matrix.iloc[ade20k_object_names.index(map_coco2ade[target_object_name][1])]
        # non target, ade object --> emb
        sde_name_distr = object_scene_rel_matrix.iloc[ade20k_object_names.index(ade_name)]
        # target - non target cos sim
        cos_sim = cosine_similarity(target_distr,sde_name_distr)

        # target size
        # coco obj --> ade obj --> things obj
        things_name_target = map_ade2things[map_coco2ade[target_object_name][1]]
        if target_object_name in things_name_target:
            target_idx = things_words.index(target_object_name)
            target_size_score = size_mean_matrix.at[target_idx, 'Size_mean']
        else:
            target_idx = [things_words.index(n) for n in things_name_target]
            target_size_score = 0
            for id in target_idx:
                target_size_score += size_mean_matrix.at[id, 'Size_mean']
            target_size_score = target_size_score/len(target_idx)

        # ade obj size
        # ade obj --> things obj
        things_name_ade_name = map_ade2things[ade_name]
        if len(things_name_ade_name) == 1:
            ade_idx = things_words.index(things_name_ade_name[0])
            ade_size_score = size_mean_matrix.at[ade_idx, 'Size_mean']
        else:
            ade_idx = [things_words.index(n) for n in things_name_ade_name]
            ade_size_score = 0
            for id in ade_idx:
                ade_size_score += size_mean_matrix.at[id, 'Size_mean']
            ade_size_score = ade_size_score/len(ade_idx)

        size_diff = abs(target_size_score - ade_size_score)
        total_score = size_diff + scene_relatedness_score #+ cos_sim
        scores.append(total_score)
    # get top k lower scores idxs
    kidxs, kvls = lowest_k(scores, 100)
    adeknames = [list(map_ade2things.keys())[i] for i in kidxs[1:]]
    things_names = [map_ade2things[ade_name] for ade_name in adeknames]
    return things_names

import numpy as np

def cosine_similarity(vec1, vec2):
    # Compute the dot product of the vectors
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitudes of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cosine_sim

def lowest_k(alist, k):
    # Step 1: Enumerate the list to pair each element with its index
    enumerated_list = list(enumerate(alist))
    
    # Step 2: Sort the enumerated list by the element values
    sorted_list = sorted(enumerated_list, key=lambda x: x[1])
    
    # Step 3: Extract the indices of the first k elements
    lowest_k_indices = [index for index, value in sorted_list[:k]]
    lowest_k_values = [value for index, value in sorted_list[:k]]
    
    return lowest_k_indices, lowest_k_values

# get object scene relatedness score

def object_scene_rel(object_name, scene_name):
    object_idx = map_coco2ade[object_name][0]
    relatedness_score = object_scene_rel_matrix.at[object_idx, scene_name].item()
    return relatedness_score

def reverse_dict(data):
    """
    This function reverses a dictionary by swapping keys and values.

    Args:
        data: A dictionary to be reversed.

    Returns:
        A new dictionary where keys become values and vice versa, handling duplicates appropriately.
    """
    reversed_dict = {}
    for key, value in data.items():
        for l in value:
            reversed_dict[str(l)] = key
    return reversed_dict


from transformers import AutoImageProcessor, ViTModel, ViTConfig 
import torch
vitc_image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
vitc_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)


things_images_path = '/Users/filippomerlo/Desktop/Datasets/osfstorage-archive/THINGS/Images'

divisions = {
  "object_images_A-C": range(ord('a'), ord('c') + 1),  # a-c (inclusive)
  "object_images_D-K": range(ord('d'), ord('k') + 1),  # d-k (inclusive)
  "object_images_L-Q": range(ord('l'), ord('q') + 1),  # l-q (inclusive)
  "object_images_R-S": range(ord('r'), ord('s') + 1),  # r-s (inclusive)
  "object_images_T-Z": range(ord('t'), ord('z') + 1),  # t-z (inclusive)
}
# Convert ranges to lists of characters for clarity (optional)
letters = {}
for division, char_range in divisions.items():
  letters[division] = [chr(c) for c in char_range]
letter_to_division = reverse_dict(letters)


import re
from PIL import Image
def get_images_names(substitutes_list):
    # get things images paths [(name, path)...]
    things_names = list(set([n[0] for n in substitutes_list]))
    images_names_list = []
    images_path_list = []
    for object_name in things_names:
        folders_path = os.path.join(things_images_path, letter_to_division[object_name[0]])
        images_paths = get_all_names(folders_path)
        for i_p in images_paths:
            things_obj_name = re.sub(r"\d+", "",i_p.split('/')[-2]).replace('_',' ')
            if object_name == things_obj_name:
                images_names_list.append(object_name)
                images_path_list.append(i_p)
    return images_path_list, images_names_list


def compare_imgs(target_patch, substitutes_list):
    # get things images paths [(name, path)...]
    images_path_list, images_names_list = get_images_names(substitutes_list)

    # embed images
    images_embeddings = []
    print(images_path_list)
    with torch.no_grad():
        for i_path in images_path_list:
            image = Image.open(i_path)
            image_input = vitc_image_processor(image, return_tensors="pt").to(device)
            image_outputs = vitc_model(**image_input)
            image_embeds = image_outputs.last_hidden_state[0][0].to('cpu')#.squeeze().mean(dim=1).to('cpu')
            images_embeddings.append(image_embeds)
        # embed target 
        target_input = vitc_image_processor(target_patch, return_tensors="pt").to(device)
        target_outputs = vitc_model(**target_input)
        target_embeds = target_outputs.last_hidden_state[0][0].to('cpu')#.squeeze().mean(dim=1).to('cpu')

    # compare
    similarities = []
    for i_embed in images_embeddings:
        similarities.append(cosine_similarity(target_embeds.detach().numpy(), i_embed.detach().numpy()))
    # get top 5
    k = 5
    v, indices = torch.topk(torch.tensor(similarities), k)
    print(v)
    print([images_names_list[i] for i in indices])
    return [images_path_list[i] for i in indices]

from PIL import Image

def visualize_images(image_paths):
  """
  This function takes a list of image paths and displays them using PIL.

  Args:
      image_paths: A list containing paths to the images.
  """
  for path in image_paths:
    try:
      # Open the image using PIL
      image = Image.open(path)

      # Display the image using Image.show()
      image.show()
    except FileNotFoundError:
      print(f"Error: File not found: {path}")

import os

def get_all_names(path):
    """
    This function retrieves all file and folder names within a directory and its subdirectories.

    Args:
        path: The directory path to search.

    Returns:
        A list containing all file and folder names.
    """
    names = []
    for root, dirs, files in os.walk(path):
        for name in files:
            names.append(os.path.join(root, name))
        for name in dirs:
            names.append(os.path.join(root, name))
    return names

#%%
'''
# find ade object in things 
path = '/Users/filippomerlo/Desktop/Datasets/ADE20K_2021_17_01/index_ade20k.pkl'
# Load the ADE20K index file
with open(path, 'rb') as f:
    ade20k_index = pkl.load(f)
ade20k_object_names = ade20k_index['objectnames']


import re
mapping_ade2things = {}
for ade_names in ade20k_object_names:
    ade_name_list = ade_names.lower().replace(' ','').replace('-','').split(',')
    for ade_name in ade_name_list:
        for things_name in things_words:
            if re.sub(r's$', '',ade_name) == re.sub(r's$', '',things_name.replace(' ','').replace('-','')):
                if ade_names not in mapping_ade2things:
                    mapping_ade2things[ade_names] = [things_name]
                else:
                    mapping_ade2things[ade_names].append(things_name)


# coco 63 obj --> ade20k
map_coco2ade
ade20k_coco_o = set([object_name[1] for object_name in map_coco2ade.values()])
len(ade20k_coco_o)

#%%
# ade20k --> things
mapping_ade2things
ade20k_things_o = set(list(mapping_ade2things.keys()))
len(ade20k_things_o)

#%%

ade20k_coco_o - ade20k_things_o # = {'tennis racket'}

# manually add tennis racket
mapping_ade2things['tennis racket'] = ['racket']
mapping_ade2things['minibike, motorbike'] = ['motorcycle']
mapping_ade2things['potted fern'] = ['plate']
mapping_ade2things['street sign'] = ['road sign']
with open('object_map_ade2things.json', 'w') as f:
    json.dump(mapping_ade2things, f, indent=4)
'''