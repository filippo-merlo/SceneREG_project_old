import os 
from config import *
from pprint import pprint
import torch

# Is MPS even available? macOS 12.3+
print(torch.backends.mps.is_available())
# Was the current version of PyTorch built with MPS activated?
print(torch.backends.mps.is_built())
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

from transformers import AutoTokenizer, AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Classify scene
def classify_scene(image_picture, image_captions):
        text_inputs = tokenizer(image_captions, padding=True, return_tensors="pt").to(device)
        cat_inputs = processor(text=scene_labels_context, return_tensors="pt", padding=True).to(device)
        img_inputs = processor(images=image_picture, return_tensors="pt").to(device)
        with torch.no_grad():
            # Get the image and text features
            text_features = model.get_text_features(**text_inputs).to('cpu')
            cat_features = model.get_text_features(**cat_inputs).to('cpu')
            image_features = model.get_image_features(**img_inputs).to('cpu')

        # Normalize the features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cat_features = cat_features / cat_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        caption_similarities = []
        for caption in text_features:
            caption_similarities.append(similarity_score(caption, cat_features))
        
        img_similarities = similarity_score(image_features, cat_features)

        # Apply softmax along the specified dimension
        # img
        img_prob = torch.nn.functional.softmax(img_similarities, dim=0)
        # caption
        caption_prob = torch.nn.functional.softmax(sum(caption_similarities), dim=0)
        # mix
        mix_prob = torch.nn.functional.softmax(img_similarities*5 + sum(caption_similarities))

        k = 3
        _ , txt_indices = torch.topk(caption_prob, k)
        txt_indices = txt_indices.tolist()

        _ , img_indices = torch.topk(img_prob, k)
        img_indices = img_indices.tolist()

        _ , mix_indices = torch.topk(mix_prob, k)
        mix_indices = mix_indices.tolist()

        print('txt: ',[scene_labels[i] for i in txt_indices])
        print('img: ',[scene_labels[i] for i in img_indices])
        print('mix: ',[scene_labels[i] for i in mix_indices])

def similarity_score(tensor, tensor_list):
    similarities = []
    for c in tensor_list:
        similarities.append(torch.matmul(tensor, c.T).item())
    return torch.tensor(similarities)
    