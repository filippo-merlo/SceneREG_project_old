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
scene_labels_vit = ['natural', 'street', 'river', 'bathroom', 'highway', 'staircase', 'building_facade', 'home_office', 'house', 'skyscraper', 'kitchen', 'attic', 'living_room', 'reception', 'bedroom', 'corridor', 'exterior', 'art_gallery', 'garage_indoor', 'alley', 'apartment_building_outdoor', 'hotel_room', 'game_room', 'mountain', 'office', 'beach', 'conference_room', 'broadleaf', 'dining_room', 'waiting_room', 'pasture', 'warehouse_indoor', 'cultivated', 'childs_room', 'airport_terminal', 'castle', 'coast', 'nursery', 'shop', 'parlor', 'bridge', 'art_studio', 'lobby', 'classroom', 'mountain_snowy', 'poolroom_home', 'dorm_room', 'closet', 'bar', 'needleleaf', 'roundabout', 'casino_indoor', 'park']
# Create the label to ID mapping
label2id = {label: idx for idx, label in enumerate(scene_labels_vit)}

# Reverse the mapping to create ID to label mapping
id2label = {idx: label for label, idx in label2id.items()}
# Create a new run
with wandb.init(project="vit-base-patch16-224") as run:
    # Pass the name and version of Artifact
    my_model_name = "model-u9yxgyhs:v0"
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

    predicted_label = logits.argmax(-1).item()
    print(vit_model.config.id2label[predicted_label])


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

from transformers import AutoProcessor, LlavaForConditionalGeneration
# LLaVa with Ollama 
from langchain_community.llms import Ollama
llava = Ollama(model="llava_short")
llava_prompt = "Where is the picture taken?"

def generate_llava_caption(image_picture, prompt):
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