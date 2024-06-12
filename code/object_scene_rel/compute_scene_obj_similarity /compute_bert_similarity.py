#%% COUMPUTE SIMILARITY WITH BERT
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from torch.nn.functional import softmax
import pandas as pd
import pickle as pkl
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

CACHE_DIR = '/mnt/cimec-storage6/users/filippo.merlo'
# Load the pre-trained model and tokenizer
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir = CACHE_DIR)
model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir = CACHE_DIR).to(device)

# Define the input sentence with a masked word
input_text = "There is a [MASK] in the [SCENE]."
# get objects and scenes names 
# Load index with global information about ADE20K
DATASET_PATH = '/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

candidates = index_ade20k['objectnames']
from datasets import load_dataset
ade_hf_data = load_dataset("scene_parse_150", cache_dir=CACHE_DIR)
scenes_categories = ade_hf_data['train'].features['scene_category'].names

# Function to calculate the probability of a candidate
def get_candidate_probability(candidate_tokens):

    # Replace the masked token with the candidate tokens
    tokenized_candidate = ["[CLS]"] + tokenized_text[:mask_token_index] + candidate_tokens + tokenized_text[mask_token_index + 1:]

    # Convert tokenized sentence to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_candidate)

    # Convert input IDs to tensors
    input_tensor = torch.tensor([input_ids]).to(device)

    # Get the logits from the model
    with torch.no_grad():
        logits = model(input_tensor).logits[0].to('cpu')

    # Calculate the probability of the candidate word
    probs = softmax(logits, dim=-1)
    probs = probs[range(len(input_ids)), input_ids]
    prob = (
        torch.prod(probs[1:mask_token_index+1])
        * torch.prod(probs[mask_token_index+len(candidate_tokens)+1:])
    )

    return prob.item()

def name2idx(name, name_list):
    return name_list.index(name)

bert_similarities_mat = pd.DataFrame(columns=scenes_categories, index=range(len(index_ade20k['objectnames'])))

for scene in tqdm(scenes_categories):
    input_text = input_text.replace("[SCENE]", scene.replace('_', ' '))
    # Tokenize the input sentence
    tokenized_text = tokenizer.tokenize(input_text)
    mask_token_index = tokenized_text.index("[MASK]")
    
    # Evaluate the probability of each candidate word
    for candidate in candidates:
        candidate_tokens = tokenizer.tokenize(candidate)
        candidate_probability = get_candidate_probability(candidate_tokens)
        bert_similarities_mat.loc[name2idx(candidate, candidates), scene] = candidate_probability

bert_similarities_mat.head()
bert_similarities_mat.to_pickle('{}/{}'.format(CACHE_DIR, "ade_scenes_bert_similarities.pkl"))

