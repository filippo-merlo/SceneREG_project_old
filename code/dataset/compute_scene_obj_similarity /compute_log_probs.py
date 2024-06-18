import  torch
import numpy as np
device = "cuda:1"

def get_log_probs(prompt, option, model=None, tokenizer=None, log=False):
    '''
        Parameters:
                prompt (str): A input for a language model
                options (list(str)): A list of possible continuations for the given input
        Returns:
                results (list(tuple)): A list of tuples of option and associated scores
    '''
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    target_ids = tokenizer(option, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

     # list to store logits of each token in the option
    current_option_logits = []

    # Get the initial input tokens
    current_input_ids = input_ids

    # Loop through each target token
    for i in range(target_ids.size(1)):
        with torch.no_grad():
            # Get the model output (logits) and compute log probabilities
            outputs = model(input_ids=current_input_ids)
        logprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1).to('cpu')

        # Store the logits for the current step
        next_target_token_id = target_ids[:, i].item()
        target_logproba = logprobs[:, -1, next_target_token_id].unsqueeze(1)
        current_option_logits.append(target_logproba.item())

        # Get the next target token and append it to the input
        next_target_token = target_ids[:, i].unsqueeze(1)
        current_input_ids = torch.cat((current_input_ids, next_target_token), dim=1)
        
    # Append option and sequence score to results
    result = -1*np.sum(current_option_logits)
    return result

#%%
CACHE_DIR = '/mnt/cimec-storage6/shared'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ACCESS_TOKEN = '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'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, cache_dir=CACHE_DIR, token=ACCESS_TOKEN)

#%%
import pickle as pkl
# get objects and scenes names 

# Load object categories from ADE20K 
DATASET_PATH = '/mnt/cimec-storage6/users/filippo.merlo/ADE20K_2016_07_26'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)
candidates = index_ade20k['objectnames']

# Load scene categories from ADE20K hf
from datasets import load_dataset
ade_hf_data = load_dataset("scene_parse_150", cache_dir='/mnt/cimec-storage6/shared/hf_datasets')
scenes_categories = ade_hf_data['train'].features['scene_category'].names
from tqdm import tqdm
for scene_name in tqdm(scenes_categories[:4]):
    candidate_scores = []
    for candidate in candidates:
        single_candidate_list = candidate.split(', ')
        single_candidate_list_scores = []
        for single_candidate in single_candidate_list:
            if single_candidate[0] in ['a', 'e', 'i', 'o', 'u']:
                article = 'an'
            else:
                article = 'a'
            prompt = f"You are a helpful assistant. Your job is to complete the following sentence with the name of an object that is higly related to the place mentioned in the sentence. In the {scene_name.replace('_',' ')} there is " + article + ' '
            option = single_candidate
            single_candidate_list_scores.append(get_log_probs(prompt, option, model=model, tokenizer=tokenizer))
        candidate_scores.append((candidate, np.mean([score for score in single_candidate_list_scores])))

    sorted_candidate_score = sorted(candidate_scores, key=lambda x: x[1]) # higer first
    print(f"Scene: {prompt}")
    for i, (option, score) in enumerate(sorted_candidate_score[:20]):
        print(f"{i+1}. {option}: {score:.2f}")
    print("\n")
