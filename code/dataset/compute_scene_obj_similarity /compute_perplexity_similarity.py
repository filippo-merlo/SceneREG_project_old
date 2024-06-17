import  torch
import numpy as np
device = "cuda:1"
def get_perplexity_target_only(prompt, options, model=None, tokenizer=None, log=False):
    '''
            Parameters:
                    prompt (str): A input for a language model
                    options (list(str)): A list of possible continuations for the given input
            Returns:
                    results (list(tuple)): A list of tuples of option and associated scores
    '''
    results = []
    for option in options:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(option, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        # list to store logits of each token in the option
        current_option_logits = []
        # Get the initial input tokens
        current_input_ids = input_ids
        # Loop through each target token
        for i in range(target_ids.size(1)):
            # Get the model output (logits) and compute log probabilities
            outputs = model(input_ids=current_input_ids)
            logprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            # Store the logits for the current step
            next_target_token_id = target_ids[:, i].item()
            target_logproba = logprobs[:, -1, next_target_token_id].unsqueeze(1)
            current_option_logits.append(target_logproba.item())
            # Get the next target token and append it to the input
            next_target_token = target_ids[:, i].unsqueeze(1)
            current_input_ids = torch.cat((current_input_ids, next_target_token), dim=1)
        # Append option and sequence score to results
        results.append((option, (-1*np.mean(current_option_logits))))
    ## Sort result in ascending order
    #results = sorted(results, key=lambda x: x[1], reverse=True)
    return results


def get_perplexity_full_prompt(prompt, options, model=None, tokenizer=None, log=False):
    '''
            Parameters:
                    prompt (str): A input for a language model
                    options (list(str)): A list of possible continuations for the given input
            Returns:
                    results (list(tuple)): A list of tuples of option and associated scores
    '''
    results = []
    for option in options:
        start_ids = tokenizer('Complete the sentence:', return_tensors="pt").input_ids.to(device)
        prompt = prompt + ' ' + option + '.'
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        # list to store logits of each token in the option
        current_option_logits = []
        # Get the initial input tokens
        current_input_ids = start_ids
        # Loop through each target token
        for i in range(input_ids.size(1)):
            # Get the model output (logits) and compute log probabilities
            outputs = model(input_ids=current_input_ids)
            logprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            # Store the logits for the current step
            next_target_token_id = input_ids[:, i].item()
            target_logproba = logprobs[:, -1, next_target_token_id].unsqueeze(1)
            current_option_logits.append(target_logproba.item())
            # Get the next target token and append it to the input
            next_target_token = input_ids[:, i].unsqueeze(1)
            current_input_ids = torch.cat((current_input_ids, next_target_token), dim=1)
        # Append option and sequence score to results
        results.append((option, (-1*np.mean(current_option_logits))))
    return results

#%%
CACHE_DIR = '/mnt/cimec-storage6/shared'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ACCESS_TOKEN = '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'
model_id = "meta-llama/Meta-Llama-Guard-2-8B"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, cache_dir=CACHE_DIR, token=ACCESS_TOKEN).to(device)

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

for scene_name in scenes_categories[:1]:

    prompt = f"In the {scene_name.replace('_',' ')} there is a"
    candidate_scores = []
    for candidate in candidates:
        single_candidate_list = candidate.split(', ')
        results = get_perplexity_full_prompt(prompt, single_candidate_list, model=model, tokenizer=tokenizer)
        candidate_scores.append((candidate, np.mean([score for _, score in results])))

    sorted_candidate_score = sorted(candidate_scores, key=lambda x: x[1])
    print(f"Scene: {prompt}")
    for i, (option, score) in enumerate(sorted_candidate_score[:100]):
        print(f"{i+1}. {option}: {score:.2f}")
    print("\n")
