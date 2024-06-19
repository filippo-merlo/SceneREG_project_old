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
    result = -1 * np.sum(current_option_logits)
    # try with mean 
    return result

### IMPORT MODEL
CACHE_DIR = '/mnt/cimec-storage6/shared'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ACCESS_TOKEN = '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'
model_id = "meta-llama/Meta-Llama-3-8B" # try with no instruct
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device, cache_dir=CACHE_DIR, token=ACCESS_TOKEN)

### IMPORT DATA
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

### COMPUTE LOG PROBS
from tqdm import tqdm
for scene_name in tqdm(scenes_categories[:4]):
    candidate_scores = []
    for candidate in candidates:
        single_candidate_list = candidate.split(', ')
        single_candidate_list_scores = []
        for single_candidate in single_candidate_list:
            if single_candidate[0] in ['a', 'e', 'i', 'o', 'u']:
                article = 'an '
            else:
                article = 'a '
            prompt = f"In the {scene_name.replace('_',' ')} there is " + article # try this
            #prompt = f"You are a helpful assistant. Your job is to complete the following sentence with the name of an object that is highly related to the place mentioned in the sentence. For example, if the place is 'kitchen', a related object could be 'refrigerator'. In the {scene_name.replace('_',' ')} there is " + article 
            option = single_candidate
            single_candidate_list_scores.append(get_log_probs(prompt, option, model=model, tokenizer=tokenizer))
        candidate_scores.append((candidate, np.mean([score for score in single_candidate_list_scores])))

    sorted_candidate_score = sorted(candidate_scores, key=lambda x: x[1]) # higer first
    print(f"Scene: {prompt}")
    for i, (option, score) in enumerate(sorted_candidate_score[:20]):
        print(f"{i+1}. {option}: {score:.2f}")
    print("\n")

'''
### RECORDS
# Model: meta-llama/Meta-Llama-3-8B-Instruct

# Prompt:
# "You are a helpful assistant. Your job is to complete the following sentence with the name of an object that is highly related to the place mentioned in the sentence. For example, if the place is 'kitchen', a related object could be 'refrigerator'. In the {scene_name.replace('_',' ')} there is " + article 

# In the airport terminal there is a 
1. 1: 6.03
2.  : 6.90
3. outhouse: 8.65
4. scanner: 11.09
5. sockets: 13.08
6. plane: 13.28
7. office: 13.41
8. bag: 13.53
9. baggage: 13.53
10. terminal: 13.62
11. equipment: 14.02
12. symbol: 14.08
13. entrance: 14.22
14. desk: 14.30
15. loudspeaker: 14.36
16. eye: 14.41
17. notice: 14.41
18. calculator: 14.47
19. sink: 14.49
20. printer: 14.53


# In the art gallery there is a 
1. 1: 5.77
2.  : 7.18
3. outhouse: 8.22
4. painting, picture: 11.89
5. obelisk: 12.13
6. piece: 12.38
7. art: 12.51
8. sockets: 12.71
9. statue: 12.83
10. scanner: 13.07
11. display: 13.21
12. eye: 13.53
13. stone: 14.00
14. calculator: 14.03
15. canvas: 14.03
16. arc: 14.08
17. office: 14.17
18. rod: 14.23
19. paintbrush: 14.27
20. lamp: 14.38


# In the badlands there is a 
1. 1: 4.60
2. outhouse: 4.64
3.  : 5.97
4. stalactite: 11.59
5. rock: 12.13
6. rock, stone: 12.47
7. hole: 12.61
8. scanner: 12.79
9. stone: 12.80
10. sandbox: 13.43
11. mountain, mount: 13.47
12. turtle: 13.48
13. oil: 13.61
14. castle: 13.98
15. tractor: 13.98
16. rock formation: 14.15
17. legs: 14.18
18. ridge: 14.19
19. pick: 14.20
20. camel: 14.22


# In the ball pit there is a 
1. 1: 3.81
2.  : 6.18
3. outhouse: 6.31
4. eye: 11.06
5. sink: 11.62
6. sandbox: 11.66
7. foot: 11.90
8. basket: 11.93
9. pool: 11.99
10. hole: 12.54
11. sockets: 12.57
12. obelisk: 12.65
13. bucket: 12.73
14. egg: 12.81
15. pond: 12.82
16. stalactite: 12.96
17. elephant: 13.01
18. buckets: 13.05
19. slide: 13.06
20. tractor: 13.13

'''
