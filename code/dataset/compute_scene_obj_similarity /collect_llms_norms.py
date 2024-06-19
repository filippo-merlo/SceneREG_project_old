#%%
device = "cuda:0"
CACHE_DIR = '/mnt/cimec-storage6/shared'
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

ACCESS_TOKEN = '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'
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
dtype = torch.bfloat16

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device,
    cache_dir=CACHE_DIR, 
    token=ACCESS_TOKEN
)

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


answers = {}
for scene_name in scenes_categories[:10]:
    answers[scene_name] = {}
    candidate_scores = []
    for candidate in candidates:
        candidate_list = candidate.split(', ')

        for single_candidate in candidate_list:
            prompt = "Is it possible to find the object '{word}' in the place '{scene}'?".format(word=single_candidate, scene=scene_name.replace("_", " "))
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Your job is to say if an object can be found in a specific place. You can answer only with YES or NO."},
                {"role": "user", "content": prompt}
            ]

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            attention_mask = attention_mask = input_ids["attention_mask"]

            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
            )

            response = outputs[0][input_ids.shape[-1]:]
            decoded_response = tokenizer.decode(response, skip_special_tokens=True)

            # Add answer for the list of candidates only if a YES was not obtained yet
            if decoded_response in ['YES', 'NO']:
                try:
                    if answers[scene_name][candidate] == 'YES':
                        continue
                    else:
                        answers[scene_name][candidate] = decoded_response
                except KeyError:
                    answers[scene_name][candidate] = decoded_response

from pprint import pprint
pprint(answers)

'''
{'airport_terminal': {'-': 'YES',
                      'aarm panel': 'YES',
                      'abacus': 'NO',
                      'accordion, piano accordion, squeeze box': 'NO',
                      'acropolis': 'NO',
                      'ad, advertisement, advertizement, advertising, advertizing, advert': 'YES',
                      'adding machine': 'NO',
                      'advertisement board': 'YES',
                      'aerial': 'YES',
                      'air conditioner, air conditioning': 'YES',
                      'air hockey table': 'NO',
                      'air machine': 'YES',
                      'aircraft carrier': 'NO',
                      'airplane, aeroplane, plane': 'YES',
                      'airport cart': 'YES',
                      'alarm': 'YES',
                      'alarm clock': 'YES',
                      'alembic': 'NO',
                      'alga': 'NO',
                      'algae': 'NO',
                      "altar, communion table, Lord's table": 'NO',
                      'altarpiece': 'NO',
                      'amphitheater': 'NO',
                      'amphora': 'NO',
                      'anchor': 'NO',
                      'andiron': 'NO',
                      'andirons': 'NO',
                      'animal toy': 'YES',
                      'animal, animate being, beast, brute, creature, fauna': 'NO',
                      'animals': 'YES',
                      'antenna': 'YES',
                      'antenna, aerial, transmitting aerial': 'YES',
                      'antler': 'NO',
                      'antlers': 'NO',
                      'anvil': 'NO',
                      'aperture': 'YES',
                      'apparatus': 'YES',
                      'apparel, wearing apparel, dress, clothes': 'YES',
                      'apple': 'YES',
                      'apples': 'YES',
                      'appliance': 'NO',
                      'apron': 'NO',
                      'aquarium': 'NO',
                      'aqueduct': 'NO',
                      'arbor': 'NO',
                      'arcade': 'NO',
                      'arcade machine': 'NO',
                      'arcade machines': 'NO',
                      'arcade, colonnade': 'NO',
                      'arcades': 'NO',
                      'arch': 'YES',
                      'arch, archway': 'YES',
                      'arches': 'YES',
                      'arm': 'YES',
                      'arm panel': 'YES',
                      'arm support': 'NO',
                      'armchair': 'NO',
                      'armor': 'NO',
                      'armrest': 'YES',
                      'art': 'YES',
                      'art mannequin': 'NO',
                      'articulated lamp': 'NO',
                      'artificial golf green': 'NO',
                      'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin': 'YES',
                      'ashtray': 'YES',
                      'asymmetric bars': 'NO',
                      'athletic field': 'NO',
                      'athletics track': 'NO',
                      'atm': 'YES',
                      'autoclave': 'NO',
                      'autopsy table': 'NO',
                      'auxiliary trolley': 'YES',
                      'aviary': 'YES',
                      'avocados': 'NO',
                      'award': 'YES',
                      'awards': 'YES',
                      'awning, sunshade, sunblind': 'NO',
                      'ax': 'NO',
                      'baby buggy, baby carriage, carriage, perambulator, pram, stroller, go-cart, pushchair, pusher': 'YES',
                      'baby chair': 'NO',
                      'baby walker': 'NO',
                      'baby weighs': 'NO',
                      'back': 'YES',
                      'back control': 'YES',
                      'back cushion': 'NO',
                      'back pillow': 'NO',
                      'backdrop': 'YES',
                      'backdrops': 'NO',
                      'background': 'YES',
                      'backpack, back pack, knapsack, packsack, rucksack, haversack': 'YES',
                      'backpacks': 'YES',
                      'backplate': 'YES',
                      'badge': 'YES',
                      'badlands': 'NO',
                      'bag': 'YES',
                      'bag, handbag, pocketbook, purse': 'YES',
                      'bag, traveling bag, travelling bag, grip, suitcase': 'YES',
                      'baggage': 'YES',
                      'baggage carts': 'YES',
                      'bagpipes': 'NO'}}

'''