#%%
import json 
from pprint import pprint

with open('/Users/filippomerlo/Desktop/Datasets/data/coco_search18_annotated.json', 'r') as f:
    data = json.load(f)

print(len(data.keys()))
print(list(data.keys())[0])
pprint(list(data.values())[0])
