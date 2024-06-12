#%%
### PREPARE THE DATASET   
from config import *
import random
import torchvision
import torch
from torch.utils.data import DataLoader

sun_data = torchvision.datasets.SUN397(root = cache_dir, download = True)
sun_classes = [x.replace('/', '_') for x in list(sun_data.class_to_idx.keys())]

from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ade_data = load_dataset("scene_parse_150", cache_dir=cache_dir)
ade_classes = list(ade_data['train'].features['scene_category'].names)

from pprint import pprint
print('SUN classes:', len(sun_classes))
print(sun_classes[0:20])
print('ADE classes:', len(ade_classes))
print(ade_classes[0:20])
print('Classes in SUN but not in ADE:')
not_matched_sun = set(sun_classes)-set(ade_classes)
not_matched_ade = set(ade_classes)-set(sun_classes)

possible_matches = {}
for x in not_matched_sun:
    possible_matches[x] = []
    l = x.split('_')
    if 'indoor' in l:
        l.remove('indoor')
    if 'outdoor' in l:
        l.remove('outdoor')
    for y in not_matched_ade:
        l2 = y.split('_')
        for i in l:
            if i in l2:
                possible_matches[x].append(y)
pprint(possible_matches)
'''
# Ajust manually possible matches
possible_matches = {'bakery_shop': [] ,
 'balcony_exterior':  'upper_balcony' ,
 'canal_natural':  [] ,
 'canal_urban':  'urban' ,
 'car_interior_backseat':  'backseat' ,
 'car_interior_frontseat':  'frontseat' ,
 'covered_bridge_exterior':  'bridge' ,
 'cubicle_office':  'office' ,
 'desert_sand':  'sand' ,
 'desert_vegetation':  'desert_road' ,
 'dinette_vehicle':'vehicle' ,
 'elevator_door': 'elevator_interior' ,
 'field_cultivated': 'cultivated',
 'field_wild':'wild' ,
 'forest_broadleaf': 'broadleaf' ,
 'forest_needleleaf': 'needleleaf' ,
 'gazebo_exterior': 'gazebo_interior' ,
 'lake_natural': ['natural','natural_spring'] ,
 'moat_water': 'water_gate' ,
 'poolroom_establishment': 'poolroom_home' ,
 'skatepark': [] ,
 'stadium_baseball': ['baseball_field', 'baseball'] ,
 'stadium_football': ['football', 'football_field'] ,
 'subway_station_platform': ['subway_interior',
                             'platform'] ,
 'temple_east_asia': 'east_asia' ,
 'temple_south_asia': 'south_asia' ,
 'theater_indoor_procenium': 'indoor_procenium' ,
 'theater_indoor_seats': ['movie_theater_indoor',
                          'indoor_seats'] ,
 'train_station_platform': ['platform',
                            'train_station_outdoor',
                            'station'],
 'underwater_coral_reef':  'coral_reef' ,
 'waterfall_block': 'block',
 'waterfall_fan': 'fan',
 'waterfall_plunge': 'plunge',
 'wine_cellar_barrel_storage': 'barrel_storage',
 'wine_cellar_bottle_storage': 'bottle_storage'}
'''
