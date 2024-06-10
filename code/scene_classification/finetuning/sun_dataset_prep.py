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
not_matched = set(sun_classes)-set(ade_classes)
pprint(not_matched)

possible_matches = {}
for x in not_matched:
    possible_matches[x] = []
    l = x.split('_')
    if 'indoor' in l:
        l.remove('indoor')
    if 'outdoor' in l:
        l.remove('outdoor')
    for y in ade_classes:
        l2 = y.split('_')
        for i in l:
            if i in l2:
                possible_matches[x].append(y)

pprint(possible_matches)
{'bakery_shop': [],
 'balcony_exterior': ['upper_balcony'],
 'canal_natural': [],
 'canal_urban': ['urban'],
 'car_interior_backseat': ['backseat'],
 'car_interior_frontseat': ['frontseat'],
 'covered_bridge_exterior': ['bridge'],
 'cubicle_office': ['office'],
 'desert_sand': ['sand'],
 'desert_vegetation': ['desert_road'],
 'dinette_vehicle': ['vehicle'],
 'elevator_door': ['elevator_interior',
                   'door',
                   'freight_elevator',
                   'elevator_lobby',
                   'elevator_shaft',
                   'revolving_door'],
 'field_cultivated': ['athletic_field_indoor',
                      'athletic_field_outdoor',
                      'baseball_field',
                      'corn_field',
                      'cultivated',
                      'field_road',
                      'football_field',
                      'field_house',
                      'field_tent_indoor',
                      'field_tent_outdoor',
                      'wheat_field'],
 'field_wild': ['athletic_field_indoor',
                'athletic_field_outdoor',
                'baseball_field',
                'corn_field',
                'wild',
                'field_road',
                'football_field',
                'field_house',
                'field_tent_indoor',
                'field_tent_outdoor',
                'wheat_field'],
 'forest_broadleaf': ['forest_road',
                      'bamboo_forest',
                      'broadleaf',
                      'forest_fire',
                      'forest_path'],
 'forest_needleleaf': ['forest_road',
                       'bamboo_forest',
                       'needleleaf',
                       'forest_fire',
                       'forest_path'],
 'gazebo_exterior': ['exterior', 'gazebo_interior'],
 'lake_natural': ['natural', 'natural_history_museum', 'natural_spring'],
 'moat_water': ['water',
                'water_fountain',
                'water_gate',
                'water_mill',
                'water_park',
                'water_tower',
                'water_treatment_plant_indoor',
                'water_treatment_plant_outdoor'],
 'poolroom_establishment': ['establishment', 'poolroom_home'],
 'skatepark': [],
 'stadium_baseball': ['baseball_field', 'baseball', 'stadium_outdoor'],
 'stadium_football': ['football', 'football_field', 'stadium_outdoor'],
 'subway_station_platform': ['gas_station',
                             'subway_interior',
                             'platform',
                             'bus_station_indoor',
                             'bus_station_outdoor',
                             'fire_station',
                             'lookout_station_indoor',
                             'lookout_station_outdoor',
                             'observation_station',
                             'pumping_station',
                             'police_station',
                             'train_station_outdoor',
                             'station'],
 'temple_east_asia': ['east_asia', 'east_asia', 'south_asia'],
 'temple_south_asia': ['east_asia', 'south_asia', 'south_asia'],
 'theater_indoor_procenium': ['indoor_procenium',
                              'aquatic_theater',
                              'home_theater',
                              'movie_theater_indoor',
                              'movie_theater_outdoor',
                              'theater_outdoor'],
 'theater_indoor_seats': ['aquatic_theater',
                          'home_theater',
                          'movie_theater_indoor',
                          'movie_theater_outdoor',
                          'indoor_seats',
                          'theater_outdoor'],
 'train_station_platform': ['gas_station',
                            'platform',
                            'train_railway',
                            'bus_station_indoor',
                            'bus_station_outdoor',
                            'fire_station',
                            'lookout_station_indoor',
                            'lookout_station_outdoor',
                            'observation_station',
                            'pumping_station',
                            'police_station',
                            'train_interior',
                            'train_station_outdoor',
                            'train_station_outdoor',
                            'station'],
 'underwater_coral_reef': ['coral_reef', 'coral_reef'],
 'waterfall_block': ['block'],
 'waterfall_fan': ['fan'],
 'waterfall_plunge': ['plunge'],
 'wine_cellar_barrel_storage': ['bottle_storage',
                                'cellar',
                                'root_cellar',
                                'storage_room',
                                'storm_cellar',
                                'barrel_storage',
                                'barrel_storage'],
 'wine_cellar_bottle_storage': ['bottle_storage',
                                'bottle_storage',
                                'cellar',
                                'root_cellar',
                                'storage_room',
                                'storm_cellar',
                                'barrel_storage']}
