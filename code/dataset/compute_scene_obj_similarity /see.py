#%%
import pickle as pkl 
with open('/Users/filippomerlo/Documents/GitHub/SceneREG_project/code/dataset/compute_scene_obj_similarity /ade_scenes_bert_similarities.pkl', "rb") as file:
    bert_scene_object_rel_matrix = pkl.load(file)

bert_scene_object_rel_matrix