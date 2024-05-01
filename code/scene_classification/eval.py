#%%
from dataset import *
from model import *

#Evaluate CLIP 

dataset = CustomImageDataset(index_ade20k, DATASET_PATH, True)
model = clipModel()

#%%
labels = list(set([parse_category_name(l) for l in index_ade20k['scene']]))
n_predict = 100
idxs = [i for i in range(0, dataset.__len__())][:n_predict]

score = 0

for idx in idxs:
    image, label = dataset.__getitem__(idx)
    probs = model.predict(labels, image)
    idx = torch.argmax(probs).item()
    if labels[idx] == label:
        score += 1
acc = score/n_predict
print(acc)



# %%
