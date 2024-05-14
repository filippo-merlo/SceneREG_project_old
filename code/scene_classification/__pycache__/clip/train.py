import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
from utils import *
from model import train_model
import os
import argparse
from nnet import model_ft
from apex import amp


FOLD_NAME = os.environ["FOLD_NAME"]
MODEL_NAME = os.environ["MODEL_NAME"]
TRAINING_BATCH_SIZE = int(os.environ["TRAINING_BATCH_SIZE"])
TEST_BATCH_SIZE = int(os.environ["TEST_BATCH_SIZE"])
NUM_CLASSES = int(os.environ["NUM_CLASSES"])
IMAGE_SIZE = int(os.environ["IMAGE_SIZE"])
EPOCHS = int(os.environ["EPOCHS"])


if torch.cuda.is_available():
    device = torch.device("cuda")
    print('CUDA Ok')
elif torch.backends.mps.is_available():
   device = torch.device("mps")
   print('mps Ok')
else:
   device = torch.device('cpu')
   print ("device not found.")


train_dataset = CollectionsDataset(final_dataset['train'], transform=processor)
valid_dataset = CollectionsDataset(final_dataset['test'], transform=processor)

train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=TRAINING_BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=4)

valid_dataset_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=TEST_BATCH_SIZE,
                                                   shuffle=False,
                                                   num_workers=4)

model_ft = model_ft.to(device)

lr_min = 1e-4
lr_max = 1e-3

plist = [
        {'params': model_ft.classifier_head.parameters(), 'lr': 5e-4}
        ]

optimizer_ft = optim.Adam(plist, lr=0.001)
model_ft, optimizer_ft = amp.initialize(model_ft, optimizer_ft, opt_level="O1", verbosity=0)
lr_sch = lr_scheduler.ReduceLROnPlateau(optimizer_ft, verbose=True, factor=0.3, mode="max", patience=1, threshold=0.01)

dataset_sizes = {}
dataset_sizes["train"] = len(train_dataset)
dataset_sizes["val"] = len(valid_dataset)

data_loader = {}
data_loader["train"] = train_dataset_loader
data_loader["val"] = valid_dataset_loader

model_ft = train_model(model_ft,
                       data_loader,
                       dataset_sizes,
                       device,
                       optimizer_ft,
                       lr_sch,
                       num_epochs=EPOCHS,
                       fold_name=FOLD_NAME,
                       use_amp=True)
torch.save(model_ft.state_dict(), os.path.join(FOLD_NAME, "model.bin"))