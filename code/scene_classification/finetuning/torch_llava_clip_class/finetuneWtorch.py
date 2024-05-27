#%%
# Import necessary modules and packages
from nnet import *
from dataset import CollectionsDataset
from dataset import *
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
from dataset_prep import final_dataset

# Initialize Weights and Biases (wandb)
cache_dir = '/mnt/cimec-storage6/users/filippo.merlo'
import os
os.environ['HF_HOME'] = cache_dir

import wandb
wandb.login()

project_name = "clip_llava_attention_scene_classifier"
config = {
    "batch_size": 16,
    "num_epochs": 10,
    "lr": 2e-4,
    "momentum": 0.9
}
wandb.init(project=project_name, config=config, dir=cache_dir)


#%%
# Specify Device (GPU/CPU)
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Initialize DataLoader and Preprocessor
from transformers import AutoProcessor, CLIPModel, pipeline

processor = {
    'clip_processor': AutoProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    'clip_model': CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device),
    'llava_pipeline': pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf")
}
train_dataloader = DataLoader(CollectionsDataset(final_dataset['train'], processor), shuffle=True, batch_size=wandb.config['batch_size'])
eval_dataloader = DataLoader(CollectionsDataset(final_dataset['test'], processor), shuffle=True, batch_size=wandb.config['batch_size'])

#%%
# Initialize Model
n_labels = len(final_dataset['train'].features['scene_category'].names)
model = AttentionClassifier(num_labels=n_labels,feature_size=512+768)

#%%
# Create Optimizer and Learning Rate Scheduler
from torch.optim import SGD

optimizer = SGD(model.parameters(), lr=wandb.config['lr'], momentum=wandb.config['momentum'])


#%%
# Setup Weights and Biases Logging
log_freq = 100
wandb.watch(model, log_freq=log_freq)

#%%
# Initialize Progress Bar
from tqdm.auto import tqdm

num_epochs = wandb.config['num_epochs']
num_training_steps = num_epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

#%%
# Initialize Evaluation Metrics
import evaluate

metric_test = evaluate.load("accuracy", cache_dir=cache_dir)
metric_train = evaluate.load("accuracy", cache_dir=cache_dir)

# Define Loss Function
criterion = torch.nn.CrossEntropyLoss()

#%%
# Training and Evaluation Loop
for epoch in range(num_epochs):

    # Evaluation Phase
    model.eval()
    for batch in tqdm(eval_dataloader):
        # Move data to device
        labels = batch['labels'].to(device)
        input = {k:v.squeeze().to(device) for k, v in batch['image'].items()}
        with torch.no_grad():
            outputs = model(input)
        # Evaluate
        predictions = torch.argmax(outputs, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        metric_test.add_batch(predictions=predictions, references=labels)
    wandb.log({'eval_acc' : metric_test.compute()['accuracy']})

    # Training Phase
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        # Move data to device
        labels = batch['labels'].to(device)
        input = {k:v.squeeze().to(device) for k, v in batch['image'].items()}
        # Forward pass
        outputs = model(input)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        # Evaluate
        labels = torch.argmax(labels, dim=-1)
        predictions = torch.argmax(outputs, dim=-1)
        metric_train.add_batch(predictions=predictions, references=labels)
        if batch_idx % log_freq == 0:
            wandb.log({"loss": loss})
            wandb.log({'train_acc' : metric_train.compute()['accuracy']})

