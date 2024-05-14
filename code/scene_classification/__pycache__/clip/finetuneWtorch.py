#%%
from nnet import *
from dataset import CollectionsDataset
from dataset import *

from torch.utils.data import DataLoader

# Initialize DataLoader
train_dataloader = DataLoader(CollectionsDataset(final_dataset['train'], processor), shuffle=True, batch_size=8)
eval_dataloader = DataLoader(CollectionsDataset(final_dataset['test'], processor), shuffle=True, batch_size=8)

# Initialize model
n_labels = len(final_dataset['train'].features['scene_category'].names)
model = ClipModelWithClassifier(n_labels)

# # Create an optimizer and learning rate scheduler to fine-tune the model. Let's use the AdamW optimizer from PyTorch:
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

#Create the default learning rate scheduler from Trainer:
from transformers import get_scheduler
num_epochs = 4
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# specify device to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes.
import torch

if torch.backends.mps.is_available():
   device = torch.device("mps")
   print('mps Ok')
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print('cuda Ok')
else:
   device = torch.device('cpu')
   print ("cpu OK")

model.to(device)

# To keep track of your training progress, use the tqdm library to add a progress bar over the number of training steps:
from tqdm.auto import tqdm
import torch.nn.functional as F

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        actual = batch['labels'].to(device)
        input = {k:v.squeeze().to(device) for k, v in batch['image'].items()}
        outputs = model(input)
        loss = F.cross_entropy(outputs, torch.argmax(actual, dim=1))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


import evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()