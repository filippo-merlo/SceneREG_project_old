#%%
from transformers import ViTImageProcessor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

from datasets import load_dataset
from PIL import Image

ds = load_dataset("sezer12138/ade20k_image_classification", cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')

#%%
# Iterate through the dataset
exclude_idx = []
for i, ex in enumerate(ds['val']):
    # Open the image
    image = ex['image']

    # Check if the mode is not 'L'
    if image.mode == 'L':
        exclude_idx.append(i)

ds['val'] = ds['val'].select(
    (
        i for i in range(len(ds['val'])) 
        if i not in set(exclude_idx)
    )
)

exclude_idx = []
for i, ex in enumerate(ds['train']):
    # Open the image
    image = ex['image']

    # Check if the mode is not 'L'
    if image.mode == 'L':
        exclude_idx.append(i)

ds['train'] = ds['train'].select(
    (
        i for i in range(len(ds['train'])) 
        if i not in set(exclude_idx)
    )
)

#%%
def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['label']
    return inputs

prepared_ds = ds.with_transform(transform)

import torch

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

import numpy as np
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

from transformers import ViTForImageClassification

labels = ds['train'].features['label'].names

import json
with open('./code/scene_classification/data_id2label.json', 'r') as f:
    data_id2label = json.load(f)
with open('./code/scene_classification/data_label2id.json', 'r') as f:
    data_label2id = json.load(f)
model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    #id2label={str(i): c for i, c in enumerate(labels)},
    #label2id={c: str(i) for i, c in enumerate(labels)}
    id2label=data_id2label,
    label2id=data_label2id
)

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="/mnt/cimec-storage6/users/filippo.merlo/vit-base-ade20k",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  fp16=True,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["val"],
    tokenizer=processor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['val'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)