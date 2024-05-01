#%% CUDA
import torch 

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print("Available GPUs:", device_count)
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Choose the GPU you want to use
    gpu_index = 1  # Replace 0 with the index of the GPU you want to use
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda")
    print('CUDA Ok')
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")

#%% IMPORT DATASET 
from datasets import load_dataset

ds = load_dataset("sezer12138/ade20k_image_classification", cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')

#%% PREPARE DATASET
from transformers import ViTImageProcessor
import torch

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

#%%
#def process_example(example):
#    inputs = processor(example['image'], return_tensors='pt')
#    inputs['pixel_values'] = torch.mean(inputs['pixel_values'], dim=1)
#    inputs['labels'] = example['label']
#    return inputs
#ex = ds['train'][0]
#inputs = process_example(ex)
#print(inputs['pixel_values'].shape)
#%%
#def transform(example_batch):
#    # Take a list of PIL images and turn them to pixel values
#    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
#
#    # Don't forget to include the labels!
#    inputs['labels'] = example_batch['label']
#    return inputs

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x for x in example_batch['image']], return_tensors='pt')

    # Calculate and store the mean pixel values for each image
    inputs['pixel_values'] = torch.mean(inputs['pixel_values'], dim=1)

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
from datasets import load_metric

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

from transformers import ViTForImageClassification

import json 
with open('data_id2label.json', 'r') as f:
    data_id2label = json.load(f)
with open('data_label2id.json', 'r') as f:
    data_label2id = json.load(f)

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(data_id2label.keys()),
    id2label=data_id2label,
    label2id=data_label2id
)

from transformers import TrainingArguments


training_args = TrainingArguments(
  output_dir="./vit-base-beans",
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
  report_to='wandb',
  load_best_model_at_end=True
)

# WANDB
import wandb
import os
os.environ["WANDB_PROJECT"]="Finetuning VIT for classification"

from transformers import Trainer

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["val"],
    tokenizer=processor
)

import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--out_path',
                help='Model memory output path', required=True)

    args = argparser.parse_args()

    # TRAIN
    train_results = trainer.train()
    trainer.save_model(output_dir=args.out_path)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # EVAL
    metrics = trainer.evaluate(prepared_ds['val'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)