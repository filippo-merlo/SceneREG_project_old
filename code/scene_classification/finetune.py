# WANDB
import wandb
wandb.login()
import os
project_name = 'vit-large-patch16-224-in21k'
# Set a single environment variable
os.environ["WANDB_PROJECT"] = project_name
os.environ["WANDB_LOG_MODEL"] = 'true'
#%%
from transformers import ViTImageProcessor

cache_dir = '/mnt/cimec-storage6/users/filippo.merlo'
checkpoint = 'google/vit-large-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(checkpoint, cache_dir= cache_dir)

from datasets import load_dataset

datasets = load_dataset("scene_parse_150", cache_dir= cache_dir)
labels = datasets['train'].features['scene_category'].names

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x.convert('RGB') for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['scene_category']
    return inputs

datasets_processed = datasets.with_transform(transform)

import torch

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# define function to compute metrics
import numpy as np
import evaluate

def compute_metrics_fn(eval_preds):
  metrics = dict()
  
  accuracy_metric = evaluate.load('accuracy', cache_dir= cache_dir)
  precision_metric = evaluate.load('precision', cache_dir= cache_dir)
  recall_metric = evaluate.load('recall', cache_dir= cache_dir)
  f1_metric = evaluate.load('f1', cache_dir= cache_dir)


  logits = eval_preds.predictions
  labels = eval_preds.label_ids
  preds = np.argmax(logits, axis=-1)  
  
  metrics.update(accuracy_metric.compute(predictions=preds, references=labels))
  metrics.update(precision_metric.compute(predictions=preds, references=labels, average='weighted'))
  metrics.update(recall_metric.compute(predictions=preds, references=labels, average='weighted'))
  metrics.update(f1_metric.compute(predictions=preds, references=labels, average='weighted'))

  return metrics

# INIT MODEL
from transformers import ViTForImageClassification

id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

def model_init():
    vit_model = ViTForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir= cache_dir
    )
    return vit_model


from transformers import TrainingArguments, Trainer

# set training arguments
training_args = TrainingArguments(
    output_dir=f'/mnt/cimec-storage6/users/filippo.merlo/{project_name}',
    report_to='wandb',  # Turn on Weights & Biases logging
    num_train_epochs=10,
    learning_rate=float(2e-5),
    weight_decay=0.4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    save_strategy='epoch',
    evaluation_strategy='epoch',
    logging_strategy='epoch',
    load_best_model_at_end=True,
    remove_unused_columns=False,
    fp16=True
)

# define training loop
trainer = Trainer(
    # model,
    model_init=model_init,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=datasets_processed['train'],
    eval_dataset=datasets_processed['validation'],
    compute_metrics=compute_metrics_fn
)

# start training loop
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Eval
metrics = trainer.evaluate(datasets['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
