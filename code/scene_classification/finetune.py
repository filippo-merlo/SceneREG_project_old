# WANDB
import wandb
import os

wandb.login()

# Set a single environment variable
project_name = 'vit-base-patch16-224-in21k'
os.environ["WANDB_PROJECT"] = project_name
os.environ["WANDB_LOG_MODEL"] = 'true'
#%%
from transformers import ViTImageProcessor

cache_dir = '/mnt/cimec-storage6/users/filippo.merlo'
#cache_dir = '/Users/filippomerlo/Documents/GitHub/SceneReg_project/code/scene_classification/cache'

checkpoint = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(checkpoint, cache_dir= cache_dir)

# Load the dataset
from datasets import load_dataset, concatenate_datasets, DatasetDict, ClassLabel
ds = load_dataset("scene_parse_150", cache_dir= cache_dir)

# Remove test split
dataset = DatasetDict()
dataset['train'] = ds['train']
dataset['validation'] = ds['validation']

# Inspect the dataset
from collections import Counter
import numpy as np

names = dataset['train'].features['scene_category'].names
names2id = dict(zip(names, range(len(names))))
id2names = dict(zip(range(len(names)), names))

# Count the occurrences of each label
tot_labs = dataset['train']['scene_category'] + dataset['validation']['scene_category']
counter = Counter(tot_labs)
# Get the labels
labels = list(counter.keys())

names2id_filtered = dict()
for label in labels:
    if counter[label] >= 10:
        names2id_filtered[id2names[label]] = label

filter_dataset = dataset.filter(lambda example: example['scene_category'] in names2id_filtered.values())
ds =  concatenate_datasets([filter_dataset['train'], filter_dataset['validation']])
splitted_dataset = ds.train_test_split(test_size=0.1, shuffle=True, seed=42)
final_dataset = DatasetDict()
final_dataset['train'] = splitted_dataset['train']
final_dataset['validation'] = splitted_dataset['test']

cl_lab = ClassLabel(names=list(names2id_filtered.keys()), num_classes=len(names2id_filtered.keys()))
final_dataset['train'] =  final_dataset['train'].cast_column('scene_category', cl_lab)
final_dataset['validation'] = final_dataset['validation'].cast_column('scene_category', cl_lab)

new_names2id = dict()
for i, name in enumerate(names2id_filtered.keys()):
    new_names2id[name] = i

old_2_new_map = dict()
for name, old_id in names2id_filtered.items():
    new_id = new_names2id[name]
    old_2_new_map[old_id] = new_id

new_train_l= [old_2_new_map[x] for x in final_dataset['train']['scene_category']]
new_valid_l = [old_2_new_map[x] for x in final_dataset['validation']['scene_category']]
final_dataset['train'] = final_dataset['train'].remove_columns('scene_category').add_column('scene_category', new_train_l)
final_dataset['validation'] = final_dataset['validation'].remove_columns('scene_category').add_column('scene_category', new_valid_l)

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = processor([x.convert('RGB') for x in example_batch['image']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['labels'] = example_batch['scene_category']
    return inputs

datasets_processed = final_dataset.with_transform(transform)

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

id2label = {str(v):k for k,v in new_names2id.items()}
label2id = new_names2id
label_len = len(new_names2id.keys())

def model_init():
    vit_model = ViTForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(names2id_filtered.keys()),
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
metrics = trainer.evaluate(final_dataset['validation'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
