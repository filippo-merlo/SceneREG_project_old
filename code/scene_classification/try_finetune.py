# WANDB
import wandb
wandb.login()
import os

# Set a single environment variable
os.environ["WANDB_PROJECT"] = 'vit_snacks_sweeps'
os.environ["WANDB_LOG_MODEL"] = 'true'

#%%
from transformers import ViTImageProcessor, ViTFeatureExtractor

#checkpoint = 'openai/clip-vit-large-patch14'
checkpoint = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(checkpoint, cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
feature_extractor = ViTFeatureExtractor.from_pretrained(checkpoint, cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
#processor = ViTImageProcessor.from_pretrained(checkpoint)
#feature_extractor = ViTFeatureExtractor.from_pretrained(checkpoint)

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

ds = load_dataset("scene_parse_150", cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
#ds = load_dataset("scene_parse_150")
#%%

# Remove 'L'
## Iterate through the dataset
#def exclude_l(ds, split):
#    exclude_idx = []
#    for i, ex in tqdm(enumerate(ds[split])):
#        # Open the image
#        image = ex['image']
#
#        # Check if the mode is not 'L'
#        if image.mode == 'L':
#            exclude_idx.append(i)
#
#    ds[split] = ds[split].select(
#        (
#            i for i in range(len(ds[split])) 
#            if i not in set(exclude_idx)
#        )
#    )
#    return ds
#
#ds = exclude_l(ds, 'validation')
#ds = exclude_l(ds, 'train')
#%%

# data augmentation transformations
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    RandomResizedCrop,
    RandomHorizontalFlip,
    RandomAdjustSharpness,
    ToTensor,
    ToPILImage
)

# train
train_aug_transforms = Compose([
    RandomResizedCrop(size=feature_extractor.size),
    RandomHorizontalFlip(p=0.5),
    RandomAdjustSharpness(sharpness_factor=5, p=0.5),
    ToTensor(),
    Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

# validation/test
valid_aug_transforms = Compose([
    Resize(size=(feature_extractor.size, feature_extractor.size)),
    ToTensor(),
    Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
])

def apply_train_aug_transforms(examples):
  examples['pixel_values'] = [train_aug_transforms(img.convert('RGB')) for img in examples['image']]
  return examples


def apply_valid_aug_transforms(examples):
  examples['pixel_values'] = [valid_aug_transforms(img.convert('RGB')) for img in examples['image']]
  return examples


ds['train'].set_transform(apply_train_aug_transforms)
ds['validation'].set_transform(apply_valid_aug_transforms)
ds['test'].set_transform(apply_valid_aug_transforms)
datasets_processed = ds.rename_column('scene_category', 'labels')

#%%
#
#def transform(example_batch):
#    # Take a list of PIL images and turn them to pixel values
#    inputs = processor([x for x in example_batch['image']], return_tensors='pt')
#
#    # Don't forget to include the labels!
#    inputs['labels'] = example_batch['scene_category']
#    return inputs
#
#prepared_ds = ds.with_transform(transform)
#
#%%

import torch

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

# define function to compute metrics
import numpy as np
import evaluate

def compute_metrics(p):
    metric = evaluate.load("accuracy", cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def compute_metrics_fn(eval_preds):
  metrics = dict()
  
  accuracy_metric = evaluate.load('accuracy', cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
  precision_metric = evaluate.load('precision', cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
  recall_metric = evaluate.load('recall', cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
  f1_metric = evaluate.load('f1', cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')


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

labels = ds['train'].features['scene_category'].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

def model_init():
    vit_model = ViTForImageClassification.from_pretrained(
        checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        cache_dir= '/mnt/cimec-storage6/users/filippo.merlo'
    )
    return vit_model


## SWEEPS
# method
sweep_config = {
    'method': 'random'
}

# hyperparameters
parameters_dict = {
    'epochs': {
        'value': 1
        },
    'batch_size': {
        'values': [8, 16, 32, 64]
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-3
    },
    'weight_decay': {
        'values': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    },
}

sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project='vit-snacks-sweeps')

from transformers import TrainingArguments, Trainer


def train(config=None):
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config

    # set training arguments
    training_args = TrainingArguments(
        output_dir='/mnt/cimec-storage6/users/filippo.merlo/vit-sweeps',
	    report_to='wandb',  # Turn on Weights & Biases logging
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
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
    trainer.train()

wandb.agent(sweep_id, train, count=20)
#train_results = trainer.train()
#trainer.save_model()
#trainer.log_metrics("train", train_results.metrics)
#trainer.save_metrics("train", train_results.metrics)
#trainer.save_state()
#
#metrics = trainer.evaluate(prepared_ds['validation'])
#trainer.log_metrics("eval", metrics)
#trainer.save_metrics("eval", metrics)