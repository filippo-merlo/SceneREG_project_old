### WANDB
import wandb
import os

wandb.login()

### Set a single environment variable
project_name = 'vit-base-patch16-224'
os.environ["WANDB_PROJECT"] = project_name
os.environ["WANDB_LOG_MODEL"] = 'true'

### PREPARE THE DATASET   
from config import *
import torchvision
import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

checkpoint = 'google/vit-base-patch16-224'
processor = ViTImageProcessor.from_pretrained(checkpoint, cache_dir= cache_dir)

def transform(image):
    # Take a list of PIL images and turn them to pixel values
    image = processor(image.convert('RGB'), return_tensors='pt')
    # Don't forget to include the labels!
    return image

sun_data = torchvision.datasets.SUN397(root = cache_dir, transform=transform,  download = True)
generator = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(sun_data, [0.8, 0.2], generator=generator)
train_dl = DataLoader(train_set, batch_size = 16)
test_dl = DataLoader(val_set, batch_size = 16)

def collate_fn(batch):
    return {
        'pixel_values': batch[0],
        'labels': batch[1]
    }

import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
        ignore_mismatched_sizes=True,
        num_labels=label_len,
        id2label=id2label,
        label2id=label2id,
        cache_dir= cache_dir
    )
    for param in vit_model.parameters():
        param.requires_grad = False
    vit_model.classifier.weight.requires_grad = True
    vit_model.classifier.bias.requires_grad = True
    return vit_model

from transformers import TrainingArguments, Trainer

# set training arguments
training_args = TrainingArguments(
    output_dir=f'/mnt/cimec-storage6/users/filippo.merlo/{project_name}',
    report_to='wandb',  # Turn on Weights & Biases logging
    num_train_epochs=10,
    learning_rate=float(2e-4),
    weight_decay=0.1,
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
    model_init=model_init,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dl,
    eval_dataset=test_dl,
    compute_metrics=compute_metrics_fn
)

# start training loop
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Eval
metrics = trainer.evaluate(test_dl)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
