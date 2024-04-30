#%% CUDA
import torch

if torch.backends.mps.is_available():
   device = torch.device("gpu")
   print('CUDA Ok')
else:
   print ("Device not found.")

#%% IMPORT DATASET 
from datasets import load_dataset

ds = load_dataset("sezer12138/ade20k_image_classification")
#%% PREPARE DATASET
from transformers import ViTImageProcessor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

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
from datasets import load_metric

metric = load_metric("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

from transformers import ViTForImageClassification

labels = ds['train'].features['label'].names

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
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
  report_to='tensorboard',
  load_best_model_at_end=True
)

# WANDB
import wandb
wandb.init(project='Finetuning VIT for classification', entity='ViT_ade20k_classification')
wandb.config.update(training_args)


from transformers import Trainer

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["val"],
    tokenizer=processor,
    callbacks=[wandb.WandbCallback()]
)

import argparse
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--out_path', '-o',
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