#%%
from transformers import ViTImageProcessor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

from datasets import load_dataset
from PIL import Image

ds = load_dataset("sezer12138/ade20k_image_classification", cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')

filtered_examples = []

# Iterate through the dataset
for ex in ds['train']:
    # Open the image
    image = ex['image']

    # Check if the mode is not 'L'
    if image.mode != 'L':
        filtered_examples.append(ex)

# Create a new dataset with filtered examples
ds['train'] = filtered_examples 

filtered_examples = []
for ex in ds['val']:
    # Open the image
    image = ex['image']

    # Check if the mode is not 'L'
    if image.mode != 'L':
        filtered_examples.append(ex)

ds['val'] = filtered_examples 

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

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="./vit-base-ade20k",
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