#%%
from transformers import ViTImageProcessor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

from datasets import load_dataset
from PIL import Image

#ds = load_dataset("scene_parse_150", cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
ds = load_dataset("scene_parse_150")
#%%
len(ds['train'].features['scene_category'].names)
#%%
# Iterate through the dataset
def exclude_l(ds, split):
    exclude_idx = []
    for i, ex in enumerate(ds[split]):
        # Open the image
        image = ex['image']

        # Check if the mode is not 'L'
        if image.mode == 'L':
            exclude_idx.append(i)

    ds[split] = ds[split].select(
        (
            i for i in range(len(ds[split])) 
            if i not in set(exclude_idx)
        )
    )
    return ds

ds = exclude_l(ds, 'test')
ds = exclude_l(ds, 'validation')
ds = exclude_l(ds, 'train')

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

labels = ds['train'].features['scene_category'].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}

model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir="/mnt/cimec-storage6/users/filippo.merlo/vit-base-ade20k",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=8,
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
    eval_dataset=prepared_ds["validation"],
    tokenizer=processor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

metrics = trainer.evaluate(prepared_ds['test'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)