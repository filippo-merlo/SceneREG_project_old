#%% CUDA
import torch 

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print("Available GPUs:", device_count)
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Choose the GPU you want to use
    gpu_index = 2  # Replace 0 with the index of the GPU you want to use
    torch.cuda.set_device(gpu_index)
    device = torch.device("cuda")
    print('CUDA Ok')
else:
    print("CUDA not available. Using CPU.")
    device = torch.device("cpu")
#%%
from transformers import ViTImageProcessor

model_name_or_path = 'openai/clip-vit-large-patch14'
processor = ViTImageProcessor.from_pretrained(model_name_or_path, cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
#%%
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

ds = load_dataset("scene_parse_150", cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')
#ds = load_dataset("scene_parse_150")
#%%
#ds['train']['scene_category']
#%%
# Iterate through the dataset
def exclude_l(ds, split):
    exclude_idx = []
    for i, ex in tqdm(enumerate(ds[split])):
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
    inputs['labels'] = example_batch['scene_category']
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
    cache_dir= '/mnt/cimec-storage6/users/filippo.merlo',
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
    model=model.to(device),
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