#%%
from transformers import ViTImageProcessor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

from datasets import load_dataset

ds = load_dataset("sezer12138/ade20k_image_classification", cache_dir= '/mnt/cimec-storage6/users/filippo.merlo')

#%%
from PIL import Image

import torch

def normalize(image, mean=[0.5], std=[0.5]):
    """
    Normalizes pixel values in an image with a specified mean and standard deviation.
    
    Parameters:
    image (torch.Tensor): Input image as a PyTorch tensor.
    mean (list of float): Mean value for each channel. Default is [0.5].
    std (list of float): Standard deviation for each channel. Default is [0.5].
    
    Returns:
    torch.Tensor: Normalized image.
    """
    # Convert pixel values to floats
    image = image.float()
    
    # Normalize pixel values
    image /= 255.0
    
    # Apply mean and standard deviation normalization
    for i in range(len(mean)):
        image[..., i] = (image[..., i] - mean[i]) / std[i]
    
    return image

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    images = [Image.open(x).convert('RGB') for x in example_batch['image']]
    inputs = processor(images, return_tensors='pt')

    # Normalize using the correct mean for grayscale images (assuming mean=0.5)
    inputs['pixel_values'] = normalize(inputs['pixel_values'], mean=[0.5], std=[0.5])

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