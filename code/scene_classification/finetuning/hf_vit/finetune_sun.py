### WANDB
import wandb
import os
import gc
wandb.login()

### Set a single environment variable
project_name = 'vit-base-patch16-224_SUN397'
os.environ["WANDB_PROJECT"] = project_name
os.environ["WANDB_LOG_MODEL"] = 'true'

### PREPARE THE DATASET   
import torchvision
import torch
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

checkpoint = 'google/vit-base-patch16-224'
cache_dir =  '/mnt/cimec-storage6/users/filippo.merlo'
processor = ViTImageProcessor.from_pretrained(checkpoint, cache_dir= cache_dir)

# Load the dataset
sun_data = torchvision.datasets.SUN397(root = cache_dir, download = True)
id2label = {v:k for k,v in sun_data.class_to_idx.items()}
label2id = sun_data.class_to_idx
label_len = len(label2id)

# Split the dataset
generator = torch.Generator().manual_seed(42)
train_set, val_set = torch.utils.data.random_split(sun_data, [0.8, 0.2], generator=generator)

from datasets import Dataset, DatasetDict
import numpy as np

# Convert to Hugging Face Dataset format
def convert_to_hf_dataset(torch_dataset):
    # Extract data and labels
    data = [torch_dataset[i][0] for i in range(len(torch_dataset))]
    labels = [torch_dataset[i][1] for i in range(len(torch_dataset))]
    
    # Create a dictionary
    data_dict = {"pixel_values": data, "labels": labels}
    
    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_dict(data_dict)
    return hf_dataset

train_hf_dataset = convert_to_hf_dataset(train_set)
test_hf_dataset = convert_to_hf_dataset(val_set)
print('Converted')
del train_set
del val_set
# Manually run garbage collection
gc.collect()

# Combine into a DatasetDict
dataset = DatasetDict({"train": train_hf_dataset, "test": test_hf_dataset})

# Remove individual datasets to free memory
del train_hf_dataset
del test_hf_dataset
gc.collect()

# Define the transform function
def preprocess_data(examples):
    # Take a list of PIL images and turn them to pixel values
    examples['pixel_values'] = [processor(img.convert('RGB'), return_tensors='pt') for img in examples['pixel_values']]
    return examples

dataset = dataset.map(preprocess_data, batched=True)

# Define the compute metrics function
import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available(): 
    print('Using GPU')

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
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics_fn
)

# start training loop
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()

# Eval
metrics = trainer.evaluate(dataset['test'])
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
