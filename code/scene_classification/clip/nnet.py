# Define clip model
from transformers import CLIPVisionModel
import torch
from config import * 
class ClipModelWithClassifier(torch.nn.Module):
    def __init__(self, num_labels):
        super(ClipModelWithClassifier, self).__init__()
        self.clip_model = CLIPVisionModel.from_pretrained(model_checkpoint, cache_dir=cache_dir)
        self.classifier_head = torch.nn.Linear(in_features=768, out_features=num_labels)

    def forward(self, input):
        outputs = self.clip_model(**input)
        #last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output  # pooled CLS states
        logits = self.classifier_head(pooled_output)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
