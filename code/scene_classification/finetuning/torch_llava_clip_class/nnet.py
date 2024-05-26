# Define clip model
from transformers import CLIPVisionModel
import torch
from config import * 
import torch.nn.functional as F

class AttentionClassifier(torch.nn.Module):
    def __init__(self, num_labels, feature_size):
        super(AttentionClassifier, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = torch.nn.Linear(feature_size, feature_size)
        self.query = torch.nn.Linear(feature_size, feature_size)
        self.value = torch.nn.Linear(feature_size, feature_size)

        self.classifier_head = torch.nn.Linear(in_features=feature_size, out_features=num_labels)

    def forward(self, x, mask=None):
          
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        # classifier head
        logits = self.classifier_head(output)

        # compute probabilities
        probabilities = torch.softmax(logits, dim=-1)

        return logits

