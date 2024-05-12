# Define clip model
from transformers import CLIPProcessor, CLIPModel
import torch

if torch.backends.mps.is_available():
   device = torch.device("mps")
   print('CUDA Ok')
else:
   print ("MPS device not found.")

class ClipModelWithClassifier():
    def __init__(self, num_labels):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device=device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.classifier_head = torch.nn.Linear(self.clip_model.config.hidden_size, num_labels, device=device)

    def forward(self, image):
        inputs = self.processor(
            images=image, 
            return_tensors="pt",
            padding=True
        )

        features = self.clip_model(**inputs).last_hidden_state[:, 0]
        logits = self.classifier_head(features)
        probabilities = torch.softmax(logits, dim=-1)
        return probabilities
