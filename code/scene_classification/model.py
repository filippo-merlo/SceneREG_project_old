# Define clip model
from transformers import AutoProcessor, CLIPModel
import torch

if torch.backends.mps.is_available():
   device = torch.device("mps")
   print('CUDA Ok')
else:
   print ("MPS device not found.")

class clipModel():
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map = device)
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def predict(self, labels, image):
        text_labels = [f'a photo of a {label.replace("_", " ")}' for label in labels]
        inputs = self.processor(
            text=text_labels, images=image, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1)
        return probs
