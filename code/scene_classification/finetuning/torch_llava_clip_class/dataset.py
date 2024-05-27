# Split the dataset
### Define Collection Dataset
import torch
from torch.utils.data import Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CollectionsDataset(Dataset):
    def __init__(self, 
                 hf_dataset, 
                 processor=None):
        
        self.data = hf_dataset
        self.clip = processor['clip_model']
        self.processor = processor['clip_processor']
        self.pipe = processor['llava_pipeline']
        self.num_classes = len(self.data.features['scene_category'].names)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        label = self.data[idx]['scene_category']
        label_tensor = torch.zeros(self.num_classes)
        label_tensor[label] = 1

        if self.transform:
            llava_caption = self.pipe(image, prompt="Where is the picture taken?", generate_kwargs={"max_new_tokens": 200})
            inputs = self.processor(text=llava_caption, images=image, return_tensors="pt", padding=True).to(device)
            outputs = self.clip(**inputs)
            txt_features = outputs.text_model_output
            img_features = outputs.vision_model_output
            reppresentation = torch.cat([txt_features, img_features], dim=1)


        return {'reppresentation': reppresentation,
                'labels': label_tensor
                }




