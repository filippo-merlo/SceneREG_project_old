# import YOLO model
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt') # load a pretrained model (recommended for training)

# Train the model
model.train(data='/kaggle/working/data', epochs=5)
