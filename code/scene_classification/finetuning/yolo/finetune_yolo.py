# import YOLO model
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt', ) # load a pretrained model (recommended for training)

# Train the model
model.train(data='/kaggle/working/data', epochs=5)

# Validate the model
metrics = model.val() # no arguments needed, dataset and settings remembered
metrics.top1 # top1 accuracy
metrics.top5 # top5 accuracy