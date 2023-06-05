from torch.cuda import empty_cache
from ultralytics import YOLO


def train_yolov8(params):
    # Loading a pretrained model
    model = YOLO('yolov8n.pt')

    # free up GPU memory
    empty_cache()

    # Training the model
    model.train(**params)

    # Loading the best performing model
    best_model = YOLO('./runs/detect/train/weights/best.pt')

    return best_model
