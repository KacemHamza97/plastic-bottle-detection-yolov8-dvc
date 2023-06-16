import argparse
import yaml
import torch

from torch.cuda import empty_cache
from ultralytics import YOLO
from typing import Text


def train_yolov8(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Loading a pretrained model
    model = YOLO('yolov8n.pt')

    # free up GPU memory
    empty_cache()

    # Training the model
    model.train(
        data=config['train']['data'],
        device=config['train']['device'],
        epochs=config['train']['epochs'],
        imgsz=config['train']['imgsz'],
        seed=config['train']['seed'],
        batch=config['train']['batch'],
        workers=config['train']['workers']
    )

    # Loading the best performing model
    best_model = YOLO(config['train']['best_model'])
    # Export the model
    torch.save(best_model, config['train']['model_path'])


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    train_yolov8(config_path=args.config)
