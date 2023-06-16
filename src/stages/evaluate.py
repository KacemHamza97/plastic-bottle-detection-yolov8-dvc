import argparse
import os
import torch
import yaml

from torch.cuda import empty_cache
from typing import Text
from src.library.visualize import plot_metrics, evaluate_model_on_test_images


def evaluate_model(config_path: Text) -> None:
    """Evaluate model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # free up GPU memory
    empty_cache()
    # Define the path of the saved YOLOv8 model
    model_path = config['train']['model_path']
    # Load the entire saved model
    model = torch.load(model_path)
    # Evaluating the model on the test dataset
    metrics = model.val(conf=0.25, split='test', device=config['evaluate']['device'])
    # Saving metrics to the reports folder
    reports_folder = config['evaluate']['reports_dir']
    metrics_image_path = os.path.join(reports_folder, config['evaluate']['metrics_image'])
    pred_images_path = os.path.join(reports_folder, config['evaluate']['predicted_images'])

    plot_metrics(metrics, save_to=metrics_image_path)
    evaluate_model_on_test_images(model, config['data_split']['test_path'], num_images=32, save_to=pred_images_path)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    evaluate_model(config_path=args.config)
