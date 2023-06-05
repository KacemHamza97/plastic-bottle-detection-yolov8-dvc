import os

ROOT_PATH = "/root/hamza/yolo"
DATASET_DIR = os.path.join(ROOT_PATH, "data", "plastic_bottle_img_dataset")
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_DIR, "train", "labels")

VALID_IMAGES = os.path.join(DATASET_DIR, "valid", "images")
VALID_LABELS = os.path.join(DATASET_DIR, "valid", "labels")

TEST_IMAGES = os.path.join(DATASET_DIR, "test", "images")
TEST_LABELS = os.path.join(DATASET_DIR, "test", "labels")

YAML_CONFIG = os.path.join(DATASET_DIR, "data.yaml")

