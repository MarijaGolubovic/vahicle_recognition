import torch

CLASSES = [
    '__background__', 'bus', 'car', 'truck'
]
NUM_CLASSES  = len(CLASSES)
RESIZE_TO = 640
NUM_EPOCHS = 1

OUT_DIR = 'outputs'
TRAIN_DIR  = "train"
VALID_DIR = "valid"
TEST_DIR = "test"

BATCH_SIZE = 5
NUM_WORKERS = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
