import torch
import os

BASE_PATH = 'dataset'
IMAGES_PATH = os.path.sep.join([BASE_PATH, 'images'])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, 'annotations'])

BASE_OUTPUT = 'output'

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, 'detector.pth'])
LE_PATH = os.path.sep.join([BASE_OUTPUT, 'le.pickle'])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, 'plots'])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, 'test_paths.txt'])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PIN_MEMORY = DEVICE == 'cuda'

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]

INIT_LR = 1e-4
NUM_EPOCHS = 20
BATCH_SIZE = 32

LABELS = 1.
BBOX = 1.
