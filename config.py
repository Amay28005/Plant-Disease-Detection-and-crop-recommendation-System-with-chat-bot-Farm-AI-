import torch
import os

DATA_DIR    = r"C:\Users\amays\OneDrive\Desktop\plant_disease\data"
OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

IMAGE_SIZE  = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

MODEL_NAME      = "efficientnet_b4"
NUM_EPOCHS      = 30
BATCH_SIZE      = 16
NUM_WORKERS     = 2

LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1

SCHEDULER       = "cosine"
WARMUP_EPOCHS   = 5
PATIENCE        = 5

MIXED_PRECISION = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
LAST_CHECKPOINT = os.path.join(OUTPUT_DIR, "last_model.pth")
