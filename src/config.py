"""
config.py - Simple configuration
"""

from pathlib import Path
import torch

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Create directories
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MODEL_PATH = MODEL_DIR / "legal_bert_classifier"
NUM_LABELS = 5
LABELS = ["ACM", "IEEE", "Springer", "Legal", "Compliance"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Text processing
MAX_LENGTH = 512
CHUNK_SIZE = 425          # 400-450 tokens
CHUNK_OVERLAP = 50
MIN_TEXT_LENGTH = 100
MAX_PAGES = 50

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
USE_FP16 = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inference
TEMPERATURE = 1.5
CONFIDENCE_THRESHOLD = 0.5