"""
config.py - Centralized configuration for document classifier

This file contains all training, model, and processing parameters.
Modify here to adjust training behavior, model selection, and text processing.
"""

from pathlib import Path
import torch

class Config:
    """Configuration for 2-model document classification system"""
    
    def __init__(self):
        # ============ PATHS ============
        self.BASE_DIR = Path(__file__).parent.parent
        self.DATA_DIR = self.BASE_DIR / "data"  # Training PDFs: acm/, compliance/, ieee/, legal/, springer/
        self.MODELS_DIR = self.BASE_DIR / "models"  # Saved fine-tuned models
        self.OUTPUTS_DIR = self.BASE_DIR / "outputs"  # Evaluation results
        
        # Create necessary directories
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.OUTPUTS_DIR.mkdir(exist_ok=True)
        
        # ============ DOCUMENT CATEGORIES ============
        self.LABELS = ["acm", "compliance", "ieee", "legal", "springer"]
        self.LABEL2ID = {label: idx for idx, label in enumerate(self.LABELS)}
        self.ID2LABEL = {idx: label for label, idx in self.LABEL2ID.items()}
        self.NUM_LABELS = len(self.LABELS)
        
        # ============ MODEL SELECTION ============
        # Simplified to 2 models for fast inference and comparison
        self.MODELS = {
            "Legal-BERT": ("nlpaueb/legal-bert-base-uncased", "legal_bert"),  # Domain-optimized for legal text
            "DeBERTa": ("microsoft/deberta-base", "deberta"),  # Improved transformer architecture
        }
        
        # ============ TEXT PROCESSING ============
        self.MAX_LENGTH = 512  # Max tokens for transformer input
        self.CHUNK_SIZE = 450  # Chunk size for long PDFs (avoid truncation)
        self.CHUNK_OVERLAP = 50  # Overlap between chunks for context
        self.MIN_TEXT_LENGTH = 50  # Minimum characters required from a PDF
        self.MAX_PAGES = 10  # Maximum pages to extract (limit processing time)
        
        # ============ TRAINING HYPERPARAMETERS ============
        self.BATCH_SIZE = 4  # Small batch for efficient memory usage on RTX 4050
        self.NUM_EPOCHS = 15  # Epochs for convergence (increased from 10)
        self.LEARNING_RATE = 5e-6  # Conservative LR for weighted loss stability (decreased from 1e-5)
        self.WARMUP_STEPS = 100  # Gradual learning rate warmup
        self.WEIGHT_DECAY = 0.01  # L2 regularization
        
        # ============ CLASS WEIGHTS ============
        # Addresses imbalanced data (10 compliance vs 15 IEEE/ACM)
        # Higher weight = model focuses more on minority classes
        self.CLASS_WEIGHTS = {
            "compliance": 5.0,  # Critical minority category
            "legal": 4.0,  # Minority category
            "springer": 3.0,  # Underrepresented
            "acm": 1.0,  # Well-represented baseline
            "ieee": 1.0,  # Well-represented baseline
        }
        
        # ============ DEVICE & TRAINING ============
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.USE_FP16 = True  # Mixed precision training (faster + less memory)
config = Config()

