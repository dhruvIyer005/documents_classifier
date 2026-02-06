"""train_legal_bert.py - Train Legal-BERT model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.train import main

if __name__ == "__main__":
    print("="*70)
    print("Training Legal-BERT Model")
    print("="*70)
    main()
