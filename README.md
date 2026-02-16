# Document Classifier

Multi-document PDF classification system using **Legal-BERT** and **DeBERTa** transformers.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Training data structure (64 PDFs total)
data/
├── acm/          (15 PDFs)
├── compliance/   (10 PDFs)
├── ieee/         (15 PDFs)
├── legal/        (10 PDFs)
└── springer/     (14 PDFs)

# 3. Train models with weighted loss for imbalanced data
python src/train_weighted.py

# 4. Run Flask web app
python web_app/app.py
# Visit http://localhost:5000
```

## Project Structure

```
llm_classifier/
├── data/                 # Training PDFs organized by category
├── models/               # Trained model weights
│   ├── legal_bert/      # Legal-BERT fine-tuned model
│   └── deberta/         # DeBERTa fine-tuned model
├── src/                  # Core modules
│   ├── config.py        # Configuration & paths
│   ├── train_weighted.py # Training script with class weights
│   ├── multi_classifier.py # Model loading & inference
│   ├── text_processor.py # PDF text extraction
│   └── requirements.txt
├── web_app/              # Flask web UI
│   ├── app.py           # Flask server
│   └── templates/
│       └── index.html   # Web interface
├── outputs/              # Evaluation results
└── README.md
```

## Models

- **Legal-BERT**: `nlpaueb/legal-bert-base-uncased` - Domain-specific legal language model
- **DeBERTa**: `microsoft/deberta-base` - Improved attention mechanism with disentangled embeddings

## Training Configuration

Models trained with **class weights** to handle imbalanced dataset:
- compliance: 5.0x (minority)
- legal: 4.0x (minority)
- springer: 3.0x (underrepresented)
- acm: 1.0x (baseline)
- ieee: 1.0x (baseline)

See `src/config.py` for training parameters:
- Epochs: 15
- Learning rate: 5e-6
- Batch size: 4
- Max tokens: 512

## Web UI Features

- ✅ Upload 1-20 PDFs
- ✅ Dual-model predictions (Legal-BERT + DeBERTa)
- ✅ Ground truth comparison (HIT/MISS badges)
- ✅ Confidence percentages
- ✅ Side-by-side model comparison
- ✅ Supports unlabeled PDFs (shows UNKNOWN)

## API

**POST** `/analyze_multiple`
- Upload PDFs and get predictions from both models
- Returns: predictions, confidences, ground truth matching

**GET** `/health`
- Server health check

## Performance

- **Inference time**: <2s per PDF
- **Current accuracy**: 50% validation (needs more training data for ACM/IEEE distinction)
- **GPU**: RTX 4050 optimized via `accelerate`

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Add training PDFs to `data/{category}/` folders
3. Train: `python src/train_weighted.py`
4. Run: `python web_app/app.py`
5. Open: http://localhost:5000
