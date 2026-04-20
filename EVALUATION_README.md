"""
EVALUATION SCRIPTS README
========================

This directory contains three independent evaluation scripts for benchmarking the 
DeBERTa and LegalBERT models on your document classification task.

## Scripts Overview

### 1. evaluate_deberta.py
   - Independently loads the fine-tuned DeBERTa model from models/deberta/
   - Evaluates on all ~64 PDFs in the data/ directory
   - Calculates: overall accuracy, per-class accuracy, prediction confidence
   - Outputs: outputs/deberta_evaluation.csv
   - Does NOT modify the main classifier code
   
### 2. evaluate_legalbert.py
   - Independently loads the fine-tuned LegalBERT model from models/legal_bert/
   - Evaluates on the same ~64 PDFs
   - Same metrics and output format as DeBERTa evaluation
   - Outputs: outputs/legalbert_evaluation.csv
   - Does NOT modify the main classifier code

### 3. compare_models.py
   - Runs both evaluate_deberta.py and evaluate_legalbert.py automatically
   - Generates comprehensive side-by-side comparison
   - Outputs: outputs/model_comparison.txt (human-readable report)
   - Includes overall accuracy, per-class breakdown, confidence metrics, recommendations

## CSV Output Format

Both evaluation scripts produce CSVs with these columns:
   - filename: Name of the PDF file
   - ground_truth: Category extracted from directory (acm, compliance, ieee, legal, springer)
   - prediction: Model's predicted label
   - confidence: Softmax confidence score (0.0000 - 1.0000)
   - correct: Yes/No indicating if prediction matches ground truth

## Usage

### Run Individual Evaluations:

```bash
# Evaluate DeBERTa only
python evaluate_deberta.py

# Evaluate LegalBERT only
python evaluate_legalbert.py
```

### Run Full Comparison:

```bash
# Runs both models and generates comparison report
python compare_models.py
```

## Expected Outputs

After running, check the outputs/ directory:

```
outputs/
├── deberta_evaluation.csv       (detailed DeBERTa results)
├── legalbert_evaluation.csv     (detailed LegalBERT results)
└── model_comparison.txt         (human-readable comparison report)
```

## Key Metrics Explained

### Hit Rate (Accuracy)
- Percentage of predictions matching ground truth
- Overall: Across all 5 categories
- Per-class: Specific accuracy for each category (acm, compliance, etc.)

### Confidence Score
- Softmax probability [0.0000 - 1.0000]
- Higher = model is more certain about its prediction
- Average confidence can indicate model calibration

### Per-Class Breakdown
- Shows which categories each model handles better
- Helps identify class-specific weaknesses (e.g., ACM vs IEEE confusion)

## Interpretation

The comparison report provides:
1. Raw accuracy numbers for both models
2. Per-category accuracy breakdown
3. Average prediction confidence
4. Misclassification count and rate
5. Recommendation on which model to use (or ensemble approach)
6. Best model per category

Example insights:
- "DeBERTa outperforms with 5% higher accuracy"
- "LegalBERT excels at legal documents (85% vs 70%)"
- "Both models struggle with IEEE (45% accuracy) - need more training data"

## Independence & Safety

✓ DeBERTa evaluation is completely independent (loads only DeBERTa model)
✓ LegalBERT evaluation is completely independent (loads only LegalBERT model)
✓ Neither script modifies main classifier code in src/ or web_app/
✓ Both use read-only operations on PDFs from data/
✓ Results are written only to outputs/ directory

## Troubleshooting

**Error: "Failed to load DeBERTa model"**
- Check that models/deberta/ directory exists
- Run: `ls models/` to verify both model directories exist

**Error: "No PDF samples found"**
- Check that PDF files exist in data/acm/, data/compliance/, etc.
- Run: `ls data/ -R` to verify directory structure

**CSV files not created**
- Check outputs/ directory exists (scripts create it automatically)
- Check write permissions on outputs/ folder

**CUDA out of memory**
- Reduce BATCH_SIZE in the evaluation script (currently 4)
- Or use CPU (slower but sufficient for 64 samples)

## Performance Tips

- First run may be slower as models download/compile
- Subsequent runs are faster (cached model weights)
- GPU (CUDA) is 5-10x faster than CPU
- Each evaluation takes ~1-2 minutes on RTX 4050

## Next Steps

After comparing models:
1. Review comparison report in outputs/model_comparison.txt
2. Check which model has better accuracy
3. Consider ensemble approach (combine both models)
4. Use better-performing model in web_app/app.py (or keep both)
5. Retrain models with improved data if needed (currently 50% accuracy)
"""

# This is just a documentation file - not executed as Python code
