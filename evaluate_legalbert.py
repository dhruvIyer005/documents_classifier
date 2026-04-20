"""
evaluate_legalbert.py - Independent LegalBERT model evaluation

Standalone script to evaluate the fine-tuned LegalBERT model on all PDFs.
Does NOT modify the main classifier. Produces CSV with predictions, confidence, accuracy.
"""

import csv
import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ============ CONFIG ============
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

LEGALBERT_MODEL_PATH = MODELS_DIR / "legal_bert"
OUTPUT_CSV = OUTPUTS_DIR / "legalbert_evaluation.csv"

LABELS = ["acm", "compliance", "ieee", "legal", "springer"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

logger.info(f"Device: {DEVICE}")
logger.info(f"LegalBERT Model Path: {LEGALBERT_MODEL_PATH}")

# ============ SETUP MODEL & TOKENIZER ============
try:
    logger.info("Loading LegalBERT model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(LEGALBERT_MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(LEGALBERT_MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    logger.info("✓ LegalBERT model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load LegalBERT model: {e}")
    raise


def extract_pdf_text(pdf_path: str) -> Optional[str]:
    """Extract text from PDF using PyMuPDF with minimal processing."""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        if len(text) > 50:
            return text[:3000]  # Limit to 3000 chars
        return None
    except Exception as e:
        logger.warning(f"Error extracting {pdf_path}: {e}")
        return None


def extract_ground_truth(filename: str) -> Optional[str]:
    """Extract category from filename (e.g., 'acm7.pdf' -> 'acm')."""
    match = re.match(r'^([a-z]+)', filename.lower())
    if match:
        label = match.group(1)
        if label in LABELS:
            return label
    return None


def predict_batch(texts: List[str]) -> Tuple[List[int], List[float]]:
    """Predict labels and confidence scores for a batch of texts."""
    if not texts:
        return [], []
    
    # Tokenize
    inputs = tokenizer(
        texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract predictions and confidences
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().tolist()
    confidences = torch.softmax(logits, dim=1).max(dim=1)[0].cpu().tolist()
    
    return predictions, confidences


def collect_pdf_samples() -> List[Tuple[str, str, str]]:
    """Collect all PDFs from data directory. Returns list of (pdf_path, text, ground_truth)."""
    samples = []
    
    for category_dir in DATA_DIR.iterdir():
        if not category_dir.is_dir() or category_dir.name == "synthetic":
            continue
        
        category = category_dir.name
        pdf_files = sorted(category_dir.glob("*.pdf"))
        
        logger.info(f"Processing {category}: {len(pdf_files)} PDFs")
        
        for pdf_file in pdf_files:
            text = extract_pdf_text(str(pdf_file))
            if text:
                ground_truth = category  # Use directory name as ground truth
                samples.append((str(pdf_file), text, ground_truth))
    
    logger.info(f"✓ Collected {len(samples)} PDF samples")
    return samples


def evaluate():
    """Run evaluation on all PDFs and save results to CSV."""
    
    # Collect samples
    samples = collect_pdf_samples()
    
    if not samples:
        logger.error("No PDF samples found!")
        return
    
    results = []
    correct_predictions = 0
    per_class_correct = {label: 0 for label in LABELS}
    per_class_total = {label: 0 for label in LABELS}
    
    logger.info(f"\nRunning inference on {len(samples)} samples...")
    
    # Process in batches
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc="Evaluating"):
        batch_samples = samples[i : i + BATCH_SIZE]
        texts = [text for _, text, _ in batch_samples]
        
        predictions_ids, confidences = predict_batch(texts)
        
        for j, (pdf_path, _, ground_truth) in enumerate(batch_samples):
            pred_id = predictions_ids[j]
            pred_label = ID2LABEL[pred_id]
            confidence = confidences[j]
            
            is_correct = pred_label == ground_truth
            if is_correct:
                correct_predictions += 1
            
            per_class_total[ground_truth] += 1
            if is_correct:
                per_class_correct[ground_truth] += 1
            
            # Extract filename
            filename = Path(pdf_path).name
            
            results.append({
                "filename": filename,
                "ground_truth": ground_truth,
                "prediction": pred_label,
                "confidence": f"{confidence:.4f}",
                "correct": "Yes" if is_correct else "No",
            })
    
    # Calculate metrics
    overall_accuracy = correct_predictions / len(samples) if samples else 0
    
    logger.info("\n" + "=" * 60)
    logger.info("LEGALBERT EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Samples: {len(samples)}")
    logger.info(f"Correct Predictions: {correct_predictions}")
    logger.info(f"Overall Accuracy: {overall_accuracy:.2%}")
    logger.info("\nPer-Class Accuracy:")
    
    for label in LABELS:
        total = per_class_total[label]
        correct = per_class_correct[label]
        accuracy = correct / total if total > 0 else 0
        logger.info(f"  {label}: {correct}/{total} ({accuracy:.2%})")
    
    # Save to CSV
    OUTPUTS_DIR.mkdir(exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "ground_truth", "prediction", "confidence", "correct"]
        )
        writer.writeheader()
        writer.writerows(results)
    
    logger.info(f"\n✓ Results saved to {OUTPUT_CSV}")
    
    # Summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Model: LegalBERT (nlpaueb/legal-bert-base-uncased)")
    logger.info(f"Samples Evaluated: {len(samples)}")
    logger.info(f"Hit Rate: {overall_accuracy:.2%}")
    

if __name__ == "__main__":
    evaluate()
