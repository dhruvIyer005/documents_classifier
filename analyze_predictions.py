"""
analyze_predictions.py - Detailed analysis of individual model predictions

This utility script compares how DeBERTa and LegalBERT predict on specific PDFs,
useful for understanding model disagreements and failure cases.
"""

import csv
import logging
from pathlib import Path
from typing import Dict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"

DEBERTA_CSV = OUTPUTS_DIR / "deberta_evaluation.csv"
LEGALBERT_CSV = OUTPUTS_DIR / "legalbert_evaluation.csv"

LABELS = ["acm", "compliance", "ieee", "legal", "springer"]


def load_predictions(csv_path: Path) -> Dict[str, Dict]:
    """Load predictions from CSV into a dictionary keyed by filename."""
    predictions = {}
    
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        return predictions
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            predictions[filename] = {
                "ground_truth": row["ground_truth"],
                "prediction": row["prediction"],
                "confidence": float(row["confidence"]),
                "correct": row["correct"] == "Yes",
            }
    
    return predictions


def analyze_disagreements(deberta_preds: Dict, legalbert_preds: Dict):
    """Find and analyze cases where the two models disagree."""
    
    logger.info("\n" + "="*70)
    logger.info("MODEL DISAGREEMENT ANALYSIS")
    logger.info("="*70)
    
    disagreements = []
    agreements = 0
    
    common_files = set(deberta_preds.keys()) & set(legalbert_preds.keys())
    
    for filename in sorted(common_files):
        d_pred = deberta_preds[filename]
        l_pred = legalbert_preds[filename]
        
        if d_pred["prediction"] == l_pred["prediction"]:
            agreements += 1
        else:
            disagreements.append((filename, d_pred, l_pred))
    
    total = len(common_files)
    agreement_rate = agreements / total if total > 0 else 0
    
    logger.info(f"\nTotal files evaluated: {total}")
    logger.info(f"Agreement rate: {agreements}/{total} ({agreement_rate:.2%})")
    logger.info(f"Disagreements: {len(disagreements)}")
    
    if disagreements:
        logger.info("\n" + "-"*70)
        logger.info("DISAGREEMENT DETAILS")
        logger.info("-"*70)
        
        for filename, d_pred, l_pred in disagreements[:20]:  # Show top 20
            ground_truth = d_pred["ground_truth"]
            d_pred_label = d_pred["prediction"]
            l_pred_label = l_pred["prediction"]
            d_conf = d_pred["confidence"]
            l_conf = l_pred["confidence"]
            
            d_correct = "✓" if d_pred["correct"] else "✗"
            l_correct = "✓" if l_pred["correct"] else "✗"
            
            logger.info(f"\nFile: {filename}")
            logger.info(f"  Ground Truth:        {ground_truth}")
            logger.info(f"  DeBERTa:   {d_pred_label:<12} (conf: {d_conf:.4f}) {d_correct}")
            logger.info(f"  LegalBERT: {l_pred_label:<12} (conf: {l_conf:.4f}) {l_correct}")
            
            # Check if either correctly predicted
            if d_pred["correct"] and not l_pred["correct"]:
                logger.info("  → DeBERTa wins")
            elif l_pred["correct"] and not d_pred["correct"]:
                logger.info("  → LegalBERT wins")
            elif d_pred["correct"] and l_pred["correct"]:
                logger.info("  → Both correct (disagreed on wrong choice)")
            else:
                logger.info("  → Both incorrect")


def analyze_per_category(deberta_preds: Dict, legalbert_preds: Dict):
    """Compare model performance per category."""
    
    logger.info("\n" + "="*70)
    logger.info("PER-CATEGORY PERFORMANCE COMPARISON")
    logger.info("="*70)
    
    per_category = {label: {"deberta": [], "legalbert": []} for label in LABELS}
    
    common_files = set(deberta_preds.keys()) & set(legalbert_preds.keys())
    
    for filename in common_files:
        ground_truth = deberta_preds[filename]["ground_truth"]
        
        if ground_truth in per_category:
            per_category[ground_truth]["deberta"].append(deberta_preds[filename])
            per_category[ground_truth]["legalbert"].append(legalbert_preds[filename])
    
    logger.info(f"\n{'Category':<15} {'DeBERTa':<25} {'LegalBERT':<25} {'Winner':<10}")
    logger.info("-" * 75)
    
    for label in LABELS:
        d_results = per_category[label]["deberta"]
        l_results = per_category[label]["legalbert"]
        
        if not d_results:
            logger.info(f"{label:<15} No samples")
            continue
        
        d_correct = sum(1 for r in d_results if r["correct"])
        l_correct = sum(1 for r in l_results if r["correct"])
        total = len(d_results)
        
        d_acc = d_correct / total if total > 0 else 0
        l_acc = l_correct / total if total > 0 else 0
        
        d_str = f"{d_correct}/{total} ({d_acc:.2%})"
        l_str = f"{l_correct}/{total} ({l_acc:.2%})"
        
        if d_acc > l_acc:
            winner = "DeBERTa"
        elif l_acc > d_acc:
            winner = "LegalBERT"
        else:
            winner = "Tie"
        
        logger.info(f"{label:<15} {d_str:<25} {l_str:<25} {winner:<10}")


def analyze_confidence(deberta_preds: Dict, legalbert_preds: Dict):
    """Analyze prediction confidence by correctness."""
    
    logger.info("\n" + "="*70)
    logger.info("CONFIDENCE ANALYSIS")
    logger.info("="*70)
    
    deberta_correct_conf = []
    deberta_incorrect_conf = []
    legalbert_correct_conf = []
    legalbert_incorrect_conf = []
    
    common_files = set(deberta_preds.keys()) & set(legalbert_preds.keys())
    
    for filename in common_files:
        d_pred = deberta_preds[filename]
        l_pred = legalbert_preds[filename]
        
        if d_pred["correct"]:
            deberta_correct_conf.append(d_pred["confidence"])
        else:
            deberta_incorrect_conf.append(d_pred["confidence"])
        
        if l_pred["correct"]:
            legalbert_correct_conf.append(l_pred["confidence"])
        else:
            legalbert_incorrect_conf.append(l_pred["confidence"])
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0
    
    logger.info("\nDeBERTa Confidence:")
    logger.info(f"  Correct predictions:   {avg(deberta_correct_conf):.4f} (avg)")
    logger.info(f"  Incorrect predictions: {avg(deberta_incorrect_conf):.4f} (avg)")
    
    logger.info("\nLegalBERT Confidence:")
    logger.info(f"  Correct predictions:   {avg(legalbert_correct_conf):.4f} (avg)")
    logger.info(f"  Incorrect predictions: {avg(legalbert_incorrect_conf):.4f} (avg)")
    
    logger.info("\nCalibration Notes:")
    d_gap = avg(deberta_correct_conf) - avg(deberta_incorrect_conf)
    l_gap = avg(legalbert_correct_conf) - avg(legalbert_incorrect_conf)
    
    logger.info(f"  DeBERTa confidence gap: {d_gap:+.4f} (higher = better calibrated)")
    logger.info(f"  LegalBERT confidence gap: {l_gap:+.4f}")
    
    if d_gap > l_gap:
        logger.info("  → DeBERTa is better calibrated (confidence reflects accuracy)")
    elif l_gap > d_gap:
        logger.info("  → LegalBERT is better calibrated")
    else:
        logger.info("  → Both models have similar calibration")


def find_hard_cases(deberta_preds: Dict, legalbert_preds: Dict):
    """Find documents that both models misclassify."""
    
    logger.info("\n" + "="*70)
    logger.info("HARD CASES (Both Models Fail)")
    logger.info("="*70)
    
    hard_cases = []
    common_files = set(deberta_preds.keys()) & set(legalbert_preds.keys())
    
    for filename in common_files:
        d_pred = deberta_preds[filename]
        l_pred = legalbert_preds[filename]
        
        if not d_pred["correct"] and not l_pred["correct"]:
            hard_cases.append((filename, d_pred, l_pred))
    
    logger.info(f"\nFound {len(hard_cases)} hard cases ({len(hard_cases)/len(common_files):.2%})")
    
    if hard_cases:
        logger.info("\n" + "-"*70)
        
        for filename, d_pred, l_pred in hard_cases[:10]:  # Show top 10
            logger.info(f"\nFile: {filename}")
            logger.info(f"  Ground Truth: {d_pred['ground_truth']}")
            logger.info(f"  DeBERTa:   {d_pred['prediction']} (conf: {d_pred['confidence']:.4f})")
            logger.info(f"  LegalBERT: {l_pred['prediction']} (conf: {l_pred['confidence']:.4f})")
            logger.info(f"  → Both models agree on {d_pred['prediction']} but it's wrong")


def main():
    """Run all analyses."""
    
    logger.info("\n" + "="*70)
    logger.info("LOADING PREDICTIONS")
    logger.info("="*70)
    
    deberta_preds = load_predictions(DEBERTA_CSV)
    legalbert_preds = load_predictions(LEGALBERT_CSV)
    
    if not deberta_preds or not legalbert_preds:
        logger.error("Could not load prediction files. Run compare_models.py first.")
        exit(1)
    
    logger.info(f"✓ Loaded DeBERTa predictions: {len(deberta_preds)} files")
    logger.info(f"✓ Loaded LegalBERT predictions: {len(legalbert_preds)} files")
    
    # Run all analyses
    analyze_disagreements(deberta_preds, legalbert_preds)
    analyze_per_category(deberta_preds, legalbert_preds)
    analyze_confidence(deberta_preds, legalbert_preds)
    find_hard_cases(deberta_preds, legalbert_preds)
    
    logger.info("\n" + "="*70)
    logger.info("✓ Analysis complete")
    logger.info("="*70)


if __name__ == "__main__":
    main()
