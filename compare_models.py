"""
compare_models.py - Run both DeBERTa and LegalBERT evaluations and generate comparison report

This script runs both evaluation scripts independently and creates a comprehensive 
comparison report showing side-by-side metrics.
"""

import subprocess
import csv
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / "outputs"

DEBERTA_CSV = OUTPUTS_DIR / "deberta_evaluation.csv"
LEGALBERT_CSV = OUTPUTS_DIR / "legalbert_evaluation.csv"
COMPARISON_REPORT = OUTPUTS_DIR / "model_comparison.txt"

LABELS = ["acm", "compliance", "ieee", "legal", "springer"]


def run_evaluation(script_name: str) -> bool:
    """Run an evaluation script and return success status."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {script_name}...")
    logger.info('='*60)
    
    try:
        result = subprocess.run(
            ["python", str(BASE_DIR / script_name)],
            cwd=str(BASE_DIR),
            capture_output=False,
            check=True
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")
        return False


def load_csv_results(csv_path: Path) -> tuple:
    """Load CSV results and compute metrics."""
    if not csv_path.exists():
        logger.warning(f"CSV not found: {csv_path}")
        return None, None, None
    
    results = []
    correct = 0
    per_class_correct = {label: 0 for label in LABELS}
    per_class_total = {label: 0 for label in LABELS}
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
            ground_truth = row["ground_truth"]
            is_correct = row["correct"] == "Yes"
            
            per_class_total[ground_truth] += 1
            if is_correct:
                correct += 1
                per_class_correct[ground_truth] += 1
    
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    return results, accuracy, (per_class_correct, per_class_total)


def generate_comparison_report():
    """Generate comprehensive comparison report."""
    
    logger.info("\n" + "="*60)
    logger.info("Loading evaluation results...")
    logger.info("="*60)
    
    deberta_results, deberta_acc, deberta_classes = load_csv_results(DEBERTA_CSV)
    legalbert_results, legalbert_acc, legalbert_classes = load_csv_results(LEGALBERT_CSV)
    
    if deberta_results is None or legalbert_results is None:
        logger.error("Failed to load results from one or both evaluations")
        return False
    
    # Generate report
    report_lines = [
        "=" * 70,
        "MODEL COMPARISON REPORT",
        "=" * 70,
        "",
        f"Evaluation Date: {Path(__file__).stat().st_mtime}",
        f"Total Samples Evaluated: {len(deberta_results)}",
        "",
        "=" * 70,
        "OVERALL ACCURACY COMPARISON",
        "=" * 70,
        f"DeBERTa (microsoft/deberta-base):    {deberta_acc:.2%}",
        f"LegalBERT (nlpaueb/legal-bert):      {legalbert_acc:.2%}",
        f"Difference:                          {abs(deberta_acc - legalbert_acc):+.2%}",
        "",
        "=" * 70,
        "PER-CLASS ACCURACY",
        "=" * 70,
        f"{'Category':<15} {'DeBERTa':<20} {'LegalBERT':<20} {'Difference':<10}",
        "-" * 70,
    ]
    
    deberta_correct, deberta_total = deberta_classes
    legalbert_correct, legalbert_total = legalbert_classes
    
    for label in LABELS:
        d_acc = deberta_correct[label] / deberta_total[label] if deberta_total[label] > 0 else 0
        l_acc = legalbert_correct[label] / legalbert_total[label] if legalbert_total[label] > 0 else 0
        diff = l_acc - d_acc
        
        d_str = f"{deberta_correct[label]}/{deberta_total[label]} ({d_acc:.2%})"
        l_str = f"{legalbert_correct[label]}/{legalbert_total[label]} ({l_acc:.2%})"
        diff_str = f"{diff:+.2%}"
        
        report_lines.append(f"{label:<15} {d_str:<20} {l_str:<20} {diff_str:<10}")
    
    # Confidence metrics
    report_lines.extend([
        "",
        "=" * 70,
        "AVERAGE PREDICTION CONFIDENCE",
        "=" * 70,
    ])
    
    deberta_confidences = [float(r["confidence"]) for r in deberta_results]
    legalbert_confidences = [float(r["confidence"]) for r in legalbert_results]
    
    avg_deberta_conf = sum(deberta_confidences) / len(deberta_confidences) if deberta_confidences else 0
    avg_legalbert_conf = sum(legalbert_confidences) / len(legalbert_confidences) if legalbert_confidences else 0
    
    report_lines.extend([
        f"DeBERTa Average Confidence:   {avg_deberta_conf:.4f}",
        f"LegalBERT Average Confidence: {avg_legalbert_conf:.4f}",
        "",
    ])
    
    # Misclassifications
    deberta_misclass = [r for r in deberta_results if r["correct"] == "No"]
    legalbert_misclass = [r for r in legalbert_results if r["correct"] == "No"]
    
    report_lines.extend([
        "=" * 70,
        "MISCLASSIFICATION SUMMARY",
        "=" * 70,
        f"DeBERTa Errors:   {len(deberta_misclass)} ({len(deberta_misclass)/len(deberta_results):.2%})",
        f"LegalBERT Errors: {len(legalbert_misclass)} ({len(legalbert_misclass)/len(legalbert_results):.2%})",
        "",
        "=" * 70,
        "RECOMMENDATION",
        "=" * 70,
    ])
    
    # Determine better model
    if deberta_acc > legalbert_acc:
        better = "DeBERTa"
        margin = deberta_acc - legalbert_acc
        report_lines.append(f"✓ {better} outperforms with {margin:.2%} higher accuracy")
    elif legalbert_acc > deberta_acc:
        better = "LegalBERT"
        margin = legalbert_acc - deberta_acc
        report_lines.append(f"✓ {better} outperforms with {margin:.2%} higher accuracy")
    else:
        report_lines.append("✓ Both models have equal accuracy - choose based on inference speed")
    
    # Best per-class model
    report_lines.extend([
        "",
        "Best Model per Category:",
    ])
    
    for label in LABELS:
        d_acc = deberta_correct[label] / deberta_total[label] if deberta_total[label] > 0 else 0
        l_acc = legalbert_correct[label] / legalbert_total[label] if legalbert_total[label] > 0 else 0
        best = "DeBERTa" if d_acc >= l_acc else "LegalBERT"
        report_lines.append(f"  {label}: {best} ({max(d_acc, l_acc):.2%})")
    
    report_lines.extend([
        "",
        "=" * 70,
    ])
    
    # Write report
    report_text = "\n".join(report_lines)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    
    with open(COMPARISON_REPORT, "w") as f:
        f.write(report_text)
    
    # Print to console
    print("\n" + report_text)
    logger.info(f"\n✓ Comparison report saved to {COMPARISON_REPORT}")
    
    return True


if __name__ == "__main__":
    # Run both evaluations
    logger.info("Starting model evaluation and comparison...")
    
    deberta_ok = run_evaluation("evaluate_deberta.py")
    legalbert_ok = run_evaluation("evaluate_legalbert.py")
    
    if deberta_ok and legalbert_ok:
        logger.info("\n✓ Both evaluations completed successfully")
        generate_comparison_report()
    else:
        logger.error("One or more evaluations failed")
        exit(1)
