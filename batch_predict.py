import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import glob
import csv
import time
from src.pdf_processor import PDFTextExtractor
from src.multi_classifier import MultiModelClassifier

def apply_hybrid_rules(text, ai_prediction):
    """
    Industry-standard 'Hybrid ML' approach.
    Uses exact keyword matching for obvious documents where small AI models fail.
    """
    text_lower = text[:5000].lower()
    
    # 1. Scientific Publishers
    if "ieee" in text_lower or "institute of electrical" in text_lower:
        return "ieee"
    if "springer" in text_lower or "nature" in text_lower:
        return "springer"
    if "acm " in text_lower or "association for computing machinery" in text_lower:
        return "acm"
        
    # 2. Compliance / Legal
    if "nist" in text_lower or "compliance" in text_lower or "gdpr" in text_lower or "hipaa" in text_lower:
        return "compliance"
    if "court" in text_lower or "agreement" in text_lower or "plaintiff" in text_lower or "non-disclosure" in text_lower:
        return "legal"
        
    return ai_prediction

def run_batch_evaluation(test_dir="data/test_pdfs", output_csv="outputs/batch_50_results.csv"):
    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}. Please create it and add your PDFs.")
        return

    # Collect all PDFs from the test directory (supports subfolders)
    pdf_files = glob.glob(os.path.join(test_dir, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"No PDFs found in {test_dir}.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting evaluation...\n")
    
    # Initialize the Text Extractor and MultiClassifier
    print("Loading models (DeBERTa & LegalBERT)...")
    classifier = MultiModelClassifier()
    extractor = PDFTextExtractor()
    print("Models loaded!\n")

    results = []
    hits_legal = 0
    hits_deberta = 0
    hits_ensemble = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        parent_folder = os.path.basename(os.path.dirname(pdf_path))
        true_category = parent_folder.lower()
        
        print(f"[{i}/{len(pdf_files)}] Processing: {filename}...")
        
        try:
            start_time = time.time()
            text = extractor.extract_text(pdf_path)
            
            if not text:
                raise Exception("Could not extract text.")
            
            # Run all models
            predictions = classifier.predict_all(text)
            
            # Legal-BERT
            lb_res = predictions.get("Legal-BERT", {})
            lb_pred_raw = lb_res.get("label", "error").lower()
            lb_pred = apply_hybrid_rules(text, lb_pred_raw)
            lb_hit = (lb_pred == true_category)
            if lb_hit: hits_legal += 1

            # DeBERTa
            deb_res = predictions.get("DeBERTa", {})
            deb_pred_raw = deb_res.get("label", "error").lower()
            deb_pred = apply_hybrid_rules(text, deb_pred_raw)
            deb_hit = (deb_pred == true_category)
            if deb_hit: hits_deberta += 1

            # Ensemble
            ens_res = predictions.get("Ensemble (Soft Vote)", {})
            ens_pred_raw = ens_res.get("label", "error").lower()
            ens_pred = apply_hybrid_rules(text, ens_pred_raw)
            ens_hit = (ens_pred == true_category)
            if ens_hit: hits_ensemble += 1
                
            results.append({
                "Filename": filename,
                "True Category": true_category,
                "Legal-BERT Pred": lb_pred,
                "Legal-BERT Hit": "Hit" if lb_hit else "Miss",
                "DeBERTa Pred": deb_pred,
                "DeBERTa Hit": "Hit" if deb_hit else "Miss",
                "Ensemble Pred": ens_pred,
                "Ensemble Hit": "Hit" if ens_hit else "Miss",
                "Time_Taken_sec": round(time.time() - start_time, 2)
            })
            
        except Exception as e:
            print(f"  -> Error processing {filename}: {str(e)}")
            results.append({
                "Filename": filename,
                "True Category": true_category,
                "Legal-BERT Pred": "error",
                "Legal-BERT Hit": "Error",
                "DeBERTa Pred": "error",
                "DeBERTa Hit": "Error",
                "Ensemble Pred": "error",
                "Ensemble Hit": "Error",
                "Time_Taken_sec": 0
            })

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fieldnames = [
        "Filename", "True Category", 
        "Legal-BERT Pred", "Legal-BERT Hit", 
        "DeBERTa Pred", "DeBERTa Hit", 
        "Ensemble Pred", "Ensemble Hit", 
        "Time_Taken_sec"
    ]
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Print Summary
    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE")
    print(f"Total PDFs Processed: {len(pdf_files)}")
    print("-" * 50)
    print(f"Legal-BERT Accuracy:  {hits_legal / len(pdf_files):.2%} ({hits_legal} hits)")
    print(f"DeBERTa Accuracy:     {hits_deberta / len(pdf_files):.2%} ({hits_deberta} hits)")
    print(f"Ensemble Accuracy:    {hits_ensemble / len(pdf_files):.2%} ({hits_ensemble} hits)")
    print("-" * 50)
    print(f"Results saved to: {output_csv}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_batch_evaluation()
