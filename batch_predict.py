import os
import glob
import csv
import time
from src.pdf_processor import extract_text_from_pdf
from src.multi_classifier import EnsembleClassifier # Assuming the ensemble class is available, or we can use the main predict function.
# If you have a primary wrapper, import it here. 
# For example: from src.predict import predict_document

def run_batch_evaluation(test_dir="data/test_pdfs", output_csv="outputs/50_pdfs_results.csv"):
    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}. Please create it and add your PDFs.")
        return

    # Collect all PDFs from the test directory (supports subfolders)
    pdf_files = glob.glob(os.path.join(test_dir, "**", "*.pdf"), recursive=True)
    
    if not pdf_files:
        print(f"No PDFs found in {test_dir}.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting evaluation...\n")
    
    # Initialize the classifier (loading models from disk)
    print("Loading models (DeBERTa & LegalBERT)...")
    classifier = EnsembleClassifier() # Adjust this to your exact prediction class/function
    print("Models loaded!\n")

    results = []
    hits = 0
    
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        # Assume true category is either the parent folder name or the prefix of the filename
        # e.g., test_pdfs/acm/doc1.pdf -> true_category = 'acm'
        parent_folder = os.path.basename(os.path.dirname(pdf_path))
        true_category = parent_folder.lower()
        
        print(f"[{i}/{len(pdf_files)}] Processing: {filename}...")
        
        try:
            start_time = time.time()
            text = extract_text_from_pdf(pdf_path)
            
            # Run the Soft-Vote prediction
            prediction = classifier.predict(text) 
            predicted_category = prediction['label'].lower()
            confidence = prediction.get('confidence', 0.0)
            
            is_hit = (predicted_category == true_category)
            if is_hit:
                hits += 1
                
            results.append({
                "Filename": filename,
                "True Category": true_category,
                "Predicted Category": predicted_category,
                "Confidence": f"{confidence:.2%}",
                "Hit/Miss": "Hit" if is_hit else "Miss",
                "Time_Taken_sec": round(time.time() - start_time, 2)
            })
            
        except Exception as e:
            print(f"  -> Error processing {filename}: {str(e)}")
            results.append({
                "Filename": filename,
                "True Category": true_category,
                "Predicted Category": "ERROR",
                "Confidence": "0.00%",
                "Hit/Miss": "Error",
                "Time_Taken_sec": 0
            })

    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Filename", "True Category", "Predicted Category", "Confidence", "Hit/Miss", "Time_Taken_sec"])
        writer.writeheader()
        writer.writerows(results)

    # Print Summary
    accuracy = hits / len(pdf_files)
    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE")
    print(f"Total PDFs Processed: {len(pdf_files)}")
    print(f"Total Hits: {hits}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Results saved to: {output_csv}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # You can change the directory path here
    run_batch_evaluation(test_dir="data/test_pdfs", output_csv="outputs/batch_50_results.csv")
