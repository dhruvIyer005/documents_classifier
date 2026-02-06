"""
app.py - Flask web application for multi-model document classification
Upload N PDFs → Run 3 models → Display comparison results
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import json
import csv
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from multi_classifier import create_multi_classifier
from config import LABELS, OUTPUT_DIR

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# Load multi-model classifier
classifier = None


def init_classifier():
    """Initialize multi-model classifier on startup"""
    global classifier
    try:
        classifier = create_multi_classifier()
        print("[✓] Multi-Model Classifier loaded (Legal-BERT, DistilBERT, SciBERT)")
        return True
    except Exception as e:
        print(f"[✗] Failed to load classifier: {e}")
        return False


@app.route("/")
def index():
    """Home page with upload form"""
    return render_template("index.html", labels=LABELS)


@app.route("/api/classify", methods=["POST"])
def classify():
    """Classify uploaded PDFs with all 3 models"""
    
    if classifier is None:
        return jsonify({"success": False, "error": "Model not loaded"}), 500
    
    # Get uploaded files
    files = request.files.getlist("files")
    
    if not files:
        return jsonify({"success": False, "error": "No files uploaded"}), 400
    
    results = []
    
    # Save and process each file
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        
        try:
            # Save temp file
            temp_path = OUTPUT_DIR / file.filename
            file.save(str(temp_path))
            
            # Get batch results from multi-model classifier
            batch_results = classifier.batch_predict_all([str(temp_path)])
            
            if batch_results and "predictions" in batch_results[0]:
                result_item = batch_results[0]
                results.append({
                    "filename": file.filename,
                    "true_label": result_item.get("true_label", "Unknown"),
                    "models": result_item["predictions"]
                })
            
            # Clean up
            temp_path.unlink()
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return jsonify({
        "success": True,
        "count": len(results),
        "results": results
    })


@app.route("/api/export", methods=["POST"])
def export():
    """Export results as CSV"""
    data = request.json.get("results", [])
    
    if not data:
        return jsonify({"success": False, "error": "No data to export"}), 400
    
    # Create CSV with multi-model results
    output = StringIO()
    fieldnames = ["Filename", "True Label", "Legal-BERT", "Legal-BERT Conf", 
                  "DistilBERT", "DistilBERT Conf", "SciBERT", "SciBERT Conf"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    
    for item in data:
        models = item.get("models", {})
        writer.writerow({
            "Filename": item.get("filename"),
            "True Label": item.get("true_label"),
            "Legal-BERT": models.get("Legal-BERT", {}).get("label", "N/A"),
            "Legal-BERT Conf": models.get("Legal-BERT", {}).get("confidence", "N/A"),
            "DistilBERT": models.get("DistilBERT", {}).get("label", "N/A"),
            "DistilBERT Conf": models.get("DistilBERT", {}).get("confidence", "N/A"),
            "SciBERT": models.get("SciBERT", {}).get("label", "N/A"),
            "SciBERT Conf": models.get("SciBERT", {}).get("confidence", "N/A"),
        })
    
    csv_content = output.getvalue()
    
    return {
        "success": True,
        "csv": csv_content
    }


@app.route("/api/labels", methods=["GET"])
def get_labels():
    """Get available labels"""
    return jsonify({"labels": LABELS})


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    print("="*60)
    print("Multi-Model Document Classifier - Flask App")
    print("="*60)
    
    # Initialize classifier
    if not init_classifier():
        print("Warning: Classifier not available")
    
    print(f"Starting Flask app...")
    app.run(debug=False, host="127.0.0.1", port=5000)
