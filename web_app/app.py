# web_app/app.py
"""Flask API for batch document classification with SciBERT"""

import os
import sys
from flask import Flask, render_template, request, jsonify, Response
import torch
import tempfile
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
from io import StringIO

# Add the src directory to path to import your modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your updated modules
try:
    from multi_classifier import MultiModelClassifier
    from config import config
    from pdf_processor import PDFTextExtractor
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ML modules: {e}")
    import traceback
    traceback.print_exc()
    ML_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size (batch processing)

class WebDocumentPredictor:
    def __init__(self):
        self.classifier = None
        self.pdf_extractor = None
        self.load_model()
    
    def load_model(self):
        """Load the trained models"""
        try:
            # Initialize PDF extractor first
            self.pdf_extractor = PDFTextExtractor()
            print("PDF extractor initialized")
            
            # Load multi-model classifier
            try:
                self.classifier = MultiModelClassifier()
                print("[OK] Multi-model classifier loaded successfully")
            except Exception as e:
                print(f"[ERROR] Error loading classifier: {e}")
                import traceback
                traceback.print_exc()
                self.classifier = None
            
        except Exception as e:
            print(f"[ERROR] Error in load_model: {e}")
            import traceback
            traceback.print_exc()
            self.classifier = None
            self.pdf_extractor = None
        
    def extract_text_from_pdf(self, file_stream):
        """Extract text from PDF file"""
        import tempfile
        import os
        import time

        if self.pdf_extractor is None:
            raise Exception("PDF extractor not initialized - model failed to load")

        temp_path = None
        try:
            # Create temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                temp_path = tmp.name
                file_stream.save(temp_path)
            
            # Small delay to ensure file is saved
            time.sleep(0.05)
            
            # Extract text
            text = self.pdf_extractor.extract_text(temp_path)
            
            if text is None:
                raise Exception("PDF extraction returned None - could not extract text from PDF")
            
            # Try to delete temp file
            for _ in range(3):  # Try 3 times
                try:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
                        break
                except:
                    time.sleep(0.1)  # Wait and retry
            
            return text
            
        except Exception as e:
            # Clean up on error
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            raise Exception(f"PDF extraction error: {str(e)}")
    
    def extract_text_from_pdf_file(self, filepath):
        """Extract text from a PDF file path"""
        try:
            if self.pdf_extractor is None:
                raise Exception("PDF extractor not initialized")
            return self.pdf_extractor.extract_text(filepath)
        except Exception as e:
            raise Exception(f"PDF file extraction error: {str(e)}")
    
    def process_uploaded_file(self, file):
        """Process uploaded file and extract text"""
        filename = file.filename.lower()
        
        if filename.endswith('.pdf'):
            text = self.extract_text_from_pdf(file)
        elif filename.endswith('.txt'):
            text = file.stream.read().decode('utf-8', errors='ignore')
        else:
            raise Exception("Unsupported file format. Please upload PDF or TXT files.")
        
        if not text or len(text.strip()) < 10:
            raise Exception("File appears to be empty or contains too little text.")
        
        return text
    
    def predict(self, text):
        """Predict document class using multi-model classifier"""
        if not text or len(text.strip()) < 10:
            return self._create_error_result("Text too short")
        
        try:
            # Use multi-model classifier
            if self.classifier is None:
                return self._create_error_result("Model not loaded")
            
            # Get predictions from all models
            results = self.classifier.predict_all(text)
            
            # Find the best prediction (highest confidence)
            best_label = None
            best_confidence = 0
            all_predictions = []
            
            for model_name, prediction in results.items():
                label = prediction['label']
                confidence = prediction['confidence']
                all_predictions.append({
                    'model': model_name,
                    'label': label,
                    'confidence': confidence
                })
                
                # Track best prediction
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label
            
            # Sort by confidence for top predictions
            sorted_predictions = sorted(all_predictions, key=lambda x: x['confidence'], reverse=True)
            
            # Format all models predictions for side-by-side display
            all_models_details = [
                {
                    'model': p['model'],
                    'label': p['label'],
                    'confidence': round(p['confidence'] * 100, 1),
                    'is_best': p == sorted_predictions[0]
                }
                for p in sorted_predictions
            ]
            
            return {
                'success': True,
                'prediction': best_label,
                'confidence': round(best_confidence * 100, 1),
                'all_models': all_models_details,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return self._create_error_result(f"Prediction error: {str(e)}")
    
    def _generate_analysis(self, predicted_label):
        """Generate analysis based on predicted label"""
        analysis = []
        
        # Label-specific analysis
        if predicted_label == 'ieee':
            analysis.append("IEEE documents typically include technical details, experimental results, and IEEE formatting.")
            analysis.append("Check for IEEE-specific sections: Abstract, Introduction, Methodology, Results, Conclusion, References.")
        
        elif predicted_label == 'springer':
            analysis.append("Springer documents often follow academic paper formats with clear structure.")
            analysis.append("Look for Springer-specific formatting and citation styles.")
        
        elif predicted_label == 'acm':
            analysis.append("ACM documents follow specific formatting guidelines for conferences and journals.")
            analysis.append("Check for ACM sections like Abstract, CCS Concepts, Keywords, and ACM reference format.")
        
        elif predicted_label == 'compliance':
            analysis.append("Compliance documents should clearly state policies, procedures, and regulatory requirements.")
            analysis.append("Ensure all necessary compliance sections are present and clearly defined.")
        
        elif predicted_label == 'legal':
            analysis.append("Legal documents require precise language, defined terms, and clear legal structure.")
            analysis.append("Check for legal sections like Parties, Terms, Conditions, Signatures, and Jurisdiction.")
        
        else:
            analysis.append("Document classification complete. Review content for accuracy.")
        
        return analysis
    
    def _create_error_result(self, error_msg):
        return {
            'success': False,
            'error': error_msg,
            'template': 'unknown',
            'template_confidence': 0,
            'alternative_templates': [],
            'scores': {},
            'overall_score': 0.0,
            'analysis': [error_msg],
            'word_count': 0,
            'char_count': 0,
            'all_probabilities': {}
        }

# Initialize predictor
predictor = WebDocumentPredictor()

# REST OF THE FILE REMAINS THE SAME...
# [Keep all the Flask routes and other code exactly as you have it]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_document():
    try:
        # Check if it's file upload or text input
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'success': False,
                    'error': 'No file selected'
                })
            
            # Process uploaded file
            try:
                text = predictor.process_uploaded_file(file)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
                
        else:
            # Text input
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'No data provided'
                })
            text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            })
        
        if len(text) < 10:
            return jsonify({
                'success': False,
                'error': 'Text too short (minimum 10 characters required)'
            })
        
        # Get prediction
        result = predictor.predict(text)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/analyze_multiple', methods=['POST'])
def analyze_multiple():
    """Analyze multiple documents"""
    try:
        if 'files' not in request.files:
            return jsonify({'success': False, 'error': 'No files provided', 'results': []})
        
        files = request.files.getlist('files')
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            filename = file.filename.lower()
            
            # Only accept PDF and TXT
            if not (filename.endswith('.pdf') or filename.endswith('.txt')):
                results.append({
                    'filename': file.filename,
                    'status': 'failed',
                    'error': 'Unsupported file format (PDF/TXT only)'
                })
                continue
            
            try:
                # Extract text from file
                text = predictor.process_uploaded_file(file)
                
                if not text or len(text.strip()) < 10:
                    results.append({
                        'filename': file.filename,
                        'status': 'failed',
                        'error': 'File too small or empty'
                    })
                    continue
                
                # Predict
                prediction = predictor.predict(text)
                
                if prediction['success']:
                    results.append({
                        'filename': file.filename,
                        'status': 'success',
                        'prediction': prediction.get('prediction'),
                        'confidence': prediction.get('confidence'),
                        'all_models': prediction.get('all_models', [])
                    })
                else:
                    results.append({
                        'filename': file.filename,
                        'status': 'failed',
                        'error': prediction.get('error', 'Prediction failed')
                    })
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                results.append({
                    'filename': file.filename,
                    'status': 'failed',
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'results': []
        })

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': predictor.classifier is not None,
        'labels': config.LABELS if ML_AVAILABLE else []
    })

@app.route('/debug')
def debug_info():
    """Debug information about model state"""
    model_path = Path(__file__).parent.parent / "models"
    
    debug_info = {
        'model_path': str(model_path),
        'classifier_loaded': predictor.classifier is not None,
        'available_labels': config.LABELS if ML_AVAILABLE else [],
        'models_in_dir': []
    }
    
    if model_path.exists():
        try:
            debug_info['models_in_dir'] = os.listdir(str(model_path))
        except:
            debug_info['models_in_dir'] = []
    
    return jsonify(debug_info)

class BulkPDFProcessor:
    def __init__(self, predictor):
        self.predictor = predictor

    def process_zip_file(self, zip_file):
        """Process ZIP containing multiple PDFs"""
        results = []
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        filepath = os.path.join(root, file)
                        try:
                            text = self.predictor.extract_text_from_pdf_file(filepath)
                            if not text:
                                results.append({
                                    'filename': file,
                                    'status': 'failed',
                                    'error': 'No text extracted from PDF'
                                })
                                continue
                            
                            prediction = self.predictor.predict(text)
                            
                            if prediction['success']:
                                results.append({
                                    'filename': file,
                                    'status': 'success',
                                    'prediction': prediction.get('prediction'),
                                    'confidence': prediction.get('confidence'),
                                    'all_models': prediction.get('all_models', [])
                                })
                            else:
                                results.append({
                                    'filename': file,
                                    'status': 'failed',
                                    'error': prediction.get('error', 'Unknown error')
                                })
                        except Exception as e:
                            results.append({
                                'filename': file,
                                'status': 'failed',
                                'error': str(e)
                            })
        return results

    def process_multiple_files(self, file_list):
        """Process multiple uploaded files concurrently"""
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for file in file_list:
                if file.filename.lower().endswith('.pdf'):
                    futures.append(executor.submit(self._process_single_file, file))
            
            for future in futures:
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    results.append({
                        'filename': 'unknown',
                        'status': 'failed',
                        'error': str(e)
                    })
        return results

    def _process_single_file(self, file_storage):
        """Process single PDF file storage"""
        try:
            text = self.predictor.extract_text_from_pdf(file_storage)
            if not text:
                return {
                    'filename': file_storage.filename,
                    'status': 'failed',
                    'error': 'No text extracted'
                }
            
            prediction = self.predictor.predict(text)
            
            if prediction['success']:
                return {
                    'filename': file_storage.filename,
                    'status': 'success',
                    'prediction': prediction.get('prediction'),
                    'confidence': prediction.get('confidence'),
                    'all_models': prediction.get('all_models', [])
                }
            else:
                return {
                    'filename': file_storage.filename,
                    'status': 'failed',
                    'error': prediction.get('error', 'Prediction failed')
                }
        except Exception as e:
            return {
                'filename': file_storage.filename,
                'status': 'failed',
                'error': str(e)
            }

@app.route('/bulk_upload', methods=['POST'])
def bulk_upload():
    try:
        if 'files[]' not in request.files:
            return jsonify({'success': False, 'error': 'No files uploaded'})
        
        files = request.files.getlist('files[]')
        if len(files) == 0:
            return jsonify({'success': False, 'error': 'No files selected'})
        
        if len(files) > 50:
            return jsonify({'success': False, 'error': f'Maximum 50 files allowed. You uploaded {len(files)}.'})

        if predictor is None or predictor.classifier is None:
            return jsonify({'success': False, 'error': 'Classifier not initialized on server'})

        bulk_processor = BulkPDFProcessor(predictor)
        results = bulk_processor.process_multiple_files(files)

        # Create summary
        summary = {
            'total_files': len(files),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed'])
        }

        return jsonify({
            'success': True, 
            'results': results, 
            'summary': summary
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_results', methods=['POST'])
def download_results():
    try:
        data = request.get_json()
        if not data or 'results' not in data:
            return jsonify({'success': False, 'error': 'No results data provided'})
        
        results = data['results']
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Filename', 'Status', 'Document Type', 'Confidence %', 'Alternative Predictions'])
        
        # Write data
        for result in results:
            if result['status'] == 'success':
                models_str = ' | '.join([f"{m['model']}: {m['label']} ({m['confidence']}%)" for m in result.get('all_models', [])])
            else:
                models_str = result.get('error', 'Error')
            
            writer.writerow([
                result.get('filename', ''),
                result.get('status', ''),
                result.get('prediction', ''),
                result.get('confidence', ''),
                models_str
            ])
        
        output.seek(0)
        
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=document_classification_results.csv"}
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("="*60)
    print("DOCUMENT CLASSIFIER WEB APPLICATION")
    print("="*60)
    
    if ML_AVAILABLE:
        print(f"[OK] ML modules loaded")
        print(f"[OK] Target labels: {config.LABELS}")
    else:
        print("[FAIL] ML modules not available")
    
    print(f"[OK] Classifier initialized: {predictor.classifier is not None}")
    print(f"[OK] PDF extractor initialized: {predictor.pdf_extractor is not None}")
    print("[OK] Supported file formats: PDF, TXT")
    print("[OK] Bulk processing: Enabled (up to 50 files)")
    print("="*60)
    
    try:
        print("[INFO] Starting Flask server...")
        app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
    except OSError as e:
        print(f"[ERROR] Port 5000 in use: {e}")
        print("[INFO] Trying port 8000...")
        try:
            app.run(debug=False, host='0.0.0.0', port=8000, use_reloader=False, threaded=True)
        except Exception as e2:
            print(f"[ERROR] Failed to start Flask: {e2}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"[ERROR] Failed to start Flask: {e}")
        import traceback
        traceback.print_exc()