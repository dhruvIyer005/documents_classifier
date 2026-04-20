"""multi_classifier.py - Load 2 models, classify up to 20 PDFs"""

import sys
from pathlib import Path
from typing import List, Dict
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

sys.path.insert(0, str(Path(__file__).parent))
from config import config
from text_processor import TextProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModelClassifier:
    """Load 2 models (Legal-BERT, DeBERTa), classify documents"""
    
    def __init__(self):
        logger.info(f"Loading {len(config.MODELS)} models...")
        self.models = {}
        self.tokenizers = {}
        self.device = config.DEVICE
        
        for display_name, (model_name, folder_name) in config.MODELS.items():
            model_path = config.MODELS_DIR / folder_name
            
            try:
                logger.info(f"  Loading {display_name}...")
                self.tokenizers[display_name] = AutoTokenizer.from_pretrained(model_path)
                self.models[display_name] = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.models[display_name] = self.models[display_name].to(self.device)
                self.models[display_name].eval()
                logger.info("    ✓ OK")
            except Exception as e:
                logger.warning(f"    ✗ Failed: {e}")
        
        if not self.models:
            raise Exception("No models loaded! Run training first.")
        
        self.text_processor = TextProcessor()
        logger.info(f"Loaded {len(self.models)} models")
    
    def predict_single_model(self, text: str, model_name: str) -> tuple:
        """Predict with one model. Returns (label, confidence, probabilities_tensor)"""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        if not text or len(text) < 50:
            return "unknown", 0.0, torch.zeros(config.NUM_LABELS)
        
        # Tokenize (Limit to max_length for the model, usually 512 tokens)
        inputs = tokenizer(
            text,
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Inference
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Predict
        probs = F.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()
        label = config.ID2LABEL[pred_id]
        
        return label, confidence, probs[0]
    
    def predict_all(self, text: str) -> Dict[str, Dict]:
        """Run all models, plus a Soft Voting Ensemble, and return results."""
        results = {}
        all_probs = []

        for model_name in self.models.keys():
            try:
                label, confidence, probs = self.predict_single_model(text, model_name)
                results[model_name] = {
                    "label": label,
                    "confidence": round(confidence, 4),
                }
                all_probs.append(probs)
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                results[model_name] = {"label": "Error", "confidence": 0.0}

        # Add Ensemble (Soft Voting)
        if all_probs:
            try:
                # Average the probabilities across all successful models
                avg_probs = torch.stack(all_probs).mean(dim=0)
                ensemble_pred_id = torch.argmax(avg_probs, dim=-1).item()
                ensemble_confidence = avg_probs[ensemble_pred_id].item()
                ensemble_label = config.ID2LABEL[ensemble_pred_id]

                results["Ensemble (Soft Vote)"] = {
                    "label": ensemble_label,
                    "confidence": round(ensemble_confidence, 4),
                }
            except Exception as e:
                logger.error(f"Error computing ensemble: {e}")

        return results

    def batch_predict_all(self, pdf_paths: List[str]) -> List[Dict]:
        """Classify 1-20 PDFs with all models"""
        
        if len(pdf_paths) > 20:
            logger.warning(f"Limiting to 20 PDFs (requested {len(pdf_paths)})")
            pdf_paths = pdf_paths[:20]
        
        results = []
        total = len(pdf_paths)

        for idx, pdf_path in enumerate(pdf_paths, 1):
            logger.info(f"Processing {idx}/{total}: {Path(pdf_path).name}")
            
            try:
                # Extract text
                text = self.text_processor.extract_text(str(pdf_path))

                if not text:
                    results.append({
                        "filename": Path(pdf_path).name,
                        "error": "Failed to extract text",
                        "status": "error"
                    })
                    continue

                # Run all models
                model_results = self.predict_all(text)

                # Determine true label from folder name
                filename = Path(pdf_path).name
                true_label = None
                for label_name in config.LABELS:
                    if label_name in str(pdf_path).lower():
                        true_label = label_name
                        break

                results.append({
                    "filename": filename,
                    "true_label": true_label,
                    "predictions": model_results,
                    "status": "success"
                })
                logger.info(f"  Classified {filename}")

            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results.append({
                    "filename": Path(pdf_path).name, 
                    "error": str(e),
                    "status": "error"
                })

        logger.info(f"Batch processing complete: {len(results)}/{total} documents")
        return results


def create_multi_classifier() -> MultiModelClassifier:
    """Factory function."""
    return MultiModelClassifier()
