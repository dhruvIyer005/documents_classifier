"""multi_classifier.py - Load and run 3 models (Legal-BERT, DistilBERT, SciBERT) for comparison."""

import torch
import torch.nn.functional as F
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import DEVICE, TEMPERATURE, ID2LABEL, LABELS, NUM_LABELS, USE_FP16
from text_processor import TextProcessor

logger = logging.getLogger(__name__)


class MultiModelClassifier:
    """Load and run 3 models in parallel for comparison."""

    def __init__(self):
        """Initialize all 3 models."""
        self.device = DEVICE
        self.use_fp16 = USE_FP16 and self.device == "cuda"
        self.models = {}
        self.tokenizers = {}
        self.text_processor = None

        # Model configs: (model_name, save_path, display_name)
        self.model_configs = [
            ("nlpaueb/legal-bert-base-uncased", "models/legal_bert", "Legal-BERT"),
            ("distilbert-base-uncased", "models/distilbert", "DistilBERT"),
            ("allenai/scibert_scivocab_uncased", "models/scibert", "SciBERT"),
        ]

        logger.info(f"Using device: {self.device}, fp16: {self.use_fp16}")

        # Load all models
        for model_name, save_path, display_name in self.model_configs:
            try:
                logger.info(f"Loading {display_name}...")
                tokenizer = AutoTokenizer.from_pretrained(save_path)
                model = AutoModelForSequenceClassification.from_pretrained(save_path)
                logger.info(f"✓ Loaded {display_name} from {save_path}")
            except Exception as e:
                logger.warning(f"Could not load from {save_path}: {e}")
                logger.info(f"Loading base {display_name} ({model_name})...")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=NUM_LABELS
                )
                logger.info(f"✓ Loaded base {display_name}")

            model = model.to(self.device)
            if self.use_fp16:
                model = model.half()
            model.eval()

            self.models[display_name] = model
            self.tokenizers[display_name] = tokenizer

        # Initialize text processor (use first tokenizer)
        first_tokenizer = self.tokenizers[list(self.tokenizers.keys())[0]]
        self.text_processor = TextProcessor(first_tokenizer)

    def predict_single_model(
        self, text: str, model_name: str
    ) -> Tuple[str, float]:
        """Predict with a single model."""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]

        # Re-tokenize with this model's tokenizer
        tp = TextProcessor(tokenizer)
        chunks = tp.chunk_text(text)

        if not chunks:
            return "Unknown", 0.0

        all_logits = []
        with torch.no_grad():
            for input_ids, attention_mask in chunks:
                input_ids = input_ids.unsqueeze(0).to(self.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.device)

                if self.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask)
                        logits = outputs.logits
                else:
                    outputs = model(input_ids, attention_mask)
                    logits = outputs.logits

                all_logits.append(logits.cpu().float())

        avg_logits = torch.mean(torch.cat(all_logits, dim=0), dim=0, keepdim=True)
        scaled_logits = avg_logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=-1)

        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_id].item()
        pred_label = ID2LABEL.get(pred_id, "Unknown")

        return pred_label, confidence

    def predict_all(self, text: str) -> Dict[str, Dict]:
        """
        Run all 3 models and return results.

        Returns:
            {
                "Legal-BERT": {"label": "acm", "confidence": 0.92},
                "DistilBERT": {"label": "ieee", "confidence": 0.88},
                "SciBERT": {"label": "acm", "confidence": 0.95}
            }
        """
        results = {}
        for model_name in self.models.keys():
            try:
                label, confidence = self.predict_single_model(text, model_name)
                results[model_name] = {
                    "label": label,
                    "confidence": round(confidence, 4),
                }
            except Exception as e:
                logger.error(f"Error with {model_name}: {e}")
                results[model_name] = {"label": "Error", "confidence": 0.0}

        return results

    def batch_predict_all(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Predict multiple PDFs with all 3 models.

        Returns:
            [
                {
                    "filename": "doc.pdf",
                    "Legal-BERT": {"label": "acm", "confidence": 0.92},
                    "DistilBERT": {"label": "ieee", "confidence": 0.88},
                    "SciBERT": {"label": "acm", "confidence": 0.95}
                },
                ...
            ]
        """
        results = []

        for pdf_path in pdf_paths:
            try:
                # Extract text (use first tokenizer for extraction)
                text = self.text_processor.extract_text(pdf_path)

                if not text:
                    results.append(
                        {
                            "filename": Path(pdf_path).name,
                            "error": "Failed to extract text",
                        }
                    )
                    continue

                # Run all 3 models
                model_results = self.predict_all(text)

                # Determine true label from folder name
                filename = Path(pdf_path).name
                true_label = None
                for label_name in LABELS:
                    if label_name in str(pdf_path).lower():
                        true_label = label_name
                        break

                results.append(
                    {
                        "filename": filename,
                        "true_label": true_label,
                        "predictions": model_results,
                    }
                )

            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results.append(
                    {"filename": Path(pdf_path).name, "error": str(e)}
                )

        return results


def create_multi_classifier() -> MultiModelClassifier:
    """Factory function."""
    return MultiModelClassifier()
