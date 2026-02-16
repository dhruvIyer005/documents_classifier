"""
train_weighted.py - Train models with class weights to handle imbalanced data
Weights prioritize legal, compliance, springer (underrepresented categories)
"""

import sys
from pathlib import Path
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent))
from config import config
from text_processor import TextProcessor

def load_pdfs():
    """Load all PDFs from data folders"""
    docs = []
    
    for label in config.LABELS:
        label_dir = config.DATA_DIR / label
        if not label_dir.exists():
            print(f"  {label}: Not found")
            continue
        
        pdf_files = list(label_dir.glob("*.pdf"))
        print(f"  {label}: {len(pdf_files)} PDFs")
        
        for pdf_path in sorted(pdf_files):
            try:
                tp = TextProcessor()
                text = tp.extract_text(str(pdf_path))
                if text is not None and len(text) > 100:
                    docs.append({
                        "text": text,
                        "label": config.LABEL2ID[label]
                    })
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
    
    return docs

def compute_class_weights(docs):
    """Compute class weights to balance underrepresented categories"""
    # Count samples per class
    label_counts = {}
    for doc in docs:
        label = doc["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nClass distribution in training data:")
    for label_id, count in sorted(label_counts.items()):
        label_name = config.ID2LABEL[label_id]
        print(f"  {label_name}: {count} samples")
    
    # Define weights: prioritize underrepresented categories
    weights = {}
    # compliance, legal: 5x-4x weight (most underrepresented, most important)
    # springer: 3x weight (moderately underrepresented)
    # acm, ieee: 1x weight (well-represented)
    weight_map = {
        "compliance": 5.0,  # Increased from 3.0
        "legal": 4.0,       # Increased from 3.0
        "springer": 3.0,    # Increased from 2.0
        "acm": 1.0,
        "ieee": 1.0,
    }
    
    for i, label_name in config.ID2LABEL.items():
        weights[i] = weight_map.get(label_name, 1.0)
    
    print("\nClass weights:")
    for label_id, weight in sorted(weights.items()):
        label_name = config.ID2LABEL[label_id]
        print(f"  {label_name}: {weight}x")
    
    return weights

def train_model(model_name, display_name, save_path):
    """Train a single model with class weights"""
    print(f"\n{'='*60}")
    print(f"TRAINING {display_name}")
    print(f"{'='*60}\n")
    
    # Load data
    print(f"Loading PDFs...")
    docs = load_pdfs()
    print(f"Total documents: {len(docs)}\n")
    
    if len(docs) < 10:
        print("ERROR: Need at least 10 PDFs!")
        return False
    
    # Compute class weights
    class_weights = compute_class_weights(docs)
    
    # Shuffle and split
    indices = np.random.RandomState(42).permutation(len(docs))
    split_train = int(len(docs) * 0.7)
    split_val = int(len(docs) * 0.85)
    
    train_docs = [docs[i] for i in indices[:split_train]]
    val_docs = [docs[i] for i in indices[split_train:split_val]]
    
    print(f"Train: {len(train_docs)}, Val: {len(val_docs)}\n")
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        "text": [d["text"] for d in train_docs],
        "label": [d["label"] for d in train_docs]
    })
    
    val_dataset = Dataset.from_dict({
        "text": [d["text"] for d in val_docs],
        "label": [d["label"] for d in val_docs]
    })
    
    # Load tokenizer
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.MAX_LENGTH,
            padding="max_length"
        )
    
    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Load model
    print(f"Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=config.NUM_LABELS
    ).to(config.DEVICE)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=str(config.OUTPUTS_DIR / display_name),
        num_train_epochs=config.NUM_EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        warmup_steps=config.WARMUP_STEPS,
        weight_decay=config.WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=config.USE_FP16 and config.DEVICE == "cuda",
        logging_steps=5,
    )
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}
    
    # Custom trainer with weighted loss
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Compute weighted cross-entropy loss
            loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(
                    [class_weights[i] for i in range(config.NUM_LABELS)],
                    dtype=torch.float32,
                    device=self.args.device
                )
            )
            loss = loss_fn(logits, labels)
            
            return (loss, outputs) if return_outputs else loss
    
    # Train
    print(f"\nTraining with class weights...\n")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Get best validation accuracy from trainer state
    if hasattr(trainer.state, 'best_metric'):
        print(f"\n{display_name} Validation Accuracy: {trainer.state.best_metric:.2%}\n")
    else:
        print(f"\n{display_name} Training completed\n")
    
    # Save
    print(f"Saving to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[SAVED]\n")
    
    return True

def main():
    print("="*60)
    print("TRAINING WITH CLASS WEIGHTS")
    print("Prioritizing underrepresented categories")
    print("="*60)
    
    models = [
        ("nlpaueb/legal-bert-base-uncased", "Legal-BERT", config.MODELS_DIR / "legal_bert"),
        ("microsoft/deberta-base", "DeBERTa", config.MODELS_DIR / "deberta"),
    ]
    
    results = {}
    for model_name, display_name, save_path in models:
        try:
            success = train_model(model_name, display_name, save_path)
            results[display_name] = "OK" if success else "FAILED"
        except Exception as e:
            print(f"ERROR: {e}\n")
            results[display_name] = f"FAILED: {str(e)[:50]}"
    
    print("="*60)
    print("SUMMARY")
    print("="*60)
    for model, status in results.items():
        print(f"{model}: {status}")
    print("="*60)

if __name__ == "__main__":
    main()
