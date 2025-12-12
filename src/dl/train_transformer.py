"""
DistilBERT fine-tuning CLI for hate speech detection.
Trains on data/labeled_data.csv with stratified splits, fine-tunes DistilBERT,
and saves artifacts to models/dl/transformer/.
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from .utils import ensure_dir, save_json, seed_everything, load_yaml_config
from .datasets import load_local_csv, stratified_splits, build_transformer_datasetdict

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from transformers import EarlyStoppingCallback
except ImportError:
    print("Error: transformers library not installed. Install with: pip install transformers")
    raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dl/transformer.yaml")
    parser.add_argument("--out_dir", type=str, default="models/dl/transformer")
    parser.add_argument("--max_epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    return parser.parse_args()


def tokenize_function(examples, tokenizer, max_length: int = 128):
    """Tokenize texts for transformer model."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )


def compute_metrics(eval_pred):
    """Compute accuracy and F1 score for evaluation."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = (predictions == labels).mean()
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    seed_everything(42)
    ensure_dir(args.out_dir)

    # Load config
    model_name = cfg.get("model_name", "distilbert-base-uncased")
    max_len = int(cfg.get("max_len", 128))
    batch_size = int(cfg.get("batch_size", 16))
    learning_rate = float(cfg.get("learning_rate", 5.0e-5))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.06))
    early_patience = int(cfg.get("early_stop_patience", args.patience))
    max_epochs = int(cfg.get("max_epochs", args.max_epochs))
    weight_decay = float(cfg.get("weight_decay", 0.01))

    print("Loading data...")
    df = load_local_csv()
    train_df, val_df, test_df = stratified_splits(df, random_state=42)
    
    label_map = {"0": "hateful", "1": "offensive", "2": "neutral"}
    save_json(label_map, os.path.join(args.out_dir, "label_map.json"))

    print("Building HuggingFace datasets...")
    dataset_dict = build_transformer_datasetdict(train_df, val_df, test_df)

    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
    )

    print("Tokenizing datasets...")
    # Tokenize and rename 'class' to 'labels' in one step
    def tokenize_and_label(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_len,
        )
        tokenized["labels"] = examples["class"]
        return tokenized
    
    tokenized_datasets = dataset_dict.map(
        tokenize_and_label,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments (using compatible parameters)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.out_dir, "checkpoint"),
        num_train_epochs=max_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_dir=os.path.join(args.out_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",  # Try eval_strategy first
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        disable_tqdm=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_patience)],
    )

    print("Starting training...")
    train_result = trainer.train()

    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
    
    print("\n=== Test Results ===")
    print(f"Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    print(f"Test Macro F1: {test_results.get('eval_macro_f1', 0):.4f}")

    # Get predictions for confusion matrix
    predictions = trainer.predict(tokenized_datasets["test"])
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(
        y_true, y_pred,
        target_names=["Hateful", "Offensive", "Neutral"]
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - DistilBERT')
    plt.colorbar()
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, ["Hateful", "Offensive", "Neutral"])
    plt.yticks(tick_marks, ["Hateful", "Offensive", "Neutral"])
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "confusion_matrix.png"))
    print(f"\nSaved confusion matrix to {args.out_dir}/confusion_matrix.png")

    # Save best model
    print("\nSaving best model...")
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Save metrics
    metrics = {
        "train_loss": train_result.training_loss,
        "test_accuracy": float(test_results.get("eval_accuracy", 0)),
        "test_macro_f1": float(test_results.get("eval_macro_f1", 0)),
    }
    save_json(metrics, os.path.join(args.out_dir, "metrics.json"))

    print(f"\n✓ Training completed successfully!")
    print(f"Model saved to: {args.out_dir}")
    print(f"Test Macro F1: {metrics['test_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
