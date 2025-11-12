"""
BiLSTM+Attention training CLI.
Trains on data/labeled_data.csv (columns: text, class) with stratified splits,
builds a vocab, optimizes CrossEntropy with optional class weights and early stopping,
and saves artifacts to models/dl/lstm/.
"""
import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from .utils import ensure_dir, save_json, seed_everything
from .datasets import load_local_csv, stratified_splits, compute_weights
from .tokenization import build_vocab, tokenize_lstm
from .models import BiLSTMAttention

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dl/lstm.yaml")
    parser.add_argument("--out_dir", type=str, default="models/dl/lstm")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def load_yaml_config(path: str) -> Dict:
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # Fallback to minimal defaults if yaml not available
        return {
            "embedding_dim": 100,
            "hidden_dim": 128,
            "dropout": 0.3,
            "vocab_max": 30000,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "early_stop_patience": 3,
            "focal_loss": False,
        }


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        label = int(self.labels[idx])
        ids = tokenize_lstm(text, self.vocab, max_len=self.max_len)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def plot_confusion(cm: np.ndarray, classes: List[str], path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ensure_dir(os.path.dirname(path) or ".")
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    preds: List[int] = []
    trues: List[int] = []
    total, correct = 0, 0
    with torch.no_grad():
        for input_ids, labels in loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask=(input_ids != 0).long())
            pred = torch.argmax(logits, dim=-1)
            preds.extend(pred.cpu().tolist())
            trues.extend(labels.cpu().tolist())
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    acc = correct / max(total, 1)
    macro_f1 = f1_score(trues, preds, average="macro")
    return acc, macro_f1, preds, trues


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    seed_everything(42)
    ensure_dir(args.out_dir)

    batch_size = int(cfg.get("batch_size", 32))
    lr = float(cfg.get("learning_rate", 1e-3))
    early_patience = int(cfg.get("early_stop_patience", args.patience))
    max_epochs = int(cfg.get("max_epochs", args.max_epochs))
    vocab_max = int(cfg.get("vocab_max", 30000))
    embedding_dim = int(cfg.get("embedding_dim", 100))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    dropout = float(cfg.get("dropout", 0.3))

    print("Loading data...")
    df = load_local_csv()
    train_df, val_df, test_df = stratified_splits(df, random_state=42)
    label_map = {"0": "hateful", "1": "offensive", "2": "neutral"}
    save_json(label_map, os.path.join(args.out_dir, "label_map.json"))

    print("Building vocab...")
    vocab = build_vocab(train_df["text"].tolist(), min_freq=2, max_size=vocab_max)
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    print("Preparing datasets...")
    train_ds = TextDataset(train_df["text"].tolist(), train_df["class"].tolist(), vocab, max_len=128)
    val_ds = TextDataset(val_df["text"].tolist(), val_df["class"].tolist(), vocab, max_len=128)
    test_ds = TextDataset(test_df["text"].tolist(), test_df["class"].tolist(), vocab, max_len=128)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    print("Computing class weights...")
    weights = compute_weights(train_df["class"].tolist(), num_classes=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing model...")
    model = BiLSTMAttention(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_labels=3,
        dropout=dropout,
        pad_idx=vocab.get("<pad>", 0),
    ).to(device)

    if weights is not None:
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("Training...")
    best_f1 = -1.0
    epochs_no_improve = 0
    best_path = os.path.join(args.out_dir, "model.pt")
    metrics: Dict[str, float] = {}

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        total = 0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids, attention_mask=(input_ids != 0).long())
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            total += labels.size(0)
        train_loss = epoch_loss / max(total, 1)

        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            metrics.update({"best_val_macro_f1": float(best_f1), "best_val_acc": float(val_acc)})
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_patience:
                print("Early stopping.")
                break

    print("Evaluating best model on test set...")
    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state, strict=False)
    test_acc, test_f1, preds, trues = evaluate(model, test_loader, device)
    metrics.update({"test_macro_f1": float(test_f1), "test_acc": float(test_acc)})
    save_json(metrics, os.path.join(args.out_dir, "metrics.json"))

    print("Saving config...")
    save_json({"lstm": cfg}, os.path.join(args.out_dir, "config.json"))

    print("Saving confusion matrix plot...")
    cm = confusion_matrix(trues, preds, labels=[0, 1, 2])
    plot_confusion(cm, classes=["hateful", "offensive", "neutral"], path=os.path.join(args.out_dir, "confusion_matrix.png"))

    print(f"Training complete. Artifacts saved to {args.out_dir}")


if __name__ == "__main__":
    main()


