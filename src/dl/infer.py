from typing import Dict, Any, Optional, Tuple
import os

import math

from src.main import classify_text_with_models, preprocess, class_mapping  # reuse baseline processing
from .utils import file_exists, load_json

_TRANSFORMER_CACHE: Dict[str, Any] = {"model": None, "tokenizer": None, "labels": None}
_LSTM_CACHE: Dict[str, Any] = {"model": None, "vocab": None, "labels": None}

def _softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _format_baseline_output(text: str, vectorizer, nb_model, lr_model) -> Dict[str, Any]:
    label, conf, explanation, metadata = classify_text_with_models(text, vectorizer, nb_model, lr_model)
    return {
        "label": label,
        "confidence": round(float(conf) * 100.0, 1),
        "explanation": explanation,
        "metadata": metadata,
        "model_name": "Rules + Classical ML",
    }


def predict_baseline(text: str, vectorizer, nb_model, lr_model) -> Dict[str, Any]:
    return _format_baseline_output(text, vectorizer, nb_model, lr_model)


def _load_transformer_model(base_dir: str = "models/dl/transformer"):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    if _TRANSFORMER_CACHE["model"] is not None:
        return _TRANSFORMER_CACHE["tokenizer"], _TRANSFORMER_CACHE["model"], _TRANSFORMER_CACHE["labels"]
    if not os.path.isdir(base_dir):
        return None, None, None
    # Expect a HF saved model folder (config.json + pytorch_model.bin) and optional label_map.json
    config_path = os.path.join(base_dir, "config.json")
    model_path = os.path.join(base_dir)
    if not file_exists(config_path):
        return None, None, None
    try:
        tok = AutoTokenizer.from_pretrained(model_path)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
        labels = None
        label_map_path = os.path.join(base_dir, "label_map.json")
        if file_exists(label_map_path):
            labels = load_json(label_map_path)
        _TRANSFORMER_CACHE.update({"tokenizer": tok, "model": mdl, "labels": labels})
        return tok, mdl, labels
    except Exception:
        return None, None, None


def predict_transformer(text: str) -> Dict[str, Any]:
    tok, mdl, labels = _load_transformer_model()
    if tok is None or mdl is None:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "model_name": "DistilBERT (unavailable)",
        }
    try:
        inputs = tok(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        with __import__("torch").no_grad():
            out = mdl(**inputs)
            logits = out.logits[0].tolist()
        probs = _softmax(logits)
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        if labels:
            # labels stored as {"0":"hateful",...}
            label = labels.get(str(pred_idx), class_mapping.get(pred_idx, "neutral"))
        else:
            label = class_mapping.get(pred_idx, "neutral")
        return {
            "label": label,
            "confidence": round(float(probs[pred_idx]) * 100.0, 1),
            "model_name": "DistilBERT",
        }
    except Exception:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "model_name": "DistilBERT (error)",
        }


def _load_lstm_model(base_dir: str = "models/dl/lstm"):
    import json
    import torch  # type: ignore
    from .models import BiLSTMAttention  # type: ignore
    if _LSTM_CACHE["model"] is not None:
        return _LSTM_CACHE["model"], _LSTM_CACHE["vocab"], _LSTM_CACHE["labels"]
    model_path = os.path.join(base_dir, "model.pt")
    vocab_path = os.path.join(base_dir, "vocab.json")
    if not (file_exists(model_path) and file_exists(vocab_path)):
        return None, None, None
    try:
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        # Minimal config
        config_path = os.path.join(base_dir, "config.json")
        cfg = {"embedding_dim": 100, "hidden_dim": 128, "num_layers": 1, "dropout": 0.3}
        if file_exists(config_path):
            try:
                cfg = load_json(config_path).get("lstm", cfg) if "lstm" in load_json(config_path) else load_json(config_path)
            except Exception:
                pass
        model = BiLSTMAttention(
            vocab_size=len(vocab),
            embedding_dim=cfg.get("embedding_dim", 100),
            hidden_dim=cfg.get("hidden_dim", 128),
            num_layers=cfg.get("num_layers", 1),
            num_labels=3,
            dropout=cfg.get("dropout", 0.3),
            pad_idx=vocab.get("<pad>", 0),
        )
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval()
        labels = None
        label_map_path = os.path.join(base_dir, "label_map.json")
        if file_exists(label_map_path):
            labels = load_json(label_map_path)
        _LSTM_CACHE.update({"model": model, "vocab": vocab, "labels": labels})
        return model, vocab, labels
    except Exception:
        return None, None, None


def predict_lstm(text: str) -> Dict[str, Any]:
    model, vocab, labels = _load_lstm_model()
    if model is None or vocab is None:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "model_name": "BiLSTM-Attn (unavailable)",
        }
    try:
        from .tokenization import tokenize_lstm  # defer import
        import torch  # type: ignore
        ids = tokenize_lstm(preprocess(text), vocab, max_len=128)
        input_ids = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(input_ids, attention_mask=(input_ids != vocab.get("<pad>", 0)).long())
            logits = logits[0].tolist()
        probs = _softmax(logits)
        pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        if labels:
            label = labels.get(str(pred_idx), class_mapping.get(pred_idx, "neutral"))
        else:
            label = class_mapping.get(pred_idx, "neutral")
        return {
            "label": label,
            "confidence": round(float(probs[pred_idx]) * 100.0, 1),
            "model_name": "BiLSTM-Attn",
        }
    except Exception:
        return {
            "label": "neutral",
            "confidence": 0.0,
            "model_name": "BiLSTM-Attn (error)",
        }


def compare_models(text: str, vectorizer, nb_model, lr_model) -> Dict[str, Any]:
    baseline = predict_baseline(text, vectorizer, nb_model, lr_model)
    # Prefer transformer if available else fallback to lstm
    transformer = predict_transformer(text)
    deep = transformer if "unavailable" not in transformer.get("model_name", "").lower() else predict_lstm(text)
    return {
        "baseline": baseline,
        "deep_lstm": predict_lstm(text),
        "deep_transformer": transformer,
        "deep_best": deep,
    }


