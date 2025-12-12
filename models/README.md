# Models Directory Structure

This directory contains all trained models for the Hate Speech Detection system, organized by model type.

## Directory Structure

```
models/
├── ml/              # Classical Machine Learning Models
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── tfidf_vectorizer.pkl
│   └── config.json
│
└── dl/              # Deep Learning Models
    ├── lstm/        # BiLSTM-Attention Model
    │   ├── model.pt
    │   ├── vocab.json
    │   ├── config.json
    │   ├── label_map.json
    │   ├── metrics.json
    │   └── confusion_matrix.png
    │
    └── transformer/ # Transformer Model (DistilBERT)
        ├── config.json
        ├── pytorch_model.bin
        └── label_map.json
```

## Model Types

### ML Models (`models/ml/`)
**Classical Machine Learning** models trained using scikit-learn:
- **Naive Bayes** (`naive_bayes.pkl`) - Multinomial Naive Bayes classifier
- **Logistic Regression** (`logistic_regression.pkl`) - Logistic Regression classifier
- **TF-IDF Vectorizer** (`tfidf_vectorizer.pkl`) - Feature extraction for ML models
- **Config** (`config.json`) - Model configuration and class mappings

**Training:** Run `python src/main.py` to train these models.

**Usage:** These models are loaded automatically by `app.py` for baseline classification.

### DL Models (`models/dl/`)
**Deep Learning** models for advanced classification:

#### LSTM (`models/dl/lstm/`)
- **BiLSTM-Attention** model architecture
- Trained with PyTorch
- Files: `model.pt`, `vocab.json`, `config.json`, `label_map.json`, `metrics.json`

**Training:** Run `python src/dl/train_lstm.py --config configs/dl/lstm.yaml --out_dir models/dl/lstm`

#### Transformer (`models/dl/transformer/`)
- **DistilBERT** fine-tuned model
- HuggingFace Transformers format
- Files: `config.json`, `pytorch_model.bin`, `label_map.json`

**Training:** Run `python src/dl/train_transformer.py --config configs/dl/transformer.yaml --out_dir models/dl/transformer`

## Model Comparison

The application compares predictions from:
1. **Baseline**: Rule-based + ML models (Naive Bayes + Logistic Regression)
2. **Deep Learning**: Best available DL model (Transformer preferred, falls back to LSTM)

## Notes

- ML models are required for the application to run
- DL models are optional but provide enhanced accuracy
- If DL models are not available, the application will still function using only ML models
- Model files are loaded lazily and cached in memory for performance




