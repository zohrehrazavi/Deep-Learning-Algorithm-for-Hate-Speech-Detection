# Hate Speech Detection Web Application

Modern hate speech detection system that blends rule-based heuristics, classical machine learning, and deep learning models. A responsive Flask UI lets you compare predictions from the `Rules + Classical ML` baseline against the latest deep learning model side by side.

## Highlights

- Hybrid architecture that classifies text as **hateful**, **offensive**, or **neutral**.
- Classical baseline with TF-IDF features, Naive Bayes, Logistic Regression, and rule-based guardrails.
- Deep learning add-ons: BiLSTM with attention and a DistilBERT fine-tuning path.
- Web UI that surfaces both pipelines together so you can track DL progress while you continue training.
- REST API, unit tests, Docker support, and automation scripts for evaluation.

## System Overview

**Preprocessing and safeguards**

- Lower-case normalization, URL/mention stripping, and offensive token normalization through `nltk` and custom regex.
- Negation-aware stopword handling and lemmatization to capture plural/singular forms of protected groups.
- Rule-based layer flags explicit hate phrases, protected-group references, and semantic alerts before ML inference.

**Classical ML baseline**

- TF-IDF (`TfidfVectorizer`) features capped at 10k terms.
- Models: `MultinomialNB` with tuned priors and `LogisticRegression` (`class_weight='balanced'`, `C=0.3`).
- SMOTE oversampling and probability calibration; exposed via `classify_text_with_models`.

**Deep learning models**

- `src/dl/train_lstm.py` trains a BiLSTM + attention classifier with stratified splits, class weighting, early stopping, and confusion-matrix export.
- `src/dl/train_transformer.py` bootstraps a DistilBERT fine-tuning workspace (`models/dl/transformer/`) so you can drop in or continue training HF checkpoints.
- `src/dl/infer.py` caches models and exposes `compare_models` to run baseline, LSTM, and transformer predictions, selecting the best DL result for the UI.
- Artifacts are written under `models/dl/...`; you can keep iterating on DL to push performance.

## User Interface

`templates/index.html` renders a Bootstrap dashboard where a single submission shows:

- Baseline decision, confidence, and metadata badges (rule-based trigger, flagged context, semantic alerts).
- Deep learning prediction (BiLSTM or DistilBERT, whichever is available/confident) with confidence and model name.
- Tooltips and a modal explain how the system works, making model comparisons clear for stakeholders.

## Project Structure

```
.
├── app.py                   # Flask app + UI comparison
├── src/
│   ├── main.py              # Baseline training + inference helpers
│   └── dl/
│       ├── train_lstm.py    # BiLSTM training CLI
│       ├── train_transformer.py  # DistilBERT training scaffold
│       ├── infer.py         # Baseline vs DL comparison utilities
│       └── ...              # datasets, tokenization, models, utils
├── configs/dl/              # YAML configs for DL experiments
├── models/                  # Saved baseline + deep learning artifacts
├── templates/index.html     # Bootstrap UI with side-by-side cards
├── static/style.css         # UI styling
├── scripts/                 # Automation helpers (train/evaluate)
├── tests/                   # Pytest suite
└── data/labeled_data.csv    # Training dataset (text, class)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Training Workflows

**Classical baseline**

```bash
python src/main.py
```

Generates `models/{naive_bayes.pkl, logistic_regression.pkl, tfidf_vectorizer.pkl, config.json}` after preprocessing, SMOTE balancing, training, and evaluation.

**BiLSTM + Attention**

```bash
python src/dl/train_lstm.py --config configs/dl/lstm.yaml --out_dir models/dl/lstm
```

Produces `model.pt`, `vocab.json`, `label_map.json`, `metrics.json`, and a confusion matrix plot. Continue re-running as you experiment with new hyperparameters or more data.

**DistilBERT fine-tuning**

```bash
python src/dl/train_transformer.py --config configs/dl/transformer.yaml --out_dir models/dl/transformer
```

The current script scaffolds the directory so inference can load HuggingFace checkpoints. Replace the placeholder with your fine-tuned weights as you keep training the DL models.

Automation scripts are available under `scripts/` (`train_lstm.sh`, `train_transformer.sh`, `evaluate_all.sh`) for repeatable experiments.

## Running the App

```bash
python app.py
```

- UI: http://localhost:8081
- API: `POST http://localhost:8081/classify` with `{"text": "..."}`

API responses include the label, confidence (formatted percentage), explanation, and metadata describing whether rules triggered.

## Testing

```bash
python -m pytest tests/
```

The suite covers preprocessing, hybrid classification logic, model interfaces, and Flask routes.

## Docker Usage

```bash
docker-compose up --build      # recommended
```

The container exposes the app on port 8082 by default. Bind-mount `./models` and `./data` to persist artifacts. Environment variables:

- `FLASK_ENV` (`production` | `development`, default `production`)
- `PORT` (default `8082`)

## Roadmap

- Keep fine-tuning BiLSTM and DistilBERT checkpoints and drop the improved weights under `models/dl/`.
- Expand the comparison dashboard with per-class probability charts and historical metrics.
- Integrate continuous evaluation to monitor drift as new data is added.

---

Contributions and experiments are welcome—train a new DL model, drop the artifacts in `models/dl/`, and the UI will immediately reflect the latest comparison against the classical baseline.
