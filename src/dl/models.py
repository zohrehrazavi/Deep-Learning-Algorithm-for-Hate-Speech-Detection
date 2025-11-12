from typing import Optional

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from transformers import AutoModelForSequenceClassification
except Exception:
    AutoModelForSequenceClassification = None  # type: ignore


class AdditiveAttention(nn.Module):  # type: ignore
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, outputs, mask=None):
        # outputs: [batch, seq, hidden]
        scores = self.v(torch.tanh(self.W(outputs))).squeeze(-1)  # [batch, seq]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)  # [batch, seq]
        context = torch.sum(outputs * attn.unsqueeze(-1), dim=1)  # [batch, hidden]
        return context, attn


class BiLSTMAttention(nn.Module):  # type: ignore
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 100,
                 hidden_dim: int = 128,
                 num_layers: int = 1,
                 num_labels: int = 3,
                 dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.attn = AdditiveAttention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_labels),
        )

    def forward(self, input_ids, attention_mask=None):
        emb = self.embedding(input_ids)
        outputs, _ = self.lstm(emb)
        context, _ = self.attn(outputs, attention_mask)
        logits = self.classifier(context)
        return logits


def build_transformer_head(model_name: str = "distilbert-base-uncased",
                           num_labels: int = 3):
    if AutoModelForSequenceClassification is None:
        raise RuntimeError("transformers not installed.")
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


