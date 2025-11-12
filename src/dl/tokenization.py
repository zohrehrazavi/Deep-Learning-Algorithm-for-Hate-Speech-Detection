from collections import Counter
from typing import List, Dict, Tuple

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None  # type: ignore


def build_vocab(texts: List[str],
                min_freq: int = 2,
                max_size: int = 30000) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        counter.update(t.strip().split())
    vocab = {"<pad>": 0, "<unk>:": 1}
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = len(vocab)
        if len(vocab) >= max_size:
            break
    return vocab


def tokenize_lstm(text: str, vocab: Dict[str, int], max_len: int = 128) -> List[int]:
    tokens = text.strip().split()
    ids = [vocab.get(tok, vocab.get("<unk>", 1)) for tok in tokens][:max_len]
    while len(ids) < max_len:
        ids.append(vocab.get("<pad>", 0))
    return ids


def get_transformer_tokenizer(model_name: str = "distilbert-base-uncased"):
    if AutoTokenizer is None:
        raise RuntimeError("transformers not installed.")
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


