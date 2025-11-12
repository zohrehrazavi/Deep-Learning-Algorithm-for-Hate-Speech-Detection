from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore

try:
    from datasets import load_dataset, DatasetDict
except Exception:
    load_dataset = None
    DatasetDict = None


@dataclass
class TextLabelExample:
    text: str
    label: int


class SimpleTextDataset(Dataset):  # type: ignore
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], int(self.labels[idx])


def load_local_csv(path: str = "data/labeled_data.csv",
                   text_col: str = "text",
                   label_col: str = "class") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "tweet" in df.columns and text_col not in df.columns:
        df = df.rename(columns={"tweet": text_col})
    if label_col not in df.columns or text_col not in df.columns:
        raise ValueError("CSV must contain 'text' and 'class' columns.")
    return df[[text_col, label_col]].dropna()


def stratified_splits(df: pd.DataFrame,
                      test_size: float = 0.15,
                      val_size: float = 0.15,
                      random_state: int = 42,
                      label_col: str = "class"):
    train_df, temp_df = train_test_split(
        df, test_size=test_size + val_size, stratify=df[label_col], random_state=random_state
        )
    rel_val = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=1 - rel_val, stratify=temp_df[label_col], random_state=random_state
        )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_weights(y_train: List[int], num_classes: int) -> Optional[List[float]]:
    try:
        classes = list(range(num_classes))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        return weights.tolist()
    except Exception:
        return None


def build_bilstm_dataloaders(train_df: pd.DataFrame,
                             val_df: pd.DataFrame,
                             test_df: pd.DataFrame,
                             text_col: str = "text",
                             label_col: str = "class",
                             batch_size: int = 32):
    if torch is None:
        raise RuntimeError("PyTorch is required for BiLSTM dataloaders.")
    train_ds = SimpleTextDataset(train_df[text_col].tolist(), train_df[label_col].tolist())
    val_ds = SimpleTextDataset(val_df[text_col].tolist(), val_df[label_col].tolist())
    test_ds = SimpleTextDataset(test_df[text_col].tolist(), test_df[label_col].tolist())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )


def load_hf_hatexplain(split_seed: int = 42):
    if load_dataset is None:
        raise RuntimeError("datasets library not installed.")
    ds = load_dataset("hatexplain")
    # The dataset has fields 'post', 'annotators', etc. We create text/label and split deterministically
    def map_ex(example):
        text = " ".join(example.get("post", []))
        # Majority label across annotators; fallback neutral (2)
        labels = example.get("annotators", {}).get("label", [])
        if labels:
            # Map dataset labels [0,1,2] -> assume 0:hateful,1:offensive,2:neutral as is
            counts = {0: 0, 1: 0, 2: 0}
            for l in labels:
                if l in counts:
                    counts[l] += 1
            label = max(counts, key=counts.get)
        else:
            label = 2
        return {"text": text, "class": label}
    mapped = ds["train"].map(map_ex)
    df = pd.DataFrame({"text": mapped["text"], "class": mapped["class"]})
    return stratified_splits(df, random_state=split_seed)


def build_transformer_datasetdict(train_df: pd.DataFrame,
                                  val_df: pd.DataFrame,
                                  test_df: pd.DataFrame):
    if DatasetDict is None:
        raise RuntimeError("datasets library not installed.")
    from datasets import Dataset as HFDataset  # type: ignore
    return DatasetDict({
        "train": HFDataset.from_pandas(train_df),
        "validation": HFDataset.from_pandas(val_df),
        "test": HFDataset.from_pandas(test_df),
    })


