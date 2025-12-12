import json
import os
import random
from typing import Any, Dict

import numpy as np

try:
    import torch
except Exception:
    torch = None


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def file_exists(path: str) -> bool:
    return os.path.isfile(path)


def load_yaml_config(path: str) -> Dict:
    """Load YAML configuration file."""
    try:
        import yaml  # type: ignore
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        # Fallback to minimal defaults if yaml not available
        return {}


