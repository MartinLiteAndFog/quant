from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np

from visual.io import load_yaml_config


def base_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    return ap


def load_cfg(path: str) -> Dict[str, Any]:
    cfg = load_yaml_config(path)
    seed = int(cfg.get('random_seed', 1337))
    np.random.seed(seed)
    return cfg


def ensure_path(p: str | Path) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path
