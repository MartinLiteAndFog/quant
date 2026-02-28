from __future__ import annotations

from pathlib import Path
from typing import Dict


def ensure_output_dirs(out_dir: str | Path, cfg: dict) -> Dict[str, Path]:
    root = Path(out_dir)
    figures = root
    exports = root / str(cfg.get('exports', {}).get('out_dir', 'exports/graph'))
    figures.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    return {'figures': figures, 'exports': exports}
