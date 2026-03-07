from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def append_event_jsonl(path: Path, event: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")