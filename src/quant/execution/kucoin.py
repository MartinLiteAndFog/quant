"""
KuCoin execution stub.

For now: we only persist normalized signals to JSONL.
Real order placement (close->wait->open, reduceOnly, retries) comes later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ExecResult:
    ok: bool
    note: str
    details: Optional[Dict[str, Any]] = None


def execute_signal_stub(payload: Dict[str, Any]) -> ExecResult:
    """
    Placeholder for future real execution on KuCoin.
    Currently does nothing beyond acknowledging.
    """
    return ExecResult(ok=True, note="execution stub: no orders sent", details=payload)
