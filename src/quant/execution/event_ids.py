from __future__ import annotations

from typing import Optional


def make_event_id(
    family_prefix: str,
    strategy: str,
    symbol: str,
    ts_iso: str,
    seq: int,
    venue: Optional[str] = None,
) -> str:
    """
    Build a stable readable event id.

    Examples:
      signal:imba_countertrend:SOLUSDT:2026-03-06T11:23:00Z:77
      execution:kraken:SOLUSDT:2026-03-06T11:23:01Z:991
    """
    parts = [str(family_prefix).strip()]
    if venue:
        parts.append(str(venue).strip())
    else:
        parts.append(str(strategy).strip())
    parts.append(str(symbol).strip())
    parts.append(str(ts_iso).strip())
    parts.append(str(int(seq)))
    return ":".join(parts)