# src/quant/strategies/signal_io.py
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import pandas as pd


def load_signals(
    path: Union[str, Path],
    kind: Optional[str] = None,
    **kwargs: Any,
):
    """
    Backwards-compatible loader.

    renko_runner expects an object with attributes:
      - bundle.signal  (Series) OR
      - bundle.df      (DataFrame with columns ts + signal)

    We provide bundle.df because it's unambiguous.
    If kind is None, we return a Series for convenience.
    """
    _ = kwargs
    df = read_signals_jsonl(path)

    # Make Series (optional convenience path)
    if df.empty:
        s = pd.Series(dtype="int64", name="signal")
    else:
        s = df.set_index("ts")["signal"].sort_index().astype("int64").clip(-1, 1)
        s.name = "signal"

    if kind is None:
        return s

    # renko_runner bundle.df expects explicit ts + signal
    if df.empty:
        sig_df = pd.DataFrame(columns=["ts", "signal"])
    else:
        sig_df = df.copy()
        sig_df["ts"] = pd.to_datetime(sig_df["ts"], utc=True, errors="coerce")
        sig_df = sig_df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")
        sig_df["signal"] = pd.to_numeric(sig_df["signal"], errors="coerce").fillna(0).astype(int)
        sig_df = sig_df[sig_df["signal"] != 0].copy()
        sig_df = sig_df[["ts", "signal"]].reset_index(drop=True)

    return SimpleNamespace(
        df=sig_df,
        signal=None,  # present but unused
        meta={
            "kind": kind,
            "path": str(Path(path)),
            "rows": int(len(sig_df)),
        },
    )


def read_signals_jsonl(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"signals file not found: {p}")

    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            ts = _extract_ts(obj)
            if ts is None:
                continue

            sig = _extract_signal(obj)
            if sig is None:
                sig = 0

            rows.append({"ts": ts, "signal": int(sig)})

    if not rows:
        return pd.DataFrame(columns=["ts", "signal"])

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"])
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0).astype("int64")
    df["signal"] = df["signal"].clip(-1, 1).astype("int64")
    df = df.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return df


def _extract_ts(obj: Dict[str, Any]) -> Optional[pd.Timestamp]:
    candidates = []

    def add(x):
        if x is not None:
            candidates.append(x)

    add(obj.get("ts"))
    add(obj.get("timestamp"))
    add(obj.get("time"))
    add(obj.get("server_ts"))

    for k in ("payload", "data", "msg"):
        v = obj.get(k)
        if isinstance(v, dict):
            add(v.get("ts"))
            add(v.get("timestamp"))
            add(v.get("time"))

    for c in candidates:
        ts = _parse_any_ts(c)
        if ts is not None:
            return ts
    return None


def _parse_any_ts(v: Any) -> Optional[pd.Timestamp]:
    if v is None:
        return None

    if isinstance(v, pd.Timestamp):
        return v.tz_convert("UTC") if v.tzinfo is not None else v.tz_localize("UTC")

    if isinstance(v, (int, float)) and not isinstance(v, bool):
        x = int(v)
        try:
            if x > 10**17:   # ns
                return pd.to_datetime(x, utc=True, unit="ns")
            if x > 10**14:   # us
                return pd.to_datetime(x, utc=True, unit="us")
            if x > 10**11:   # ms
                return pd.to_datetime(x, utc=True, unit="ms")
            return pd.to_datetime(x, utc=True, unit="s")
        except Exception:
            return None

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return pd.to_datetime(s, utc=True, errors="raise")
        except Exception:
            try:
                return _parse_any_ts(int(s))
            except Exception:
                return None

    return None


def _extract_signal(obj: Dict[str, Any]) -> Optional[int]:
    for k in ("signal", "position"):
        if k in obj:
            return _coerce_signal(obj.get(k))

    if "action" in obj:
        a = _coerce_action(obj.get("action"))
        if a is not None:
            return a

    for k in ("payload", "data", "msg"):
        v = obj.get(k)
        if isinstance(v, dict):
            for kk in ("signal", "position"):
                if kk in v:
                    return _coerce_signal(v.get(kk))
            if "action" in v:
                a = _coerce_action(v.get("action"))
                if a is not None:
                    return a

    return None


def _coerce_signal(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return 1 if v > 0 else (-1 if v < 0 else 0)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "+1", "long", "buy"):
            return 1
        if s in ("-1", "short", "sell"):
            return -1
        if s in ("0", "flat", "neutral", "close", "exit"):
            return 0
        try:
            return _coerce_signal(float(s))
        except Exception:
            return None
    return None


def _coerce_action(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("buy", "long", "enter_long", "open_long"):
            return 1
        if s in ("sell", "short", "enter_short", "open_short"):
            return -1
        if s in ("close", "exit", "flat", "close_position", "reduce_only"):
            return 0
    return None
