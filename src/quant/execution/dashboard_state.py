from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.regime import RegimeStore

_LAST_REFRESH_TS: Optional[pd.Timestamp] = None


def _to_ts_iso(ts_like: Any) -> Optional[str]:
    ts = pd.to_datetime(ts_like, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _env_path(name: str, default_value: str) -> Path:
    return Path(os.getenv(name, default_value))


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _read_renko_df() -> pd.DataFrame:
    p = _env_path("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet")
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            return pd.DataFrame()
    need = {"open", "high", "low", "close"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def _refresh_renko_cache_if_needed(existing_df: pd.DataFrame) -> pd.DataFrame:
    global _LAST_REFRESH_TS
    if not _truthy(os.getenv("DASHBOARD_RENKO_AUTO_REFRESH_ON_READ", "1")):
        return existing_df
    now = pd.Timestamp.now("UTC")
    stale_min = int(os.getenv("DASHBOARD_RENKO_STALE_MIN", "30"))
    refresh_cooldown_sec = int(os.getenv("DASHBOARD_RENKO_REFRESH_COOLDOWN_SEC", "60"))
    is_stale = True
    if not existing_df.empty:
        last_ts = pd.Timestamp(existing_df["ts"].iloc[-1])
        is_stale = (now - last_ts) > pd.Timedelta(minutes=max(1, stale_min))
    if not is_stale:
        return existing_df
    if _LAST_REFRESH_TS is not None and (now - _LAST_REFRESH_TS) < pd.Timedelta(seconds=max(1, refresh_cooldown_sec)):
        return existing_df
    try:
        from quant.execution.renko_cache_updater import refresh_renko_cache

        refresh_renko_cache(
            symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
            box=float(os.getenv("DASHBOARD_RENKO_BOX", "0.1")),
            days_back=int(os.getenv("DASHBOARD_RENKO_DAYS_BACK", "14")),
            step_hours=int(os.getenv("DASHBOARD_RENKO_STEP_HOURS", "6")),
            out_parquet=str(_env_path("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet")),
        )
        _LAST_REFRESH_TS = now
    except Exception:
        return existing_df
    return _read_renko_df()


def load_renko_bars(max_points: int = 5000) -> List[Dict[str, Any]]:
    df = _read_renko_df()
    df = _refresh_renko_cache_if_needed(df)
    if df.empty:
        return []
    df = df.tail(int(max(1, max_points)))
    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        out.append(
            {
                "time": int(pd.Timestamp(r["ts"]).timestamp()),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            }
        )
    return out


def build_fibo_levels(max_points: int = 5000, lookback: Optional[int] = None) -> Dict[str, Any]:
    lb = int(lookback or int(os.getenv("LIVE_IMBA_LOOKBACK", "250")))
    lb = max(2, lb)
    df = _read_renko_df()
    if df.empty:
        return {"lookback": lb, "long": [], "mid": [], "short": [], "latest": {}}
    df = df.tail(int(max(lb + 5, max_points))).copy()
    hh = pd.to_numeric(df["high"], errors="coerce").rolling(lb, min_periods=lb).max()
    ll = pd.to_numeric(df["low"], errors="coerce").rolling(lb, min_periods=lb).min()
    rng = hh - ll
    fib_long = hh - rng * 0.236
    fib_mid = hh - rng * 0.5
    fib_short = hh - rng * 0.786

    out_long: List[Dict[str, Any]] = []
    out_mid: List[Dict[str, Any]] = []
    out_short: List[Dict[str, Any]] = []
    for i in range(len(df)):
        ts = int(pd.Timestamp(df.iloc[i]["ts"]).timestamp())
        a = fib_long.iloc[i]
        b = fib_mid.iloc[i]
        c = fib_short.iloc[i]
        if pd.notna(a):
            out_long.append({"time": ts, "value": float(a)})
        if pd.notna(b):
            out_mid.append({"time": ts, "value": float(b)})
        if pd.notna(c):
            out_short.append({"time": ts, "value": float(c)})

    latest = {
        "long": out_long[-1]["value"] if out_long else None,
        "mid": out_mid[-1]["value"] if out_mid else None,
        "short": out_short[-1]["value"] if out_short else None,
    }
    return {"lookback": lb, "long": out_long, "mid": out_mid, "short": out_short, "latest": latest}


def load_trade_markers(max_points: int = 5000) -> List[Dict[str, Any]]:
    p = _env_path("DASHBOARD_TRADES_PARQUET", "data/live/trades.parquet")
    if not p.exists():
        return []
    df = pd.read_parquet(p)
    if "entry_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "entry_ts"})
    if "entry_ts" not in df.columns:
        return []
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts"]).sort_values("entry_ts").tail(int(max(1, max_points)))
    markers: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        side = int(r["side"]) if "side" in df.columns and pd.notna(r.get("side")) else 0
        markers.append(
            {
                "time": int(pd.Timestamp(r["entry_ts"]).timestamp()),
                "position": "belowBar" if side >= 0 else "aboveBar",
                "shape": "arrowUp" if side >= 0 else "arrowDown",
                "color": "#2ecc71" if side >= 0 else "#f39c12",
                "text": f"entry {'L' if side >= 0 else 'S'}",
            }
        )
        if "exit_ts" in df.columns and pd.notna(r.get("exit_ts")):
            exit_ts = pd.to_datetime(r["exit_ts"], utc=True, errors="coerce")
            if pd.notna(exit_ts):
                markers.append(
                    {
                        "time": int(pd.Timestamp(exit_ts).timestamp()),
                        "position": "aboveBar" if side >= 0 else "belowBar",
                        "shape": "circle",
                        "color": "#9aa5b1",
                        "text": str(r.get("exit_event", "exit")),
                    }
                )
    return markers


def load_active_levels() -> Dict[str, Any]:
    p = _env_path("DASHBOARD_LEVELS_JSON", "data/live/execution_state.json")
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def build_regime_overlay(symbol: str, hours: int = 24 * 14) -> Dict[str, Any]:
    store = RegimeStore()
    end_ts = pd.Timestamp.now("UTC")
    start_ts = end_ts - pd.Timedelta(hours=int(max(1, hours)))
    rows = store.get_history(symbol=symbol, start_ts=start_ts.isoformat(), end_ts=end_ts.isoformat(), limit=20000)
    if not rows:
        return {"spans": [], "points": [], "latest": None}

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    df["gate_on"] = pd.to_numeric(df["gate_on"], errors="coerce").fillna(0).astype(int)
    df["confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    spans: List[Dict[str, Any]] = []
    if len(df):
        cur_gate = int(df.loc[0, "gate_on"])
        cur_conf = float(df.loc[0, "confidence"])
        start = pd.Timestamp(df.loc[0, "ts"])
        for i in range(1, len(df)):
            gate_i = int(df.loc[i, "gate_on"])
            conf_i = float(df.loc[i, "confidence"])
            ts_i = pd.Timestamp(df.loc[i, "ts"])
            if gate_i != cur_gate:
                spans.append(
                    {
                        "from": int(start.timestamp()),
                        "to": int(ts_i.timestamp()),
                        "gate_on": cur_gate,
                        "confidence": cur_conf,
                    }
                )
                start = ts_i
                cur_gate = gate_i
                cur_conf = conf_i
            else:
                cur_conf = max(cur_conf, conf_i)
        to_ts = max(pd.Timestamp(df.iloc[-1]["ts"]), end_ts)
        spans.append(
            {
                "from": int(start.timestamp()),
                "to": int(to_ts.timestamp()),
                "gate_on": cur_gate,
                "confidence": cur_conf,
            }
        )

    points = [
        {
            "time": int(pd.Timestamp(r["ts"]).timestamp()),
            "confidence": float(r["confidence"]),
            "gate_on": int(r["gate_on"]),
            "regime_state": str(r.get("regime_state") or ""),
        }
        for _, r in df.iterrows()
    ]
    latest = points[-1] if points else None
    return {"spans": spans, "points": points, "latest": latest}
