from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quant.execution.event_store import load_latest_daily_gate_from_postgres


def _live_default(rel_path: str) -> str:
    if Path("/data").exists():
        return str(Path("/data") / rel_path)
    return str(Path("data") / rel_path)


def _env_bool(name: str, default: bool = False) -> bool:
    v = str(os.getenv(name, str(default))).strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _wilder_smooth(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / float(n), adjust=False).mean()


def _choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    tr = _true_range(df)
    sum_tr = tr.rolling(n, min_periods=n).sum()
    hh = pd.to_numeric(df["high"], errors="coerce").astype(float).rolling(n, min_periods=n).max()
    ll = pd.to_numeric(df["low"], errors="coerce").astype(float).rolling(n, min_periods=n).min()
    denom = (hh - ll).replace(0.0, np.nan)
    return 100.0 * np.log10(sum_tr / denom) / np.log10(float(n))


def _adx(df: pd.DataFrame, n: int) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)

    up = high.diff()
    down = -low.diff()

    dm_plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    dm_minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

    tr = _true_range(df)
    atr = _wilder_smooth(tr, n)
    sm_plus = _wilder_smooth(dm_plus, n)
    sm_minus = _wilder_smooth(dm_minus, n)

    di_plus = 100.0 * (sm_plus / atr.replace(0.0, np.nan))
    di_minus = 100.0 * (sm_minus / atr.replace(0.0, np.nan))

    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, np.nan)
    return _wilder_smooth(dx, n)


def _efficiency_ratio(df: pd.DataFrame, n: int) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    net = (close - close.shift(n)).abs()
    denom = close.diff().abs().rolling(n, min_periods=n).sum()
    return (net / denom.replace(0.0, np.nan)).clip(0.0, 1.0)


def _quantile_threshold(x: pd.Series, q: float) -> float:
    s = pd.to_numeric(x, errors="coerce").astype(float)
    s = s[np.isfinite(s)]
    if len(s) == 0:
        return float("nan")
    q = max(0.0, min(1.0, float(q)))
    return float(s.quantile(q))


def _hysteresis_high_is_good_last(x: pd.Series, on_th: float, off_th: float) -> tuple[int, int]:
    state = False
    last_val = np.nan
    for raw in x.values:
        v = float(raw) if pd.notna(raw) else np.nan
        last_val = v
        if np.isnan(v):
            continue
        if (not state) and v >= on_th:
            state = True
        elif state and v <= off_th:
            state = False
    return int(state), int(not state)


def _hysteresis_low_is_good_last(x: pd.Series, on_th: float, off_th: float, start_on: bool = True) -> tuple[int, int]:
    state = bool(start_on)
    for raw in x.values:
        v = float(raw) if pd.notna(raw) else np.nan
        if np.isnan(v):
            continue
        if state and v >= off_th:
            state = False
        elif (not state) and v <= on_th:
            state = True
    return int(state), int(not state)


def _publish_gate_state_to_redis(state: Dict[str, Any]) -> None:
    redis_url = str(os.getenv("REDIS_URL", "")).strip()
    if not redis_url:
        return
    try:
        import redis as redis_lib

        symbol = _gate_symbol()
        canon = "".join(ch for ch in symbol if ch.isalnum())
        key = f"gate:{canon}:latest"

        r = redis_lib.from_url(redis_url, decode_responses=True)
        payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"), default=str)
        r.set(key, payload)
    except Exception:
        return


def _gate_symbol() -> str:
    return str(os.getenv("LIVE_GATE_SYMBOL", os.getenv("LIVE_SYMBOL", "SOL-USDT"))).strip().upper()


def _read_live_gate_from_redis() -> Optional[Dict[str, Any]]:
    redis_url = str(os.getenv("REDIS_URL", "")).strip()
    if not redis_url:
        return None
    try:
        import redis as redis_lib

        key = f"gate:{''.join(ch for ch in _gate_symbol() if ch.isalnum())}:latest"
        r = redis_lib.from_url(redis_url, decode_responses=True)
        raw = r.get(key)
        if not raw:
            return None
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None
        out = dict(obj)
        out["source"] = "redis"
        return out
    except Exception:
        return None


def _default_daily_gate_path() -> str:
    return str(Path(_live_default("live/gate_conf/gate_daily.csv")))


def _default_daily_gate_off_path() -> str:
    return str(Path(_live_default("live/gate_conf/gate_daily_off.csv")))


def _load_daily_gate_row(path: str, *, ts_col: str, value_col: str, now_ts: pd.Timestamp) -> Dict[str, Any]:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing_gate_csv:{csv_path}")

    df = pd.read_csv(csv_path)
    if ts_col not in df.columns:
        raise ValueError(f"gate_csv_missing_ts_col:{ts_col}")
    if value_col not in df.columns:
        raise ValueError(f"gate_csv_missing_value_col:{value_col}")

    work = df[[ts_col, value_col]].copy()
    work[ts_col] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    vals = pd.to_numeric(work[value_col], errors="coerce")
    if vals.isna().all():
        vals = work[value_col].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})
    work["gate"] = vals.fillna(0).astype(int).clip(0, 1)
    work = work.dropna(subset=[ts_col]).sort_values(ts_col).drop_duplicates(subset=[ts_col], keep="last").reset_index(drop=True)
    work = work[work[ts_col] <= now_ts].reset_index(drop=True)
    if work.empty:
        raise ValueError("gate_csv_no_applicable_row")

    row = work.iloc[-1]
    row_ts = pd.Timestamp(row[ts_col])
    return {
        "ts": row_ts,
        "gate": int(row["gate"]),
        "age_sec": float((now_ts - row_ts).total_seconds()),
        "path": str(csv_path),
        "value_col": value_col,
    }


def _daily_csv_gate_state(now_ts: pd.Timestamp) -> Dict[str, Any]:
    ts_col = str(os.getenv("GATE_DAILY_TS_COL", "ts")).strip() or "ts"
    on_path = str(os.getenv("GATE_DAILY_PATH", _default_daily_gate_path())).strip()
    on_col = str(os.getenv("GATE_DAILY_COL", "gate_on_2of3")).strip() or "gate_on_2of3"
    off_path = str(os.getenv("GATE_DAILY_OFF_PATH", _default_daily_gate_off_path())).strip()
    off_ts_col = str(os.getenv("GATE_DAILY_OFF_TS_COL", ts_col)).strip() or ts_col
    off_col = str(os.getenv("GATE_DAILY_OFF_COL", "gate_off_2of3")).strip() or "gate_off_2of3"

    on_row = _load_daily_gate_row(on_path, ts_col=ts_col, value_col=on_col, now_ts=now_ts)
    off_row = _load_daily_gate_row(off_path, ts_col=off_ts_col, value_col=off_col, now_ts=now_ts)

    gate_countertrend_on = int(on_row["gate"])
    gate_trend_on = int(off_row["gate"])

    primary = str(os.getenv("LIVE_GATE_PRIMARY", "off")).strip().lower()
    if primary in ("on", "countertrend", "flip"):
        gate_on = gate_countertrend_on
    else:
        gate_on = gate_trend_on
    gate_off = int(1 - gate_on)

    selected_ts = max(pd.Timestamp(on_row["ts"]), pd.Timestamp(off_row["ts"]))
    return {
        "ts": selected_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate_on": int(gate_on),
        "gate_off": int(gate_off),
        "source": "daily_csv",
        "primary": primary,
        "gate_countertrend_on": int(gate_countertrend_on),
        "gate_trend_on": int(gate_trend_on),
        "gate_on_ts": pd.Timestamp(on_row["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate_off_ts": pd.Timestamp(off_row["ts"]).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate_on_age_sec": round(float(on_row["age_sec"]), 1),
        "gate_off_age_sec": round(float(off_row["age_sec"]), 1),
        "gate_on_path": str(on_row["path"]),
        "gate_off_path": str(off_row["path"]),
        "gate_on_col": str(on_row["value_col"]),
        "gate_off_col": str(off_row["value_col"]),
    }


def _build_gate_state_payload(
    *,
    ts: pd.Timestamp,
    gate_countertrend_on: int,
    gate_trend_on: int,
    now_ts: pd.Timestamp,
    source: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    primary = str(os.getenv("LIVE_GATE_PRIMARY", "off")).strip().lower()
    if source == "forced_countertrend":
        gate_on = 1
        primary = "countertrend"
    elif primary in ("on", "countertrend", "flip"):
        gate_on = int(gate_countertrend_on)
    else:
        gate_on = int(gate_trend_on)
    out = {
        "ts": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate_on": int(gate_on),
        "gate_off": int(1 - gate_on),
        "source": source,
        "primary": primary,
        "gate_countertrend_on": int(gate_countertrend_on),
        "gate_trend_on": int(gate_trend_on),
        "gate_on_ts": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate_off_ts": pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate_on_age_sec": round(float((now_ts - pd.Timestamp(ts)).total_seconds()), 1),
        "gate_off_age_sec": round(float((now_ts - pd.Timestamp(ts)).total_seconds()), 1),
    }
    if error:
        out["error"] = error
    return out


def _load_live_renko() -> pd.DataFrame:
    renko_path = Path(
        os.getenv(
            "LIVE_RENKO_PATH",
            os.getenv(
                "DASHBOARD_RENKO_PARQUET",
                _live_default("live/renko_latest.parquet"),
            ),
        )
    )

    if not renko_path.exists():
        raise FileNotFoundError(f"missing_renko:{renko_path}")

    df = pd.read_parquet(renko_path).copy()
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    else:
        df = df.reset_index().rename(columns={"index": "ts"})
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    need = {"ts", "open", "high", "low", "close"}
    missing = sorted(list(need - set(df.columns)))
    if missing:
        raise ValueError(f"renko_missing_cols:{missing}")

    df = df.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts").reset_index(drop=True)
    if df.empty:
        raise ValueError("renko_empty")

    return df


def _fit_thresholds_live(hist: pd.DataFrame) -> Dict[str, float]:
    mode = str(os.getenv("LIVE_GATE_MODE", "rolling_quantiles")).strip().lower()

    if mode == "fixed":
        return {
            "chop_on": _env_float("LIVE_GATE_CHOP_ON", 58.0),
            "chop_off": _env_float("LIVE_GATE_CHOP_OFF", 52.0),
            "adx_on": _env_float("LIVE_GATE_ADX_ON", 18.0),
            "adx_off": _env_float("LIVE_GATE_ADX_OFF", 25.0),
            "er_on": _env_float("LIVE_GATE_ER_ON", 0.30),
            "er_off": _env_float("LIVE_GATE_ER_OFF", 0.40),
        }

    # rolling_quantiles:
    # Settings reference from the historical filename:
    # qch=0.4 qadx=0.6 qer=0.3
    # Hysteresis band is configurable separately.
    q_band = _env_float("LIVE_GATE_Q_BAND", 0.05)

    qch = _env_float("LIVE_GATE_CHOP_Q", 0.40)
    qadx = _env_float("LIVE_GATE_ADX_Q", 0.60)
    qer = _env_float("LIVE_GATE_ER_Q", 0.30)

    chop_on_q = min(1.0, max(0.0, qch))
    chop_off_q = min(1.0, max(0.0, qch - q_band))

    adx_on_q = min(1.0, max(0.0, qadx))
    adx_off_q = min(1.0, max(0.0, qadx + q_band))

    er_on_q = min(1.0, max(0.0, qer))
    er_off_q = min(1.0, max(0.0, qer + q_band))

    return {
        "chop_on": _quantile_threshold(hist["CHOP"], chop_on_q),
        "chop_off": _quantile_threshold(hist["CHOP"], chop_off_q),
        "adx_on": _quantile_threshold(hist["ADX"], adx_on_q),
        "adx_off": _quantile_threshold(hist["ADX"], adx_off_q),
        "er_on": _quantile_threshold(hist["ER"], er_on_q),
        "er_off": _quantile_threshold(hist["ER"], er_off_q),
    }


def get_live_gate_state() -> Dict[str, Any]:
    """
    Live daily CSV gate provider.

    Primary behavior:
    - Read the latest applicable ON/OFF rows from the daily gate CSV artifacts.
    - Expose both countertrend/trend views.
    - Map gate_on/gate_off from those views using LIVE_GATE_PRIMARY.

    If daily CSV data is missing or invalid:
    - default safe OFF
    - surface the CSV error in the payload
    """
    now_ts = pd.Timestamp.now("UTC")
    pg_error = None
    try:
        row = load_latest_daily_gate_from_postgres(
            symbol=_gate_symbol(),
            now_ts=now_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if row:
            out = _build_gate_state_payload(
                ts=pd.Timestamp(row["ts"]),
                gate_countertrend_on=int(row.get("gate_countertrend_on", 0) or 0),
                gate_trend_on=int(row.get("gate_trend_on", 0) or 0),
                now_ts=now_ts,
                source=str(row.get("source") or "postgres_daily_gate"),
            )
            _publish_gate_state_to_redis(out)
            return out
    except Exception as e:
        pg_error = str(e)

    redis_state = _read_live_gate_from_redis()
    if redis_state:
        return redis_state

    out = _build_gate_state_payload(
        ts=now_ts,
        gate_countertrend_on=1,
        gate_trend_on=0,
        now_ts=now_ts,
        source="forced_countertrend",
        error=pg_error,
    )
    _publish_gate_state_to_redis(out)
    return out