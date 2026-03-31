from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from quant.execution.CHOPgate import (
    _adx,
    _choppiness,
    _efficiency_ratio,
    _fit_thresholds_live,
)
from quant.execution.event_store import (
    load_live_renko_bricks_from_postgres,
    upsert_daily_gate_history,
)


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _gate_symbol(symbol: Optional[str] = None) -> str:
    return str(symbol or os.getenv("LIVE_GATE_SYMBOL", os.getenv("LIVE_SYMBOL", "SOL-USDT"))).strip().upper()


def _canon_symbol(symbol: str) -> str:
    return "".join(ch for ch in str(symbol or "").upper() if ch.isalnum())


def _primary_mode() -> str:
    return str(os.getenv("LIVE_GATE_PRIMARY", "off")).strip().lower()


def _compute_primary_gate(
    gate_countertrend_on: int,
    gate_trend_on: int,
) -> tuple[int, int, str]:
    primary = _primary_mode()
    if primary in ("on", "countertrend", "flip"):
        gate_on = int(gate_countertrend_on)
    else:
        gate_on = int(gate_trend_on)
    return gate_on, int(1 - gate_on), primary


def publish_latest_daily_gate_snapshot(state: dict[str, object], symbol: Optional[str] = None) -> bool:
    redis_url = str(os.getenv("REDIS_URL", "")).strip()
    if not redis_url:
        return False
    try:
        import redis as redis_lib

        canon = _canon_symbol(_gate_symbol(symbol))
        key = f"gate:{canon}:latest"
        payload = dict(state)
        payload.setdefault("upstream_source", str(state.get("source") or "postgres_daily_gate"))
        payload["source"] = "redis"
        r = redis_lib.from_url(redis_url, decode_responses=True)
        r.set(key, json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))
        return True
    except Exception:
        return False


def _hysteresis_high_is_good_series(x: pd.Series, *, on_th: float, off_th: float) -> pd.Series:
    state = False
    out: list[bool] = []
    for raw in x.values:
        v = float(raw) if pd.notna(raw) else np.nan
        if np.isnan(v):
            out.append(state)
            continue
        if (not state) and v >= on_th:
            state = True
        elif state and v <= off_th:
            state = False
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def _hysteresis_low_is_good_series(
    x: pd.Series,
    *,
    on_th: float,
    off_th: float,
    start_on: bool = True,
) -> pd.Series:
    state = bool(start_on)
    out: list[bool] = []
    for raw in x.values:
        v = float(raw) if pd.notna(raw) else np.nan
        if np.isnan(v):
            out.append(state)
            continue
        if state and v >= off_th:
            state = False
        elif (not state) and v <= on_th:
            state = True
        out.append(state)
    return pd.Series(out, index=x.index, dtype="bool")


def _compute_chop_gate_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "ts" not in work.columns:
        raise ValueError("missing_ts_column")
    need = {"open", "high", "low", "close"}
    missing = need - set(work.columns)
    if missing:
        raise ValueError(f"missing_renko_columns:{sorted(missing)}")

    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if len(work) < max(int(os.getenv("LIVE_GATE_ER_LEN", "40")) + 5, 100):
        raise ValueError("not_enough_renko_rows")

    work["CHOP"] = _choppiness(work, int(os.getenv("LIVE_GATE_CHOP_LEN", "14")))
    work["ADX"] = _adx(work, int(os.getenv("LIVE_GATE_ADX_LEN", "14")))
    work["ER"] = _efficiency_ratio(work, int(os.getenv("LIVE_GATE_ER_LEN", "40")))

    hist = work.dropna(subset=["CHOP", "ADX", "ER"]).reset_index(drop=True)
    if hist.empty:
        raise ValueError("no_finite_gate_indicators")
    th = _fit_thresholds_live(hist)

    chop_ok = _hysteresis_high_is_good_series(work["CHOP"], on_th=float(th["chop_on"]), off_th=float(th["chop_off"]))
    adx_ok = _hysteresis_low_is_good_series(work["ADX"], on_th=float(th["adx_on"]), off_th=float(th["adx_off"]), start_on=True)
    er_ok = _hysteresis_low_is_good_series(work["ER"], on_th=float(th["er_on"]), off_th=float(th["er_off"]), start_on=True)

    base_2of3 = ((chop_ok.astype(int) + adx_ok.astype(int) + er_ok.astype(int)) >= 2).astype(int)
    base_3of3 = ((chop_ok.astype(int) + adx_ok.astype(int) + er_ok.astype(int)) >= 3).astype(int)

    return pd.DataFrame(
        {
            "ts": work["ts"],
            "CHOP": work["CHOP"],
            "ADX": work["ADX"],
            "ER": work["ER"],
            "gate_chop_ok": chop_ok.astype(int),
            "gate_adx_ok": adx_ok.astype(int),
            "gate_er_ok": er_ok.astype(int),
            "gate_base_2of3": base_2of3,
            "gate_base_3of3": base_3of3,
            "chop_on": float(th["chop_on"]),
            "chop_off": float(th["chop_off"]),
            "adx_on": float(th["adx_on"]),
            "adx_off": float(th["adx_off"]),
            "er_on": float(th["er_on"]),
            "er_off": float(th["er_off"]),
        }
    )


def _load_renko_input_frame(
    *,
    symbol: str,
    input_path: str | Path,
) -> pd.DataFrame:
    end_ts = pd.Timestamp.now("UTC")
    start_ts = end_ts - pd.Timedelta(days=int(os.getenv("LIVE_RENKO_RETENTION_DAYS", "30")))
    try:
        pg_df = load_live_renko_bricks_from_postgres(
            symbol=symbol,
            start_ts=start_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_ts=end_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        if pg_df is not None and not pg_df.empty:
            return pg_df
    except Exception:
        pass

    src_path = Path(input_path)
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    return _load_frame(src_path)


def build_daily_gate_artifacts(
    df: pd.DataFrame,
    *,
    ts_col: str = "ts",
    on_source_col: str = "gate_base_2of3",
    off_source_col: Optional[str] = None,
    on_output_col: str = "gate_on_2of3",
    off_output_col: str = "gate_off_2of3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    if ts_col not in work.columns:
        raise ValueError(f"missing_ts_column:{ts_col}")
    if on_source_col not in work.columns:
        raise ValueError(f"missing_on_source_col:{on_source_col}")

    work[ts_col] = pd.to_datetime(work[ts_col], utc=True, errors="coerce")
    work = work.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    work["day"] = work[ts_col].dt.floor("D")
    work[on_source_col] = pd.to_numeric(work[on_source_col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    daily_on = work.groupby("day", as_index=False).tail(1).copy()
    daily_on = daily_on[["day", on_source_col]].rename(columns={"day": "ts", on_source_col: on_output_col})
    daily_on["ts"] = pd.to_datetime(daily_on["ts"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
    daily_on[on_output_col] = pd.to_numeric(daily_on[on_output_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    daily_on = daily_on.reset_index(drop=True)

    if off_source_col:
        if off_source_col not in work.columns:
            raise ValueError(f"missing_off_source_col:{off_source_col}")
        work[off_source_col] = pd.to_numeric(work[off_source_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        daily_off = work.groupby("day", as_index=False).tail(1).copy()
        daily_off = daily_off[["day", off_source_col]].rename(columns={"day": "ts", off_source_col: off_output_col})
        daily_off["ts"] = pd.to_datetime(daily_off["ts"], utc=True).dt.strftime("%Y-%m-%dT00:00:00Z")
        daily_off[off_output_col] = pd.to_numeric(daily_off[off_output_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        daily_off = daily_off.reset_index(drop=True)
    else:
        daily_off = pd.DataFrame(
            {
                "ts": daily_on["ts"],
                off_output_col: (1 - daily_on[on_output_col].astype(int)).astype(int),
            }
        )

    return daily_on, daily_off


def write_daily_gate_artifacts(
    *,
    input_path: str | Path,
    out_on_path: str | Path | None = None,
    out_off_path: str | Path | None = None,
    symbol: Optional[str] = None,
    persist_to_postgres: bool = True,
    publish_latest: bool = True,
    ts_col: str = "ts",
    on_source_col: str = "gate_base_2of3",
    off_source_col: Optional[str] = None,
    on_output_col: str = "gate_on_2of3",
    off_output_col: str = "gate_off_2of3",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sym = _gate_symbol(symbol)
    src_df = _load_renko_input_frame(symbol=sym, input_path=input_path)
    if on_source_col not in src_df.columns:
        src_df = _compute_chop_gate_frame(src_df)

    on_df, off_df = build_daily_gate_artifacts(
        src_df,
        ts_col=ts_col,
        on_source_col=on_source_col,
        off_source_col=off_source_col,
        on_output_col=on_output_col,
        off_output_col=off_output_col,
    )

    if persist_to_postgres:
        for on_row, off_row in zip(on_df.to_dict("records"), off_df.to_dict("records")):
            gate_countertrend_on = int(on_row[on_output_col])
            gate_trend_on = int(off_row[off_output_col])
            gate_on, gate_off, primary = _compute_primary_gate(gate_countertrend_on, gate_trend_on)
            upsert_daily_gate_history(
                {
                    "ts": on_row["ts"],
                    "symbol": sym,
                    "gate_on": gate_on,
                    "gate_off": gate_off,
                    "gate_countertrend_on": gate_countertrend_on,
                    "gate_trend_on": gate_trend_on,
                    "source": "postgres_daily_gate",
                    "payload_json": {
                        "primary": primary,
                        "source_kind": "chop_adx_er_2of3",
                        "gate_on_col": on_output_col,
                        "gate_off_col": off_output_col,
                    },
                }
            )

    if publish_latest and not on_df.empty and not off_df.empty:
        last_on = on_df.iloc[-1]
        last_off = off_df.iloc[-1]
        gate_countertrend_on = int(last_on[on_output_col])
        gate_trend_on = int(last_off[off_output_col])
        gate_on, gate_off, primary = _compute_primary_gate(gate_countertrend_on, gate_trend_on)
        publish_latest_daily_gate_snapshot(
            {
                "ts": str(last_on["ts"]),
                "gate_on": gate_on,
                "gate_off": gate_off,
                "source": "postgres_daily_gate",
                "primary": primary,
                "gate_countertrend_on": gate_countertrend_on,
                "gate_trend_on": gate_trend_on,
                "gate_on_ts": str(last_on["ts"]),
                "gate_off_ts": str(last_off["ts"]),
                "gate_on_age_sec": 0.0,
                "gate_off_age_sec": 0.0,
            },
            symbol=sym,
        )

    if out_on_path is not None and out_off_path is not None:
        out_on = Path(out_on_path)
        out_off = Path(out_off_path)
        out_on.parent.mkdir(parents=True, exist_ok=True)
        out_off.parent.mkdir(parents=True, exist_ok=True)
        on_df.to_csv(out_on, index=False)
        off_df.to_csv(out_off, index=False)
    return on_df, off_df
