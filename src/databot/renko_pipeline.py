"""Renko data pipeline — fetches 1m klines, builds bricks, writes to Redis + Postgres."""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import redis as redis_lib

from quant.execution.event_store import (
    prune_live_renko_bricks_before,
    upsert_live_renko_bricks,
)
from quant.execution.kucoin_futures import KucoinFuturesBroker, _symbol_to_contract
from quant.features.renko import renko_from_close

from databot.config import DatabotConfig

log = logging.getLogger("databot.renko")

KLINE_PAGE_LIMIT = 200
SAFE_STEP_MINUTES = 180


def _redis_client() -> Optional[redis_lib.Redis]:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    return redis_lib.from_url(url, decode_responses=True)


def _canon_symbol(symbol: str) -> str:
    return str(symbol).upper().replace("-", "")


def _publish_renko_to_redis(
    symbol: str, renko: pd.DataFrame, box: float, cfg: DatabotConfig,
) -> Dict[str, Any]:
    client = _redis_client()
    if client is None:
        return {"ok": False, "reason": "no_redis_url"}

    if renko is None or renko.empty:
        return {"ok": False, "reason": "empty_renko"}

    sym = _canon_symbol(symbol)
    tail = renko.tail(cfg.redis_bars).copy().reset_index(drop=True)
    tail["ts"] = pd.to_datetime(tail["ts"], utc=True, errors="coerce")

    last = tail.iloc[-1]
    ts = pd.Timestamp(last["ts"])

    bars = []
    for rec in tail.to_dict("records"):
        bars.append({
            "ts": pd.Timestamp(rec["ts"]).isoformat(),
            "open": float(rec["open"]),
            "high": float(rec["high"]),
            "low": float(rec["low"]),
            "close": float(rec["close"]),
        })

    payload = {
        "event_id": f"renko:{sym}:{ts.isoformat()}:{len(renko)}",
        "symbol": sym,
        "ts": ts.isoformat(),
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "box": float(box),
        "n_bars": int(len(renko)),
        "lookback_max": int(len(tail)),
        "swing_low_50": float(tail["low"].min()),
        "swing_high_50": float(tail["high"].max()),
        "bars": bars,
    }

    latest_key = f"renko:{sym}:latest"
    stream_key = f"renko:{sym}:events"
    dedupe_key = f"{latest_key}:event_id"

    prev_event_id = client.get(dedupe_key)
    is_new = prev_event_id != payload["event_id"]

    pipe = client.pipeline()
    pipe.set(latest_key, json.dumps(payload, separators=(",", ":")))
    pipe.set(dedupe_key, payload["event_id"])
    if is_new:
        pipe.xadd(
            stream_key,
            {"json": json.dumps(payload, separators=(",", ":"))},
            maxlen=10000,
            approximate=True,
        )
    pipe.execute()

    return {
        "ok": True,
        "latest_key": latest_key,
        "stream_key": stream_key,
        "published_event": bool(is_new),
        "event_id": payload["event_id"],
    }


def _build_renko_ohlc(bricks: pd.DataFrame) -> pd.DataFrame:
    """Build Renko OHLC with unique timestamps from raw bricks."""
    if bricks is None or len(bricks) == 0:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])

    b = bricks.copy().reset_index(drop=True)
    b["_seq"] = range(len(b))
    b["ts"] = pd.to_datetime(b["ts"], utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "ts": b["ts"],
            "open": pd.to_numeric(b["open"], errors="coerce"),
            "high": b[["open", "close"]].max(axis=1),
            "low": b[["open", "close"]].min(axis=1),
            "close": pd.to_numeric(b["close"], errors="coerce"),
            "_seq": b["_seq"],
        }
    ).dropna()
    out = out.sort_values(["ts", "_seq"], kind="mergesort").reset_index(drop=True)
    if len(out) > 1:
        dup = out["ts"].duplicated(keep=False)
        if dup.any():
            grp = out["ts"].astype("int64")
            idx_in_grp = out.groupby(grp).cumcount()
            out["ts"] = out["ts"] + pd.to_timedelta(idx_in_grp, unit="ns")
    out = out.drop(columns=["_seq"], errors="ignore")
    return out.reset_index(drop=True)


def _fetch_1m_close_paged(
    broker: KucoinFuturesBroker,
    symbol: str,
    days_back: int,
    step_hours: int,
) -> pd.DataFrame:
    contract = _symbol_to_contract(symbol)
    now = pd.Timestamp.now("UTC")
    start = now - pd.Timedelta(days=int(max(1, days_back)))
    requested = pd.Timedelta(hours=int(max(1, step_hours)))
    max_safe = pd.Timedelta(minutes=SAFE_STEP_MINUTES)
    step = min(requested, max_safe)

    chunks: List[pd.DataFrame] = []
    cur = start
    while cur < now:
        nxt = min(cur + step, now)
        from_ms = int(cur.timestamp() * 1000)
        to_ms = int(nxt.timestamp() * 1000)
        try:
            rows = broker._req(
                "GET",
                f"/api/v1/kline/query?symbol={contract}&granularity=1&from={from_ms}&to={to_ms}",
            )
        except Exception as exc:
            log.warning("fetch page failed symbol=%s from=%s err=%s", symbol, cur.isoformat(), exc)
            rows = []
        rows = rows if isinstance(rows, list) else []

        parsed: List[Dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, list) or len(r) < 5:
                continue
            try:
                ts_i = int(float(r[0]))
                ts = pd.to_datetime(ts_i, unit="ms" if ts_i > 10**12 else "s", utc=True)
                parsed.append({"ts": ts, "close": float(r[4])})
            except Exception:
                continue

        if parsed:
            cdf = pd.DataFrame(parsed)
            cdf = cdf[(cdf["ts"] >= cur) & (cdf["ts"] < nxt)]
            if len(cdf):
                chunks.append(cdf)
        cur = nxt

    if not chunks:
        return pd.DataFrame(columns=["ts", "close"])

    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    return out


_last_refresh: Dict[str, Dict[str, Any]] = {}


def refresh_renko(symbol: str, cfg: DatabotConfig) -> Dict[str, Any]:
    """Full pipeline for one symbol: fetch klines -> build renko -> write Redis + Postgres."""
    broker = KucoinFuturesBroker()
    close_df = _fetch_1m_close_paged(
        broker,
        symbol=symbol,
        days_back=cfg.renko_days_back,
        step_hours=cfg.renko_step_hours,
    )
    if len(close_df) == 0:
        return {"ok": False, "symbol": symbol, "reason": "no_candles"}

    box = cfg.box_for_symbol(symbol)
    bricks = renko_from_close(close_df[["ts", "close"]], box=box)
    if len(bricks) == 0:
        return {"ok": False, "symbol": symbol, "reason": "no_bricks", "candles": int(len(close_df))}

    renko = _build_renko_ohlc(bricks)

    postgres_info: Dict[str, Any]
    try:
        inserted = upsert_live_renko_bricks(
            symbol=symbol,
            renko=renko,
            source="databot",
            payload_json={
                "box": float(box),
                "days_back": cfg.renko_days_back,
                "step_hours": cfg.renko_step_hours,
            },
        )
        cutoff_ts = (
            pd.Timestamp.now("UTC") - pd.Timedelta(days=cfg.retention_days)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        pruned = prune_live_renko_bricks_before(symbol=symbol, cutoff_ts=cutoff_ts)
        postgres_info = {"ok": True, "inserted_rows": int(inserted), "pruned_rows": int(pruned)}
    except Exception as exc:
        postgres_info = {"ok": False, "error": str(exc)}

    redis_info = _publish_renko_to_redis(symbol=symbol, renko=renko, box=box, cfg=cfg)

    last_close = float(renko["close"].iloc[-1]) if len(renko) else None
    result = {
        "ok": True,
        "symbol": symbol,
        "candles": int(len(close_df)),
        "bricks": int(len(renko)),
        "box": float(box),
        "last_close": last_close,
        "postgres": postgres_info,
        "redis": redis_info,
        "ts": pd.Timestamp.now("UTC").isoformat(),
    }

    _last_refresh[symbol] = result
    return result


def get_last_refresh(symbol: str) -> Optional[Dict[str, Any]]:
    return _last_refresh.get(symbol)
