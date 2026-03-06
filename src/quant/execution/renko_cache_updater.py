from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import redis

from quant.execution.kucoin_futures import KucoinFuturesBroker, _symbol_to_contract
from quant.features.renko import renko_from_close
from quant.utils.log import get_logger, log_throttled

log = get_logger("quant.renko_cache_updater")
KLINE_PAGE_LIMIT = 200
SAFE_STEP_MINUTES = 180  # < 200 to avoid page truncation gaps on 1m candles


def _redis_client() -> Optional[redis.Redis]:
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    return redis.from_url(url, decode_responses=True)


def _redis_key_latest(symbol: str) -> str:
    sym = str(symbol).upper().replace("-", "")
    return f"renko:{sym}:latest"


def _redis_stream_key(symbol: str) -> str:
    sym = str(symbol).upper().replace("-", "")
    return f"renko:{sym}:events"


def _publish_renko_to_redis(symbol: str, renko: pd.DataFrame, box: float) -> Dict[str, Any]:
    client = _redis_client()
    if client is None:
        return {"ok": False, "reason": "no_redis_url"}

    if renko is None or renko.empty:
        return {"ok": False, "reason": "empty_renko"}

    tail_n = 50
    tail = renko.tail(tail_n).copy().reset_index(drop=True)
    tail["ts"] = pd.to_datetime(tail["ts"], utc=True, errors="coerce")

    last = tail.iloc[-1]
    ts = pd.Timestamp(last["ts"])

    bars = []
    for _, r in tail.iterrows():
        bars.append(
            {
                "ts": pd.Timestamp(r["ts"]).isoformat(),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            }
        )

    payload = {
        "event_id": f"renko:{str(symbol).upper().replace('-', '')}:{ts.isoformat()}:{len(renko)}",
        "symbol": str(symbol).upper().replace("-", ""),
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

    latest_key = _redis_key_latest(symbol)
    stream_key = _redis_stream_key(symbol)
    dedupe_key = f"{latest_key}:event_id"

    prev_event_id = client.get(dedupe_key)
    is_new = prev_event_id != payload["event_id"]

    pipe = client.pipeline()
    pipe.set(latest_key, json.dumps(payload, separators=(",", ":")))
    pipe.set(dedupe_key, payload["event_id"])
    if is_new:
        pipe.xadd(stream_key, {"json": json.dumps(payload, separators=(",", ":"))}, maxlen=10000, approximate=True)
    pipe.execute()

    return {
        "ok": True,
        "latest_key": latest_key,
        "stream_key": stream_key,
        "published_event": bool(is_new),
        "event_id": payload["event_id"],
    }


def _build_renko_ohlc(bricks: pd.DataFrame) -> pd.DataFrame:
    """
    Build Renko OHLC for dashboard rendering using real brick timestamps.
    If multiple bricks share the same timestamp, add nanosecond offsets to keep
    chart times unique while preserving chronological order.
    """
    if bricks is None or len(bricks) == 0:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])

    b = bricks.copy().reset_index(drop=True)
    # Preserve original brick emission order for identical timestamps.
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
        except Exception as e:
            log_throttled(
                log,
                logging.WARNING,
                f"renko_cache_fetch_page_failed:{symbol}",
                float(os.getenv("DASHBOARD_LOG_THROTTLE_SEC", "60")),
                "fetch page failed symbol=%s from=%s err=%s",
                symbol,
                cur.isoformat(),
                e,
            )
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


def refresh_renko_cache(
    *,
    symbol: str,
    box: float,
    days_back: int,
    step_hours: int,
    out_parquet: str,
) -> Dict[str, Any]:
    broker = KucoinFuturesBroker()
    close_df = _fetch_1m_close_paged(broker, symbol=symbol, days_back=days_back, step_hours=step_hours)
    if len(close_df) == 0:
        return {"ok": False, "reason": "no_candles"}

    bricks = renko_from_close(close_df[["ts", "close"]], box=float(box))
    if len(bricks) == 0:
        return {"ok": False, "reason": "no_bricks", "candles": int(len(close_df))}

    renko = _build_renko_ohlc(bricks)
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renko.to_parquet(out_path, index=False)

    redis_info = _publish_renko_to_redis(symbol=symbol, renko=renko, box=float(box))

    last_close = float(renko["close"].iloc[-1]) if len(renko) else None
    return {
        "ok": True,
        "symbol": symbol,
        "candles": int(len(close_df)),
        "bricks": int(len(renko)),
        "box": float(box),
        "days_back": int(days_back),
        "step_hours": int(step_hours),
        "step_effective_minutes": int(
            SAFE_STEP_MINUTES if int(step_hours) * 60 > SAFE_STEP_MINUTES else int(step_hours) * 60
        ),
        "last_close": last_close,
        "out": str(out_path),
        "redis": redis_info,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh dashboard Renko cache from KuCoin 1m data")
    p.add_argument("--symbol", default=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"))
    p.add_argument("--box", type=float, default=float(os.getenv("DASHBOARD_RENKO_BOX", "0.1")))
    p.add_argument("--days-back", type=int, default=int(os.getenv("DASHBOARD_RENKO_DAYS_BACK", "14")))
    p.add_argument("--step-hours", type=int, default=int(os.getenv("DASHBOARD_RENKO_STEP_HOURS", "6")))
    p.add_argument("--out-parquet", default=os.getenv("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet"))
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("DASHBOARD_RENKO_POLL_SEC", "60")))
    p.add_argument("--once", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        try:
            info = refresh_renko_cache(
                symbol=str(args.symbol),
                box=float(args.box),
                days_back=int(args.days_back),
                step_hours=int(args.step_hours),
                out_parquet=str(args.out_parquet),
            )
            log_throttled(
                log,
                logging.INFO,
                "renko_cache_refresh_info",
                float(os.getenv("DASHBOARD_LOG_THROTTLE_SEC", "60")),
                "renko cache refresh: %s",
                info,
            )
            print(info)
        except Exception as e:
            log_throttled(
                log,
                logging.WARNING,
                "renko_cache_refresh_failed",
                float(os.getenv("DASHBOARD_LOG_THROTTLE_SEC", "60")),
                "renko cache refresh failed: %s",
                e,
            )
            print({"ok": False, "error": str(e)})

        if args.once:
            break
        time.sleep(max(5.0, float(args.poll_sec)))


if __name__ == "__main__":
    main()