from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from quant.execution.kucoin_futures import KucoinFuturesBroker, _symbol_to_contract
from quant.features.renko import renko_from_close
from quant.utils.log import get_logger

log = get_logger("quant.renko_cache_updater")
KLINE_PAGE_LIMIT = 200
SAFE_STEP_MINUTES = 180  # < 200 to avoid page truncation gaps on 1m candles


def _build_renko_ohlc(bricks: pd.DataFrame) -> pd.DataFrame:
    """
    Build Renko OHLC for dashboard rendering using real brick timestamps.
    If multiple bricks share the same timestamp, add nanosecond offsets to keep
    chart times unique while preserving chronological order.
    """
    if bricks is None or len(bricks) == 0:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])

    b = bricks.copy().reset_index(drop=True)
    b["ts"] = pd.to_datetime(b["ts"], utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "ts": b["ts"],
            "open": pd.to_numeric(b["open"], errors="coerce"),
            "high": b[["open", "close"]].max(axis=1),
            "low": b[["open", "close"]].min(axis=1),
            "close": pd.to_numeric(b["close"], errors="coerce"),
        }
    ).dropna()
    out = out.sort_values("ts").reset_index(drop=True)
    if len(out) > 1:
        dup = out["ts"].duplicated(keep=False)
        if dup.any():
            grp = out["ts"].astype("int64")
            idx_in_grp = out.groupby(grp).cumcount()
            out["ts"] = out["ts"] + pd.to_timedelta(idx_in_grp, unit="ns")
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
            log.warning("fetch page failed symbol=%s from=%s err=%s", symbol, cur.isoformat(), e)
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

    last_close = float(renko["close"].iloc[-1]) if len(renko) else None
    return {
        "ok": True,
        "symbol": symbol,
        "candles": int(len(close_df)),
        "bricks": int(len(renko)),
        "box": float(box),
        "days_back": int(days_back),
        "step_hours": int(step_hours),
        "step_effective_minutes": int(SAFE_STEP_MINUTES if int(step_hours) * 60 > SAFE_STEP_MINUTES else int(step_hours) * 60),
        "last_close": last_close,
        "out": str(out_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refresh dashboard Renko cache from KuCoin 1m data")
    p.add_argument("--symbol", default=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"))
    p.add_argument("--box", type=float, default=float(os.getenv("DASHBOARD_RENKO_BOX", "0.1")))
    p.add_argument("--days-back", type=int, default=int(os.getenv("DASHBOARD_RENKO_DAYS_BACK", "14")))
    p.add_argument("--step-hours", type=int, default=int(os.getenv("DASHBOARD_RENKO_STEP_HOURS", "6")))
    p.add_argument("--out-parquet", default=os.getenv("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet"))
    p.add_argument("--poll-sec", type=float, default=float(os.getenv("DASHBOARD_RENKO_POLL_SEC", "300")))
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
            log.info("renko cache refresh: %s", info)
            print(info)
        except Exception as e:
            log.warning("renko cache refresh failed: %s", e)
            print({"ok": False, "error": str(e)})

        if args.once:
            break
        time.sleep(max(5.0, float(args.poll_sec)))


if __name__ == "__main__":
    main()
