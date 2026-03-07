from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.execution.kucoin_futures import KucoinFuturesBroker, list_fills
from quant.regime import RegimeStore

_LAST_REFRESH_TS: Optional[pd.Timestamp] = None
_LAST_REFRESH_ERROR: Optional[str] = None
_LAST_FILLS_REFRESH_TS: Optional[pd.Timestamp] = None
_LAST_FILLS_REFRESH_ERROR: Optional[str] = None


def _to_ts_iso(ts_like: Any) -> Optional[str]:
    ts = pd.to_datetime(ts_like, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.isoformat()


def _fill_client_oid_prefixes() -> List[str]:
    raw = str(os.getenv("DASHBOARD_FILLS_CLIENT_OID_PREFIXES", "") or "").strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _fill_row_allowed(client_oid: str) -> bool:
    """
    Optional filter to isolate fills by client_oid prefixes.
    If no prefixes configured, all rows are allowed.
    """
    prefixes = _fill_client_oid_prefixes()
    if not prefixes:
        return True
    cid = str(client_oid or "").strip()
    if not cid:
        return _truthy(os.getenv("DASHBOARD_FILLS_INCLUDE_EMPTY_CLIENT_OID", "0"))
    return any(cid.startswith(p) for p in prefixes)


def _epoch_seconds_from_any(v: Any) -> Optional[int]:
    """
    Parse epoch-like timestamps with mixed precision (s / ms / us / ns) or ISO strings.
    """
    if v is None:
        return None
    if isinstance(v, pd.Timestamp):
        ts = pd.Timestamp(v)
        return int(ts.timestamp()) if pd.notna(ts) else None
    if isinstance(v, (int, float)):
        try:
            x = float(v)
        except Exception:
            return None
        if not (x > 0):
            return None
        # Epoch magnitude in 2026:
        # s  ~1e9, ms ~1e12, us ~1e15, ns ~1e18
        if x >= 1e18:
            return int(x / 1e9)   # ns -> s
        if x >= 1e15:
            return int(x / 1e6)   # us -> s
        if x >= 1e12:
            return int(x / 1e3)   # ms -> s
        return int(x)             # s
    s = str(v).strip()
    if not s:
        return None
    try:
        return _epoch_seconds_from_any(float(s))
    except Exception:
        pass
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return int(pd.Timestamp(ts).timestamp())


def _live_default(rel_path: str) -> str:
    """Prefer Railway volume (/data/live) when available."""
    if Path("/data").exists():
        return str(Path("/data/live") / rel_path)
    return str(Path("data/live") / rel_path)


def _env_path(name: str, default_value: str) -> Path:
    return Path(os.getenv(name, default_value))


def _read_trades_df() -> pd.DataFrame:
    p = _env_path("DASHBOARD_TRADES_PARQUET", _live_default("trades.parquet"))
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()


def _read_fills_df() -> pd.DataFrame:
    p = _env_path("DASHBOARD_FILLS_PARQUET", _live_default("fills_cache.parquet"))
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    need = {"time", "side", "size", "price"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()
    df = df.copy()
    # Normalize optional metadata columns used for richer reason mapping.
    if "client_oid" not in df.columns and "clientOid" in df.columns:
        df["client_oid"] = df["clientOid"]
    if "order_id" not in df.columns and "orderId" in df.columns:
        df["order_id"] = df["orderId"]
    if "reduce_only" not in df.columns and "reduceOnly" in df.columns:
        df["reduce_only"] = df["reduceOnly"]
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["side"] = df["side"].astype(str).str.lower()
    df["size"] = pd.to_numeric(df["size"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    if "client_oid" in df.columns:
        df["client_oid"] = df["client_oid"].where(df["client_oid"].notna(), "").astype(str)
    if "order_id" in df.columns:
        df["order_id"] = df["order_id"].where(df["order_id"].notna(), "").astype(str)
    if "reduce_only" in df.columns:
        # Keep nullable/bool semantics when present.
        df["reduce_only"] = df["reduce_only"].astype("boolean")
    df = df.dropna(subset=["time", "side", "size", "price"])
    df = df[df["size"] > 0].sort_values("time").reset_index(drop=True)
    return df


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _read_renko_df() -> pd.DataFrame:
    p = _env_path("DASHBOARD_RENKO_PARQUET", _live_default("renko_latest.parquet"))
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
    global _LAST_REFRESH_TS, _LAST_REFRESH_ERROR
    if not _truthy(os.getenv("DASHBOARD_RENKO_AUTO_REFRESH_ON_READ", "1")):
        return existing_df
    now = pd.Timestamp.now("UTC")
    stale_min = int(os.getenv("DASHBOARD_RENKO_STALE_MIN", "1"))
    refresh_cooldown_sec = int(os.getenv("DASHBOARD_RENKO_REFRESH_COOLDOWN_SEC", "15"))
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
        info = refresh_renko_cache(
            symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
            box=float(os.getenv("DASHBOARD_RENKO_BOX", "0.1")),
            days_back=int(os.getenv("DASHBOARD_RENKO_DAYS_BACK", "14")),
            step_hours=int(os.getenv("DASHBOARD_RENKO_STEP_HOURS", "6")),
            out_parquet=str(_env_path("DASHBOARD_RENKO_PARQUET", _live_default("renko_latest.parquet"))),
        )
        if not bool(info.get("ok", False)):
            _LAST_REFRESH_ERROR = str(info.get("reason") or info.get("error") or "refresh_not_ok")
            return existing_df
        _LAST_REFRESH_TS = now
        _LAST_REFRESH_ERROR = None
    except Exception as e:
        _LAST_REFRESH_ERROR = f"refresh_failed:{e}"
        return existing_df
    return _read_renko_df()


def load_renko_bars(max_points: int = 5000) -> List[Dict[str, Any]]:
    df = _read_renko_df()
    if df.empty:
        return []
    df = df.tail(int(max(1, max_points)))
    out: List[Dict[str, Any]] = []
    last_t = -1
    for _, r in df.iterrows():
        t_i = int(pd.Timestamp(r["ts"]).timestamp())
        # Ensure strictly increasing chart times; Renko can emit multiple bricks
        # on one minute, and second-resolution APIs otherwise collide visually.
        if t_i <= last_t:
            t_i = last_t + 1
        last_t = t_i
        out.append(
            {
                "time": t_i,
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
            }
        )
    return out


def load_renko_health() -> Dict[str, Any]:
    df = _read_renko_df()
    if df.empty:
        return {
            "ok": False,
            "bars": 0,
            "last_ts": None,
            "age_sec": None,
            "last_refresh_ts": _LAST_REFRESH_TS.isoformat() if _LAST_REFRESH_TS is not None else None,
            "last_refresh_error": _LAST_REFRESH_ERROR,
        }
    now = pd.Timestamp.now("UTC")
    last_ts = pd.Timestamp(df["ts"].iloc[-1])
    age_sec = float(max(0.0, (now - last_ts).total_seconds()))
    return {
        "ok": True,
        "bars": int(len(df)),
        "last_ts": last_ts.isoformat(),
        "age_sec": age_sec,
        "last_refresh_ts": _LAST_REFRESH_TS.isoformat() if _LAST_REFRESH_TS is not None else None,
        "last_refresh_error": _LAST_REFRESH_ERROR,
    }


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
    df = _read_trades_df()
    if df.empty:
        return []
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


def load_trade_segments(max_points: int = 2000) -> List[Dict[str, Any]]:
    """
    Return entry->exit line segments for closed trades.
    Color is green for positive PnL, red for negative PnL.
    """
    df = _read_trades_df()
    if df.empty:
        return []

    if "entry_ts" not in df.columns and "ts" in df.columns:
        df = df.rename(columns={"ts": "entry_ts"})
    if "entry_ts" not in df.columns or "exit_ts" not in df.columns:
        return []

    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["entry_ts", "exit_ts"]).sort_values("entry_ts").tail(int(max(1, max_points)))

    entry_candidates = ["entry_px", "entry_price", "price_entry", "entry"]
    exit_candidates = ["exit_px", "exit_price", "price_exit", "exit"]
    entry_col = next((c for c in entry_candidates if c in df.columns), None)
    exit_col = next((c for c in exit_candidates if c in df.columns), None)
    if not entry_col or not exit_col:
        return []

    side_col = "side" if "side" in df.columns else None
    pnl_cols = [c for c in ("pnl_pct", "pnl", "pnl_abs", "net_pnl") if c in df.columns]
    pnl_col = pnl_cols[0] if pnl_cols else None

    segs: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        try:
            epx = float(r[entry_col])
            xpx = float(r[exit_col])
        except Exception:
            continue
        if not pd.notna(epx) or not pd.notna(xpx):
            continue

        side = int(r[side_col]) if side_col and pd.notna(r.get(side_col)) else 1
        if pnl_col and pd.notna(r.get(pnl_col)):
            pnl_positive = float(r[pnl_col]) >= 0.0
        else:
            # Fallback from direction-aware move.
            pnl_positive = ((xpx - epx) * (1 if side >= 0 else -1)) >= 0.0

        segs.append(
            {
                "from_time": int(pd.Timestamp(r["entry_ts"]).timestamp()),
                "to_time": int(pd.Timestamp(r["exit_ts"]).timestamp()),
                "from_price": float(epx),
                "to_price": float(xpx),
                "positive": bool(pnl_positive),
                "color": "#2ecc71" if pnl_positive else "#f7768e",
            }
        )
    return segs


def _refresh_fills_cache_if_needed(symbol: str, fills_path: Path) -> None:
    """
    Refresh fills cache from KuCoin with cooldown to avoid API spam.
    """
    global _LAST_FILLS_REFRESH_TS, _LAST_FILLS_REFRESH_ERROR
    if not _truthy(os.getenv("DASHBOARD_FILLS_AUTO_REFRESH_ON_READ", "1")):
        return
    now = pd.Timestamp.now("UTC")
    cooldown_sec = int(os.getenv("DASHBOARD_FILLS_REFRESH_COOLDOWN_SEC", "20"))
    if _LAST_FILLS_REFRESH_TS is not None and (now - _LAST_FILLS_REFRESH_TS) < pd.Timedelta(seconds=max(1, cooldown_sec)):
        return

    fetch_limit = int(os.getenv("DASHBOARD_FILLS_FETCH_LIMIT", "200"))
    try:
        rows = list_fills(symbol=symbol, limit=int(max(10, fetch_limit)))
        norm_rows: List[Dict[str, Any]] = []
        for r in rows:
            try:
                side = str(r.get("side", "")).lower()
                sz = float(r.get("size", 0) or 0)
                px = float(r.get("price", 0) or 0)
                t_raw = r.get("createdAt") or r.get("tradeTime") or r.get("ts")
                t_sec = _epoch_seconds_from_any(t_raw)
                if t_sec is None:
                    continue
                client_oid = str(r.get("clientOid") or r.get("client_oid") or "").strip()
                if not _fill_row_allowed(client_oid):
                    continue
                order_id = str(r.get("orderId") or r.get("order_id") or "").strip()
                reduce_only = bool(r.get("reduceOnly", r.get("reduce_only", False)))
            except Exception:
                continue
            norm_rows.append(
                {
                    "time": int(t_sec),
                    "side": side,
                    "size": float(sz),
                    "price": float(px),
                    "client_oid": client_oid or None,
                    "order_id": order_id or None,
                    "reduce_only": reduce_only,
                }
            )

        if norm_rows:
            fills_path.parent.mkdir(parents=True, exist_ok=True)
            fresh_df = pd.DataFrame(norm_rows)
            if fills_path.exists():
                try:
                    old_df = pd.read_parquet(fills_path)
                    all_df = pd.concat([old_df, fresh_df], ignore_index=True)
                except Exception:
                    all_df = fresh_df
            else:
                all_df = fresh_df
            dedupe_cols = [c for c in ("time", "side", "size", "price", "order_id", "client_oid") if c in all_df.columns]
            all_df = all_df.drop_duplicates(subset=dedupe_cols, keep="last").sort_values("time")
            all_df.to_parquet(fills_path, index=False)
        _LAST_FILLS_REFRESH_TS = now
        _LAST_FILLS_REFRESH_ERROR = None
    except Exception as e:
        _LAST_FILLS_REFRESH_TS = now
        _LAST_FILLS_REFRESH_ERROR = f"fills_refresh_failed:{e}"


def load_live_fill_markers(symbol: str, limit: int = 100, start_ts: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Build chart markers from live KuCoin fills as a fallback/augmentation
    when local trades parquet is incomplete.
    """
    fills_path = _env_path("DASHBOARD_FILLS_PARQUET", _live_default("fills_cache.parquet"))
    _refresh_fills_cache_if_needed(symbol=symbol, fills_path=fills_path)

    if fills_path.exists():
        try:
            src_df = pd.read_parquet(fills_path)
        except Exception:
            src_df = pd.DataFrame()
    else:
        src_df = pd.DataFrame()
    if src_df.empty:
        return []

    if start_ts is not None:
        src_df = src_df[pd.to_numeric(src_df["time"], errors="coerce") >= int(start_ts)]
    src_df = src_df.sort_values("time").tail(int(max(1, limit)))
    if "client_oid" in src_df.columns:
        src_df = src_df[src_df["client_oid"].map(lambda x: _fill_row_allowed(str(x or "")))]

    out: List[Dict[str, Any]] = []
    for _, r in src_df.iterrows():
        side = str(r.get("side", "")).lower()
        sz = float(r.get("size", 0) or 0)
        px = float(r.get("price", 0) or 0)
        out.append(
            {
                "time": int(r.get("time", 0)),
                "position": "belowBar" if side == "buy" else "aboveBar",
                "shape": "arrowUp" if side == "buy" else "arrowDown",
                "color": "#2ecc71" if side == "buy" else "#f7768e",
                "text": f"fill {side} {sz:g} @ {px:.3f}",
            }
        )
    out = sorted(out, key=lambda x: int(x["time"]))
    return out


def load_fills_cache_rows(max_points: int = 500, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """Expose normalized fills cache rows for diagnostics/UI."""
    fills_path = _env_path("DASHBOARD_FILLS_PARQUET", _live_default("fills_cache.parquet"))
    sym = str(symbol or os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"))
    _refresh_fills_cache_if_needed(symbol=sym, fills_path=fills_path)
    df = _read_fills_df()
    if df.empty:
        return []
    df = df.sort_values("time").tail(int(max(1, max_points)))

    expected_by_time: List[Dict[str, Any]] = []
    expected_by_client_oid: Dict[str, str] = {}
    exp = load_latest_expected_entry()
    # Build richer expected-event map from expected_trades.jsonl if available.
    p_exp = _env_path("DASHBOARD_EXPECTED_TRADES_JSONL", _live_default("expected_trades.jsonl"))
    if p_exp.exists():
        try:
            with open(p_exp, "r", encoding="utf-8") as f:
                for ln in f:
                    s = ln.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        continue
                    ts = pd.to_datetime(obj.get("ts"), utc=True, errors="coerce")
                    if pd.isna(ts):
                        continue
                    note = str(obj.get("note", "") or "")
                    action = str(obj.get("action", "") or "").lower()
                    reason = action
                    if "event=" in note:
                        try:
                            reason = note.split("event=", 1)[1].split()[0].strip()
                        except Exception:
                            reason = action
                    client_oid = str(obj.get("client_oid") or obj.get("clientOid") or "").strip()
                    expected_by_time.append(
                        {
                            "time": int(pd.Timestamp(ts).timestamp()),
                            "reason": reason or action or "unknown",
                            "client_oid": client_oid or None,
                        }
                    )
                    if client_oid:
                        expected_by_client_oid[client_oid] = reason or action or "unknown"
        except Exception:
            expected_by_time = []
            expected_by_client_oid = {}
    if not expected_by_time and exp is not None:
        expected_by_time = [{"time": int(exp["entry_time"]), "reason": "entry"}]

    expected_by_time = sorted(expected_by_time, key=lambda x: int(x["time"]))

    def infer_reason(fill_ts: int, fill_client_oid: Optional[str]) -> str:
        cid = str(fill_client_oid or "").strip()
        if cid and cid in expected_by_client_oid:
            return str(expected_by_client_oid[cid])
        if not expected_by_time:
            return "-"
        best = None
        best_dt = 10**12
        for e in expected_by_time:
            dt = abs(int(e["time"]) - int(fill_ts))
            if dt < best_dt:
                best_dt = dt
                best = e
        # Only map if reasonably close in time.
        if best is None or best_dt > 180:
            return "-"
        return str(best.get("reason") or "-")

    out: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        ts_i = int(r["time"])
        ts = pd.to_datetime(ts_i, unit="s", utc=True, errors="coerce")
        dt_utc = ts.strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(ts) else "-"
        fill_client_oid = str(r.get("client_oid", "") or "").strip()
        fill_order_id = str(r.get("order_id", "") or "").strip()
        fill_reduce_only = None
        if "reduce_only" in df.columns:
            try:
                fill_reduce_only = bool(r.get("reduce_only"))
            except Exception:
                fill_reduce_only = None
        out.append(
            {
                "time": ts_i,
                "time_utc": dt_utc,
                "side": str(r["side"]),
                "size": float(r["size"]),
                "price": float(r["price"]),
                "reason": infer_reason(ts_i, fill_client_oid),
                "client_oid": fill_client_oid or None,
                "order_id": fill_order_id or None,
                "reduce_only": fill_reduce_only,
            }
        )
    return out


def load_real_equity_history(max_points: int = 500) -> Dict[str, Any]:
    """
    Load/refresh realized account-equity history in USDT.
    Uses periodic snapshots from KuCoin account balance.
    """
    p = _env_path("DASHBOARD_EQUITY_PARQUET", _live_default("equity_history.parquet"))
    refresh_sec = int(os.getenv("DASHBOARD_EQUITY_REFRESH_SEC", "60"))
    currency = os.getenv("DASHBOARD_EQUITY_CCY", "USDT")

    if p.exists():
        try:
            df = pd.read_parquet(p)
        except Exception:
            df = pd.DataFrame(columns=["time", "equity"])
    else:
        df = pd.DataFrame(columns=["time", "equity"])

    if not df.empty:
        df["time"] = pd.to_numeric(df["time"], errors="coerce")
        df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
        df = df.dropna(subset=["time", "equity"]).sort_values("time").drop_duplicates(subset=["time"], keep="last")

    key = os.getenv("KUCOIN_FUTURES_API_KEY", "").strip()
    sec = os.getenv("KUCOIN_FUTURES_API_SECRET", "").strip()
    pp = os.getenv("KUCOIN_FUTURES_PASSPHRASE", "").strip()
    now_sec = int(pd.Timestamp.now("UTC").timestamp())
    can_refresh = bool(key and sec and pp)
    stale = True
    if not df.empty:
        last_t = int(df.iloc[-1]["time"])
        stale = (now_sec - last_t) >= max(5, refresh_sec)

    if can_refresh and stale:
        try:
            broker = KucoinFuturesBroker(api_key=key, api_secret=sec, passphrase=pp)
            bal = broker.get_account_balance(currency=currency)
            eq = float(bal.get("equity", 0.0) or 0.0)
            if eq > 0:
                snap = pd.DataFrame([{"time": now_sec, "equity": eq}])
                df = pd.concat([df, snap], ignore_index=True) if not df.empty else snap
                df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last")
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(p, index=False)
        except Exception:
            pass

    if df.empty:
        return {"points": [], "source": "none"}

    df = df.sort_values("time").tail(int(max(1, max_points))).reset_index(drop=True)
    points = [{"time": int(r["time"]), "equity": float(r["equity"])} for _, r in df.iterrows()]
    return {"points": points, "source": "kucoin_equity_snapshots"}


def load_kraken_metrics() -> Dict[str, Any]:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            import redis as redis_lib
            r = redis_lib.from_url(redis_url, decode_responses=True)
            raw = r.get("kraken:metrics:latest")
            if raw:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass

    p = _env_path("KRAKEN_METRICS_JSON", _live_default("kraken/metrics.json"))
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def load_kraken_equity_history(max_points: int = 500) -> Dict[str, Any]:
    redis_url = os.getenv("REDIS_URL", "").strip()
    if redis_url:
        try:
            import redis as redis_lib
            r = redis_lib.from_url(redis_url, decode_responses=True)
            raw = r.get("kraken:equity:latest")
            if raw:
                obj = json.loads(raw)
                ts_i = _epoch_seconds_from_any(obj.get("ts"))
                eq = pd.to_numeric(obj.get("equity_usd"), errors="coerce")
                if ts_i is not None and pd.notna(eq):
                    return {
                        "points": [{"time": int(ts_i), "equity": float(eq)}],
                        "source": "kraken_equity_redis_latest",
                    }
        except Exception:
            pass

    p = _env_path("KRAKEN_EQUITY_CSV", _live_default("kraken/equity.csv"))
    if not p.exists():
        return {"points": [], "source": "none"}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {"points": [], "source": "none"}
    if df.empty:
        return {"points": [], "source": "none"}

    df = df.copy()
    if "ts" not in df.columns:
        for c in ("time", "timestamp", "datetime"):
            if c in df.columns:
                df["ts"] = df[c]
                break
    if "equity_usd" not in df.columns:
        for c in ("equity", "portfolio_value", "portfolioValue", "value"):
            if c in df.columns:
                df["equity_usd"] = df[c]
                break
    if not {"ts", "equity_usd"}.issubset(set(df.columns)):
        return {"points": [], "source": "none"}

    df["ts"] = df["ts"].map(_epoch_seconds_from_any)
    df["equity_usd"] = pd.to_numeric(df["equity_usd"], errors="coerce")
    df = df.dropna(subset=["ts", "equity_usd"]).sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
    if df.empty:
        return {"points": [], "source": "none"}
    df = df.tail(int(max(1, max_points))).reset_index(drop=True)
    pts = [{"time": int(r["ts"]), "equity": float(r["equity_usd"])} for _, r in df.iterrows()]
    return {"points": pts, "source": "kraken_equity_snapshots_usd"}

def build_combined_equity(
    kucoin_points: List[Dict[str, Any]],
    kraken_points_usd: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Combine KuCoin USDT and Kraken USD equity into one USDT-denominated series."""
    if not kucoin_points and not kraken_points_usd:
        return {"points": [], "source": "none"}

    usdt_per_usd = float(os.getenv("DASHBOARD_USDT_PER_USD", "1.0") or 1.0)

    k1 = pd.DataFrame(kucoin_points or [])
    k2 = pd.DataFrame(kraken_points_usd or [])
    if k1.empty and k2.empty:
        return {"points": [], "source": "none"}

    if k1.empty:
        k1 = pd.DataFrame(columns=["time", "equity"])
    if k2.empty:
        k2 = pd.DataFrame(columns=["time", "equity"])

    k1["time"] = pd.to_numeric(k1.get("time"), errors="coerce")
    k1["equity"] = pd.to_numeric(k1.get("equity"), errors="coerce")
    k1 = k1.dropna(subset=["time", "equity"]).sort_values("time")

    k2["time"] = pd.to_numeric(k2.get("time"), errors="coerce")
    k2["equity"] = pd.to_numeric(k2.get("equity"), errors="coerce") * float(usdt_per_usd)
    k2 = k2.dropna(subset=["time", "equity"]).sort_values("time")

    if k1.empty and k2.empty:
        return {"points": [], "source": "none"}

    t1 = set(int(x) for x in k1["time"].tolist()) if not k1.empty else set()
    t2 = set(int(x) for x in k2["time"].tolist()) if not k2.empty else set()
    all_times = sorted(t1 | t2)
    rows: List[Dict[str, Any]] = []
    for t in all_times:
        e1 = float(k1[k1["time"] <= t]["equity"].iloc[-1]) if (not k1.empty and (k1["time"] <= t).any()) else 0.0
        e2 = float(k2[k2["time"] <= t]["equity"].iloc[-1]) if (not k2.empty and (k2["time"] <= t).any()) else 0.0
        rows.append({"time": int(t), "equity": float(e1 + e2)})
    return {"points": rows, "source": "kucoin_usdt_plus_kraken_usd_to_usdt"}


def load_active_levels() -> Dict[str, Any]:
    p = _env_path("DASHBOARD_LEVELS_JSON", _live_default("execution_state.json"))
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return {}
    return {}


def load_latest_expected_entry() -> Optional[Dict[str, Any]]:
    """
    Best-effort fallback for open-position entry context from expected_trades.jsonl.
    Returns latest entry-like event (entry / exit_flip) if available.
    """
    p = _env_path("DASHBOARD_EXPECTED_TRADES_JSONL", _live_default("expected_trades.jsonl"))
    if not p.exists():
        return None
    rows: List[Dict[str, Any]] = []
    try:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                ln = line.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                action = str(obj.get("action", "")).strip().lower()
                if action not in ("entry", "exit_flip"):
                    continue
                ts = pd.to_datetime(obj.get("ts"), utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                side_raw = str(obj.get("side", "")).strip().lower()
                if side_raw not in ("long", "short"):
                    continue
                px = pd.to_numeric(obj.get("expected_px"), errors="coerce")
                rows.append(
                    {
                        "entry_time": int(pd.Timestamp(ts).timestamp()),
                        "side": side_raw,
                        "entry_price": float(px) if pd.notna(px) else None,
                        "source": "expected_trades_jsonl",
                    }
                )
    except Exception:
        return None
    if not rows:
        return None
    rows = sorted(rows, key=lambda x: int(x["entry_time"]))
    return rows[-1]


def _cluster_fills_df(fills: pd.DataFrame, window_sec: int = 90) -> pd.DataFrame:
    """
    Cluster adjacent same-side fills within a short time window into one execution block.
    Price is size-weighted average, time is last fill timestamp in block.
    """
    if fills.empty:
        return fills
    df = fills.copy().sort_values("time").reset_index(drop=True)
    out_rows: List[Dict[str, Any]] = []
    cur_side: Optional[str] = None
    cur_start_t: Optional[int] = None
    cur_last_t: Optional[int] = None
    cur_qty = 0.0
    cur_notional = 0.0

    def flush() -> None:
        nonlocal cur_side, cur_start_t, cur_last_t, cur_qty, cur_notional
        if cur_side is None or cur_last_t is None or cur_qty <= 0:
            return
        out_rows.append(
            {
                "time": int(cur_last_t),
                "side": str(cur_side),
                "size": float(cur_qty),
                "price": float(cur_notional / cur_qty) if cur_qty > 0 else 0.0,
                "cluster_from": int(cur_start_t or cur_last_t),
                "cluster_to": int(cur_last_t),
            }
        )
        cur_side = None
        cur_start_t = None
        cur_last_t = None
        cur_qty = 0.0
        cur_notional = 0.0

    for _, r in df.iterrows():
        t = int(r["time"])
        side = str(r["side"])
        qty = float(r["size"])
        px = float(r["price"])
        if qty <= 0:
            continue
        if cur_side is None:
            cur_side = side
            cur_start_t = t
            cur_last_t = t
            cur_qty = qty
            cur_notional = qty * px
            continue
        if side == cur_side and (t - int(cur_last_t or t)) <= int(max(1, window_sec)):
            cur_last_t = t
            cur_qty += qty
            cur_notional += qty * px
            continue
        flush()
        cur_side = side
        cur_start_t = t
        cur_last_t = t
        cur_qty = qty
        cur_notional = qty * px
    flush()

    if not out_rows:
        return pd.DataFrame(columns=df.columns)
    out = pd.DataFrame(out_rows)
    return out.sort_values("time").reset_index(drop=True)


def build_trading_diary(max_points: int = 500) -> Dict[str, Any]:
    """
    Build a normalized trading diary online from closed trades parquet;
    fallback to reconstructed closed trades from fills cache.
    """
    out: List[Dict[str, Any]] = []

    df = _read_trades_df()
    if not df.empty:
        if "entry_ts" not in df.columns and "ts" in df.columns:
            df = df.rename(columns={"ts": "entry_ts"})
        if "entry_ts" in df.columns and "exit_ts" in df.columns:
            df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
            df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
            df = df.dropna(subset=["entry_ts", "exit_ts"]).sort_values("exit_ts").tail(int(max(1, max_points)))

            entry_col = next((c for c in ("entry_px", "entry_price", "price_entry", "entry") if c in df.columns), None)
            exit_col = next((c for c in ("exit_px", "exit_price", "price_exit", "exit") if c in df.columns), None)
            pnl_cols = [c for c in ("pnl_pct", "pnl", "pnl_abs", "net_pnl") if c in df.columns]
            pnl_col = pnl_cols[0] if pnl_cols else None
            side_col = "side" if "side" in df.columns else None
            qty_col = next((c for c in ("qty", "size", "contracts") if c in df.columns), None)

            for _, r in df.iterrows():
                epx = float(r[entry_col]) if entry_col and pd.notna(r.get(entry_col)) else None
                xpx = float(r[exit_col]) if exit_col and pd.notna(r.get(exit_col)) else None
                side = int(r[side_col]) if side_col and pd.notna(r.get(side_col)) else 1
                qty = float(r[qty_col]) if qty_col and pd.notna(r.get(qty_col)) else None
                pnl_pct = None
                if pnl_col and pd.notna(r.get(pnl_col)):
                    pnl_pct = float(r[pnl_col])
                    if pnl_col not in ("pnl_pct",) and epx and epx > 0:
                        pnl_pct = pnl_pct / epx * 100.0
                elif epx and xpx and epx > 0:
                    pnl_pct = ((xpx - epx) / epx * 100.0) * (1 if side >= 0 else -1)
                if pnl_pct is None:
                    continue
                out.append(
                    {
                        "id": f"p_{int(pd.Timestamp(r['exit_ts']).timestamp())}_{'L' if side >= 0 else 'S'}",
                        "entry_time": int(pd.Timestamp(r["entry_ts"]).timestamp()),
                        "time": int(pd.Timestamp(r["exit_ts"]).timestamp()),
                        "side": "long" if side >= 0 else "short",
                        "qty": qty,
                        "entry_price": epx,
                        "exit_price": xpx,
                        "pnl_pct": round(float(pnl_pct), 4),
                        "source": "trades_parquet",
                    }
                )
            if out:
                out = sorted(out, key=lambda x: int(x["time"]))[-int(max(1, max_points)) :]
                return {"entries": out, "source": "trades_parquet"}

    fills = _read_fills_df()
    if fills.empty:
        return {"entries": [], "source": "none"}
    cluster_window_sec = int(os.getenv("DASHBOARD_FILLS_CLUSTER_SEC", "90"))
    fills = _cluster_fills_df(fills, window_sec=cluster_window_sec)
    if fills.empty:
        return {"entries": [], "source": "none"}

    pos_qty = 0.0
    avg_entry = 0.0
    pos_open_ts: Optional[int] = None
    events: List[Dict[str, Any]] = []

    for _, r in fills.iterrows():
        t = int(r["time"])
        side = str(r["side"])
        qty = float(r["size"])
        px = float(r["price"])
        signed = qty if side == "buy" else -qty

        if pos_qty == 0 or (pos_qty > 0 and signed > 0) or (pos_qty < 0 and signed < 0):
            new_abs = abs(pos_qty) + abs(signed)
            if new_abs > 0:
                avg_entry = ((abs(pos_qty) * avg_entry) + (abs(signed) * px)) / new_abs
            pos_qty += signed
            if pos_open_ts is None:
                pos_open_ts = t
            continue

        close_qty = min(abs(pos_qty), abs(signed))
        direction = 1.0 if pos_qty > 0 else -1.0
        pnl_per_unit = (px - avg_entry) * direction
        pnl_pct = (pnl_per_unit / avg_entry * 100.0) if avg_entry > 0 else 0.0
        events.append(
            {
                "id": f"f_{t}_{len(events)}",
                "entry_time": int(pos_open_ts or t),
                "time": t,
                "side": "long" if direction > 0 else "short",
                "qty": float(close_qty),
                "entry_price": float(avg_entry) if avg_entry > 0 else None,
                "exit_price": float(px),
                "pnl_pct": round(float(pnl_pct), 4),
                "source": "fills_reconstructed_clustered",
            }
        )

        remainder = abs(signed) - close_qty
        if remainder <= 1e-12:
            pos_qty += signed
            if abs(pos_qty) <= 1e-12:
                pos_qty = 0.0
                avg_entry = 0.0
                pos_open_ts = None
            continue

        pos_qty = remainder if signed > 0 else -remainder
        avg_entry = px
        pos_open_ts = t

    events = sorted(events, key=lambda x: int(x["time"]))[-int(max(1, max_points)) :]
    return {"entries": events, "source": "fills_reconstructed_clustered"}


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


def build_equity_curve(max_points: int = 500) -> Dict[str, Any]:
    """Cumulative equity curve from normalized diary entries."""
    diary = build_trading_diary(max_points=max_points)
    entries = diary.get("entries", [])
    cum = 0.0
    curve: List[Dict[str, Any]] = []
    for e in entries:
        pnl_pct = float(e.get("pnl_pct", 0.0))
        cum += pnl_pct
        curve.append(
            {
                "time": int(e.get("time", 0)),
                "pnl_pct": round(pnl_pct, 4),
                "cum_pct": round(cum, 4),
                "side": e.get("side"),
                "entry_price": e.get("entry_price"),
                "exit_price": e.get("exit_price"),
                "qty": e.get("qty"),
                "source": e.get("source"),
            }
        )
    return {"trades": curve, "source": diary.get("source", "none")}


def build_regime_scores(symbol: str, hours: int = 24 * 14) -> Dict[str, List]:
    """Extract regime_score time series for the gradient band."""
    store = RegimeStore()
    end_ts = pd.Timestamp.now("UTC")
    start_ts = end_ts - pd.Timedelta(hours=int(max(1, hours)))
    rows = store.get_history(symbol=symbol, start_ts=start_ts.isoformat(), end_ts=end_ts.isoformat(), limit=20000)
    if not rows:
        return {"scores": [], "forecast": []}

    scores = []
    for r in rows:
        ts = pd.to_datetime(r.get("ts"), utc=True, errors="coerce")
        rs = pd.to_numeric(r.get("regime_score"), errors="coerce")
        if pd.notna(ts) and pd.notna(rs):
            scores.append({"time": int(ts.timestamp()), "score": round(float(rs), 4)})

    return {"scores": scores, "forecast": []}
