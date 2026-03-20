from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from quant.execution.event_store import (
    get_conn,
    insert_equity_snapshot,
    upsert_closed_trade,
)
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


def load_renko_bars(max_points: int = 5000, _df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    df = _df if _df is not None else _read_renko_df()
    if df.empty:
        return []
    df = df.tail(int(max(1, max_points)))
    out: List[Dict[str, Any]] = []
    last_t = -1
    for _, r in df.iterrows():
        t_i = int(pd.Timestamp(r["ts"]).timestamp())
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


def load_renko_health(_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    df = _df if _df is not None else _read_renko_df()
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


def build_fibo_levels(
    max_points: int = 5000,
    lookback: Optional[int] = None,
    _df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    lb = int(lookback or int(os.getenv("LIVE_IMBA_LOOKBACK", "250")))
    lb = max(2, lb)
    df = _df if _df is not None else _read_renko_df()
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


def load_closed_trades_from_postgres(
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
    max_points: int = 5000,
) -> pd.DataFrame:
    try:
        if venue is None and symbol is None:
            sql = """
                select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
                       entry_price, exit_price, pnl_pct, exit_event
                from closed_trades
                order by exit_ts desc
                limit %(limit)s
            """
            params = {"limit": int(max(1, max_points))}
        elif venue is None:
            sql = """
                select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
                       entry_price, exit_price, pnl_pct, exit_event
                from closed_trades
                where symbol = %(symbol)s
                order by exit_ts desc
                limit %(limit)s
            """
            params = {"symbol": symbol, "limit": int(max(1, max_points))}
        elif symbol is None:
            sql = """
                select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
                       entry_price, exit_price, pnl_pct, exit_event
                from closed_trades
                where venue = %(venue)s
                order by exit_ts desc
                limit %(limit)s
            """
            params = {"venue": venue, "limit": int(max(1, max_points))}
        else:
            sql = """
                select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
                       entry_price, exit_price, pnl_pct, exit_event
                from closed_trades
                where venue = %(venue)s
                  and symbol = %(symbol)s
                order by exit_ts desc
                limit %(limit)s
            """
            params = {
                "venue": venue,
                "symbol": symbol,
                "limit": int(max(1, max_points)),
            }

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(
            rows,
            columns=[
                "trade_id",
                "venue",
                "symbol",
                "entry_ts",
                "exit_ts",
                "side",
                "qty",
                "entry_price",
                "exit_price",
                "pnl_pct",
                "exit_event",
            ],
        )
    except Exception:
        return pd.DataFrame()


def load_trade_markers(
    max_points: int = 5000,
    _trades_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    if _trades_df is not None:
        df = _trades_df
    else:
        df = load_closed_trades_from_postgres(
            venue="kucoin",
            symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
            max_points=max_points,
        )
        if df.empty:
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
        side_raw = r.get("side") if "side" in df.columns else 0
        if pd.isna(side_raw):
            side = 0
        elif isinstance(side_raw, str):
            s = side_raw.strip().lower()
            side = 1 if s in ("long", "l", "buy", "1") else (-1 if s in ("short", "s", "sell", "-1") else 0)
        else:
            side = int(side_raw)

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


def load_trade_segments(
    max_points: int = 2000,
    _trades_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    if _trades_df is not None:
        df = _trades_df
    else:
        df = load_closed_trades_from_postgres(
            venue="kucoin",
            symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
            max_points=max_points,
        )
        if df.empty:
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

        if side_col and pd.notna(r.get(side_col)):
            side_raw = r.get(side_col)
            if isinstance(side_raw, str):
                s = side_raw.strip().lower()
                side = 1 if s in ("long", "l", "buy", "1") else (-1 if s in ("short", "s", "sell", "-1") else 1)
            else:
                side = int(side_raw)
        else:
            side = 1

        if pnl_col and pd.notna(r.get(pnl_col)):
            pnl_positive = float(r[pnl_col]) >= 0.0
        else:
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
    return sorted(out, key=lambda x: int(x["time"]))


def load_fills_cache_rows(max_points: int = 500, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
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


def load_equity_history_from_postgres(
    venue: str,
    account: Optional[str] = None,
    max_points: int = 500,
) -> Dict[str, Any]:
    try:
        if account is None:
            sql = """
                select ts, equity, currency, source
                from equity_snapshots
                where venue = %(venue)s
                order by ts desc
                limit %(limit)s
            """
            params = {"venue": venue, "limit": int(max(1, max_points))}
        else:
            sql = """
                select ts, equity, currency, source
                from equity_snapshots
                where venue = %(venue)s
                  and account = %(account)s
                order by ts desc
                limit %(limit)s
            """
            params = {
                "venue": venue,
                "account": account,
                "limit": int(max(1, max_points)),
            }

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            return {"points": [], "source": "none"}

        rows = list(reversed(rows))
        pts = [
            {
                "time": int(pd.Timestamp(r[0]).timestamp()),
                "equity": float(r[1]),
            }
            for r in rows
        ]
        src = rows[-1][3] if rows and len(rows[-1]) >= 4 else "postgres"
        return {"points": pts, "source": f"postgres:{src}"}
    except Exception:
        return {"points": [], "source": "none"}


def _maybe_refresh_kucoin_equity() -> None:
    refresh_sec = int(os.getenv("DASHBOARD_EQUITY_REFRESH_SEC", "60"))
    currency = os.getenv("DASHBOARD_EQUITY_CCY", "USDT")
    key = os.getenv("KUCOIN_FUTURES_API_KEY", "").strip()
    sec = os.getenv("KUCOIN_FUTURES_API_SECRET", "").strip()
    pp = os.getenv("KUCOIN_FUTURES_PASSPHRASE", "").strip()
    if not (key and sec and pp):
        return

    now_sec = int(pd.Timestamp.now("UTC").timestamp())

    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT EXTRACT(EPOCH FROM MAX(ts))::bigint FROM equity_snapshots "
                "WHERE venue = 'kucoin' AND account = 'futures'"
            )
            row = cur.fetchone()
            last_ts = int(row[0]) if row and row[0] else 0
    except Exception:
        last_ts = 0

    if (now_sec - last_ts) < max(5, refresh_sec):
        return

    try:
        broker = KucoinFuturesBroker(api_key=key, api_secret=sec, passphrase=pp)
        bal = broker.get_account_balance(currency=currency)
        eq = float(bal.get("equity", 0.0) or 0.0)
        if eq > 0:
            insert_equity_snapshot(
                {
                    "ts": pd.to_datetime(now_sec, unit="s", utc=True),
                    "venue": "kucoin",
                    "account": "futures",
                    "symbol": None,
                    "equity": eq,
                    "currency": currency,
                    "source": "dashboard_state.load_real_equity_history",
                    "payload_json": {"time": now_sec, "equity": eq, "currency": currency},
                }
            )
    except Exception:
        pass


def load_real_equity_history(max_points: int = 500) -> Dict[str, Any]:
    _maybe_refresh_kucoin_equity()

    pg = load_equity_history_from_postgres(
        venue="kucoin",
        account="futures",
        max_points=max_points,
    )
    if pg.get("points"):
        return pg

    return {"points": [], "source": "none"}


def _latest_equity_value_from_history(history: Dict[str, Any]) -> Optional[float]:
    if not isinstance(history, dict):
        return None
    pts = history.get("points", [])
    if not isinstance(pts, list) or not pts:
        return None
    try:
        eq = pd.to_numeric(pts[-1].get("equity"), errors="coerce")
        if pd.notna(eq):
            return float(eq)
    except Exception:
        pass
    return None


def load_kraken_metrics() -> Dict[str, Any]:
    exec_state_path = _env_path(
        "KRAKEN_EXECUTION_STATE_JSON",
        _live_default("execution_state.json"),
    )

    if exec_state_path.exists():
        try:
            obj = json.loads(exec_state_path.read_text(encoding="utf-8"))
        except Exception:
            obj = None

        if isinstance(obj, dict) and obj:
            live_pos_num = pd.to_numeric(obj.get("live_pos"), errors="coerce")
            entry_px_num = pd.to_numeric(obj.get("entry_px"), errors="coerce")
            best_fav_num = pd.to_numeric(obj.get("best_fav"), errors="coerce")
            side_raw = str(obj.get("side") or "").strip().lower()

            venue_pos_side = 0
            if pd.notna(live_pos_num):
                if float(live_pos_num) > 0:
                    venue_pos_side = 1
                elif float(live_pos_num) < 0:
                    venue_pos_side = -1
            elif side_raw in ("long", "buy", "1"):
                venue_pos_side = 1
            elif side_raw in ("short", "sell", "-1"):
                venue_pos_side = -1

            equity_hist = load_kraken_equity_history(max_points=1)
            latest_equity = _latest_equity_value_from_history(equity_hist)

            return {
                "ts": _to_ts_iso(obj.get("updated_at") or obj.get("ts")),
                "equity_usd": latest_equity,
                "wallet_usd": latest_equity,
                "upnl_usd": None,
                "mark_price": None,
                "target_size": None,
                "gate_on": None,
                "gate_source": "execution_state",
                "engine": "flip",
                "mode": obj.get("mode"),
                "pos_side": venue_pos_side,
                "entry_px": float(entry_px_num) if pd.notna(entry_px_num) else None,
                "best_fav": float(best_fav_num) if pd.notna(best_fav_num) else None,
                "size_rem": float(live_pos_num) if pd.notna(live_pos_num) else 0.0,
                "tp1_done": False,
                "signal": None,
                "signal_ts": None,
                "venue_pos_side": venue_pos_side,
                "venue_pos_size": float(live_pos_num) if pd.notna(live_pos_num) else 0.0,
                "dry_run": None,
            }

    p = _env_path("KRAKEN_METRICS_JSON", _live_default("kraken/metrics.json"))
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def load_kraken_equity_history(max_points: int = 500) -> Dict[str, Any]:
    pg = load_equity_history_from_postgres(
        venue="kraken",
        account="main",
        max_points=max_points,
    )
    if pg.get("points"):
        return pg

    p = _env_path("KRAKEN_EQUITY_CSV", _live_default("kraken/equity.csv"))

    pts: List[Dict[str, Any]] = []
    source = "none"

    if p.exists():
        try:
            df = pd.read_csv(p)
            if not df.empty:
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
                if {"ts", "equity_usd"}.issubset(set(df.columns)):
                    df["ts"] = df["ts"].map(_epoch_seconds_from_any)
                    df["equity_usd"] = pd.to_numeric(df["equity_usd"], errors="coerce")
                    df = (
                        df.dropna(subset=["ts", "equity_usd"])
                        .sort_values("ts")
                        .drop_duplicates(subset=["ts"], keep="last")
                    )
                    if not df.empty:
                        pts = [
                            {"time": int(r["ts"]), "equity": float(r["equity_usd"])}
                            for _, r in df.iterrows()
                        ]
                        source = "kraken_equity_snapshots_usd"
        except Exception:
            pts = []
            source = "none"

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
                    latest_pt = {"time": int(ts_i), "equity": float(eq)}
                    if not pts:
                        pts = [latest_pt]
                        source = "kraken_equity_redis_latest"
                    else:
                        last_ts = int(pts[-1]["time"])
                        if int(ts_i) > last_ts:
                            pts.append(latest_pt)
                            source = "kraken_equity_snapshots_usd+redis_latest"
        except Exception:
            pass

    if not pts:
        return {"points": [], "source": "none"}

    pts = sorted(pts, key=lambda x: int(x["time"]))
    if len(pts) > int(max(1, max_points)):
        pts = pts[-int(max(1, max_points)) :]
    return {"points": pts, "source": source}


def build_combined_equity(
    kucoin_points: List[Dict[str, Any]],
    kraken_points_usd: List[Dict[str, Any]],
) -> Dict[str, Any]:
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
    return pd.DataFrame(out_rows).sort_values("time").reset_index(drop=True)


def build_trading_diary(max_points: int = 500, _trades_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []

    if _trades_df is not None:
        df = _trades_df
        df_source = "preloaded"
    else:
        df = load_closed_trades_from_postgres(
            venue="kucoin",
            symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
            max_points=max_points,
        )
        df_source = "postgres:closed_trades"
        if df.empty:
            df = _read_trades_df()
            df_source = "trades_parquet"

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

                if side_col and pd.notna(r.get(side_col)):
                    side_raw = r.get(side_col)
                    if isinstance(side_raw, str):
                        s = side_raw.strip().lower()
                        side = 1 if s in ("long", "l", "buy", "1") else (-1 if s in ("short", "s", "sell", "-1") else 1)
                    else:
                        side = int(side_raw)
                else:
                    side = 1

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
                        "source": df_source,
                    }
                )

            if out:
                out = sorted(out, key=lambda x: int(x["time"]))[-int(max(1, max_points)) :]
                return {"entries": out, "source": df_source}

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

    try:
        trades_path = _env_path("DASHBOARD_TRADES_PARQUET", _live_default("trades.parquet"))
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        if events:
            write_rows = []
            for e in events:
                side_i = 1 if str(e.get("side")) == "long" else -1
                write_rows.append(
                    {
                        "entry_ts": pd.to_datetime(int(e["entry_time"]), unit="s", utc=True),
                        "exit_ts": pd.to_datetime(int(e["time"]), unit="s", utc=True),
                        "side": side_i,
                        "qty": e.get("qty"),
                        "entry_price": e.get("entry_price"),
                        "exit_price": e.get("exit_price"),
                        "pnl_pct": e.get("pnl_pct"),
                        "exit_event": "fills_reconstructed",
                    }
                )
                try:
                    upsert_closed_trade(
                        {
                            "trade_id": str(e["id"]),
                            "venue": "kucoin",
                            "symbol": os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
                            "entry_ts": pd.to_datetime(int(e["entry_time"]), unit="s", utc=True),
                            "exit_ts": pd.to_datetime(int(e["time"]), unit="s", utc=True),
                            "side": str(e.get("side")),
                            "qty": e.get("qty"),
                            "entry_price": e.get("entry_price"),
                            "exit_price": e.get("exit_price"),
                            "pnl_pct": e.get("pnl_pct"),
                            "exit_event": "fills_reconstructed",
                            "strategy": "dashboard_fills_reconstruction",
                            "strategy_instance": None,
                            "config_hash": None,
                            "source_action_event_id": None,
                            "payload_json": dict(e),
                        }
                    )
                except Exception:
                    pass

            new_df = pd.DataFrame(write_rows)
            if trades_path.exists():
                try:
                    old_df = pd.read_parquet(trades_path)
                    all_df = pd.concat([old_df, new_df], ignore_index=True)
                except Exception:
                    all_df = new_df
            else:
                all_df = new_df

            dedupe_cols = [
                c for c in ("entry_ts", "exit_ts", "side", "qty", "entry_price", "exit_price")
                if c in all_df.columns
            ]
            all_df = all_df.drop_duplicates(subset=dedupe_cols, keep="last").sort_values("exit_ts")
            all_df.to_parquet(trades_path, index=False)
    except Exception:
        pass

    events = sorted(events, key=lambda x: int(x["time"]))[-int(max(1, max_points)) :]
    return {"entries": events, "source": "fills_reconstructed_clustered"}


def build_regime_overlay(
    symbol: str,
    hours: int = 24 * 14,
    _rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    end_ts = pd.Timestamp.now("UTC")
    if _rows is not None:
        rows = _rows
    else:
        store = RegimeStore()
        start_ts = end_ts - pd.Timedelta(hours=int(max(1, hours)))
        rows = store.get_history(
            symbol=symbol,
            start_ts=start_ts.isoformat(),
            end_ts=end_ts.isoformat(),
            limit=20000,
        )
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


def build_equity_curve(max_points: int = 500, _trades_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    diary = build_trading_diary(max_points=max_points, _trades_df=_trades_df)
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


def build_regime_scores(symbol: str, hours: int = 24 * 14, _rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List]:
    if _rows is not None:
        rows = _rows
    else:
        store = RegimeStore()
        end_ts = pd.Timestamp.now("UTC")
        start_ts = end_ts - pd.Timedelta(hours=int(max(1, hours)))
        rows = store.get_history(
            symbol=symbol,
            start_ts=start_ts.isoformat(),
            end_ts=end_ts.isoformat(),
            limit=20000,
        )
    if not rows:
        return {"scores": [], "forecast": []}

    scores = []
    for r in rows:
        ts = pd.to_datetime(r.get("ts"), utc=True, errors="coerce")
        rs = pd.to_numeric(r.get("regime_score"), errors="coerce")
        if pd.notna(ts) and pd.notna(rs):
            scores.append({"time": int(ts.timestamp()), "score": round(float(rs), 4)})

    return {"scores": scores, "forecast": []}


def _normalize_strategy_label(raw_state: Any, gate_on: Any) -> Dict[str, Any]:
    raw = str(raw_state or "").strip().lower()

    countertrend_states = {
        "countertrend",
        "counter_trend",
        "mean_revert",
        "mean_reversion",
    }
    trend_states = {
        "trend",
        "trendfollower",
        "trend_follower",
        "trendfollow",
    }

    if raw in countertrend_states:
        return {
            "strategy_label": "Countertrend",
            "regime_state": raw,
            "source": "regime_store_latest",
        }
    if raw in trend_states:
        return {
            "strategy_label": "Trendfollower",
            "regime_state": raw,
            "source": "regime_store_latest",
        }

    gate_i = pd.to_numeric(gate_on, errors="coerce")
    if pd.notna(gate_i):
        return {
            "strategy_label": "Countertrend" if int(gate_i) == 1 else "Trendfollower",
            "regime_state": raw or None,
            "source": "gate_fallback",
        }

    return {
        "strategy_label": "Trendfollower",
        "regime_state": raw or None,
        "source": "default_fallback",
    }


def load_dashboard_strategy(symbol: str) -> Dict[str, Any]:
    now = pd.Timestamp.now("UTC")
    try:
        latest = RegimeStore().get_latest_state(symbol=symbol) or {}
    except Exception:
        latest = {}

    mapped = _normalize_strategy_label(
        raw_state=latest.get("regime_state"),
        gate_on=latest.get("gate_on"),
    )

    return {
        "symbol": symbol,
        "strategy_label": mapped["strategy_label"],
        "regime_state": mapped["regime_state"],
        "source": mapped["source"],
        "ts": now.isoformat(),
    }


def _performance_trade_frame(
    symbol: str,
    venue: str,
    max_points: int,
) -> tuple[pd.DataFrame, str]:
    df = load_closed_trades_from_postgres(
        venue=venue,
        symbol=symbol,
        max_points=max_points,
    )
    source = "postgres:closed_trades"

    if df.empty:
        df = _read_trades_df()
        source = "trades_parquet"

        if not df.empty:
            if "symbol" in df.columns:
                df = df[df["symbol"].astype(str) == str(symbol)]
            if "venue" in df.columns:
                df = df[df["venue"].astype(str) == str(venue)]

    if df.empty:
        return pd.DataFrame(), "none"

    df = df.copy()

    if "exit_ts" not in df.columns:
        return pd.DataFrame(), source

    df["exit_ts"] = pd.to_datetime(df["exit_ts"], utc=True, errors="coerce")
    df["pnl_pct"] = pd.to_numeric(df.get("pnl_pct"), errors="coerce")
    df = df.dropna(subset=["exit_ts", "pnl_pct"]).sort_values("exit_ts").reset_index(drop=True)

    return df, source


def _compound_trade_returns_pct(s: pd.Series) -> Optional[float]:
    if s is None or len(s) == 0:
        return None
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if vals.empty:
        return None
    return float(((1.0 + vals / 100.0).prod() - 1.0) * 100.0)


def build_dashboard_performance(
    symbol: str,
    venue: str = "kucoin",
    max_points: int = 5000,
) -> Dict[str, Any]:
    now = pd.Timestamp.now("UTC")
    df, source = _performance_trade_frame(
        symbol=symbol,
        venue=venue,
        max_points=max_points,
    )

    if df.empty:
        return {
            "symbol": symbol,
            "venue": venue,
            "as_of": now.isoformat(),
            "window": "lifetime",
            "pnl_pct": None,
            "winrate": None,
            "monthly_growth": None,
            "average_gain": None,
            "trade_count": 0,
            "winning_trade_count": 0,
            "losing_trade_count": 0,
            "source": source,
        }

    winners = df[df["pnl_pct"] > 0]
    losers = df[df["pnl_pct"] <= 0]

    pnl_pct = _compound_trade_returns_pct(df["pnl_pct"])
    monthly_growth = _compound_trade_returns_pct(
        df.loc[df["exit_ts"] >= (now - pd.Timedelta(days=30)), "pnl_pct"]
    )
    average_gain = float(winners["pnl_pct"].mean()) if not winners.empty else None

    trade_count = int(len(df))
    winning_trade_count = int(len(winners))
    losing_trade_count = int(len(losers))
    winrate = (winning_trade_count / trade_count * 100.0) if trade_count > 0 else None

    return {
        "symbol": symbol,
        "venue": venue,
        "as_of": now.isoformat(),
        "window": "lifetime",
        "pnl_pct": round(float(pnl_pct), 4) if pnl_pct is not None else None,
        "winrate": round(float(winrate), 4) if winrate is not None else None,
        "monthly_growth": round(float(monthly_growth), 4) if monthly_growth is not None else None,
        "average_gain": round(float(average_gain), 4) if average_gain is not None else None,
        "trade_count": trade_count,
        "winning_trade_count": winning_trade_count,
        "losing_trade_count": losing_trade_count,
        "source": source,
    }