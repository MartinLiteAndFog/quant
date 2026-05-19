from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quant.execution.CHOPgate import get_live_gate_state
from quant.execution.event_store import (
    get_conn,
    insert_equity_snapshot,
    upsert_closed_trade,
)
from quant.execution.kucoin_futures import KucoinFuturesBroker, list_fills
from quant.regime import RegimeStore
from quant.strategies.imba import ImbaParams, get_imba_barrier_series


_LAST_REFRESH_TS: Optional[pd.Timestamp] = None
_LAST_REFRESH_ERROR: Optional[str] = None
_LAST_FILLS_REFRESH_TS: Optional[pd.Timestamp] = None
_LAST_FILLS_REFRESH_ERROR: Optional[str] = None


# In-process result caches keyed by query parameters. Each entry stores the
# wall-clock timestamp at which it was produced, plus the cached value. Keep
# TTLs short so the dashboard remains responsive to live trades while still
# absorbing the bulk of repeated requests within the same refresh tick.
_RENKO_DF_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "ts": 0.0, "df": None}
_CLOSED_TRADES_CACHE: Dict[str, Dict[str, Any]] = {}
_REGIME_HISTORY_CACHE: Dict[str, Dict[str, Any]] = {}
_EQUITY_HISTORY_CACHE: Dict[str, Dict[str, Any]] = {}


def _cache_ttl_env(env_key: str, default_sec: float) -> float:
    raw = os.getenv(env_key)
    if raw:
        try:
            return float(raw)
        except Exception:
            pass
    return float(default_sec)


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


def _normalize_symbol_token(v: Optional[str]) -> str:
    s = str(v or "").strip().upper()
    return s.replace("-", "").replace("_", "").replace("/", "")


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
        st = p.stat()
        cache_path = _RENKO_DF_CACHE.get("path")
        cache_mtime = _RENKO_DF_CACHE.get("mtime")
        cache_size = _RENKO_DF_CACHE.get("size")
        if (
            isinstance(_RENKO_DF_CACHE.get("df"), pd.DataFrame)
            and str(cache_path) == str(p)
            and cache_mtime == st.st_mtime
            and cache_size == st.st_size
        ):
            return _RENKO_DF_CACHE["df"]
    except Exception:
        st = None
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
    df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    if st is not None:
        _RENKO_DF_CACHE["path"] = str(p)
        _RENKO_DF_CACHE["mtime"] = st.st_mtime
        _RENKO_DF_CACHE["size"] = st.st_size
        _RENKO_DF_CACHE["ts"] = time.time()
        _RENKO_DF_CACHE["df"] = df
    return df


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
    # Vectorized epoch (seconds) conversion. Use ``Timedelta`` arithmetic
    # rather than ``// 1_000_000_000`` so we stay correct regardless of the
    # underlying datetime64 precision. Pandas 2.x can return
    # ``datetime64[us, UTC]`` from ``pd.to_datetime`` / parquet, which made
    # the previous ``astype("int64") // 1e9`` path silently produce
    # ``seconds // 1000``.
    try:
        ts_int = (
            (pd.to_datetime(df["ts"], utc=True, errors="coerce") - _EPOCH_UTC)
            // pd.Timedelta(seconds=1)
        ).to_numpy(dtype="int64")
    except Exception:
        ts_int = (
            df["ts"]
            .map(lambda t: int(pd.Timestamp(t).timestamp()))
            .to_numpy(dtype="int64")
        )

    if ts_int.size:
        # Equivalent to:
        #   last = -1
        #   for i, t in enumerate(ts_int):
        #       t = max(t, last + 1); last = t; mono[i] = t
        # Vectorized via the substitution g[i] = mono[i] - i, which yields
        # g[i] = cummax(ts_int[i] - i), and then mono[i] = g[i] + i.
        import numpy as np

        idx = np.arange(ts_int.size, dtype="int64")
        adjusted = ts_int - idx
        ts_int = np.maximum.accumulate(adjusted) + idx

    out = pd.DataFrame(
        {
            "time": ts_int,
            "open": df["open"].astype(float).to_numpy(),
            "high": df["high"].astype(float).to_numpy(),
            "low": df["low"].astype(float).to_numpy(),
            "close": df["close"].astype(float).to_numpy(),
        }
    )
    return out.to_dict("records")


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
    barriers = get_imba_barrier_series(
        df,
        ImbaParams(
            lookback=lb,
            fixed_sl_abs=float(os.getenv("LIVE_IMBA_SL_ABS", "1.5")),
        ),
    )
    return {"lookback": lb, **barriers}


def load_closed_trades_from_postgres(
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
    max_points: int = 5000,
    strategy_whitelist: Optional[List[str]] = None,
    exclude_exit_events: Optional[List[str]] = None,
) -> pd.DataFrame:
    # Short-lived in-process cache to absorb back-to-back identical reads
    # (e.g. chart endpoint + diary + performance all fanning out).
    ttl = _cache_ttl_env("DASHBOARD_CLOSED_TRADES_CACHE_SEC", 4.0)
    if ttl > 0:
        sw = ",".join(sorted(strategy_whitelist)) if strategy_whitelist else ""
        ee = ",".join(sorted(exclude_exit_events)) if exclude_exit_events else ""
        cache_key = f"{venue or ''}|{symbol or ''}|{int(max_points)}|{sw}|{ee}"
        entry = _CLOSED_TRADES_CACHE.get(cache_key)
        now = time.time()
        if entry is not None and (now - float(entry.get("ts", 0.0))) <= ttl:
            cached_df = entry.get("df")
            if isinstance(cached_df, pd.DataFrame):
                return cached_df.copy()
    else:
        cache_key = None

    try:
        symbol_norm = _normalize_symbol_token(symbol) if symbol is not None else None
        if venue is None and symbol is None:
            sql = """
                select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
                       entry_price, exit_price, pnl_pct, exit_event, strategy, strategy_instance
                from closed_trades
                order by exit_ts desc
                limit %(limit)s
            """
            params = {"limit": int(max(1, max_points))}
        elif venue is None:
            sql = """
                select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
                       entry_price, exit_price, pnl_pct, exit_event, strategy, strategy_instance
                from closed_trades
                where replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s
                order by exit_ts desc
                limit %(limit)s
            """
            params = {"symbol_norm": symbol_norm, "limit": int(max(1, max_points))}
        elif symbol is None:
            sql = """
                select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
                       entry_price, exit_price, pnl_pct, exit_event, strategy, strategy_instance
                from closed_trades
                where venue = %(venue)s
                order by exit_ts desc
                limit %(limit)s
            """
            params = {"venue": venue, "limit": int(max(1, max_points))}
        else:
            sql = """
                select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
                       entry_price, exit_price, pnl_pct, exit_event, strategy, strategy_instance
                from closed_trades
                where venue = %(venue)s
                  and replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s
                order by exit_ts desc
                limit %(limit)s
            """
            params = {
                "venue": venue,
                "symbol_norm": symbol_norm,
                "limit": int(max(1, max_points)),
            }

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        if not rows:
            out = pd.DataFrame()
        else:
            out = pd.DataFrame(
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
                    "strategy",
                    "strategy_instance",
                ],
            )

        if cache_key is not None:
            _CLOSED_TRADES_CACHE[cache_key] = {"ts": time.time(), "df": out}
        return out.copy() if not out.empty else out
    except Exception:
        return pd.DataFrame()


def load_execution_fills_from_postgres(
    venue: str,
    symbol: str,
    max_points: int = 20000,
) -> pd.DataFrame:
    try:
        sql = """
            select ts, seq, side, qty, price, reduce_only, status, execution_stage, payload_json
            from execution_events
            where venue = %(venue)s
              and replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(symbol_norm)s
              and execution_stage = 'fill'
            order by ts desc, seq desc
            limit %(limit)s
        """
        params = {
            "venue": str(venue),
            "symbol_norm": _normalize_symbol_token(symbol),
            "limit": int(max(1, max_points)),
        }
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(
            rows,
            columns=[
                "ts",
                "seq",
                "side",
                "qty",
                "price",
                "reduce_only",
                "status",
                "execution_stage",
                "payload_json",
            ],
        )
        out["ts"] = pd.to_datetime(out["ts"], utc=True, errors="coerce")
        out["qty"] = pd.to_numeric(out.get("qty"), errors="coerce")
        out["price"] = pd.to_numeric(out.get("price"), errors="coerce")
        out["seq"] = pd.to_numeric(out.get("seq"), errors="coerce")
        out = out.dropna(subset=["ts", "qty", "price"])
        out = out[(out["qty"] > 0) & (out["price"] > 0)]
        return out.sort_values(["ts", "seq"], na_position="last").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _signed_fill_qty(side: Any, qty: float) -> float:
    s = str(side or "").strip().lower()
    if s in ("buy", "long", "1"):
        return abs(float(qty))
    if s in ("sell", "short", "-1"):
        return -abs(float(qty))
    return 0.0


def _reconstruct_trades_from_execution_fills_df(
    fills_df: pd.DataFrame,
    max_points: int = 500,
    source: str = "postgres:execution_events",
) -> List[Dict[str, Any]]:
    if fills_df.empty:
        return []
    df = fills_df.copy()
    if "ts" not in df.columns or "side" not in df.columns or "qty" not in df.columns or "price" not in df.columns:
        return []
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["ts", "qty", "price"]).sort_values(["ts", "seq"], na_position="last").reset_index(drop=True)
    if df.empty:
        return []

    entries: List[Dict[str, Any]] = []
    pos_qty = 0.0
    avg_entry = 0.0
    open_ts: Optional[pd.Timestamp] = None
    open_side: Optional[str] = None
    realized_pnl_abs = 0.0
    realized_notional_abs = 0.0
    realized_qty = 0.0

    def _flush(close_ts: pd.Timestamp, exit_price: float) -> None:
        nonlocal realized_pnl_abs, realized_notional_abs, realized_qty, open_ts, open_side
        if open_ts is None or open_side is None or realized_notional_abs <= 0:
            realized_pnl_abs = 0.0
            realized_notional_abs = 0.0
            realized_qty = 0.0
            open_ts = None
            open_side = None
            return
        pnl_pct = (realized_pnl_abs / realized_notional_abs) * 100.0
        entries.append(
            {
                "id": f"x_{int(pd.Timestamp(close_ts).timestamp())}_{len(entries)}",
                "entry_time": int(pd.Timestamp(open_ts).timestamp()),
                "time": int(pd.Timestamp(close_ts).timestamp()),
                "side": str(open_side),
                "qty": float(realized_qty) if realized_qty > 0 else None,
                "entry_price": float(avg_entry) if avg_entry > 0 else None,
                "exit_price": float(exit_price),
                "pnl_pct": round(float(pnl_pct), 4),
                "source": source,
            }
        )
        realized_pnl_abs = 0.0
        realized_notional_abs = 0.0
        realized_qty = 0.0
        open_ts = None
        open_side = None

    eps = 1e-12
    for _, r in df.iterrows():
        ts = pd.Timestamp(r["ts"])
        qty = float(r["qty"])
        px = float(r["price"])
        signed = _signed_fill_qty(r.get("side"), qty)
        if abs(signed) <= eps:
            continue

        if abs(pos_qty) <= eps:
            pos_qty = signed
            avg_entry = px
            open_ts = ts
            open_side = "long" if pos_qty > 0 else "short"
            continue

        if pos_qty * signed > 0:
            new_abs = abs(pos_qty) + abs(signed)
            avg_entry = ((abs(pos_qty) * avg_entry) + (abs(signed) * px)) / new_abs
            pos_qty += signed
            continue

        close_qty = min(abs(pos_qty), abs(signed))
        direction = 1.0 if pos_qty > 0 else -1.0
        realized_pnl_abs += (px - avg_entry) * direction * close_qty
        realized_notional_abs += avg_entry * close_qty
        realized_qty += close_qty

        pos_after_abs = abs(pos_qty) - close_qty
        remainder_abs = abs(signed) - close_qty

        if pos_after_abs <= eps:
            _flush(ts, px)
            pos_qty = 0.0
            avg_entry = 0.0
            if remainder_abs > eps:
                pos_qty = remainder_abs if signed > 0 else -remainder_abs
                avg_entry = px
                open_ts = ts
                open_side = "long" if pos_qty > 0 else "short"
            continue

        pos_qty = (1.0 if pos_qty > 0 else -1.0) * pos_after_abs

    if not entries:
        return []
    return entries[-int(max(1, max_points)) :]


# Module-level epoch anchor used for vectorised UTC datetime -> epoch seconds
# conversions. Created once so hot paths (e.g. marker building per refresh)
# don't allocate a fresh ``pd.Timestamp`` on every call.
_EPOCH_UTC = pd.Timestamp("1970-01-01", tz="UTC")

_LONG_TOKENS = frozenset({"long", "l", "buy", "1"})
_SHORT_TOKENS = frozenset({"short", "s", "sell", "-1"})


def _side_value_to_int(v: Any) -> int:
    if v is None:
        return 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in _LONG_TOKENS:
            return 1
        if s in _SHORT_TOKENS:
            return -1
        return 0
    try:
        if pd.isna(v):
            return 0
    except Exception:
        pass
    try:
        x = int(v)
    except Exception:
        try:
            x = int(float(v))
        except Exception:
            return 0
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def load_trade_markers(
    max_points: int = 5000,
    _trades_df: Optional[pd.DataFrame] = None,
    open_entry_ts: Optional[int] = None,
    open_side: Any = None,
) -> List[Dict[str, Any]]:
    """
    Render closed-trade markers on the price chart with a single entry arrow
    per trade plus a co-located text label carrying the realized pnl.

    Contract (rewritten):

    * One arrow marker per closed trade at the *entry* timestamp:
        - Long  -> ``arrowUp`` ``belowBar`` in green.
        - Short -> ``arrowDown`` ``aboveBar`` in red.
      The arrow's color is *direction only* — never pnl-based — so the user
      can read "where did the trade start and which way" at a glance.
    * When the trade has a known realized pnl, a second invisible-shape
      marker is appended at the same timestamp/position carrying the pnl
      text ("``+1.23%``" / "``-1.23%``") colored by sign (green/red).
      Lightweight-charts uses the marker's ``color`` for both shape *and*
      text, so we split the arrow and the label into two markers to keep
      direction (arrow) and outcome (text) independently colorable.
    * Exit markers are NOT emitted — the entry arrow + co-located pnl text
      carries the full per-trade story.
    * The currently-open trade (which never appears in ``closed_trades``)
      gets a plain direction-colored arrow with empty text. The open
      timestamp is resolved from ``open_entry_ts`` (caller hint) or
      ``load_active_levels()`` (live execution state). The matching
      ``time + shape`` will dedup the legacy live-entry marker emitted by
      the chart endpoint, so the new direction-colored arrow wins.
    """
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

    open_side_int = _side_value_to_int(open_side) if open_side is not None else 0
    # Auto-detect the open-trade entry from live execution state whenever
    # the caller didn't fully specify it. This runs regardless of whether
    # ``_trades_df`` was injected because the chart endpoint always passes
    # closed_trades through ``_trades_df`` (which by definition never
    # contains the open trade). If ``execution_state.json`` is empty / has
    # no side, ``open_side_int`` stays 0 and the open marker is skipped.
    if open_entry_ts is None or open_side_int == 0:
        try:
            live_levels = load_active_levels()
        except Exception:
            live_levels = {}
        if isinstance(live_levels, dict) and live_levels:
            if open_entry_ts is None:
                raw_ts = live_levels.get("entry_bar_ts")
                if raw_ts is None:
                    raw_ts = live_levels.get("ts")
                if raw_ts is not None:
                    open_entry_ts = _epoch_seconds_from_any(raw_ts)
            if open_side_int == 0:
                open_side_int = _side_value_to_int(live_levels.get("side"))

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        df = pd.DataFrame()

    if not df.empty:
        if "entry_ts" not in df.columns and "ts" in df.columns:
            df = df.rename(columns={"ts": "entry_ts"})
        if "entry_ts" not in df.columns:
            df = pd.DataFrame()

    if not df.empty:
        df = df.copy()
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
        df = (
            df.dropna(subset=["entry_ts"])
            .sort_values("entry_ts")
            .tail(int(max(1, max_points)))
        )

    # Direction palette (entry arrow color) and pnl palette (text marker).
    # Both share the same hex values for green/red so the entire chart
    # legend stays in a single palette.
    LONG_COLOR = "#22c55e"
    SHORT_COLOR = "#ef4444"
    PNL_WIN_COLOR = "#22c55e"
    PNL_LOSS_COLOR = "#ef4444"
    ARROW_SIZE = 2
    # Lightweight-charts v4 uses ``size`` as a multiplier on its built-in
    # marker shape size; ``size=0`` collapses the shape to a single pixel
    # while keeping the text label rendered next to it. That gives us a
    # de-facto "text-only" marker without resorting to a primitive plugin.
    TEXT_MARKER_SIZE = 0

    markers: List[Dict[str, Any]] = []
    entry_seen_ts: set[int] = set()

    if not df.empty:
        cols = set(df.columns)
        entry_px_col = next(
            (c for c in ("entry_px", "entry_price", "price_entry", "entry") if c in cols),
            None,
        )
        exit_px_col = next(
            (c for c in ("exit_px", "exit_price", "price_exit", "exit") if c in cols),
            None,
        )
        pnl_col = next(
            (c for c in ("pnl_pct", "pnl", "pnl_abs", "net_pnl") if c in cols),
            None,
        )

        side_series = df["side"] if "side" in cols else pd.Series(0, index=df.index)
        side_int = side_series.map(_side_value_to_int).astype("int64").to_numpy()

        # Convert tz-aware datetime series to integer epoch seconds using
        # timedelta arithmetic. Casting via ``.astype("int64") // 1_000_000_000``
        # silently breaks under pandas 2.x because ``pd.to_datetime`` now returns
        # ``datetime64[us, UTC]`` (microsecond resolution) rather than the
        # ``datetime64[ns, UTC]`` it used to. That regression collapsed every
        # marker timestamp to ``epoch_seconds // 1000``, which lightweight-charts
        # then rendered as a tight stack at the chart's left edge — the visible
        # "summary of old trades" artefact.
        entry_ts_int = (
            (df["entry_ts"] - _EPOCH_UTC) // pd.Timedelta(seconds=1)
        ).to_numpy(dtype="int64")

        entry_px_num = (
            pd.to_numeric(df[entry_px_col], errors="coerce").to_numpy()
            if entry_px_col
            else None
        )
        exit_px_num = (
            pd.to_numeric(df[exit_px_col], errors="coerce").to_numpy()
            if exit_px_col
            else None
        )
        pnl_num = (
            pd.to_numeric(df[pnl_col], errors="coerce").to_numpy()
            if pnl_col
            else None
        )

        for i in range(len(df)):
            side_i = int(side_int[i])
            is_long = side_i >= 0
            entry_ts_val = int(entry_ts_int[i])

            arrow_color = LONG_COLOR if is_long else SHORT_COLOR
            position = "belowBar" if is_long else "aboveBar"
            shape = "arrowUp" if is_long else "arrowDown"

            # Compute realized pnl: prefer the stored pnl_pct, otherwise
            # derive from entry/exit prices weighted by direction.
            pnl_value: Optional[float] = None
            if pnl_num is not None:
                p = pnl_num[i]
                if p == p:  # not NaN
                    pnl_value = float(p)
            if (
                pnl_value is None
                and entry_px_num is not None
                and exit_px_num is not None
            ):
                ep = entry_px_num[i]
                xp = exit_px_num[i]
                if ep == ep and xp == xp and float(ep) != 0.0:
                    pnl_value = (
                        ((float(xp) - float(ep)) / float(ep))
                        * 100.0
                        * (1 if is_long else -1)
                    )

            suppress_text = (
                open_entry_ts is not None and entry_ts_val == int(open_entry_ts)
            )

            markers.append(
                {
                    "time": entry_ts_val,
                    "position": position,
                    "shape": shape,
                    "color": arrow_color,
                    "text": "",
                    "size": ARROW_SIZE,
                }
            )
            entry_seen_ts.add(entry_ts_val)

            if pnl_value is not None and not suppress_text:
                pnl_color = PNL_WIN_COLOR if pnl_value >= 0 else PNL_LOSS_COLOR
                markers.append(
                    {
                        "time": entry_ts_val,
                        "position": position,
                        "shape": "circle",
                        "color": pnl_color,
                        "text": f"{pnl_value:+.2f}%",
                        "size": TEXT_MARKER_SIZE,
                    }
                )

    # Emit the open-trade arrow when we know it and we haven't already
    # rendered a closed-trade arrow at the same entry timestamp. No text
    # label — the trade is still running, so the realized pnl is undefined.
    if (
        open_entry_ts is not None
        and open_side_int != 0
        and int(open_entry_ts) not in entry_seen_ts
    ):
        is_long_open = open_side_int >= 0
        markers.append(
            {
                "time": int(open_entry_ts),
                "position": "belowBar" if is_long_open else "aboveBar",
                "shape": "arrowUp" if is_long_open else "arrowDown",
                "color": LONG_COLOR if is_long_open else SHORT_COLOR,
                "text": "",
                "size": ARROW_SIZE,
            }
        )

    return markers


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

    w = src_df.copy()
    w["side"] = w["side"].fillna("").astype(str).str.lower()
    w["size"] = pd.to_numeric(w["size"], errors="coerce").fillna(0.0)
    w["price"] = pd.to_numeric(w["price"], errors="coerce").fillna(0.0)
    is_buy = w["side"] == "buy"
    w["time"] = w["time"].astype(int)
    w["position"] = is_buy.map({True: "belowBar", False: "aboveBar"})
    w["shape"] = is_buy.map({True: "arrowUp", False: "arrowDown"})
    w["color"] = is_buy.map({True: "#2ecc71", False: "#f7768e"})
    w["text"] = "fill " + w["side"] + " " + w["size"].map(lambda v: f"{v:g}") + " @ " + w["price"].map(lambda v: f"{v:.3f}")
    out = w[["time", "position", "shape", "color", "text"]].to_dict("records")
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
                select ts, equity, currency, source, payload_json
                from equity_snapshots
                where venue = %(venue)s
                order by ts desc
                limit %(limit)s
            """
            params = {"venue": venue, "limit": int(max(1, max_points))}
        else:
            sql = """
                select ts, equity, currency, source, payload_json
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
                "payload_json": r[4] if len(r) >= 5 else None,
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


def _aggregate_logical_trades(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    w = df.copy()
    if "entry_ts" not in w.columns and "ts" in w.columns:
        w = w.rename(columns={"ts": "entry_ts"})
    if "exit_ts" not in w.columns:
        return pd.DataFrame()

    w["entry_ts"] = pd.to_datetime(w.get("entry_ts"), utc=True, errors="coerce")
    w["exit_ts"] = pd.to_datetime(w.get("exit_ts"), utc=True, errors="coerce")

    def _side_sign(v: Any) -> int:
        if pd.isna(v):
            return 0
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("long", "l", "buy", "1"):
                return 1
            if s in ("short", "s", "sell", "-1"):
                return -1
            return 0
        try:
            x = float(v)
        except Exception:
            return 0
        if x > 0:
            return 1
        if x < 0:
            return -1
        return 0

    side_raw = w["side"] if "side" in w.columns else pd.Series([1] * len(w), index=w.index)
    w["side_sign"] = side_raw.map(_side_sign).astype(int)
    w = w[w["side_sign"] != 0]
    if w.empty:
        return pd.DataFrame()

    qty_col = next((c for c in ("qty", "size", "contracts") if c in w.columns), None)
    if qty_col:
        w["qty_raw"] = pd.to_numeric(w[qty_col], errors="coerce").abs()
    else:
        w["qty_raw"] = float("nan")

    entry_col = next((c for c in ("entry_px", "entry_price", "price_entry", "entry") if c in w.columns), None)
    exit_col = next((c for c in ("exit_px", "exit_price", "price_exit", "exit") if c in w.columns), None)
    pnl_col = next((c for c in ("pnl_pct", "pnl", "pnl_abs", "net_pnl") if c in w.columns), None)

    w["entry_price_num"] = pd.to_numeric(w[entry_col], errors="coerce") if entry_col else float("nan")
    w["exit_price_num"] = pd.to_numeric(w[exit_col], errors="coerce") if exit_col else float("nan")

    if pnl_col:
        w["pnl_pct_num"] = pd.to_numeric(w[pnl_col], errors="coerce")
        if pnl_col != "pnl_pct":
            valid = w["entry_price_num"] > 0
            w.loc[valid, "pnl_pct_num"] = (w.loc[valid, "pnl_pct_num"] / w.loc[valid, "entry_price_num"]) * 100.0
    else:
        w["pnl_pct_num"] = float("nan")

    recompute_mask = w["pnl_pct_num"].isna() & (w["entry_price_num"] > 0) & w["exit_price_num"].notna()
    w.loc[recompute_mask, "pnl_pct_num"] = (
        ((w.loc[recompute_mask, "exit_price_num"] - w.loc[recompute_mask, "entry_price_num"]) / w.loc[recompute_mask, "entry_price_num"])
        * 100.0
        * w.loc[recompute_mask, "side_sign"]
    )

    w = w.dropna(subset=["exit_ts", "pnl_pct_num"]).sort_values("exit_ts").reset_index(drop=True)
    if w.empty:
        return pd.DataFrame()

    w["qty_w"] = w["qty_raw"].where(w["qty_raw"].notna() & (w["qty_raw"] > 0), 1.0)
    # Use a sentinel int for "missing entry anchor" so we can do vectorized
    # equality checks instead of paired-None comparisons in a Python loop.
    _NO_ANCHOR_SENTINEL = -1
    dt_entry = pd.to_datetime(w["entry_ts"], utc=True, errors="coerce")
    entry_anchor_int = (
        (dt_entry.astype("int64") // 1_000_000_000)
        .where(dt_entry.notna(), _NO_ANCHOR_SENTINEL)
        .astype("int64")
    )
    w["entry_anchor"] = entry_anchor_int

    side_arr = w["side_sign"].astype("int64").to_numpy()
    anchor_arr = entry_anchor_int.to_numpy(dtype="int64")
    if side_arr.size == 0:
        w["logical_gid"] = []
    else:
        side_changed = np.empty(side_arr.size, dtype=bool)
        side_changed[0] = True
        side_changed[1:] = (
            (side_arr[1:] != side_arr[:-1])
            | (anchor_arr[1:] != anchor_arr[:-1])
        )
        w["logical_gid"] = np.cumsum(side_changed.astype("int64")) - 1

    # Vectorized aggregation: precompute per-row contributions then a single
    # groupby.sum over the logical_gid column. Avoids a Python-level groupby
    # iteration that becomes the dominant cost when there are thousands of
    # logical trades.
    qty_w = pd.to_numeric(w["qty_w"], errors="coerce").fillna(1.0).astype("float64")
    pnl_num = pd.to_numeric(w["pnl_pct_num"], errors="coerce").fillna(0.0).astype("float64")
    entry_num = pd.to_numeric(w["entry_price_num"], errors="coerce").astype("float64")
    exit_num = pd.to_numeric(w["exit_price_num"], errors="coerce").astype("float64")
    qty_raw_num = pd.to_numeric(w["qty_raw"], errors="coerce").astype("float64")

    work = pd.DataFrame(
        {
            "logical_gid": w["logical_gid"].astype("int64"),
            "qty_w": qty_w,
            "pnl_x_qw": pnl_num * qty_w,
            "entry_x_qw": entry_num.fillna(0.0) * qty_w,
            "exit_x_qw": exit_num.fillna(0.0) * qty_w,
            "qty_raw": qty_raw_num,
            "qty_raw_notna": qty_raw_num.notna().astype("int64"),
            "entry_notna": entry_num.notna().astype("int64"),
            "exit_notna": exit_num.notna().astype("int64"),
            "entry_ts": w["entry_ts"],
            "exit_ts": w["exit_ts"],
            "side_sign": w["side_sign"].astype("int64"),
            "row_idx": np.arange(len(w), dtype="int64"),
        }
    )

    grouped_sum = work.groupby("logical_gid", sort=True).agg(
        qty_w_sum=("qty_w", "sum"),
        pnl_x_qw_sum=("pnl_x_qw", "sum"),
        entry_x_qw_sum=("entry_x_qw", "sum"),
        exit_x_qw_sum=("exit_x_qw", "sum"),
        qty_raw_sum=("qty_raw", "sum"),
        qty_raw_any=("qty_raw_notna", "max"),
        entry_any=("entry_notna", "max"),
        exit_any=("exit_notna", "max"),
        entry_ts_min=("entry_ts", "min"),
        exit_ts_max=("exit_ts", "max"),
        last_row_idx=("row_idx", "max"),
        slice_count=("row_idx", "count"),
    )

    n_groups = len(grouped_sum)
    if n_groups == 0:
        return pd.DataFrame()

    # Weighted means; fall back to simple count when total weight is zero.
    qty_w_sum = grouped_sum["qty_w_sum"].to_numpy()
    use_count_fallback = qty_w_sum <= 0
    effective_w = np.where(use_count_fallback, grouped_sum["slice_count"].to_numpy(), qty_w_sum).astype("float64")
    effective_w = np.where(effective_w == 0, 1.0, effective_w)

    pnl_pct = grouped_sum["pnl_x_qw_sum"].to_numpy() / effective_w
    entry_price = grouped_sum["entry_x_qw_sum"].to_numpy() / effective_w
    exit_price = grouped_sum["exit_x_qw_sum"].to_numpy() / effective_w

    last_row_idx = grouped_sum["last_row_idx"].to_numpy().astype("int64")
    side_sign_last = w["side_sign"].astype("int64").to_numpy()[last_row_idx]
    side_strs = np.where(side_sign_last >= 0, "long", "short")

    qty_raw_any = grouped_sum["qty_raw_any"].astype(bool).to_numpy()
    entry_any = grouped_sum["entry_any"].astype(bool).to_numpy()
    exit_any = grouped_sum["exit_any"].astype(bool).to_numpy()

    qty_out_arr = np.where(qty_raw_any, grouped_sum["qty_raw_sum"].to_numpy(), np.nan)
    entry_out_arr = np.where(entry_any, entry_price, np.nan)
    exit_out_arr = np.where(exit_any, exit_price, np.nan)

    # Fall back to exit_ts when entry_ts_min is NaT (group had no valid entry_ts).
    entry_ts_filled = grouped_sum["entry_ts_min"].fillna(grouped_sum["exit_ts_max"])

    out_df = pd.DataFrame(
        {
            "logical_trade_id": [
                f"lt_{int(gid)}" for gid in grouped_sum.index.to_list()
            ],
            "entry_ts": entry_ts_filled.to_numpy(),
            "exit_ts": grouped_sum["exit_ts_max"].to_numpy(),
            "side": side_strs,
            "qty": qty_out_arr,
            "entry_price": entry_out_arr,
            "exit_price": exit_out_arr,
            "pnl_pct": pnl_pct.astype("float64"),
            "slice_count": grouped_sum["slice_count"].astype("int64").to_numpy(),
        }
    )
    return out_df.sort_values("exit_ts").reset_index(drop=True)


def build_trading_diary(
    max_points: int = 500,
    symbol: Optional[str] = None,
    venue: str = "kucoin",
    live_only: bool = False,
    include_reconstructed: bool = False,
    allow_file_fallback: bool = True,
    allow_fill_reconstruction: bool = True,
    _trades_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []
    symbol_eff = str(symbol or os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"))
    venue_eff = str(venue or "kucoin")
    if venue_eff.lower() != "kucoin":
        return {"entries": [], "source": "unsupported_venue"}

    if live_only and _trades_df is None:
        fills_df = load_execution_fills_from_postgres(
            venue=venue_eff,
            symbol=symbol_eff,
            max_points=int(max(100, max_points * 40)),
        )
        fills_entries = _reconstruct_trades_from_execution_fills_df(
            fills_df=fills_df,
            max_points=max_points,
            source="postgres:execution_events_reconstructed",
        )
        if fills_entries:
            return {"entries": fills_entries, "source": "postgres:execution_events_reconstructed"}

    if _trades_df is not None:
        df = _trades_df.copy()
        df_source = "preloaded"
        if not df.empty and "symbol" in df.columns:
            sym_norm = _normalize_symbol_token(symbol_eff)
            df = df[df["symbol"].map(lambda x: _normalize_symbol_token(str(x))).astype(str) == sym_norm]
        if not df.empty and "venue" in df.columns:
            df = df[df["venue"].astype(str) == venue_eff]
        if live_only:
            if "strategy" in df.columns:
                live_map = {"kucoin": {"live_executor"}}
                allowed = live_map.get(venue_eff.lower(), set())
                if allowed:
                    df = df[df["strategy"].astype(str).isin(allowed)]
            if not include_reconstructed and "exit_event" in df.columns:
                df = df[df["exit_event"].astype(str).str.lower() != "fills_reconstructed"]
    else:
        strategy_whitelist = None
        if live_only:
            live_map = {"kucoin": ["live_executor"]}
            strategy_whitelist = live_map.get(venue_eff.lower())
        df = load_closed_trades_from_postgres(
            venue=venue_eff,
            symbol=symbol_eff,
            max_points=max_points,
            strategy_whitelist=strategy_whitelist,
            exclude_exit_events=(None if include_reconstructed else ["fills_reconstructed"]),
        )
        df_source = "postgres:closed_trades"
        if df.empty and allow_file_fallback:
            df = _read_trades_df()
            df_source = "trades_parquet"
            if not df.empty and "symbol" in df.columns:
                sym_norm = _normalize_symbol_token(symbol_eff)
                df = df[df["symbol"].map(lambda x: _normalize_symbol_token(str(x))).astype(str) == sym_norm]
            if not df.empty and "venue" in df.columns:
                df = df[df["venue"].astype(str) == venue_eff]
            if live_only:
                if "strategy" in df.columns:
                    live_map = {"kucoin": {"live_executor"}}
                    allowed = live_map.get(venue_eff.lower(), set())
                    if allowed:
                        df = df[df["strategy"].astype(str).isin(allowed)]
                if not include_reconstructed and "exit_event" in df.columns:
                    df = df[df["exit_event"].astype(str).str.lower() != "fills_reconstructed"]

    if not df.empty:
        # For live Postgres closed_trades we want exact stored trade performance (pnl_pct),
        # not additional grouping heuristics that may merge distinct trades.
        use_direct_rows = bool(live_only)
        rows_df = df.copy()
        if use_direct_rows:
            if "exit_ts" not in rows_df.columns:
                rows_df = pd.DataFrame()
            else:
                rows_df["entry_ts"] = pd.to_datetime(rows_df.get("entry_ts"), utc=True, errors="coerce")
                rows_df["exit_ts"] = pd.to_datetime(rows_df.get("exit_ts"), utc=True, errors="coerce")
                rows_df["pnl_pct"] = pd.to_numeric(rows_df.get("pnl_pct"), errors="coerce")
                rows_df = rows_df.dropna(subset=["exit_ts", "pnl_pct"]).sort_values("exit_ts")
                rows_df = rows_df.tail(int(max(1, max_points))).reset_index(drop=True)
        else:
            rows_df = _aggregate_logical_trades(rows_df).tail(int(max(1, max_points)))

        # Reject pre-2000 sentinel entry timestamps so the equity tooltip
        # never falls back to "1/1/1970" when older NaT->0 writers leaked a
        # bogus row into ``closed_trades``. ``exit_ts`` is required (no
        # row makes sense without a close) but ``entry_ts`` is allowed to
        # be null — the frontend treats that as "—".
        _entry_min_valid = pd.Timestamp("2000-01-01", tz="UTC")
        for i, r in rows_df.iterrows():
            entry_ts = pd.to_datetime(r.get("entry_ts"), utc=True, errors="coerce")
            exit_ts = pd.to_datetime(r.get("exit_ts"), utc=True, errors="coerce")
            pnl_pct = pd.to_numeric(r.get("pnl_pct"), errors="coerce")
            if pd.isna(exit_ts) or pd.isna(pnl_pct):
                continue
            if pd.notna(entry_ts) and entry_ts >= _entry_min_valid:
                entry_time_val: Optional[int] = int(
                    (pd.Timestamp(entry_ts) - _EPOCH_UTC) // pd.Timedelta(seconds=1)
                )
            else:
                entry_time_val = None
            out.append(
                {
                    "id": str(r.get("logical_trade_id") or f"lt_{i}"),
                    "entry_time": entry_time_val,
                    "time": int((pd.Timestamp(exit_ts) - _EPOCH_UTC) // pd.Timedelta(seconds=1)),
                    "side": str(r.get("side") or "long"),
                    "qty": (float(r["qty"]) if pd.notna(pd.to_numeric(r.get("qty"), errors="coerce")) else None),
                    "entry_price": (float(r["entry_price"]) if pd.notna(pd.to_numeric(r.get("entry_price"), errors="coerce")) else None),
                    "exit_price": (float(r["exit_price"]) if pd.notna(pd.to_numeric(r.get("exit_price"), errors="coerce")) else None),
                    "pnl_pct": round(float(pnl_pct), 4),
                    "source": df_source,
                }
            )
        if out:
            out = sorted(out, key=lambda x: int(x["time"]))[-int(max(1, max_points)) :]
            return {"entries": out, "source": df_source}

    if not allow_fill_reconstruction:
        return {"entries": [], "source": "none"}

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
    if df.empty:
        return {"spans": [], "points": [], "latest": None}

    df["gate_on"] = pd.to_numeric(df["gate_on"], errors="coerce").fillna(0).astype(int)
    df["confidence"] = (
        pd.to_numeric(df.get("confidence"), errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
    )

    ts_int = (
        (df["ts"] - _EPOCH_UTC) // pd.Timedelta(seconds=1)
    ).to_numpy(dtype="int64")
    gate_arr = df["gate_on"].to_numpy(dtype="int64")
    conf_arr = df["confidence"].to_numpy(dtype="float64")
    regime_state_arr = (
        df["regime_state"].fillna("").astype(str).to_numpy()
        if "regime_state" in df.columns
        else None
    )

    # Group consecutive equal-gate runs vectorially.
    n = gate_arr.size
    end_ts_int = int(max(int(ts_int[-1]), int(end_ts.timestamp())))
    if n == 1:
        spans = [
            {
                "from": int(ts_int[0]),
                "to": end_ts_int,
                "gate_on": int(gate_arr[0]),
                "confidence": float(conf_arr[0]),
            }
        ]
    else:
        change_mask = gate_arr[1:] != gate_arr[:-1]
        change_idx = list(map(int, change_mask.nonzero()[0] + 1))
        boundaries = [0, *change_idx, n]
        spans = []
        for k in range(len(boundaries) - 1):
            start_i = boundaries[k]
            stop_i = boundaries[k + 1]
            from_ts = int(ts_int[start_i])
            if stop_i < n:
                to_ts = int(ts_int[stop_i])
            else:
                to_ts = end_ts_int
            spans.append(
                {
                    "from": from_ts,
                    "to": to_ts,
                    "gate_on": int(gate_arr[start_i]),
                    "confidence": float(conf_arr[start_i:stop_i].max()),
                }
            )

    # ``points`` is consumed by the legacy dashboard HTML and tests; build it
    # using zip() over numpy arrays which is ~10x faster than DataFrame.iterrows.
    if regime_state_arr is None:
        points = [
            {
                "time": int(t),
                "confidence": float(c),
                "gate_on": int(g),
                "regime_state": "",
            }
            for t, c, g in zip(ts_int.tolist(), conf_arr.tolist(), gate_arr.tolist())
        ]
    else:
        points = [
            {
                "time": int(t),
                "confidence": float(c),
                "gate_on": int(g),
                "regime_state": str(rs),
            }
            for t, c, g, rs in zip(
                ts_int.tolist(),
                conf_arr.tolist(),
                gate_arr.tolist(),
                regime_state_arr.tolist(),
            )
        ]

    latest = points[-1] if points else None
    return {"spans": spans, "points": points, "latest": latest}


def build_equity_curve(
    max_points: int = 500,
    symbol: Optional[str] = None,
    venue: str = "kucoin",
    live_only: bool = False,
    include_reconstructed: bool = False,
    allow_file_fallback: bool = True,
    allow_fill_reconstruction: bool = True,
    _trades_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    diary = build_trading_diary(
        max_points=max_points,
        symbol=symbol,
        venue=venue,
        live_only=live_only,
        include_reconstructed=include_reconstructed,
        allow_file_fallback=allow_file_fallback,
        allow_fill_reconstruction=allow_fill_reconstruction,
        _trades_df=_trades_df,
    )
    entries = diary.get("entries", [])
    cum = 0.0
    curve: List[Dict[str, Any]] = []
    for e in entries:
        pnl_pct = float(e.get("pnl_pct", 0.0))
        cum += pnl_pct
        exit_time_val = int(e.get("time", 0))
        entry_time_raw = e.get("entry_time")
        try:
            entry_time_val = int(entry_time_raw) if entry_time_raw is not None else None
        except (TypeError, ValueError):
            entry_time_val = None
        curve.append(
            {
                "time": exit_time_val,
                "entry_time": entry_time_val,
                "exit_time": exit_time_val,
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

    df = pd.DataFrame(rows)
    if "ts" not in df.columns or "regime_score" not in df.columns:
        return {"scores": [], "forecast": []}
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(df["ts"], utc=True, errors="coerce"),
            "regime_score": pd.to_numeric(df["regime_score"], errors="coerce"),
        }
    ).dropna()
    if df.empty:
        return {"scores": [], "forecast": []}

    ts_int = (
        (df["ts"] - _EPOCH_UTC) // pd.Timedelta(seconds=1)
    ).to_numpy(dtype="int64")
    score_round = df["regime_score"].astype(float).round(4).to_numpy()
    scores = [
        {"time": int(t), "score": float(s)}
        for t, s in zip(ts_int.tolist(), score_round.tolist())
    ]
    return {"scores": scores, "forecast": []}


def _normalize_strategy_label(raw_state: Any, gate_on: Any, exec_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = str(raw_state or "").strip().lower()
    exec_state = exec_state or {}

    exec_strategy = str(exec_state.get("strategy") or exec_state.get("exit_engine") or "").strip().lower()
    exec_mode = str(exec_state.get("mode") or "").strip().upper()

    if exec_strategy in ("tp2", "follow_tp2", "trend", "trendfollower"):
        return {
            "strategy_label": "TP2",
            "regime_state": raw or None,
            "source": "execution_state.strategy",
        }

    if exec_strategy in ("flip", "countertrend", "counter_trend"):
        return {
            "strategy_label": "Flip",
            "regime_state": raw or None,
            "source": "execution_state.strategy",
        }

    if exec_mode == "TP2":
        return {
            "strategy_label": "TP2",
            "regime_state": raw or None,
            "source": "execution_state.mode",
        }

    if exec_mode in ("TTP", "WAIT"):
        return {
            "strategy_label": "Flip",
            "regime_state": raw or None,
            "source": "execution_state.mode",
        }

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
            "strategy_label": "Flip",
            "regime_state": raw,
            "source": "regime_store_latest",
        }
    if raw in trend_states:
        return {
            "strategy_label": "TP2",
            "regime_state": raw,
            "source": "regime_store_latest",
        }

    gate_i = pd.to_numeric(gate_on, errors="coerce")
    if pd.notna(gate_i):
        return {
            "strategy_label": "Flip" if int(gate_i) == 1 else "TP2",
            "regime_state": raw or None,
            "source": "gate_fallback",
        }

    return {
        "strategy_label": "TP2",
        "regime_state": raw or None,
        "source": "default_fallback",
    }


def _daily_gate_strategy_label(gate_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    gate_state = gate_state or {}

    try:
        if int(gate_state.get("gate_countertrend_on", 0) or 0) == 1:
            return {
                "strategy_label": "countertrend",
                "regime_state": "countertrend",
                "source": "daily_gate",
            }
        if int(gate_state.get("gate_trend_on", 0) or 0) == 1:
            return {
                "strategy_label": "trend",
                "regime_state": "trend",
                "source": "daily_gate",
            }
    except Exception:
        pass

    gate_on = pd.to_numeric(gate_state.get("gate_on"), errors="coerce")
    if pd.notna(gate_on):
        gate_on_means = str(os.getenv("GATE_ON_MEANS", "countertrend")).strip().lower()
        label = "trend" if (gate_on_means == "trend") == (int(gate_on) == 1) else "countertrend"
        return {
            "strategy_label": label,
            "regime_state": label,
            "source": "daily_gate_fallback",
        }
    return None

def load_dashboard_strategy(symbol: str) -> Dict[str, Any]:
    now = pd.Timestamp.now("UTC")
    try:
        day_gate = get_live_gate_state()
    except Exception:
        day_gate = None
    day_mapped = _daily_gate_strategy_label(day_gate)
    if day_mapped:
        return {
            "symbol": symbol,
            "strategy_label": day_mapped["strategy_label"],
            "regime_state": day_mapped["regime_state"],
            "source": day_mapped["source"],
            "ts": now.isoformat(),
        }

    try:
        latest = RegimeStore().get_latest_state(symbol=symbol) or {}
    except Exception:
        latest = {}

    exec_state = load_active_levels()

    mapped = _normalize_strategy_label(
        raw_state=latest.get("regime_state"),
        gate_on=latest.get("gate_on"),
        exec_state=exec_state,
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
    if str(venue or "").lower() != "kucoin":
        return pd.DataFrame(), "unsupported_venue"

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
    df = df[df["exit_ts"] >= pd.Timestamp("2026-03-10T00:00:00Z")]

    if "strategy" in df.columns:
        if venue == "kucoin":
            df = df[df["strategy"].astype(str) == "live_executor"]

    df = df.reset_index(drop=True)
    logical_df = _aggregate_logical_trades(df)
    if logical_df.empty:
        return pd.DataFrame(), source
    return logical_df.reset_index(drop=True), source


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
    losers = df[df["pnl_pct"] < 0]

    pnl_pct = _compound_trade_returns_pct(df["pnl_pct"])
    monthly_growth = _compound_trade_returns_pct(
        df.loc[df["exit_ts"] >= (now - pd.Timedelta(days=30)), "pnl_pct"]
    )
    average_gain = float(pd.to_numeric(df["pnl_pct"], errors="coerce").dropna().mean()) if not df.empty else None

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