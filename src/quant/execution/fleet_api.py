"""Fleet aggregator: multi-bot performance board for the desktop cockpit.

Reads shared Postgres rows tagged with ``strategy_instance``, builds absolute
and percent account-equity curves (hero board), compounded trade-PnL % curves,
a white portfolio sum line, and fans out to each bot's public ``/health``.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from quant.execution.event_store import get_conn
from quant.utils.log import get_logger

log = get_logger("quant.fleet_api")

_KRAKEN_DIRECT_TRADE_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
_KRAKEN_POSITION_EVENT_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}

# Default registry — overridden by FLEET_BOTS_JSON env (array of bot dicts).
_DEFAULT_BOTS: List[Dict[str, Any]] = [
    {
        "id": "imba-runner",
        "display_name": "imba5",
        "strategy_instance": "sol-pilot-canonical",
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "health_url": "https://sol-pilot-canonical-production.up.railway.app/health",
        "color": "#c4a35a",
    },
    {
        "id": "pure-imbatp",
        "display_name": "imbatp",
        "strategy_instance": "sol-pilot-pc3axis",
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "health_url": "https://sol-pilot-pc3axis-production.up.railway.app/health",
        "color": "#6b9e7a",
    },
    {
        "id": "countervariante",
        "display_name": "Countervariante",
        "strategy_instance": "sol-pilot-countertrend",
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "health_url": "https://sol-pilot-countertrend-production.up.railway.app/health",
        "color": "#5b8fad",
    },
    {
        "id": "counter-sl-reverse",
        "display_name": "Counter SL Reverse",
        "strategy_instance": "sol-pilot-countertrend-sl-reverse",
        # No historic equity_snapshots / closed_trades yet in shared Postgres;
        # account key reserved for when the pilot starts sparse equity writes.
        "equity_account": "sol-pilot-countertrend-sl-reverse",
        "trade_instances": ["sol-pilot-countertrend-sl-reverse"],
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "health_url": "https://sol-pilot-countertrend-sl-reverse-production.up.railway.app/health",
        "color": "#8a7a9a",
    },
    {
        "id": "quant-main",
        "display_name": "Quant (KuCoin main)",
        "strategy_instance": "quant",
        # Dashboard equity_snapshots historically used account='futures'.
        "equity_account": "futures",
        "trade_instances": ["quant", "live_executor"],
        # Legacy dashboard closed_trades often lack strategy_instance.
        "include_null_instance_trades": True,
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "health_url": "https://quant-production-5533.up.railway.app/health",
        "color": "#9a8f6a",
    },
    {
        "id": "kraken-legacy",
        "display_name": "Kraken Legacy",
        "strategy_instance": "kraken_bot",
        "equity_account": "main",
        "trade_instances": ["kraken_bot", "live_executor_2"],
        "venue_trade_fallback": True,
        "venue": "kraken",
        "symbol": "SOL-USD",
        "health_url": "https://kraken-production-cb57.up.railway.app/health",
        "color": "#b07050",
    },
]


def fleet_bot_registry() -> List[Dict[str, Any]]:
    raw = (os.getenv("FLEET_BOTS_JSON") or "").strip()
    if not raw:
        return [dict(b) for b in _DEFAULT_BOTS]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and parsed:
            out = []
            for row in parsed:
                if not isinstance(row, dict):
                    continue
                if not row.get("strategy_instance") or not row.get("id"):
                    continue
                out.append(dict(row))
            return out or [dict(b) for b in _DEFAULT_BOTS]
    except Exception as e:
        log.warning("FLEET_BOTS_JSON parse failed: %s", e)
    return [dict(b) for b in _DEFAULT_BOTS]


def _normalize_symbol_token(symbol: str) -> str:
    return (
        str(symbol or "")
        .upper()
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
        .replace(".", "")
    )


def _hours_cutoff_ts(hours: Optional[float]) -> Optional[pd.Timestamp]:
    if hours is None or float(hours) <= 0:
        return None
    return pd.Timestamp.now("UTC") - pd.Timedelta(hours=float(hours))


def _history_start_ts() -> Optional[pd.Timestamp]:
    """Global floor for fleet history (FLEET_HISTORY_START, ISO date).

    Pre-cutoff Postgres rows are relics from earlier experiments (old seeds,
    retired bots); the board counts from the fresh start (2026-07-22 audit).
    Set FLEET_HISTORY_START=off to include everything.
    """
    raw = (os.getenv("FLEET_HISTORY_START") or "2026-07-16").strip()
    if raw.lower() in {"", "0", "off", "none", "all"}:
        return None
    try:
        ts = pd.Timestamp(raw)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    except Exception:
        return None


def _effective_since(hours: Optional[float]) -> Optional[pd.Timestamp]:
    since = _hours_cutoff_ts(hours)
    floor = _history_start_ts()
    if floor is None:
        return since
    if since is None or floor > since:
        return floor
    return since


def _fetch_health(url: str, timeout: float = 8.0) -> Dict[str, Any]:
    if not url:
        return {"ok": False, "error": "no_health_url"}
    try:
        req = Request(url, method="GET", headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            if isinstance(body, dict):
                return {"ok": True, **body}
            return {"ok": True, "raw": body}
    except HTTPError as e:
        return {"ok": False, "error": f"http_{e.code}", "detail": str(e.reason)}
    except URLError as e:
        return {"ok": False, "error": "unreachable", "detail": str(e.reason)}
    except Exception as e:
        return {"ok": False, "error": "health_failed", "detail": str(e)}


def list_fleet_bots(*, probe_health: bool = True) -> Dict[str, Any]:
    bots = fleet_bot_registry()
    health_rows: List[Dict[str, Any]] = []
    if probe_health and bots:
        # Health probes are independent read-only calls.  Running them
        # concurrently bounds Fleet startup by the slowest bot instead of the
        # sum of six network timeouts, while preserving registry order below.
        workers = min(8, len(bots))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            health_rows = list(
                pool.map(
                    lambda bot: _fetch_health(str(bot.get("health_url") or "")),
                    bots,
                )
            )
    out = []
    for index, b in enumerate(bots):
        row = {
            "id": b.get("id"),
            "display_name": b.get("display_name") or b.get("id"),
            "strategy_instance": b.get("strategy_instance"),
            "equity_account": b.get("equity_account"),
            "trade_instances": b.get("trade_instances"),
            "venue": b.get("venue") or "kucoin",
            "symbol": b.get("symbol") or "SOL-USDT",
            "health_url": b.get("health_url"),
            "color": b.get("color"),
        }
        if probe_health:
            health = health_rows[index]
            row["health"] = health
            row["executor_ready"] = bool(health.get("executor_ready")) if health.get("ok") else False
            # Explicit false stays false; missing on older Kraken health was null → UI "off".
            if "live_trading_enabled" in health:
                row["live_trading_enabled"] = bool(health.get("live_trading_enabled"))
            else:
                row["live_trading_enabled"] = None
            row["dry_run"] = health.get("dry_run")
            live_on = bool(row["live_trading_enabled"])
            ready = bool(row["executor_ready"])
            row["status"] = (
                "live"
                if health.get("ok") and ready and live_on
                else (
                    "dry"
                    if health.get("ok") and health.get("dry_run")
                    else ("up" if health.get("ok") else "down")
                )
            )
            # Surface live equity on bot rows for consumers that skip capitalization.
            if health.get("equity") is not None:
                row["equity"] = health.get("equity")
                row["available"] = health.get("available")
                row["currency"] = health.get("currency")
                row["equity_source"] = health.get("equity_source")
        out.append(row)
    return {"ok": True, "bots": out, "ts": pd.Timestamp.now("UTC").isoformat()}


def _load_closed_trades_for_instance(
    *,
    strategy_instance: str,
    venue: Optional[str] = None,
    symbol: Optional[str] = None,
    since: Optional[pd.Timestamp] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    where = ["strategy_instance = %(instance)s"]
    params: Dict[str, Any] = {
        "instance": str(strategy_instance),
        "limit": int(max(1, limit)),
    }
    if venue:
        where.append("venue = %(venue)s")
        params["venue"] = str(venue)
    if symbol:
        where.append(
            "replace(replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', ''), '.', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    if since is not None:
        where.append("exit_ts >= %(since)s")
        params["since"] = since.to_pydatetime()

    sql = f"""
        select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
               entry_price, exit_price, pnl_pct, exit_event, strategy, strategy_instance
        from closed_trades
        where {' and '.join(where)}
        order by exit_ts asc
        limit %(limit)s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
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
                "strategy",
                "strategy_instance",
            ],
        )
    except Exception as e:
        log.warning("fleet closed_trades load failed for %s: %s", strategy_instance, e)
        return pd.DataFrame()


def _load_closed_trades_null_instance(
    *,
    venue: str,
    symbol: Optional[str] = None,
    since: Optional[pd.Timestamp] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """Legacy dashboard rows where strategy_instance was never set."""
    where = ["venue = %(venue)s", "(strategy_instance is null or strategy_instance = '')"]
    params: Dict[str, Any] = {"venue": str(venue), "limit": int(max(1, limit))}
    if symbol:
        where.append(
            "replace(replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', ''), '.', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    if since is not None:
        where.append("exit_ts >= %(since)s")
        params["since"] = since.to_pydatetime()
    sql = f"""
        select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
               entry_price, exit_price, pnl_pct, exit_event, strategy, strategy_instance
        from closed_trades
        where {' and '.join(where)}
        order by exit_ts asc
        limit %(limit)s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
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
                "strategy",
                "strategy_instance",
            ],
        )
    except Exception as e:
        log.warning("fleet null-instance closed_trades load failed: %s", e)
        return pd.DataFrame()


def _load_closed_trades_by_venue(
    *,
    venue: str,
    symbol: Optional[str] = None,
    since: Optional[pd.Timestamp] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    where = ["venue = %(venue)s"]
    params: Dict[str, Any] = {"venue": str(venue), "limit": int(max(1, limit))}
    if symbol:
        where.append(
            "replace(replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', ''), '.', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    if since is not None:
        where.append("exit_ts >= %(since)s")
        params["since"] = since.to_pydatetime()
    sql = f"""
        select trade_id, venue, symbol, entry_ts, exit_ts, side, qty,
               entry_price, exit_price, pnl_pct, exit_event, strategy, strategy_instance
        from closed_trades
        where {' and '.join(where)}
        order by exit_ts asc
        limit %(limit)s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
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
                "strategy",
                "strategy_instance",
            ],
        )
    except Exception as e:
        log.warning("fleet venue closed_trades load failed for %s: %s", venue, e)
        return pd.DataFrame()


def _trade_instances_for_bot(bot: Dict[str, Any]) -> List[str]:
    raw = bot.get("trade_instances")
    if isinstance(raw, list) and raw:
        return [str(x) for x in raw if str(x).strip()]
    inst = str(bot.get("strategy_instance") or "").strip()
    return [inst] if inst else []


def _load_closed_trades_for_bot(
    bot: Dict[str, Any],
    *,
    since: Optional[pd.Timestamp] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    venue = str(bot.get("venue") or "kucoin")
    symbol = str(bot.get("symbol") or "SOL-USDT")
    frames: List[pd.DataFrame] = []
    for inst in _trade_instances_for_bot(bot):
        df = _load_closed_trades_for_instance(
            strategy_instance=inst,
            venue=venue,
            symbol=symbol if venue != "kraken" else None,
            since=since,
            limit=limit,
        )
        if df.empty and venue == "kraken":
            df = _load_closed_trades_for_instance(
                strategy_instance=inst,
                venue=venue,
                since=since,
                limit=limit,
            )
        if not df.empty:
            frames.append(df)

    if bot.get("include_null_instance_trades"):
        null_df = _load_closed_trades_null_instance(
            venue=venue,
            symbol=symbol,
            since=since,
            limit=limit,
        )
        if not null_df.empty:
            frames.append(null_df)

    if not frames and bot.get("venue_trade_fallback"):
        venue_df = _load_closed_trades_by_venue(
            venue=venue,
            symbol=None if venue == "kraken" else symbol,
            since=since,
            limit=limit,
        )
        if not venue_df.empty:
            frames.append(venue_df)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if "trade_id" in out.columns:
        out = out.drop_duplicates(subset=["trade_id"], keep="last")
    elif "exit_ts" in out.columns:
        out = out.drop_duplicates(subset=["exit_ts", "side", "pnl_pct"], keep="last")
    return out.sort_values("exit_ts") if "exit_ts" in out.columns else out


def _execution_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _activity_execution_is_partial(row: Dict[str, Any]) -> bool:
    """True when a reducing activity is a partial TP, not a completed trade."""
    payload = _execution_payload(row.get("payload_json"))
    reason = str(payload.get("reason_code") or "").strip().lower()
    client_oid = str(row.get("client_oid") or "").strip().lower()
    if reason in {"tv_tp1", "tp1", "tp1_partial"} or ":tp1:" in client_oid:
        return True
    before = payload.get("position_before")
    after = payload.get("position_after")
    try:
        return float(before) != 0.0 and float(after) == float(before)
    except (TypeError, ValueError):
        return False


def _trades_from_execution_activity(
    rows: Sequence[Sequence[Any]],
    *,
    bot: Dict[str, Any],
    since: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Infer completed round trips from the Fleet execution activity stream.

    This is a read-model fallback for historic pilot activity that predates
    durable ``closed_trades`` writes. Opening market activity is paired with a
    later full reducing activity for the same instance and symbol. Partial TP1
    reductions deliberately leave the leg open and do not become standalone
    trades.
    """
    columns = [
        "event_id",
        "ts",
        "venue",
        "symbol",
        "strategy_instance",
        "side",
        "qty",
        "price",
        "reduce_only",
        "status",
        "client_oid",
        "payload_json",
    ]
    records = [dict(zip(columns, row)) for row in rows]
    records.sort(key=lambda row: pd.Timestamp(row.get("ts") or 0))
    open_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    trades: List[Dict[str, Any]] = []
    accepted_statuses = {"sent", "filled", "done", "success", "completed", "closed", "ok"}

    for row in records:
        status = str(row.get("status") or "").strip().lower()
        if status and status not in accepted_statuses:
            continue
        side = str(row.get("side") or "").strip().lower()
        if side not in {"buy", "sell"}:
            continue
        inst = str(row.get("strategy_instance") or "").strip()
        symbol = str(row.get("symbol") or bot.get("symbol") or "")
        key = (inst, _normalize_symbol_token(symbol))
        is_reducing = bool(row.get("reduce_only"))
        if not is_reducing:
            opened = open_by_key.get(key)
            opened_side = str((opened or {}).get("side") or "").strip().lower()
            if opened is not None and opened_side != side:
                # Historic pilot reversals were persisted as a new opening fill
                # (reduce_only=false, position_before=0) even though the
                # opposite-side transaction completed the prior displayed leg.
                # Close the read-model leg at this activity, then keep the same
                # activity as the opening of the new direction.
                open_by_key.pop(key, None)
            else:
                open_by_key[key] = row
                continue
        else:
            opened = open_by_key.pop(key, None)
        if is_reducing and _activity_execution_is_partial(row):
            if opened is not None:
                open_by_key[key] = opened
            continue
        if opened is None:
            continue

        entry_ts = pd.Timestamp(opened.get("ts"))
        exit_ts = pd.Timestamp(row.get("ts"))
        if since is not None and exit_ts < since:
            continue
        entry_side = str(opened.get("side") or "").strip().lower()
        trade_side = "long" if entry_side == "buy" else "short"
        entry_price = float(opened["price"]) if _is_finite(opened.get("price")) else None
        exit_price = float(row["price"]) if _is_finite(row.get("price")) else None
        pnl_pct = None
        if entry_price is not None and entry_price > 0 and exit_price is not None:
            direction = 1.0 if trade_side == "long" else -1.0
            pnl_pct = (exit_price / entry_price - 1.0) * 100.0 * direction
        payload = _execution_payload(row.get("payload_json"))
        exit_event = (
            str(payload.get("reason_code") or "activity_close")
            if is_reducing
            else "activity_reversal"
        )
        trades.append(
            {
                "trade_id": f"activity:{row.get('event_id')}",
                "venue": row.get("venue") or bot.get("venue"),
                "symbol": symbol,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "side": trade_side,
                "qty": row.get("qty"),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_pct": pnl_pct,
                "exit_event": exit_event,
                "strategy": "execution_activity",
                "strategy_instance": inst or bot.get("strategy_instance"),
                "display_source": "execution_activity",
            }
        )
        if not is_reducing:
            open_by_key[key] = row
    return pd.DataFrame(trades)


def _load_execution_activity_trades_for_bot(
    bot: Dict[str, Any],
    *,
    since: Optional[pd.Timestamp] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    instances = _trade_instances_for_bot(bot)
    if not instances:
        return pd.DataFrame()
    venue = str(bot.get("venue") or "kucoin")
    symbol = str(bot.get("symbol") or "SOL-USDT")
    # Read from the global display-history floor rather than the selected range
    # so a close inside the range can still pair with its earlier opening leg.
    query_since = _history_start_ts()
    where = [
        "strategy_instance = any(%(instances)s::text[])",
        "venue = %(venue)s",
        "execution_stage = 'market_fill'",
        "lower(coalesce(side, '')) in ('buy', 'sell')",
    ]
    params: Dict[str, Any] = {
        "instances": instances,
        "venue": venue,
        "limit": int(max(2, limit * 4)),
    }
    if venue != "kraken":
        where.append(
            "replace(replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', ''), '.', '') = %(symbol_norm)s"
        )
        params["symbol_norm"] = _normalize_symbol_token(symbol)
    if query_since is not None:
        where.append("ts >= %(query_since)s")
        params["query_since"] = query_since.to_pydatetime()
    sql = f"""
        select event_id, ts, venue, symbol, strategy_instance, side, qty, price,
               reduce_only, status, client_oid, payload_json
        from execution_events
        where {' and '.join(where)}
        order by ts asc
        limit %(limit)s
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
    except Exception as e:
        log.warning("fleet execution activity load failed for %s: %s", bot.get("id"), e)
        return pd.DataFrame()
    return _trades_from_execution_activity(rows, bot=bot, since=since)


def _kraken_position_events_frame(
    events: Sequence[Dict[str, Any]],
    *,
    bot: Dict[str, Any],
    since: Optional[pd.Timestamp],
) -> pd.DataFrame:
    """Aggregate Kraken closing fills into realized price-move batches.

    Kraken's position history emits one ``PositionUpdate`` per fill.  A single
    economic reduction can therefore span several rows with the same fill
    timestamp.  Fleet keeps the raw rows in Activity, but the performance
    read-model combines those split fills by timestamp, side and entry basis
    before compounding them.  This prevents an order split into ten fills from
    receiving ten times the weight of an otherwise identical execution.

    Fees, funding and leverage are deliberately excluded from ``pnl_pct``:
    this frame is the unlevered underlying price-move contract.  Exchange PnL
    and fees are retained as audit columns for future complete strategy-return
    inputs.
    """

    def number(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except Exception:
            return None
        return parsed if _is_finite(parsed) else None

    fills: List[Dict[str, Any]] = []
    for event in events:
        if str(event.get("updateReason") or "").lower() != "trade":
            continue
        execution_uid = str(event.get("executionUid") or "").strip()
        execution_price = number(event.get("executionPrice"))
        execution_size = number(event.get("executionSize"))
        ts_raw = event.get("fillTime") or event.get("timestamp")
        try:
            ts = pd.to_datetime(int(ts_raw), unit="ms", utc=True)
        except Exception:
            continue
        if since is not None and ts < since:
            continue
        if not execution_uid or execution_price is None or execution_size is None:
            continue

        old_position = number(event.get("oldPosition")) or 0.0
        new_position = number(event.get("newPosition")) or 0.0
        change = str(event.get("positionChange") or "unknown").strip().lower()
        closes_position = change in {"close", "decrease", "reverse"}
        closed_qty = 0.0
        if closes_position:
            if change == "reverse" or new_position == 0 or old_position * new_position < 0:
                closed_qty = abs(old_position)
            else:
                closed_qty = max(0.0, abs(old_position) - abs(new_position))

        entry_price = number(event.get("oldAverageEntryPrice"))
        if not closes_position:
            entry_price = number(event.get("newAverageEntryPrice")) or execution_price
        pnl_pct: Optional[float] = None
        if closes_position and entry_price is not None and entry_price > 0:
            direction = 1.0 if old_position > 0 else -1.0
            pnl_pct = ((execution_price - entry_price) / entry_price) * 100.0 * direction

        position_side = (
            "long"
            if (old_position > 0 if closes_position else new_position > 0)
            else "short"
        )
        if not closes_position or closed_qty <= 0 or pnl_pct is None:
            continue
        fills.append(
            {
                "trade_id": f"kraken-position:{execution_uid}",
                "venue": "kraken",
                "symbol": str(bot.get("symbol") or event.get("tradeable") or "SOL-USD"),
                "entry_ts": ts,
                "exit_ts": ts,
                "side": position_side,
                "qty": closed_qty if closed_qty > 0 else abs(execution_size),
                "entry_price": entry_price,
                "exit_price": execution_price,
                "pnl_pct": pnl_pct,
                "exit_event": f"kraken_position_{change.replace('nochange', 'no_change')}",
                "strategy": "kraken_exchange_history",
                "strategy_instance": str(bot.get("strategy_instance") or "kraken_bot"),
                "fee": number(event.get("fee")),
                "fee_currency": event.get("feeCurrency"),
                "realized_pnl": number(event.get("realizedPnL")),
                "realized_funding": number(event.get("realizedFunding")),
                "execution_uid": execution_uid,
            }
        )
    if not fills:
        return pd.DataFrame()

    # ``fillTime`` is stable across the split fills of one Kraken execution.
    # Entry basis and side keep unrelated position mutations at the same
    # millisecond from being merged accidentally.
    grouped: List[Dict[str, Any]] = []
    frame = pd.DataFrame(fills).sort_values(["exit_ts", "execution_uid"])
    keys = ["exit_ts", "side", "entry_price", "strategy_instance", "symbol"]
    for key, batch in frame.groupby(keys, sort=True, dropna=False):
        qty = pd.to_numeric(batch["qty"], errors="coerce").fillna(0.0)
        total_qty = float(qty.sum())
        if total_qty <= 0:
            continue
        weights = qty / total_qty
        exit_price = float(
            (pd.to_numeric(batch["exit_price"], errors="coerce") * weights).sum()
        )
        move_pct = float(
            (pd.to_numeric(batch["pnl_pct"], errors="coerce") * weights).sum()
        )
        fee_values = pd.to_numeric(batch["fee"], errors="coerce")
        pnl_values = pd.to_numeric(batch["realized_pnl"], errors="coerce")
        funding_values = pd.to_numeric(batch["realized_funding"], errors="coerce")
        execution_uids = [str(value) for value in batch["execution_uid"] if value]
        exit_ts, side, entry_price, strategy_instance, symbol = key
        grouped.append(
            {
                "trade_id": "kraken-batch:" + ",".join(execution_uids),
                "venue": "kraken",
                "symbol": symbol,
                "entry_ts": exit_ts,
                "exit_ts": exit_ts,
                "side": side,
                "qty": total_qty,
                "entry_price": float(entry_price),
                "exit_price": exit_price,
                "pnl_pct": move_pct,
                "exit_event": "kraken_realized_batch",
                "strategy": "kraken_exchange_position_events",
                "strategy_instance": strategy_instance,
                "fee": float(fee_values.sum()) if fee_values.notna().all() else None,
                "fee_currency": next(
                    (str(value) for value in batch["fee_currency"] if value), None
                ),
                "realized_pnl": (
                    float(pnl_values.sum()) if pnl_values.notna().all() else None
                ),
                "realized_funding": (
                    float(funding_values.sum()) if funding_values.notna().all() else None
                ),
                "execution_uids": execution_uids,
                "fill_count": len(batch),
            }
        )
    return pd.DataFrame(grouped).sort_values("exit_ts") if grouped else pd.DataFrame()


def _kraken_position_event_activity_items(
    events: Sequence[Dict[str, Any]],
    *,
    bot: Dict[str, Any],
    since: Optional[pd.Timestamp],
) -> List[Dict[str, Any]]:
    """Map immutable Kraken position-history rows into Fleet activity items.

    Kraken emits one authoritative row for each position mutation, including
    entries, reductions, closes, reversals, fees and funding realisations.
    These are not synthesized round trips: each item keeps the exchange UID
    and position before/after values so the UI can distinguish a partial
    reduction from a completed trade.
    """

    def number(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if _is_finite(parsed) else None

    items: List[Dict[str, Any]] = []
    for event in events:
        reason = str(event.get("updateReason") or "unknown").strip().lower()
        ts_raw = (
            event.get("fundingRealizationTime") or event.get("timestamp")
            if reason == "fundingrealisation"
            else event.get("fillTime") or event.get("timestamp")
        )
        try:
            ts = pd.to_datetime(int(ts_raw), unit="ms", utc=True)
        except (TypeError, ValueError):
            continue
        if since is not None and ts < since:
            continue

        change = str(event.get("positionChange") or "noChange").strip().lower()
        old_position = number(event.get("oldPosition")) or 0.0
        new_position = number(event.get("newPosition")) or 0.0
        execution_size = number(event.get("executionSize"))
        execution_price = number(event.get("executionPrice"))
        delta = new_position - old_position
        side = "buy" if delta > 0 else ("sell" if delta < 0 else None)
        if side is None and execution_size is not None:
            side = "buy" if new_position >= old_position else "sell"
        action = {
            "open": "entry",
            "increase": "scale_in",
            "decrease": "reduce",
            "close": "close",
            "reverse": "reverse",
            "nochange": "execution",
        }.get(change, "execution")
        if reason == "fundingrealisation":
            action = "funding"
        elif reason == "settlement":
            action = "settlement"

        execution_uid = str(event.get("executionUid") or "").strip()
        stable_id = execution_uid or f"{reason}:{event.get('timestamp')}:{change}:{old_position}:{new_position}"
        fee = number(event.get("fee"))
        realized_pnl = number(event.get("realizedPnL"))
        realized_funding = number(event.get("realizedFunding"))
        closes_position = change in {"decrease", "close", "reverse"}
        entry_price = (
            number(event.get("oldAverageEntryPrice"))
            if closes_position
            else execution_price
        )
        exit_price = execution_price if closes_position else None
        position_ref = str(event.get("accountUid") or "")
        if position_ref:
            position_ref = f"{position_ref}:{event.get('tradeable') or bot.get('symbol') or ''}"
        else:
            position_ref = str(event.get("tradeable") or bot.get("symbol") or "")
        items.append(
            {
                "id": f"kraken-position-event:{stable_id}",
                "kind": "event",
                "t": int(ts.timestamp()),
                "ts": ts.isoformat(),
                "venue": "kraken",
                "symbol": str(event.get("tradeable") or bot.get("symbol") or "SOL-USD"),
                "strategy_instance": str(bot.get("strategy_instance") or "kraken_bot"),
                "bot_id": bot.get("id"),
                "display_name": bot.get("display_name") or "Kraken Legacy",
                "action": action,
                "side": side,
                "qty": abs(execution_size) if execution_size is not None else None,
                "price": execution_price,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "status": "reported",
                "pnl_pct": None,
                "realized_pnl": realized_pnl,
                "fee": fee,
                "fee_currency": event.get("feeCurrency"),
                "realized_funding": realized_funding,
                "position_before": old_position,
                "position_after": new_position,
                "position_ref": position_ref,
                "execution_uid": execution_uid or None,
                "source": "kraken_position_history",
                "color": bot.get("color"),
            }
        )
    return items


def _load_kraken_position_events_for_bot(
    bot: Dict[str, Any],
    *,
    since: Optional[pd.Timestamp],
    limit: int,
    include_funding: bool = True,
) -> List[Dict[str, Any]]:
    """Read the authoritative Kraken event ledger, including funding.

    This is a bounded, cached read-through synchronisation. It never submits
    an order and intentionally falls back to the durable Fleet event store
    when the credentials or the upstream history API are unavailable.
    """
    since_ms = int(since.timestamp() * 1000) if since is not None else None
    cache_key = f"{bot.get('id')}:{since_ms}:{int(limit)}:{int(include_funding)}"
    cached = _KRAKEN_POSITION_EVENT_CACHE.get(cache_key)
    if cached and (time.monotonic() - cached[0]) <= 300.0:
        return [dict(row) for row in cached[1]]
    rows: List[Dict[str, Any]] = []
    if (os.getenv("KRAKEN_FUTURES_KEY") or "").strip() and (
        os.getenv("KRAKEN_FUTURES_SECRET") or ""
    ).strip():
        try:
            from quant.execution.kraken_futures import KrakenFuturesClient

            rows = KrakenFuturesClient().get_position_events(
                symbol=os.getenv("KRAKEN_FUTURES_SYMBOL", "PF_SOLUSD"),
                since_ms=since_ms,
                limit=int(max(1, min(limit, 10_000))),
                include_funding=include_funding,
            )
        except Exception as exc:
            log.warning("Kraken position-event history failed: %s", type(exc).__name__)
    else:
        remote_url = str(
            bot.get("direct_events_url")
            or os.getenv("FLEET_KRAKEN_DIRECT_EVENTS_URL")
            or ""
        ).strip()
        # The raw ledger is account-specific. Never send an unauthenticated
        # request or accidentally reuse quant's unrelated webhook token.
        token = (os.getenv("FLEET_KRAKEN_READ_TOKEN") or "").strip()
        if remote_url and token:
            query: Dict[str, Any] = {
                "limit": int(max(1, min(limit, 10_000))),
                "include_funding": int(include_funding),
            }
            if since_ms is not None:
                query["since_ms"] = since_ms
            req = Request(
                f"{remote_url.rstrip('/')}?{urlencode(query)}",
                method="GET",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                },
            )
            try:
                # A complete historical page can span many Kraken continuation
                # requests. Keep this bounded but do not truncate it at the
                # old short proxy timeout.
                with urlopen(req, timeout=90.0) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                remote_events = payload.get("events") if isinstance(payload, dict) else None
                if isinstance(remote_events, list):
                    rows = [dict(row) for row in remote_events if isinstance(row, dict)]
            except Exception as exc:
                log.warning("Kraken position-event proxy failed: %s", type(exc).__name__)
        elif remote_url:
            log.warning("Kraken position-event proxy is configured without read token")
    _KRAKEN_POSITION_EVENT_CACHE[cache_key] = (time.monotonic(), [dict(row) for row in rows])
    return rows


def _load_kraken_exchange_trades_for_bot(
    bot: Dict[str, Any],
    *,
    since: Optional[pd.Timestamp],
    limit: int,
    allow_remote: bool,
) -> pd.DataFrame:
    """Read Kraken account history locally or through the Kraken bot service."""
    since_ms = int(since.timestamp() * 1000) if since is not None else None
    cache_key = f"{bot.get('id')}:{since_ms}:{int(limit)}:{int(allow_remote)}"
    cached = _KRAKEN_DIRECT_TRADE_CACHE.get(cache_key)
    if cached and (time.monotonic() - cached[0]) <= 30.0:
        return cached[1].copy()

    frame = pd.DataFrame()
    if (os.getenv("KRAKEN_FUTURES_KEY") or "").strip() and (
        os.getenv("KRAKEN_FUTURES_SECRET") or ""
    ).strip():
        try:
            from quant.execution.kraken_futures import KrakenFuturesClient

            events = KrakenFuturesClient().get_position_events(
                symbol=os.getenv("KRAKEN_FUTURES_SYMBOL", "PF_SOLUSD"),
                since_ms=since_ms,
                limit=limit,
            )
            frame = _kraken_position_events_frame(events, bot=bot, since=since)
        except Exception as exc:
            log.warning("direct Kraken position history failed: %s", type(exc).__name__)
    elif allow_remote:
        remote_url = str(
            bot.get("direct_trades_url")
            or os.getenv("FLEET_KRAKEN_DIRECT_TRADES_URL")
            or ""
        ).strip()
        if remote_url:
            query: Dict[str, Any] = {"limit": int(limit)}
            if since_ms is not None:
                query["since_ms"] = since_ms
            token = (
                os.getenv("FLEET_KRAKEN_READ_TOKEN")
                or os.getenv("WEBHOOK_TOKEN")
                or ""
            ).strip()
            headers = {"Accept": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            req = Request(
                f"{remote_url.rstrip('/')}?{urlencode(query)}",
                method="GET",
                headers=headers,
            )
            try:
                with urlopen(req, timeout=12.0) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                trades = payload.get("trades") if isinstance(payload, dict) else None
                if isinstance(trades, list) and trades:
                    frame = pd.DataFrame(trades)
                    frame["entry_ts"] = pd.to_datetime(
                        frame["entry_ts"], utc=True, errors="coerce"
                    )
                    frame["exit_ts"] = pd.to_datetime(
                        frame["exit_ts"], utc=True, errors="coerce"
                    )
            except Exception as exc:
                log.warning("Kraken trade proxy failed: %s", type(exc).__name__)

    _KRAKEN_DIRECT_TRADE_CACHE[cache_key] = (time.monotonic(), frame.copy())
    return frame


def _load_display_trades_for_bot(
    bot: Dict[str, Any],
    *,
    since: Optional[pd.Timestamp] = None,
    limit: int = 5000,
) -> pd.DataFrame:
    """Closed-trade SoT plus inferred and read-only exchange history."""
    closed = _load_closed_trades_for_bot(bot, since=since, limit=limit)
    inferred = _load_execution_activity_trades_for_bot(bot, since=since, limit=limit)
    if inferred.empty:
        out = closed
    elif closed.empty:
        out = inferred
    else:
        # Durable closed_trades remain authoritative. Suppress an inferred copy
        # when its close activity lands within five seconds of an existing close.
        closed_exit: List[Tuple[pd.Timestamp, str]] = []
        for _, row in closed.iterrows():
            ts = pd.to_datetime(row.get("exit_ts"), utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            closed_exit.append(
                (ts, _normalize_symbol_token(str(row.get("symbol") or "")))
            )
        keep: List[bool] = []
        for _, row in inferred.iterrows():
            ts = pd.to_datetime(row.get("exit_ts"), utc=True, errors="coerce")
            symbol = _normalize_symbol_token(str(row.get("symbol") or ""))
            duplicate = any(
                other_symbol == symbol
                and abs((ts - other_ts).total_seconds()) <= 5.0
                for other_ts, other_symbol in closed_exit
            )
            keep.append(not duplicate)
        supplemental = inferred.loc[keep]
        out = (
            closed
            if supplemental.empty
            else pd.concat([closed, supplemental], ignore_index=True)
        )

    if str(bot.get("venue") or "").lower() == "kraken":
        # Performance and Activity must share the same complete, read-only
        # position-event ledger.  The legacy direct-trades proxy is retained
        # only as a compatibility fallback for deployments that have not yet
        # exposed the event endpoint.
        events = _load_kraken_position_events_for_bot(
            bot,
            since=since,
            limit=10_000,
            include_funding=False,
        )
        direct = _kraken_position_events_frame(events, bot=bot, since=since)
        event_ledger_available = not direct.empty
        if not event_ledger_available:
            direct = _load_kraken_exchange_trades_for_bot(
                bot,
                since=since,
                limit=limit,
                allow_remote=True,
            )
        if not direct.empty:
            out = (
                direct
                if event_ledger_available or out.empty
                else pd.concat([out, direct], ignore_index=True)
            )

    if out.empty:
        return out
    if "trade_id" in out.columns:
        out = out.drop_duplicates(subset=["trade_id"], keep="first")
    return out.sort_values("exit_ts").tail(int(max(1, limit)))


def _is_finite(v: Any) -> bool:
    try:
        x = float(v)
    except Exception:
        return False
    return x == x and abs(x) != float("inf")


def _downsample_points(
    points: List[Dict[str, Any]],
    *,
    max_points: int = 180,
    value_key: str = "equity",
    min_interval_sec: int = 900,
) -> List[Dict[str, Any]]:
    """Thin equity snapshots into a drawable line.

    Health polls previously wrote ~30–60s snapshots; without thinning the chart
    looks like a dense point cloud (often only on the right of a wider window).
    Keep plateau edges, enforce a minimum spacing, then bucket to max_points.
    """
    if not points:
        return []

    # 1) Collapse flat plateaus — keep start + end of each constant run.
    simple: List[Dict[str, Any]] = []
    for p in points:
        item = {"t": int(p["t"]), value_key: p.get(value_key)}
        if not simple:
            simple.append(item)
            continue
        try:
            same = abs(float(item[value_key]) - float(simple[-1][value_key])) < 1e-9
        except Exception:
            same = False
        if same:
            if len(simple) >= 2:
                try:
                    prev_flat = (
                        abs(float(simple[-1][value_key]) - float(simple[-2][value_key])) < 1e-9
                    )
                except Exception:
                    prev_flat = False
                if prev_flat:
                    simple[-1] = item  # move plateau end forward
                else:
                    simple.append(item)  # first extension of plateau
            else:
                simple.append(item)
        else:
            simple.append(item)

    if len(simple) == 1:
        return simple

    # 2) Hard min interval for dense history. Tiny live wobbles must NOT keep
    # ~20s points (that looked like a point cloud). Already-sparse / short
    # unique paths keep their shape.
    min_iv = max(60, int(min_interval_sec))
    span = int(simple[-1]["t"]) - int(simple[0]["t"])
    max_by_interval = max(3, (span // min_iv) + 2)
    if len(simple) <= max_by_interval:
        spaced = simple
    else:
        spaced = [simple[0]]
        for p in simple[1:-1]:
            if int(p["t"]) - int(spaced[-1]["t"]) >= min_iv:
                spaced.append(p)
        if int(simple[-1]["t"]) != int(spaced[-1]["t"]):
            spaced.append(simple[-1])

    if len(spaced) <= max_points:
        return spaced
    if max_points < 3:
        return spaced[:max_points]

    # 3) Time-bucket to max_points.
    t0 = int(spaced[0]["t"])
    t1 = int(spaced[-1]["t"])
    if t1 <= t0:
        return [spaced[0], spaced[-1]]
    bucket = max(1, (t1 - t0) // (max_points - 1))
    out: List[Dict[str, Any]] = [spaced[0]]
    next_t = t0 + bucket
    for p in spaced[1:-1]:
        if int(p["t"]) >= next_t:
            out.append(p)
            next_t = int(p["t"]) + bucket
    out.append(spaced[-1])
    dedup: Dict[int, Dict[str, Any]] = {}
    for p in out:
        dedup[int(p["t"])] = p
    return [dedup[k] for k in sorted(dedup)]


def _choose_grid_sec(span_sec: int, *, target_points: int = 280) -> int:
    """Pick a regular clock step so the chart axis is uniform, not event-spaced."""
    span = max(60, int(span_sec))
    target = max(40, int(target_points))
    # Prefer human-friendly steps, then stretch if the window is huge.
    for step in (60, 300, 900, 1800, 3600, 7200, 14400, 21600, 43200, 86400):
        if (span // step) + 1 <= target:
            return step
    return max(86400, span // max(1, target - 1))


def _forward_fill_on_grid(
    points: List[Dict[str, Any]],
    *,
    value_key: str,
    t0: int,
    t1: int,
    interval_sec: int,
) -> List[Dict[str, Any]]:
    """Resample sparse snapshots onto a uniform UTC grid with forward-fill.

    Points before the series' first real observation are omitted (no invented
    history). From the first observation through ``t1``, values hold until the
    next observation — so bots share one clock even when flat.
    """
    if not points or int(t1) < int(t0) or int(interval_sec) <= 0:
        return []
    by_t: Dict[int, float] = {}
    for p in points:
        try:
            t = int(p["t"])
            v = float(p[value_key])
        except Exception:
            continue
        if not _is_finite(v):
            continue
        by_t[t] = v
    if not by_t:
        return []
    times = sorted(by_t)
    first_t = times[0]
    end = int(t1)
    step = max(1, int(interval_sec))
    # First emitted sample is the real first observation (may be mid-bucket).
    start = max(int(first_t), int(t0))
    out: List[Dict[str, Any]] = []
    idx = 0
    last = by_t[times[0]]
    # Walk a regular grid from the next step boundary after start, but always
    # keep the exact first observation so the line origin is honest.
    out.append({"t": start, value_key: round(last, 6)})
    # Next grid tick strictly after start, aligned to epoch step.
    nxt = ((start // step) + 1) * step
    t = nxt
    while t <= end:
        while idx + 1 < len(times) and times[idx + 1] <= t:
            idx += 1
            last = by_t[times[idx]]
        out.append({"t": int(t), value_key: round(last, 6)})
        t += step
    # Pin the window end so all series share the same last timestamp.
    while idx + 1 < len(times) and times[idx + 1] <= end:
        idx += 1
        last = by_t[times[idx]]
    if end >= start:
        if out and int(out[-1]["t"]) == end:
            out[-1] = {"t": end, value_key: round(last, 6)}
        else:
            out.append({"t": end, value_key: round(last, 6)})
    dedup: Dict[int, Dict[str, Any]] = {}
    for p in out:
        dedup[int(p["t"])] = p
    return [dedup[k] for k in sorted(dedup)]


def _pct_from_abs_curve(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rebase absolute equity to % without thinning (preserves uniform grid)."""
    if not points:
        return []
    base = float(points[0]["equity"])
    if base <= 0:
        return [{"t": int(p["t"]), "equity_pct": 0.0} for p in points]
    return [
        {
            "t": int(p["t"]),
            "equity_pct": round((float(p["equity"]) / base - 1.0) * 100.0, 6),
        }
        for p in points
    ]


def _shared_curve_window(
    series: List[Dict[str, Any]],
    *,
    hours: Optional[float],
    now_ts: int,
) -> Tuple[int, int, int]:
    """Return (t0, t1, interval_sec) for a uniform shared clock."""
    t1 = int(now_ts)
    if hours is not None and float(hours) > 0:
        t0 = t1 - int(float(hours) * 3600)
    else:
        firsts: List[int] = []
        for s in series:
            for key in (
                "account_curve_abs",
                "account_curve",
                "trade_curve",
                "price_move_curve_bps",
                "strategy_curve",
                "corrected_curve",
            ):
                curve = s.get(key) or []
                if curve:
                    firsts.append(int(curve[0]["t"]))
                    break
        t0 = min(firsts) if firsts else t1 - 7 * 86400
    if t0 >= t1:
        t0 = t1 - 3600
    interval = _choose_grid_sec(t1 - t0)
    return t0, t1, interval


def _align_series_to_shared_clock(
    series: List[Dict[str, Any]],
    *,
    hours: Optional[float],
    now_ts: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Forward-fill every bot onto one regular UTC grid for true-time charts."""
    t0, t1, interval = _shared_curve_window(series, hours=hours, now_ts=now_ts)
    aligned: List[Dict[str, Any]] = []
    for s in series:
        row = dict(s)
        abs_curve = s.get("account_curve_abs") or []
        if abs_curve:
            # Never invent history past a bot's last real observation: filling
            # a stale series to "now" painted month-old equity as current.
            fill_end = min(t1, int(abs_curve[-1]["t"]))
            row["account_curve_abs"] = _forward_fill_on_grid(
                abs_curve, value_key="equity", t0=t0, t1=fill_end, interval_sec=interval
            )
            # Forward-fill the precomputed % curve (TWR, deposit-adjusted).
            # Recomputing % from the abs curve here reintroduced deposit
            # jumps as fake returns and silently discarded the TWR result.
            pct_curve = s.get("account_curve") or []
            if pct_curve:
                row["account_curve"] = _forward_fill_on_grid(
                    pct_curve,
                    value_key="equity_pct",
                    t0=t0,
                    t1=fill_end,
                    interval_sec=interval,
                )
            else:
                row["account_curve"] = _pct_from_abs_curve(row["account_curve_abs"])
        else:
            row["account_curve_abs"] = []
            row["account_curve"] = []
        trade_curve = s.get("trade_curve") or []
        if trade_curve:
            row["trade_curve"] = _forward_fill_on_grid(
                trade_curve,
                value_key="equity_pct",
                t0=t0,
                t1=min(t1, int(trade_curve[-1]["t"])),
                interval_sec=interval,
            )
        price_move_curve = s.get("price_move_curve_bps") or []
        if price_move_curve:
            row["price_move_curve_bps"] = _forward_fill_on_grid(
                price_move_curve,
                value_key="equity_pct",
                t0=t0,
                t1=min(t1, int(price_move_curve[-1]["t"])),
                interval_sec=interval,
            )
        else:
            row["price_move_curve_bps"] = []
        strategy_curve = s.get("strategy_curve") or []
        if strategy_curve:
            row["strategy_curve"] = _forward_fill_on_grid(
                strategy_curve,
                value_key="equity_pct",
                t0=t0,
                t1=min(t1, int(strategy_curve[-1]["t"])),
                interval_sec=interval,
            )
        else:
            row["strategy_curve"] = []
        corrected_curve = s.get("corrected_curve") or []
        if corrected_curve:
            row["corrected_curve"] = _forward_fill_on_grid(
                corrected_curve,
                value_key="equity_pct",
                t0=t0,
                t1=min(t1, int(corrected_curve[-1]["t"])),
                interval_sec=interval,
            )
        else:
            row["corrected_curve"] = []
        aligned.append(row)
    meta = {
        "t0": t0,
        "t1": t1,
        "interval_sec": interval,
        "note": "forward_fill_uniform_utc_grid",
    }
    return aligned, meta


def _absolute_account_curve(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw = [
        {"t": int(p["t"]), "equity": round(float(p["equity"]), 6)}
        for p in points
        if p.get("equity") is not None and _is_finite(p.get("equity"))
    ]
    return _downsample_points(raw, max_points=180, value_key="equity", min_interval_sec=900)


def _stitch_live_equity(
    points: List[Dict[str, Any]],
    *,
    live_equity: Optional[float],
    now_ts: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if live_equity is None:
        return list(points)
    try:
        eq = float(live_equity)
    except Exception:
        return list(points)
    if not _is_finite(eq) or eq <= 0:
        return list(points)
    t = int(now_ts if now_ts is not None else pd.Timestamp.now("UTC").timestamp())
    out = list(points)
    if out and int(out[-1]["t"]) >= t:
        out[-1] = {
            "t": t,
            "equity": eq,
            "currency": out[-1].get("currency"),
            "account": out[-1].get("account"),
            "source": "live_stitch",
        }
    else:
        out.append({"t": t, "equity": eq, "source": "live_stitch"})
    return out


def _load_account_points_for_bot(
    bot: Dict[str, Any],
    *,
    since: Optional[pd.Timestamp] = None,
) -> List[Dict[str, Any]]:
    venue = str(bot.get("venue") or "kucoin")
    instance = str(bot.get("strategy_instance") or "")
    snap_account = str(bot.get("equity_account") or instance)
    acct_pts = _load_equity_snapshots(venue=venue, account=snap_account, since=since)
    if not acct_pts and snap_account != instance:
        acct_pts = _load_equity_snapshots(venue=venue, account=instance, since=since)
    if not acct_pts and venue == "kraken":
        acct_pts = _load_equity_snapshots(venue=venue, account="main", since=since)
        if not acct_pts:
            acct_pts = _load_equity_snapshots(venue=venue, account=None, since=since)
    return acct_pts


def _compounded_trade_curve(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build compounded return % series rebased to 0 at first trade in window."""
    if df is None or df.empty:
        return [], {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_count": 0,
            "win_rate": None,
            "profit_factor": None,
            "wins": 0,
            "losses": 0,
        }

    work = df.copy()
    work["exit_ts"] = pd.to_datetime(work["exit_ts"], utc=True, errors="coerce")
    work["pnl_pct"] = pd.to_numeric(work["pnl_pct"], errors="coerce")
    work = work.dropna(subset=["exit_ts", "pnl_pct"]).sort_values("exit_ts")
    if work.empty:
        return [], {
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trade_count": 0,
            "win_rate": None,
            "profit_factor": None,
            "wins": 0,
            "losses": 0,
        }

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    points: List[Dict[str, Any]] = []
    # Anchor at 0% just before first trade
    first_ts = work.iloc[0]["exit_ts"]
    t0 = int((pd.Timestamp(first_ts) - pd.Timedelta(seconds=1)).timestamp())
    points.append({"t": t0, "equity_pct": 0.0})

    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0

    for _, row in work.iterrows():
        pnl = float(row["pnl_pct"])
        equity *= 1.0 + (pnl / 100.0)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        t = int(pd.Timestamp(row["exit_ts"]).timestamp())
        points.append({"t": t, "equity_pct": round((equity - 1.0) * 100.0, 6)})
        if pnl > 0:
            wins += 1
            gross_win += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += abs(pnl)

    n = wins + losses
    profit_factor: Optional[float] = None
    if gross_loss > 0:
        profit_factor = round(gross_win / gross_loss, 6)
    stats = {
        "return_pct": round((equity - 1.0) * 100.0, 6),
        "max_drawdown_pct": round(max_dd * 100.0, 6),
        "trade_count": int(len(work)),
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / n, 6) if n else None,
        "profit_factor": profit_factor,
    }
    return points, stats


def _strategy_return_payload(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a leverage/notional-aware strategy curve only from complete rows.

    A price move is not a strategy return.  Fleet therefore refuses to infer
    leverage from current account state or multiply by a configured default.
    Producers may opt in by persisting a net ``strategy_return_pct`` for every
    displayed economic trade together with ``strategy_return_complete=true``.
    Until then the metric is explicitly unavailable.
    """
    unavailable = {
        "strategy_curve": [],
        "strategy_meta": {
            "available": False,
            "method": "unavailable",
            "reason": "historical_notional_leverage_or_cost_basis_incomplete",
            "costs_included": False,
            "funding_included": False,
            "trade_count": 0,
        },
    }
    if df is None or df.empty:
        return {**unavailable, "strategy_meta": {**unavailable["strategy_meta"], "reason": "no_completed_trades"}}
    required = {"strategy_return_pct", "strategy_return_complete"}
    if not required.issubset(df.columns):
        return unavailable
    complete = df["strategy_return_complete"].fillna(False).astype(bool)
    returns = pd.to_numeric(df["strategy_return_pct"], errors="coerce")
    if not bool(complete.all()) or bool(returns.isna().any()):
        return unavailable
    work = df.copy()
    work["pnl_pct"] = returns
    curve, stats = _compounded_trade_curve(work)
    return {
        "strategy_curve": curve,
        "strategy_meta": {
            "available": bool(curve),
            "method": "net_realized_on_strategy_risk_capital",
            "reason": None if curve else "no_completed_trades",
            "costs_included": True,
            "funding_included": True,
            "trade_count": int(stats["trade_count"]),
            "return_pct": stats["return_pct"],
        },
    }


def _curve_value_at(points: List[Dict[str, Any]], timestamp: int) -> Optional[float]:
    value: Optional[float] = None
    for point in points:
        if int(point["t"]) > int(timestamp):
            break
        if _is_finite(point.get("equity_pct")):
            value = float(point["equity_pct"])
    return value


def _risk_normalized_allocation_payload(
    series: List[Dict[str, Any]],
    portfolio_corrected: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare corrected portfolio growth with an equal-risk strategy basket.

    Every included strategy is first scaled to the median realized volatility
    on the common time window.  Only then are interval returns equal weighted.
    The allocation contribution is the geometric relative return of the real
    corrected portfolio versus that benchmark, never an average of headline
    percentages.
    """
    eligible = [
        row
        for row in series
        if (row.get("strategy_meta") or {}).get("available")
        and len(row.get("strategy_curve") or []) >= 2
    ]
    excluded = [str(row.get("id")) for row in series if row not in eligible]
    base = {
        "available": False,
        "method": "equal_weight_common_window_equal_risk",
        "reason": "insufficient_complete_strategy_returns",
        "included_bot_ids": [str(row.get("id")) for row in eligible],
        "excluded_bot_ids": excluded,
        "benchmark_curve": [],
        "contribution_curve": [],
        "contribution_pct": None,
        "common_start": None,
        "common_end": None,
    }
    if len(eligible) < 2 or len(portfolio_corrected) < 2:
        return base
    start = max(int(row["strategy_curve"][0]["t"]) for row in eligible)
    end = min(int(row["strategy_curve"][-1]["t"]) for row in eligible)
    start = max(start, int(portfolio_corrected[0]["t"]))
    end = min(end, int(portfolio_corrected[-1]["t"]))
    if end <= start:
        return {**base, "reason": "no_common_time_window"}
    times = sorted(
        {
            int(point["t"])
            for row in eligible
            for point in row["strategy_curve"]
            if start <= int(point["t"]) <= end
        }
        | {
            int(point["t"])
            for point in portfolio_corrected
            if start <= int(point["t"]) <= end
        }
        | {start, end}
    )
    if len(times) < 3:
        return {**base, "reason": "insufficient_common_observations"}

    return_paths: List[List[float]] = []
    volatilities: List[float] = []
    for row in eligible:
        values = [_curve_value_at(row["strategy_curve"], timestamp) for timestamp in times]
        if any(value is None for value in values):
            return {**base, "reason": "incomplete_common_time_window"}
        growth = [1.0 + float(value) / 100.0 for value in values]
        increments = [growth[index] / growth[index - 1] - 1.0 for index in range(1, len(growth))]
        volatility = float(pd.Series(increments, dtype="float64").std(ddof=0))
        if not _is_finite(volatility) or volatility <= 0:
            return {**base, "reason": "strategy_risk_not_estimable"}
        return_paths.append(increments)
        volatilities.append(volatility)

    target_volatility = float(pd.Series(volatilities).median())
    benchmark_growth = 1.0
    benchmark_curve = [{"t": times[0], "equity_pct": 0.0}]
    for index, timestamp in enumerate(times[1:]):
        scaled = [
            path[index] * target_volatility / volatility
            for path, volatility in zip(return_paths, volatilities)
        ]
        benchmark_growth *= 1.0 + sum(scaled) / len(scaled)
        benchmark_curve.append(
            {"t": timestamp, "equity_pct": round((benchmark_growth - 1.0) * 100.0, 6)}
        )

    portfolio_values = [_curve_value_at(portfolio_corrected, timestamp) for timestamp in times]
    if any(value is None for value in portfolio_values):
        return {**base, "reason": "corrected_portfolio_window_incomplete"}
    portfolio_base = 1.0 + float(portfolio_values[0]) / 100.0
    contribution_curve: List[Dict[str, Any]] = []
    for timestamp, value, benchmark in zip(times, portfolio_values, benchmark_curve):
        portfolio_growth = (1.0 + float(value) / 100.0) / portfolio_base
        benchmark_growth_at_t = 1.0 + float(benchmark["equity_pct"]) / 100.0
        contribution_curve.append(
            {
                "t": timestamp,
                "equity_pct": round((portfolio_growth / benchmark_growth_at_t - 1.0) * 100.0, 6),
            }
        )
    return {
        **base,
        "available": True,
        "reason": None,
        "benchmark_curve": benchmark_curve,
        "contribution_curve": contribution_curve,
        "contribution_pct": contribution_curve[-1]["equity_pct"],
        "common_start": start,
        "common_end": end,
        "target_interval_volatility": round(target_volatility, 10),
    }


def _load_equity_snapshots(
    *,
    venue: str,
    account: Optional[str] = None,
    since: Optional[pd.Timestamp] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    where = ["venue = %(venue)s"]
    params: Dict[str, Any] = {"venue": str(venue)}
    if account:
        where.append("account = %(account)s")
        params["account"] = str(account)
    if since is not None:
        where.append("ts >= %(since)s")
        params["since"] = since.to_pydatetime()
    if limit is None:
        # Performance curves need the complete requested time span. The result
        # is downsampled only after loading, so no historical boundary vanishes.
        sql = f"""
            select ts, equity, currency, account, source
            from equity_snapshots
            where {' and '.join(where)}
            order by ts asc
        """
    else:
        params["limit"] = int(max(1, limit))
        # Explicit bounded reads (for example Capitalization's latest point)
        # select newest rows first, then restore chronological curve order.
        sql = f"""
            select ts, equity, currency, account, source
            from (
                select ts, equity, currency, account, source
                from equity_snapshots
                where {' and '.join(where)}
                order by ts desc
                limit %(limit)s
            ) latest
            order by ts asc
        """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
        out = []
        for r in rows:
            ts = r[0]
            eq = r[1]
            if ts is None or eq is None:
                continue
            out.append(
                {
                    "t": int(pd.Timestamp(ts, tz="UTC").timestamp())
                    if getattr(ts, "tzinfo", None) is None
                    else int(pd.Timestamp(ts).timestamp()),
                    "equity": float(eq),
                    "currency": r[2],
                    "account": r[3],
                    "source": r[4],
                }
            )
        return out
    except Exception as e:
        log.warning("fleet equity_snapshots load failed: %s", e)
        return []


def _load_cashflow_data(
    *,
    venue: str,
    account: str,
    since: pd.Timestamp,
    until: pd.Timestamp,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load confirmed account cashflows and their authoritative coverage."""
    state_sql = """
        select coverage_start, coverage_end, last_success_at, last_error, source
        from cashflow_sync_state
        where venue = %(venue)s and account = %(account)s
        limit 1
    """
    flow_sql = """
        select ts, amount, reporting_amount, currency, direction, flow_type,
               status, equity_after, source_ref
        from cashflow_events
        where venue = %(venue)s
          and account = %(account)s
          and ts >= %(since)s
          and ts <= %(until)s
          and lower(status) in ('completed', 'success')
        order by ts asc, event_id asc
    """
    params = {
        "venue": str(venue),
        "account": str(account),
        "since": since.to_pydatetime(),
        "until": until.to_pydatetime(),
    }
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(state_sql, params)
            state_row = cur.fetchone()
            cur.execute(flow_sql, params)
            rows = cur.fetchall() or []
    except Exception as exc:
        log.warning(
            "fleet cashflow load failed venue=%s account=%s error=%s",
            venue,
            account,
            type(exc).__name__,
        )
        return [], None

    state = None
    if state_row:
        state = {
            "coverage_start": state_row[0],
            "coverage_end": state_row[1],
            "last_success_at": state_row[2],
            "last_error": state_row[3],
            "source": state_row[4],
        }
    flows = [
        {
            "t": int(pd.Timestamp(row[0]).timestamp()),
            "amount": float(row[1]),
            "reporting_amount": (
                float(row[2]) if row[2] is not None else None
            ),
            "currency": row[3],
            "direction": row[4],
            "flow_type": row[5],
            "status": row[6],
            "equity_after": float(row[7]) if row[7] is not None else None,
            "source_ref": row[8],
            "venue": venue,
            "account": account,
        }
        for row in rows
    ]
    return flows, state


def _cashflow_corrected_curve(
    points: List[Dict[str, Any]],
    cashflows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compound equity growth after removing confirmed deposits/withdrawals."""
    clean = [
        {"t": int(point["t"]), "equity": float(point["equity"])}
        for point in points
        if point.get("equity") is not None
        and _is_finite(point.get("equity"))
        and float(point["equity"]) > 0
    ]
    clean.sort(key=lambda point: point["t"])
    if len(clean) < 2:
        return []
    flows = sorted(
        [
            flow
            for flow in cashflows
            if flow.get("reporting_amount") is not None
            and _is_finite(flow.get("reporting_amount"))
        ],
        key=lambda flow: int(flow["t"]),
    )
    growth = 1.0
    flow_index = 0
    last_value = clean[0]["equity"]
    out: List[Dict[str, Any]] = [{"t": clean[0]["t"], "equity_pct": 0.0}]
    for point in clean[1:]:
        interval_flows: List[Dict[str, Any]] = []
        while flow_index < len(flows) and int(flows[flow_index]["t"]) <= point["t"]:
            if int(flows[flow_index]["t"]) > clean[0]["t"]:
                interval_flows.append(flows[flow_index])
            flow_index += 1
        unresolved = 0.0
        for flow in interval_flows:
            amount = float(flow["reporting_amount"])
            equity_after = flow.get("equity_after")
            if (
                equity_after is None
                or not _is_finite(equity_after)
                or float(equity_after) <= 0
            ):
                unresolved += amount
                continue
            before = float(equity_after) - amount
            if before <= 0 or last_value <= 0:
                return []
            growth *= before / last_value
            last_value = float(equity_after)
        adjusted_end = float(point["equity"]) - unresolved
        if adjusted_end <= 0 or last_value <= 0:
            return []
        growth *= adjusted_end / last_value
        last_value = float(point["equity"])
        out.append(
            {"t": int(point["t"]), "equity_pct": round((growth - 1.0) * 100.0, 6)}
        )
    return out


def _bot_corrected_payload(
    *,
    abs_points: List[Dict[str, Any]],
    venue: str,
    account: str,
    since: Optional[pd.Timestamp],
    until_ts: int,
    bot_status: Optional[str],
    bot_disabled: bool,
) -> Dict[str, Any]:
    """Return a verified-ledger curve, or a conservative jump-TWR fallback."""
    clean_abs = [
        {"t": int(point["t"]), "equity": float(point["equity"])}
        for point in abs_points
        if point.get("equity") is not None
        and _is_finite(point.get("equity"))
        and float(point["equity"]) > 0
    ]
    clean_abs.sort(key=lambda point: point["t"])
    load_start = (
        since
        if since is not None
        else pd.Timestamp(
            clean_abs[0]["t"] if clean_abs else 0,
            unit="s",
            tz="UTC",
        )
    )
    flows, state = _load_cashflow_data(
        venue=venue,
        account=account,
        since=load_start,
        until=pd.Timestamp(int(until_ts), unit="s", tz="UTC"),
    )
    api_flows = [
        {
            "t": int(flow["t"]),
            "direction": flow.get("direction"),
            "reporting_amount": flow.get("reporting_amount"),
            "currency": flow.get("currency"),
            "flow_type": flow.get("flow_type"),
        }
        for flow in flows
    ]
    reportable = [
        flow
        for flow in flows
        if flow.get("reporting_amount") is not None
        and _is_finite(flow.get("reporting_amount"))
    ]
    base_meta: Dict[str, Any] = {
        "method": "unavailable",
        "available": False,
        "reason": None,
        "flow_count": len(reportable),
        "net_cashflow": round(
            sum(float(flow["reporting_amount"]) for flow in reportable), 6
        ),
        "source": state.get("source") if state else None,
    }

    def result(
        curve: List[Dict[str, Any]],
        *,
        method: str,
        available: bool,
        reason: Optional[str],
    ) -> Dict[str, Any]:
        return {
            "corrected_curve": curve,
            "corrected_meta": {
                **base_meta,
                "method": method,
                "available": available,
                "reason": reason,
            },
            "cashflows": api_flows,
        }

    if bot_disabled:
        return result([], method="unavailable", available=False, reason="disabled")
    if str(bot_status or "").strip().lower() == "down":
        return result([], method="unavailable", available=False, reason="inactive")
    if len(clean_abs) < 2:
        return result(
            [],
            method="unavailable",
            available=False,
            reason="insufficient_equity",
        )

    curve_start = int(clean_abs[0]["t"])
    curve_end = int(clean_abs[-1]["t"])
    coverage_ok = bool(
        state
        and state.get("last_success_at")
        and state.get("coverage_start") is not None
        and state.get("coverage_end") is not None
        and int(pd.Timestamp(state["coverage_start"]).timestamp()) <= curve_start
        and int(pd.Timestamp(state["coverage_end"]).timestamp()) >= curve_end
    )
    if coverage_ok:
        ledger_curve = _cashflow_corrected_curve(clean_abs, flows)
        if ledger_curve:
            return result(
                _downsample_points(
                    ledger_curve,
                    max_points=180,
                    value_key="equity_pct",
                    min_interval_sec=900,
                ),
                method="ledger",
                available=True,
                reason=None,
            )

    fallback = _twr_account_curve(clean_abs)
    if not fallback:
        return result(
            [],
            method="unavailable",
            available=False,
            reason="insufficient_equity",
        )
    return result(
        _downsample_points(
            fallback,
            max_points=180,
            value_key="equity_pct",
            min_interval_sec=900,
        ),
        method="jump_twr",
        available=True,
        reason=None if coverage_ok else "ledger_coverage_incomplete",
    )


def _twr_account_curve(
    points: List[Dict[str, Any]],
    *,
    jump_threshold_pct: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Time-weighted return %: deposits/withdrawals are excluded.

    Any single-step equity change larger than FLEET_TWR_JUMP_PCT (default 10%)
    is treated as a transfer, not trading PnL, and contributes 0% to the
    compounded curve. The absolute $ curve stays honest about capital steps;
    this % curve shows only what trading earned (Martin's decision 2026-07-22).
    """
    if jump_threshold_pct is None:
        try:
            jump_threshold_pct = max(
                1.0, float(os.getenv("FLEET_TWR_JUMP_PCT", "10"))
            )
        except Exception:
            jump_threshold_pct = 10.0
    clean = [
        {"t": int(p["t"]), "equity": float(p["equity"])}
        for p in points
        if p.get("equity") is not None
        and _is_finite(p.get("equity"))
        and float(p["equity"]) > 0
    ]
    clean.sort(key=lambda p: p["t"])
    if not clean:
        return []
    growth = 1.0
    out = [{"t": clean[0]["t"], "equity_pct": 0.0}]
    for prev, cur in zip(clean, clean[1:]):
        r = cur["equity"] / prev["equity"] - 1.0
        if abs(r) * 100.0 > jump_threshold_pct:
            r = 0.0  # transfer (deposit/withdrawal), not PnL
        growth *= 1.0 + r
        out.append({"t": cur["t"], "equity_pct": round((growth - 1.0) * 100.0, 6)})
    return out


def _normalize_account_curve(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not points:
        return []
    base = float(points[0]["equity"])
    if base <= 0:
        raw = [{"t": int(p["t"]), "equity_pct": 0.0} for p in points]
    else:
        raw = [
            {"t": int(p["t"]), "equity_pct": round((float(p["equity"]) / base - 1.0) * 100.0, 6)}
            for p in points
        ]
    return _downsample_points(raw, max_points=180, value_key="equity_pct", min_interval_sec=900)


def _equal_weight_pct_mean(
    curves: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Equal-weight, forward-filled mean on the union of curve timestamps."""
    clean: List[List[Dict[str, Any]]] = []
    for curve in curves:
        points = [
            {"t": int(point["t"]), "equity_pct": float(point["equity_pct"])}
            for point in curve
            if point.get("equity_pct") is not None
            and _is_finite(point.get("equity_pct"))
        ]
        if points:
            clean.append(points)
    if not clean:
        return []
    times = sorted({int(point["t"]) for curve in clean for point in curve})
    indices = [0] * len(clean)
    last_values: List[Optional[float]] = [None] * len(clean)
    out: List[Dict[str, Any]] = []
    for timestamp in times:
        values: List[float] = []
        for index, curve in enumerate(clean):
            while (
                indices[index] < len(curve)
                and int(curve[indices[index]]["t"]) <= timestamp
            ):
                last_values[index] = float(curve[indices[index]]["equity_pct"])
                indices[index] += 1
            if last_values[index] is not None:
                values.append(float(last_values[index]))
        if values:
            out.append(
                {
                    "t": timestamp,
                    "equity_pct": round(sum(values) / len(values), 6),
                }
            )
    return out

def _build_portfolio_curve(
    series: List[Dict[str, Any]],
    *,
    thin: bool = True,
) -> Dict[str, Any]:
    """Build portfolio cash equity plus organic, deposit-adjusted growth.

    Absolute (Equity $): sum forward-filled equities (nominal USD/USDT 1:1).

    Percent (Return): time-weighted growth of the summed portfolio equity.
    Large capital steps are treated as transfers by the same jump filter used
    for individual accounts, while organic gains compound by portfolio size.
    """
    curves: List[List[Dict[str, Any]]] = []
    live_sum = 0.0
    live_n = 0
    for s in series:
        abs_curve = s.get("account_curve_abs") or []
        if abs_curve:
            curves.append(
                [
                    {"t": int(p["t"]), "equity": float(p["equity"])}
                    for p in abs_curve
                    if p.get("equity") is not None and _is_finite(p.get("equity"))
                ]
            )
        le = s.get("live_equity")
        if le is not None and _is_finite(le):
            live_sum += float(le)
            live_n += 1

    if not curves:
        return {
            "id": "portfolio",
            "display_name": "Portfolio",
            "color": "#ffffff",
            "currency": "USD",
            "live_equity": live_sum if live_n else None,
            "account_curve_abs": [],
            "account_curve": [],
            "bot_count": 0,
            "note": "portfolio_twr_from_abs_sum",
        }

    times = sorted({int(p["t"]) for c in curves for p in c})
    abs_idxs = [0] * len(curves)
    abs_lasts: List[Optional[float]] = [None] * len(curves)
    portfolio_abs: List[Dict[str, Any]] = []
    for t in times:
        total = 0.0
        contributing = 0
        for i, curve in enumerate(curves):
            while abs_idxs[i] < len(curve) and int(curve[abs_idxs[i]]["t"]) <= t:
                abs_lasts[i] = float(curve[abs_idxs[i]]["equity"])
                abs_idxs[i] += 1
            if abs_lasts[i] is not None:
                total += abs_lasts[i]
                contributing += 1
        if contributing:
            portfolio_abs.append({"t": t, "equity": round(total, 6)})

    portfolio_pct = _twr_account_curve(portfolio_abs)

    # Only thin dense boards when caller has not already put curves on a
    # uniform shared clock (thin=False preserves true-time regularity).
    if thin and len(portfolio_abs) > 80:
        portfolio_abs = _downsample_points(
            portfolio_abs, max_points=220, value_key="equity", min_interval_sec=600
        )
    if thin and len(portfolio_pct) > 80:
        portfolio_pct = _downsample_points(
            portfolio_pct, max_points=220, value_key="equity_pct", min_interval_sec=600
        )
    return {
        "id": "portfolio",
        "display_name": "Portfolio",
        "color": "#ffffff",
        "currency": "USD",
        "live_equity": live_sum if live_n else (
            float(portfolio_abs[-1]["equity"]) if portfolio_abs else None
        ),
        "account_curve_abs": portfolio_abs,
        "account_curve": portfolio_pct,
        "bot_count": len(curves),
        "note": "portfolio_twr_from_abs_sum",
    }


def build_fleet_performance(
    *,
    hours: Optional[float] = 168.0,
    instance_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    registry = fleet_bot_registry()
    if instance_ids:
        want = {str(x) for x in instance_ids}
        registry = [
            b
            for b in registry
            if str(b.get("id")) in want or str(b.get("strategy_instance")) in want
        ]

    since = _effective_since(hours)
    # One health fan-out for live equity stitch (also keeps curves fresh).
    health_by_id: Dict[str, Dict[str, Any]] = {}
    try:
        probed = list_fleet_bots(probe_health=True)
        for row in probed.get("bots") or []:
            health_by_id[str(row.get("id"))] = row if isinstance(row, dict) else {}
    except Exception as e:
        log.warning("fleet performance health probe failed: %s", e)

    now_ts = int(pd.Timestamp.now("UTC").timestamp())
    # Live equity is only stitched onto curves whose snapshots are fresh.
    # Stitching onto a stale seed fabricates a flat-line-plus-cliff "curve"
    # (audit 2026-07-22: +261% jumps from Nov-2025 seeds). Stale series are
    # reported honestly via needs_backfill + snapshot_age_sec instead.
    try:
        stitch_max_age_sec = max(
            300.0, float(os.getenv("FLEET_STITCH_MAX_AGE_SEC", "3600"))
        )
    except Exception:
        stitch_max_age_sec = 3600.0
    # Trade/equity reads are independent read-only I/O.  Loading them in
    # registry order through a bounded pool removes the previous additive
    # database/proxy latency without changing the deterministic response order.
    def load_bot_inputs(bot: Dict[str, Any]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        return (
            _load_display_trades_for_bot(bot, since=since),
            _load_account_points_for_bot(bot, since=since),
        )

    if registry:
        with ThreadPoolExecutor(max_workers=min(8, len(registry))) as pool:
            bot_inputs = list(pool.map(load_bot_inputs, registry))
    else:
        bot_inputs = []

    series = []
    for b, (trades, acct_pts) in zip(registry, bot_inputs):
        instance = str(b.get("strategy_instance"))
        venue = str(b.get("venue") or "kucoin")
        symbol = str(b.get("symbol") or "SOL-USDT")
        bot_id = str(b.get("id") or instance)

        trade_curve, stats = _compounded_trade_curve(trades)
        price_move_curve_bps = [
            {
                "t": int(point["t"]),
                # The legacy API retains percent for compatibility; the new
                # contract is explicit and exact: 1 percent = 100 bps.
                "equity_pct": round(float(point["equity_pct"]) * 100.0, 6),
            }
            for point in trade_curve
        ]
        strategy = _strategy_return_payload(trades)

        last_snapshot_ts = int(acct_pts[-1]["t"]) if acct_pts else None
        snapshot_age_sec = (
            max(0, now_ts - last_snapshot_ts) if last_snapshot_ts is not None else None
        )
        snapshots_fresh = (
            snapshot_age_sec is not None and snapshot_age_sec <= stitch_max_age_sec
        )
        live_row = health_by_id.get(bot_id) or {}
        live_eq = live_row.get("equity")
        if live_eq is None and isinstance(live_row.get("health"), dict):
            live_eq = live_row["health"].get("equity")
        if snapshots_fresh:
            acct_pts = _stitch_live_equity(acct_pts, live_equity=live_eq, now_ts=now_ts)

        account_curve_abs = _absolute_account_curve(acct_pts)
        # % curve is deposit-adjusted (TWR): transfers don't count as returns.
        account_curve = _twr_account_curve(acct_pts)
        corrected = _bot_corrected_payload(
            abs_points=acct_pts,
            venue=venue,
            account=str(b.get("equity_account") or instance),
            since=since,
            until_ts=now_ts,
            bot_status=live_row.get("status") if live_row else None,
            bot_disabled=bool(b.get("disabled")),
        )

        series.append(
            {
                "id": b.get("id"),
                "display_name": b.get("display_name") or b.get("id"),
                "strategy_instance": instance,
                "venue": venue,
                "symbol": symbol,
                "color": b.get("color"),
                "currency": live_row.get("currency")
                or (acct_pts[-1].get("currency") if acct_pts else None)
                or ("USD" if venue == "kraken" else "USDT"),
                "live_equity": float(live_eq) if live_eq is not None else (
                    float(acct_pts[-1]["equity"]) if acct_pts else None
                ),
                "trade_curve": trade_curve,
                "price_move_curve_bps": price_move_curve_bps,
                "price_move_meta": {
                    "available": bool(price_move_curve_bps),
                    "unit": "bps",
                    "method": "equal_weight_compounded_unlevered_realized_price_moves",
                    "fees_included": False,
                    "funding_included": False,
                    "position_size_included": False,
                    "leverage_included": False,
                    "return_bps": round(float(stats["return_pct"]) * 100.0, 6),
                    "realized_event_ts": (
                        int(pd.Timestamp(trades["exit_ts"].max()).timestamp())
                        if not trades.empty
                        else None
                    ),
                },
                "strategy_curve": strategy["strategy_curve"],
                "strategy_meta": strategy["strategy_meta"],
                "account_curve": account_curve,
                "account_curve_abs": account_curve_abs,
                "corrected_curve": corrected["corrected_curve"],
                "corrected_meta": corrected["corrected_meta"],
                "cashflows": corrected["cashflows"],
                "stats": stats,
                "needs_backfill": bool(trades.empty) or not snapshots_fresh,
                "last_snapshot_ts": last_snapshot_ts,
                "snapshot_age_sec": snapshot_age_sec,
            }
        )

    # Uniform UTC clock + forward-fill so sparse bots share one time domain.
    # (Persist may be off — live stitch alone leaves 2–3 uneven snapshot times.)
    series, clock = _align_series_to_shared_clock(
        series, hours=hours, now_ts=now_ts
    )
    portfolio = _build_portfolio_curve(series, thin=False)
    # A mean of per-bot percentages is not the real portfolio's capital
    # growth.  Until every account ledger covers the common portfolio window,
    # use the explicitly labelled jump-TWR of the summed account equity.
    portfolio["corrected_curve"] = list(portfolio.get("account_curve") or [])
    portfolio["corrected_meta"] = {
        "method": "jump_twr",
        "available": bool(portfolio["corrected_curve"]),
        "reason": "portfolio_ledger_coverage_incomplete",
        "flow_count": sum(
            int((row.get("corrected_meta") or {}).get("flow_count") or 0)
            for row in series
        ),
        "net_cashflow": round(
            sum(
                float((row.get("corrected_meta") or {}).get("net_cashflow") or 0.0)
                for row in series
            ),
            6,
        ),
        "source": "summed_account_equity_jump_twr",
    }
    portfolio["allocation"] = _risk_normalized_allocation_payload(
        series, portfolio["corrected_curve"]
    )
    return {
        "ok": True,
        "hours": hours,
        "since": since.isoformat() if since is not None else None,
        "series": series,
        "portfolio": portfolio,
        "clock": clock,
        "ts": pd.Timestamp.now("UTC").isoformat(),
    }


def _bot_registry_by_instance() -> Dict[str, Dict[str, Any]]:
    """Map every known strategy_instance alias → bot (primary + trade_instances)."""
    out: Dict[str, Dict[str, Any]] = {}
    for b in fleet_bot_registry():
        for inst in _trade_instances_for_bot(b):
            out.setdefault(inst, b)
        primary = str(b.get("strategy_instance") or "").strip()
        if primary:
            out.setdefault(primary, b)
    return out


def _activity_item_from_event(
    *,
    ts: Any,
    venue: Any,
    symbol: Any,
    strategy_instance: str,
    side: Any,
    qty: Any,
    price: Any,
    stage: Any,
    status: Any,
    event_id: Any,
    bot: Dict[str, Any],
) -> Dict[str, Any]:
    t_unix = int(pd.Timestamp(ts).timestamp()) if ts is not None else None
    ts_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    action = str(stage or "event")
    return {
        "id": str(event_id or f"event:{strategy_instance}:{ts_iso}:{action}"),
        "kind": "event",
        "t": t_unix,
        "ts": ts_iso,
        "venue": venue,
        "symbol": symbol,
        "strategy_instance": strategy_instance,
        "bot_id": bot.get("id"),
        "display_name": bot.get("display_name") or strategy_instance,
        "action": action,
        "side": side,
        "qty": float(qty) if qty is not None else None,
        "price": float(price) if price is not None else None,
        "status": status,
        "pnl_pct": None,
        "color": bot.get("color"),
        # Backward-compat aliases consumed by older clients.
        "stage": action,
        "event_id": event_id,
    }


def _activity_item_from_fill(
    *,
    row: pd.Series,
    bot: Dict[str, Any],
) -> Dict[str, Any]:
    exit_ts = row.get("exit_ts")
    t_unix = (
        int(pd.Timestamp(exit_ts).timestamp())
        if exit_ts is not None and pd.notna(exit_ts)
        else None
    )
    ts_iso = (
        exit_ts.isoformat()
        if hasattr(exit_ts, "isoformat")
        else (str(exit_ts) if exit_ts is not None else "")
    )
    action = str(row.get("exit_event") or "fill")
    trade_id = str(row.get("trade_id") or f"fill:{bot.get('id')}:{ts_iso}")
    inst = str(row.get("strategy_instance") or bot.get("strategy_instance") or "")
    return {
        "id": f"fill:{trade_id}",
        "kind": "fill",
        "t": t_unix,
        "ts": ts_iso,
        "venue": row.get("venue"),
        "symbol": row.get("symbol"),
        "strategy_instance": inst,
        "bot_id": bot.get("id"),
        "display_name": bot.get("display_name") or inst,
        "action": action,
        "side": row.get("side"),
        "qty": float(row["qty"]) if pd.notna(row.get("qty")) else None,
        "price": float(row["exit_price"]) if pd.notna(row.get("exit_price")) else None,
        "status": "closed",
        "pnl_pct": float(row["pnl_pct"]) if pd.notna(row.get("pnl_pct")) else None,
        "color": bot.get("color"),
        "trade_id": trade_id,
        "exit_event": action,
        "entry_ts": (
            row["entry_ts"].isoformat()
            if hasattr(row.get("entry_ts"), "isoformat")
            else str(row.get("entry_ts") or "")
        ),
        "exit_ts": ts_iso,
    }


def _limit_activity_items(items: List[Dict[str, Any]], *, cap: int) -> List[Dict[str, Any]]:
    """Keep the complete bounded Kraken ledger before generic activity rows.

    Kraken history is independently paginated to ``cap`` exchange records.
    A global post-merge limit must not silently discard its older real fills
    merely because unrelated bots also produced recent dashboard activity.
    """
    ordered = sorted(items, key=lambda x: (x.get("t") is None, -(x.get("t") or 0)))
    kraken_history = [
        item for item in ordered if item.get("source") == "kraken_position_history"
    ]
    others = [
        item for item in ordered if item.get("source") != "kraken_position_history"
    ]
    kept = kraken_history + others[: max(0, int(cap) - len(kraken_history))]
    return sorted(kept, key=lambda x: (x.get("t") is None, -(x.get("t") or 0)))


def build_fleet_activity(
    *,
    hours: Optional[float] = 168.0,
    limit: int = 500,
) -> Dict[str, Any]:
    """Unified activity feed: execution events + closed-trade fills.

    Single source of truth for the Fleet Cockpit Activity panel. Fills carry
    PnL; events carry execution stage/status (entries, exits, TPs, SLs, flips).
    """
    since = _effective_since(hours)
    registry = _bot_registry_by_instance()
    bots = fleet_bot_registry()
    instances = list(registry.keys())
    cap = int(max(1, min(limit, 10_000)))
    if not instances and not bots:
        return {
            "ok": True,
            "items": [],
            "events": [],
            "count": 0,
            "ts": pd.Timestamp.now("UTC").isoformat(),
        }

    items: List[Dict[str, Any]] = []

    # --- execution_events (all trade_instance aliases) ---
    if instances:
        where = ["strategy_instance = any(%(instances)s::text[])"]
        params: Dict[str, Any] = {
            "instances": instances,
            "limit": cap,
        }
        if since is not None:
            where.append("ts >= %(since)s")
            params["since"] = since.to_pydatetime()

        sql = f"""
            select ts, venue, symbol, strategy_instance, side, qty, price,
                   execution_stage, status, event_id
            from execution_events
            where {' and '.join(where)}
            order by ts desc
            limit %(limit)s
        """
        try:
            with get_conn() as conn, conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() or []
            for r in rows:
                inst = str(r[3] or "")
                bot = registry.get(inst) or {}
                items.append(
                    _activity_item_from_event(
                        ts=r[0],
                        venue=r[1],
                        symbol=r[2],
                        strategy_instance=inst,
                        side=r[4],
                        qty=r[5],
                        price=r[6],
                        stage=r[7],
                        status=r[8],
                        event_id=r[9],
                        bot=bot,
                    )
                )
        except Exception as e:
            log.warning("fleet activity events query failed: %s", e)
            return {
                "ok": False,
                "items": [],
                "events": [],
                "error": str(e),
                "ts": pd.Timestamp.now("UTC").isoformat(),
            }

    # --- closed trades and direct Kraken position history ---
    per_bot_limit = min(max(cap, 50), 2000)
    for bot in bots:
        try:
            # Use the display read model, not only ``closed_trades``: Kraken
            # history is exchange-authoritative and older deployments did not
            # persist every fill locally.
            df = _load_display_trades_for_bot(bot, since=since, limit=per_bot_limit)
        except Exception as e:
            log.warning(
                "fleet activity fills load failed for %s: %s",
                bot.get("id"),
                e,
            )
            continue
        if not df.empty:
            work = df.sort_values("exit_ts", ascending=False).head(per_bot_limit)
            for _, row in work.iterrows():
                items.append(_activity_item_from_fill(row=row, bot=bot))

        if str(bot.get("venue") or "").lower() == "kraken":
            # Kraken's exchange ledger is primary evidence, not an old local
            # research table: the All-range view must not inherit the Fleet
            # Postgres start-date floor. The client paginates up to this cap.
            kraken_since = _hours_cutoff_ts(hours)
            raw_events = _load_kraken_position_events_for_bot(
                bot,
                since=kraken_since,
                limit=10_000 if kraken_since is None else per_bot_limit,
            )
            items.extend(
                _kraken_position_event_activity_items(
                    raw_events, bot=bot, since=kraken_since
                )
            )

    items = _limit_activity_items(items, cap=cap)

    # Legacy `events` = execution rows only (older clients).
    events = [i for i in items if i.get("kind") == "event"]
    return {
        "ok": True,
        "items": items,
        "events": events,
        "count": len(items),
        "fill_count": sum(1 for i in items if i.get("kind") == "fill"),
        "event_count": len(events),
        "ts": pd.Timestamp.now("UTC").isoformat(),
    }


def build_fleet_trades(
    *,
    strategy_instance: str,
    hours: Optional[float] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    bot = next(
        (
            b
            for b in fleet_bot_registry()
            if str(b.get("strategy_instance")) == strategy_instance
            or str(b.get("id")) == strategy_instance
        ),
        None,
    )
    instance = str((bot or {}).get("strategy_instance") or strategy_instance)
    venue = str((bot or {}).get("venue") or "kucoin")
    since = _effective_since(hours)
    if bot:
        df = _load_display_trades_for_bot(bot, since=since, limit=max(limit * 3, limit))
    else:
        df = _load_closed_trades_for_instance(
            strategy_instance=instance,
            venue=venue,
            since=since,
            limit=limit,
        )

    trades = []
    if not df.empty:
        work = df.sort_values("exit_ts", ascending=False).head(int(limit))
        for _, r in work.iterrows():
            trades.append(
                {
                    "trade_id": r.get("trade_id"),
                    "side": r.get("side"),
                    "qty": float(r["qty"]) if pd.notna(r.get("qty")) else None,
                    "entry_price": float(r["entry_price"]) if pd.notna(r.get("entry_price")) else None,
                    "exit_price": float(r["exit_price"]) if pd.notna(r.get("exit_price")) else None,
                    "pnl_pct": float(r["pnl_pct"]) if pd.notna(r.get("pnl_pct")) else None,
                    "entry_ts": r["entry_ts"].isoformat() if hasattr(r.get("entry_ts"), "isoformat") else str(r.get("entry_ts")),
                    "exit_ts": r["exit_ts"].isoformat() if hasattr(r.get("exit_ts"), "isoformat") else str(r.get("exit_ts")),
                    "exit_event": r.get("exit_event"),
                    "strategy_instance": r.get("strategy_instance"),
                    "venue": r.get("venue"),
                    "symbol": r.get("symbol"),
                    "bot_id": (bot or {}).get("id"),
                    "display_name": (bot or {}).get("display_name") or instance,
                }
            )
    return {
        "ok": True,
        "strategy_instance": instance,
        "display_name": (bot or {}).get("display_name") or instance,
        "trades": trades,
        "count": len(trades),
        "ts": pd.Timestamp.now("UTC").isoformat(),
    }


def build_kraken_exchange_trades(
    *,
    since_ms: Optional[int] = None,
    limit: int = 500,
) -> Dict[str, Any]:
    """Return account-specific Kraken position fills from the exchange API."""
    bot = next(
        (
            row
            for row in fleet_bot_registry()
            if str(row.get("venue") or "").lower() == "kraken"
        ),
        {
            "id": "kraken-legacy",
            "display_name": "Kraken Legacy",
            "strategy_instance": "kraken_bot",
            "venue": "kraken",
            "symbol": "SOL-USD",
        },
    )
    since = (
        pd.Timestamp(int(since_ms), unit="ms", tz="UTC")
        if since_ms is not None
        else _history_start_ts()
    )
    frame = _load_kraken_exchange_trades_for_bot(
        bot,
        since=since,
        limit=int(max(1, min(limit, 2000))),
        allow_remote=False,
    )
    trades: List[Dict[str, Any]] = []
    if not frame.empty:
        for _, row in frame.sort_values("exit_ts", ascending=False).iterrows():
            trades.append(
                {
                    "trade_id": row.get("trade_id"),
                    "side": row.get("side"),
                    "qty": float(row["qty"]) if pd.notna(row.get("qty")) else None,
                    "entry_price": (
                        float(row["entry_price"])
                        if pd.notna(row.get("entry_price"))
                        else None
                    ),
                    "exit_price": (
                        float(row["exit_price"])
                        if pd.notna(row.get("exit_price"))
                        else None
                    ),
                    "pnl_pct": (
                        float(row["pnl_pct"])
                        if pd.notna(row.get("pnl_pct"))
                        else None
                    ),
                    "entry_ts": row["entry_ts"].isoformat(),
                    "exit_ts": row["exit_ts"].isoformat(),
                    "exit_event": row.get("exit_event"),
                    "strategy_instance": row.get("strategy_instance"),
                    "venue": "kraken",
                    "symbol": row.get("symbol"),
                    "bot_id": bot.get("id"),
                    "display_name": bot.get("display_name"),
                }
            )
    return {
        "ok": True,
        "source": "kraken_futures_position_history",
        "trades": trades,
        "count": len(trades),
        "ts": pd.Timestamp.now("UTC").isoformat(),
    }


def _public_kraken_position_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Return only display-safe exchange fields for the Fleet read proxy."""
    keys = (
        "updateReason", "positionChange", "fillTime", "fundingRealizationTime",
        "timestamp", "executionUid", "executionPrice", "executionSize",
        "oldPosition", "newPosition", "oldAverageEntryPrice",
        "newAverageEntryPrice", "fee", "feeCurrency", "realizedPnL",
        "realizedFunding", "tradeable",
    )
    return {key: event[key] for key in keys if key in event}


def build_kraken_position_events(
    *,
    since_ms: Optional[int] = None,
    before_ms: Optional[int] = None,
    limit: int = 10_000,
    include_funding: bool = True,
) -> Dict[str, Any]:
    """Expose bounded, authenticated, read-only Kraken position history.

    The client follows Kraken's continuation token internally. Callers receive
    up to 10,000 real position mutations, not merely the exchange's first page.
    Account identifiers and API metadata are stripped before the service hop.
    """
    from quant.execution.kraken_futures import KrakenFuturesClient

    cap = int(max(1, min(limit, 10_000)))
    events = KrakenFuturesClient().get_position_events(
        symbol=os.getenv("KRAKEN_FUTURES_SYMBOL", "PF_SOLUSD"),
        since_ms=since_ms,
        before_ms=before_ms,
        limit=cap,
        include_funding=include_funding,
    )
    safe_events = [_public_kraken_position_event(row) for row in events]

    def event_time(row: Dict[str, Any]) -> Optional[int]:
        for key in ("fillTime", "fundingRealizationTime", "timestamp"):
            try:
                return int(row[key])
            except (KeyError, TypeError, ValueError):
                continue
        return None

    times = [value for value in (event_time(row) for row in safe_events) if value is not None]
    return {
        "ok": True,
        "source": "kraken_futures_position_history",
        "events": safe_events,
        "count": len(safe_events),
        "oldest_ms": min(times) if times else None,
        "newest_ms": max(times) if times else None,
        "limit": cap,
        "include_funding": include_funding,
    }


def build_fleet_capitalization() -> Dict[str, Any]:
    """Health + live equity (preferred) or latest equity snapshot per bot."""
    bots_payload = list_fleet_bots(probe_health=True)
    accounts = []
    for b in bots_payload.get("bots") or []:
        venue = str(b.get("venue") or "kucoin")
        instance = str(b.get("strategy_instance") or "")
        # Optional snapshot account override (quant dashboard wrote account='futures').
        snap_account = str(b.get("equity_account") or instance)
        health = b.get("health") if isinstance(b.get("health"), dict) else {}

        live_equity = health.get("equity")
        live_available = health.get("available")
        live_upnl = health.get("unrealised_pnl")
        live_currency = health.get("currency")
        live_source = health.get("equity_source")

        snaps = _load_equity_snapshots(venue=venue, account=snap_account, limit=1)
        if not snaps and snap_account != instance:
            snaps = _load_equity_snapshots(venue=venue, account=instance, limit=1)
        if not snaps and venue == "kraken":
            snaps = _load_equity_snapshots(venue=venue, account=None, limit=1)
            if not snaps:
                snaps = _load_equity_snapshots(venue=venue, account="main", limit=1)
        # Do NOT fall back to venue-wide account='futures' for arbitrary pilots —
        # that mis-attributes the dashboard KuCoin main key.
        latest = snaps[-1] if snaps else None

        equity = float(live_equity) if live_equity is not None else (
            float(latest["equity"]) if latest else None
        )
        currency = live_currency or (latest.get("currency") if latest else None)
        equity_ts = (
            int(health.get("ts"))
            if isinstance(health.get("ts"), (int, float))
            else (latest.get("t") if latest else None)
        )
        equity_source = live_source or ("equity_snapshots" if latest else None)

        accounts.append(
            {
                "id": b.get("id"),
                "display_name": b.get("display_name"),
                "strategy_instance": instance,
                "venue": venue,
                "status": b.get("status"),
                "executor_ready": b.get("executor_ready"),
                "live_trading_enabled": b.get("live_trading_enabled"),
                "dry_run": b.get("dry_run"),
                "health": health,
                "equity": equity,
                "available": float(live_available) if live_available is not None else None,
                "unrealised_pnl": float(live_upnl) if live_upnl is not None else None,
                "equity_ts": equity_ts,
                "currency": currency,
                "equity_source": equity_source,
            }
        )
    return {"ok": True, "accounts": accounts, "ts": pd.Timestamp.now("UTC").isoformat()}
