"""Fleet aggregator: multi-bot percent performance + activity for the desktop cockpit.

Reads shared Postgres rows tagged with ``strategy_instance``, builds compounded
trade-PnL % curves (hero) and account-equity % curves (drawer), and fans out
to each bot's public ``/health`` endpoint for live status.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from quant.execution.event_store import get_conn
from quant.utils.log import get_logger

log = get_logger("quant.fleet_api")

# Default registry — overridden by FLEET_BOTS_JSON env (array of bot dicts).
_DEFAULT_BOTS: List[Dict[str, Any]] = [
    {
        "id": "imba-runner",
        "display_name": "Imba Runner",
        "strategy_instance": "sol-pilot-canonical",
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "health_url": "https://sol-pilot-canonical-production.up.railway.app/health",
        "color": "#c4a35a",
    },
    {
        "id": "pure-imbatp",
        "display_name": "Pure ImbaTP",
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
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "health_url": "https://sol-pilot-countertrend-sl-reverse-production.up.railway.app/health",
        "color": "#8a7a9a",
    },
    {
        "id": "kraken-legacy",
        "display_name": "Kraken Legacy",
        "strategy_instance": "kraken_bot",
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
    out = []
    for b in bots:
        row = {
            "id": b.get("id"),
            "display_name": b.get("display_name") or b.get("id"),
            "strategy_instance": b.get("strategy_instance"),
            "venue": b.get("venue") or "kucoin",
            "symbol": b.get("symbol") or "SOL-USDT",
            "health_url": b.get("health_url"),
            "color": b.get("color"),
        }
        if probe_health:
            health = _fetch_health(str(b.get("health_url") or ""))
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


def _load_equity_snapshots(
    *,
    venue: str,
    account: Optional[str] = None,
    since: Optional[pd.Timestamp] = None,
    limit: int = 5000,
) -> List[Dict[str, Any]]:
    where = ["venue = %(venue)s"]
    params: Dict[str, Any] = {"venue": str(venue), "limit": int(max(1, limit))}
    if account:
        where.append("account = %(account)s")
        params["account"] = str(account)
    if since is not None:
        where.append("ts >= %(since)s")
        params["since"] = since.to_pydatetime()
    sql = f"""
        select ts, equity, currency, account, source
        from equity_snapshots
        where {' and '.join(where)}
        order by ts asc
        limit %(limit)s
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


def _normalize_account_curve(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not points:
        return []
    base = float(points[0]["equity"])
    if base <= 0:
        return [{"t": p["t"], "equity_pct": 0.0} for p in points]
    return [
        {"t": int(p["t"]), "equity_pct": round((float(p["equity"]) / base - 1.0) * 100.0, 6)}
        for p in points
    ]


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

    since = _hours_cutoff_ts(hours)
    series = []
    for b in registry:
        instance = str(b.get("strategy_instance"))
        venue = str(b.get("venue") or "kucoin")
        symbol = str(b.get("symbol") or "SOL-USDT")
        trades = _load_closed_trades_for_instance(
            strategy_instance=instance,
            venue=venue,
            symbol=symbol if venue != "kraken" else None,
            since=since,
        )
        # Kraken symbol variants — retry without strict symbol filter if empty
        if trades.empty and venue == "kraken":
            trades = _load_closed_trades_for_instance(
                strategy_instance=instance,
                venue=venue,
                since=since,
            )

        trade_curve, stats = _compounded_trade_curve(trades)

        # Account equity: prefer account=strategy_instance, else venue-level snapshots
        acct_pts = _load_equity_snapshots(
            venue=venue,
            account=instance,
            since=since,
        )
        if not acct_pts:
            acct_pts = _load_equity_snapshots(venue=venue, account=None, since=since)
            # Only attach shared venue curve for single-account venues (kraken)
            if venue == "kucoin" and len(registry) > 1:
                acct_pts = []

        account_curve = _normalize_account_curve(acct_pts)

        series.append(
            {
                "id": b.get("id"),
                "display_name": b.get("display_name") or b.get("id"),
                "strategy_instance": instance,
                "venue": venue,
                "symbol": symbol,
                "color": b.get("color"),
                "trade_curve": trade_curve,
                "account_curve": account_curve,
                "stats": stats,
                "needs_backfill": bool(trades.empty),
            }
        )

    return {
        "ok": True,
        "hours": hours,
        "since": since.isoformat() if since is not None else None,
        "series": series,
        "ts": pd.Timestamp.now("UTC").isoformat(),
    }


def build_fleet_activity(
    *,
    hours: Optional[float] = 168.0,
    limit: int = 500,
) -> Dict[str, Any]:
    since = _hours_cutoff_ts(hours)
    registry = {str(b["strategy_instance"]): b for b in fleet_bot_registry()}
    instances = list(registry.keys())
    if not instances:
        return {"ok": True, "events": [], "ts": pd.Timestamp.now("UTC").isoformat()}

    where = ["strategy_instance = any(%(instances)s::text[])"]
    params: Dict[str, Any] = {
        "instances": instances,
        "limit": int(max(1, limit)),
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
    events: List[Dict[str, Any]] = []
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
        for r in rows:
            inst = str(r[3] or "")
            bot = registry.get(inst) or {}
            events.append(
                {
                    "t": int(pd.Timestamp(r[0]).timestamp()) if r[0] is not None else None,
                    "ts": r[0].isoformat() if hasattr(r[0], "isoformat") else str(r[0]),
                    "venue": r[1],
                    "symbol": r[2],
                    "strategy_instance": inst,
                    "bot_id": bot.get("id"),
                    "display_name": bot.get("display_name") or inst,
                    "side": r[4],
                    "qty": float(r[5]) if r[5] is not None else None,
                    "price": float(r[6]) if r[6] is not None else None,
                    "stage": r[7],
                    "status": r[8],
                    "event_id": r[9],
                    "color": bot.get("color"),
                }
            )
    except Exception as e:
        log.warning("fleet activity query failed: %s", e)
        return {"ok": False, "events": [], "error": str(e), "ts": pd.Timestamp.now("UTC").isoformat()}

    return {"ok": True, "events": events, "count": len(events), "ts": pd.Timestamp.now("UTC").isoformat()}


def build_fleet_trades(
    *,
    strategy_instance: str,
    hours: Optional[float] = None,
    limit: int = 200,
) -> Dict[str, Any]:
    bot = next(
        (b for b in fleet_bot_registry() if str(b.get("strategy_instance")) == strategy_instance or str(b.get("id")) == strategy_instance),
        None,
    )
    instance = str((bot or {}).get("strategy_instance") or strategy_instance)
    venue = str((bot or {}).get("venue") or "kucoin")
    since = _hours_cutoff_ts(hours)
    df = _load_closed_trades_for_instance(
        strategy_instance=instance,
        venue=venue,
        since=since,
        limit=limit,
    )
    if df.empty and venue == "kraken":
        df = _load_closed_trades_for_instance(strategy_instance=instance, venue=venue, since=since, limit=limit)

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


def build_fleet_capitalization() -> Dict[str, Any]:
    """Health + live equity (preferred) or latest equity snapshot per bot."""
    bots_payload = list_fleet_bots(probe_health=True)
    accounts = []
    for b in bots_payload.get("bots") or []:
        venue = str(b.get("venue") or "kucoin")
        instance = str(b.get("strategy_instance") or "")
        health = b.get("health") if isinstance(b.get("health"), dict) else {}

        live_equity = health.get("equity")
        live_available = health.get("available")
        live_upnl = health.get("unrealised_pnl")
        live_currency = health.get("currency")
        live_source = health.get("equity_source")

        snaps = _load_equity_snapshots(venue=venue, account=instance, limit=1)
        if not snaps and venue == "kraken":
            snaps = _load_equity_snapshots(venue=venue, account=None, limit=1)
            if not snaps:
                snaps = _load_equity_snapshots(venue=venue, account="main", limit=1)
        # Do NOT fall back to venue-wide account='futures' here — that is the
        # dashboard's single KuCoin key and mis-attributes capital to dry pilots
        # that have no credentials (e.g. Counter SL Reverse with PASTE_ME keys).
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
