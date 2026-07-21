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
            "equity_account": b.get("equity_account"),
            "trade_instances": b.get("trade_instances"),
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
    # ~20s points (that looked like a point cloud). Short spans keep shape.
    min_iv = max(60, int(min_interval_sec))
    span = int(simple[-1]["t"]) - int(simple[0]["t"])
    if span < 2 * min_iv:
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
        raw = [{"t": int(p["t"]), "equity_pct": 0.0} for p in points]
    else:
        raw = [
            {"t": int(p["t"]), "equity_pct": round((float(p["equity"]) / base - 1.0) * 100.0, 6)}
            for p in points
        ]
    return _downsample_points(raw, max_points=180, value_key="equity_pct", min_interval_sec=900)

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
    # One health fan-out for live equity stitch (also keeps curves fresh).
    health_by_id: Dict[str, Dict[str, Any]] = {}
    try:
        probed = list_fleet_bots(probe_health=True)
        for row in probed.get("bots") or []:
            health_by_id[str(row.get("id"))] = row if isinstance(row, dict) else {}
    except Exception as e:
        log.warning("fleet performance health probe failed: %s", e)

    now_ts = int(pd.Timestamp.now("UTC").timestamp())
    series = []
    for b in registry:
        instance = str(b.get("strategy_instance"))
        venue = str(b.get("venue") or "kucoin")
        symbol = str(b.get("symbol") or "SOL-USDT")
        bot_id = str(b.get("id") or instance)

        trades = _load_closed_trades_for_bot(b, since=since)
        trade_curve, stats = _compounded_trade_curve(trades)

        acct_pts = _load_account_points_for_bot(b, since=since)
        live_row = health_by_id.get(bot_id) or {}
        live_eq = live_row.get("equity")
        if live_eq is None and isinstance(live_row.get("health"), dict):
            live_eq = live_row["health"].get("equity")
        acct_pts = _stitch_live_equity(acct_pts, live_equity=live_eq, now_ts=now_ts)

        account_curve_abs = _absolute_account_curve(acct_pts)
        account_curve = _normalize_account_curve(acct_pts)

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
                "account_curve": account_curve,
                "account_curve_abs": account_curve_abs,
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
    since = _hours_cutoff_ts(hours)
    if bot:
        df = _load_closed_trades_for_bot(bot, since=since, limit=max(limit * 3, limit))
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
