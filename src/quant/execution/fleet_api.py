"""Fleet aggregator: multi-bot performance board for the desktop cockpit.

Reads shared Postgres rows tagged with ``strategy_instance``, builds absolute
and percent account-equity curves (hero board), compounded trade-PnL % curves,
a white portfolio sum line, and fans out to each bot's public ``/health``.
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
            for key in ("account_curve_abs", "account_curve", "trade_curve"):
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

def _build_portfolio_curve(
    series: List[Dict[str, Any]],
    *,
    thin: bool = True,
) -> Dict[str, Any]:
    """Build portfolio abs sum + equal-weight % of bot returns.

    Absolute (Equity $): sum forward-filled equities (nominal USD/USDT 1:1).

    Percent (Equity %): equal-weight mean of each bot's own-base equity_pct,
    forward-filled on the union of timestamps. Do NOT normalize the summed
    absolute curve from its first point — when a large account (e.g. Kraken
    ~$360) joins after tiny pilots (~$15), that false rebase spikes to
    thousands of percent while every bot is still near ± a few percent.
    """
    curves: List[List[Dict[str, Any]]] = []
    twr_pcts: List[Optional[List[Dict[str, Any]]]] = []
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
            # Prefer the bot's deposit-adjusted (TWR) % curve when present so
            # portfolio % inherits it; recomputing from abs re-added deposits.
            pc = [
                {"t": int(p["t"]), "equity_pct": float(p["equity_pct"])}
                for p in (s.get("account_curve") or [])
                if p.get("equity_pct") is not None and _is_finite(p.get("equity_pct"))
            ]
            twr_pcts.append(pc or None)
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
            "note": "equal_weight_pct_mean_abs_sum",
        }

    # Per-bot %: use the bot's own (TWR) curve when available; else fall back
    # to first-equity base from the abs curve.
    pct_curves: List[List[Dict[str, Any]]] = []
    for curve, twr in zip(curves, twr_pcts):
        if twr:
            pct_curves.append(twr)
            continue
        base = float(curve[0]["equity"])
        if base <= 0:
            pct_curves.append(
                [{"t": int(p["t"]), "equity_pct": 0.0} for p in curve]
            )
        else:
            pct_curves.append(
                [
                    {
                        "t": int(p["t"]),
                        "equity_pct": round((float(p["equity"]) / base - 1.0) * 100.0, 6),
                    }
                    for p in curve
                ]
            )

    times = sorted({int(p["t"]) for c in curves for p in c})
    abs_idxs = [0] * len(curves)
    abs_lasts: List[Optional[float]] = [None] * len(curves)
    pct_idxs = [0] * len(pct_curves)
    pct_lasts: List[Optional[float]] = [None] * len(pct_curves)
    portfolio_abs: List[Dict[str, Any]] = []
    portfolio_pct: List[Dict[str, Any]] = []
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

        pct_vals: List[float] = []
        for i, curve in enumerate(pct_curves):
            while pct_idxs[i] < len(curve) and int(curve[pct_idxs[i]]["t"]) <= t:
                pct_lasts[i] = float(curve[pct_idxs[i]]["equity_pct"])
                pct_idxs[i] += 1
            if pct_lasts[i] is not None:
                pct_vals.append(pct_lasts[i])
        if pct_vals:
            portfolio_pct.append(
                {"t": t, "equity_pct": round(sum(pct_vals) / len(pct_vals), 6)}
            )

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
        "note": "equal_weight_pct_mean_abs_sum",
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
    series = []
    for b in registry:
        instance = str(b.get("strategy_instance"))
        venue = str(b.get("venue") or "kucoin")
        symbol = str(b.get("symbol") or "SOL-USDT")
        bot_id = str(b.get("id") or instance)

        trades = _load_closed_trades_for_bot(b, since=since)
        trade_curve, stats = _compounded_trade_curve(trades)

        acct_pts = _load_account_points_for_bot(b, since=since)
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
    return {
        "ok": True,
        "hours": hours,
        "since": since.isoformat() if since is not None else None,
        "series": series,
        "portfolio": portfolio,
        "clock": clock,
        "ts": pd.Timestamp.now("UTC").isoformat(),
    }


def build_fleet_activity(
    *,
    hours: Optional[float] = 168.0,
    limit: int = 500,
) -> Dict[str, Any]:
    since = _effective_since(hours)
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
    since = _effective_since(hours)
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
