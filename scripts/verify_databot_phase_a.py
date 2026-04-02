#!/usr/bin/env python3
"""
Phase A verification for DATABOT: Redis Renko keys, Postgres live_renko_bricks, optional HTTP health.

Required environment:
  REDIS_URL          — Redis connection URL (same as DATABOT)
  POSTGRES_URL       — Postgres DSN (or DATABASE_URL)

Optional:
  DATABOT_HEALTH_URL — Base URL of the deployed DATABOT service (e.g. https://…railway.app);
                       script requests …/health unless the URL already ends with /health
  PHASE_A_SYMBOLS    — Comma-separated symbols to check (default: DATABOT_SYMBOLS, then DASHBOARD_SYMBOL, then SOL-USDT)

Exit codes:
  0 — all required checks passed (and optional health passed if DATABOT_HEALTH_URL was set)
  1 — one or more required checks failed
  2 — missing configuration or usage error
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def _norm_symbol(symbol: str) -> str:
    """Match Postgres matching in event_store._normalize_symbol_token."""
    return "".join(ch for ch in str(symbol or "").upper() if ch.isalnum())


def _redis_canon_symbol(symbol: str) -> str:
    """Match databot renko_pipeline._canon_symbol (Redis key suffix)."""
    return str(symbol).upper().replace("-", "")


def _parse_symbols() -> List[str]:
    raw = os.getenv("PHASE_A_SYMBOLS", "").strip()
    if not raw:
        raw = os.getenv("DATABOT_SYMBOLS", "").strip()
    if not raw:
        raw = os.getenv("DASHBOARD_SYMBOL", "").strip()
    if not raw:
        raw = "SOL-USDT"
    return [s.strip() for s in raw.split(",") if s.strip()]


def _pg_dsn() -> str:
    dsn = (os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        raise RuntimeError("POSTGRES_URL or DATABASE_URL is not set")
    return dsn


def _print_usage() -> None:
    print(
        __doc__.strip()
        if __doc__
        else "See script docstring for REDIS_URL, POSTGRES_URL / DATABASE_URL, optional DATABOT_HEALTH_URL, PHASE_A_SYMBOLS."
    )


def check_redis(redis_url: str, symbol: str) -> Tuple[bool, str]:
    try:
        import redis as redis_lib
    except ImportError:
        return False, "redis package not installed"

    sym = _redis_canon_symbol(symbol)
    latest_key = f"renko:{sym}:latest"
    try:
        client = redis_lib.from_url(redis_url, decode_responses=True)
        client.ping()
        raw = client.get(latest_key)
    except Exception as exc:
        return False, f"redis error: {exc}"

    if not raw:
        return False, f"missing key {latest_key}"
    try:
        payload: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, f"{latest_key} is not valid JSON: {exc}"

    if not isinstance(payload, dict):
        return False, f"{latest_key} JSON is not an object"
    n_bars = payload.get("n_bars")
    bars = payload.get("bars")
    if isinstance(n_bars, int) and n_bars > 0:
        return True, f"{latest_key} ok (n_bars={n_bars})"
    if isinstance(bars, list) and len(bars) > 0:
        return True, f"{latest_key} ok (bars={len(bars)})"
    return False, f"{latest_key} present but empty or missing n_bars/bars"


def check_postgres(symbol: str) -> Tuple[bool, str]:
    try:
        import psycopg
    except ImportError:
        return False, "psycopg package not installed"

    norm = _norm_symbol(symbol)
    sql = """
    select count(*)::bigint as n, max(ts) as max_ts
    from live_renko_bricks
    where replace(replace(replace(upper(symbol), '-', ''), '_', ''), '/', '') = %(norm)s
    """
    try:
        with psycopg.connect(_pg_dsn(), autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"norm": norm})
                row = cur.fetchone()
    except Exception as exc:
        return False, f"postgres error: {exc}"

    if not row:
        return False, "no row returned from live_renko_bricks query"
    n, max_ts = row[0], row[1]
    if n and int(n) > 0:
        return True, f"live_renko_bricks ok for symbol_norm={norm} (rows={n}, max_ts={max_ts})"
    return False, f"no rows in live_renko_bricks for symbol_norm={norm}"


def check_health_url(base: str) -> Tuple[bool, str]:
    b = base.rstrip("/")
    url = b if b.lower().endswith("/health") else b + "/health"
    req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.getcode()
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code} from {url}"
    except urllib.error.URLError as exc:
        return False, f"request failed: {exc.reason}"
    except Exception as exc:
        return False, f"health request error: {exc}"

    if status != 200:
        return False, f"HTTP {status} from {url}"
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return False, f"health body is not JSON: {exc}"
    if not isinstance(data, dict):
        return False, "health JSON is not an object"
    if data.get("service") != "databot":
        return False, f"unexpected service field: {data.get('service')!r}"
    overall = data.get("status")
    if overall == "ok":
        return True, f"health ok ({url})"
    if overall == "degraded":
        return True, f"health reachable but degraded ({url}) — check pipelines in JSON"
    return True, f"health 200 + service=databot ({url}), status={overall!r}"


def main() -> int:
    redis_url = os.getenv("REDIS_URL", "").strip()
    pg_ok = bool((os.getenv("POSTGRES_URL") or os.getenv("DATABASE_URL") or "").strip())

    if not redis_url or not pg_ok:
        print("verify_databot_phase_a: missing required environment variables.\n", file=sys.stderr)
        if not redis_url:
            print("  - REDIS_URL is not set", file=sys.stderr)
        if not pg_ok:
            print("  - POSTGRES_URL and DATABASE_URL are both unset", file=sys.stderr)
        print(file=sys.stderr)
        _print_usage()
        return 2

    symbols = _parse_symbols()
    health_base = os.getenv("DATABOT_HEALTH_URL", "").strip()

    results: List[Tuple[str, str, bool, str]] = []

    for sym in symbols:
        ok_r, msg_r = check_redis(redis_url, sym)
        results.append(("redis", sym, ok_r, msg_r))
        ok_p, msg_p = check_postgres(sym)
        results.append(("postgres", sym, ok_p, msg_p))

    health_result: Optional[Tuple[bool, str]] = None
    if health_base:
        health_result = check_health_url(health_base)

    print("DATABOT Phase A verification")
    print(f"  symbols: {', '.join(symbols)}")
    print()

    all_ok = True
    for kind, sym, ok, msg in results:
        label = f"{kind}/{sym}"
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {label}: {msg}")

    if health_result is not None:
        ok_h, msg_h = health_result
        status = "PASS" if ok_h else "FAIL"
        if not ok_h:
            all_ok = False
        print(f"  [{status}] health: {msg_h}")
    else:
        print("  [SKIP] health: DATABOT_HEALTH_URL not set")

    print()
    if all_ok:
        print("RESULT: PASS — Redis and Postgres checks succeeded for all symbols" + (
            "; health check passed" if health_result and health_result[0] else
            ("; health skipped" if health_result is None else "")
        ))
        if health_result and not health_result[0]:
            pass  # unreachable if all_ok
        return 0

    print("RESULT: FAIL — fix the items above, then re-run", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
