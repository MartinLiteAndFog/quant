"""Read-only exchange-ledger synchronization for Fleet performance.

Each bot process keeps its existing venue credentials local, reads only its
own futures-account transfer history, and persists normalized cashflows in the
shared Postgres database.  The central Fleet API can then calculate a
reproducible cashflow-corrected return without distributing API secrets.

This module never places orders or initiates transfers.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from quant.execution.event_store import (
    delete_closed_trade,
    ensure_cashflow_schema,
    upsert_closed_trade,
    upsert_cashflow_event,
    upsert_cashflow_sync_state,
)
from quant.execution.fleet_history import fleet_history_start
from quant.utils.log import get_logger

log = get_logger("quant.cashflow_sync")

_STARTED: set[tuple[str, str]] = set()
_START_LOCK = threading.Lock()


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _event_id(venue: str, account: str, source_ref: str) -> str:
    token = f"{venue}:{account}:{source_ref}".encode("utf-8")
    return f"cashflow:{hashlib.sha256(token).hexdigest()[:32]}"


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def normalize_kucoin_cashflows(
    rows: Iterable[Dict[str, Any]],
    *,
    account: str,
) -> List[Dict[str, Any]]:
    """Keep only completed Funding↔Futures transfer ledger rows."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        flow_type = str(row.get("type") or "").strip()
        flow_key = flow_type.lower()
        if flow_key not in {"transferin", "transferout"}:
            continue
        if str(row.get("status") or "").strip().lower() != "completed":
            continue
        ts_ms = _to_float(row.get("time"))
        amount = _to_float(row.get("amount"))
        if ts_ms is None or amount is None:
            continue
        direction = "in" if flow_key == "transferin" else "out"
        signed_amount = abs(amount) if direction == "in" else -abs(amount)
        currency = str(row.get("currency") or "").upper()
        source_ref = str(row.get("offset") or f"{int(ts_ms)}:{flow_type}:{amount}")
        reporting_amount = signed_amount if currency in {"USD", "USDT", "USDC"} else None
        out.append(
            {
                "event_id": _event_id("kucoin", account, source_ref),
                "ts": datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc),
                "venue": "kucoin",
                "account": account,
                "currency": currency,
                "amount": signed_amount,
                "reporting_currency": "USD",
                "reporting_amount": reporting_amount,
                "fee": abs(_to_float(row.get("fee")) or 0.0),
                "direction": direction,
                "flow_type": flow_type,
                "status": "completed",
                "source_ref": source_ref,
                "equity_after": _to_float(row.get("accountEquity")),
                "boundary_scope": "futures",
                "payload_json": {"remark": row.get("remark")},
            }
        )
    return out


def normalize_kraken_cashflows(
    rows: Iterable[Dict[str, Any]],
    *,
    account: str,
) -> List[Dict[str, Any]]:
    """Keep only successful Spot↔Futures boundary transfers."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "").strip().lower() != "success":
            continue
        source = str(row.get("from") or "").strip().lower()
        destination = str(row.get("to") or "").strip().lower()
        if (source, destination) == ("spot", "futures"):
            direction = "in"
        elif (source, destination) == ("futures", "spot"):
            direction = "out"
        else:
            continue
        try:
            ts = pd.Timestamp(row.get("date"))
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            continue
        amount = _to_float(row.get("amount"))
        if amount is None:
            continue
        signed_amount = abs(amount) if direction == "in" else -abs(amount)
        currency = str(row.get("asset") or "").upper()
        source_ref = str(row.get("id") or row.get("reference") or f"{int(ts.timestamp())}:{amount}")
        # Stablecoins and USD are nominally equivalent to the Fleet USD
        # reporting unit. Other collateral (notably EUR) stays explicitly
        # unavailable until an authoritative event-time conversion is stored.
        reporting_amount = signed_amount if currency in {"USD", "USDT", "USDC"} else None
        out.append(
            {
                "event_id": _event_id("kraken", account, source_ref),
                "ts": ts.to_pydatetime(),
                "venue": "kraken",
                "account": account,
                "currency": currency,
                "amount": signed_amount,
                "reporting_currency": "USD",
                "reporting_amount": reporting_amount,
                "fee": 0.0,
                "direction": direction,
                "flow_type": f"{source}_to_{destination}",
                "status": "success",
                "source_ref": source_ref,
                "equity_after": None,
                "boundary_scope": "futures",
                "payload_json": {"from": source, "to": destination},
            }
        )
    return out


def normalize_kraken_closed_trades(
    rows: Iterable[Dict[str, Any]],
    *,
    symbol: str = "PF_SOLUSD",
) -> List[Dict[str, Any]]:
    """Convert authoritative Kraken position closes/reversals to Fleet trades.

    Kraken emits a position update for every fill fragment.  Only ``close`` and
    ``reverse`` complete the prior position; ``decrease`` is deliberately
    ignored so partial take-profit fills never become standalone trades.
    """
    wrappers = [row for row in rows if isinstance(row, dict)]
    wrappers.sort(key=lambda row: _to_float(row.get("timestamp")) or 0.0)
    out: List[Dict[str, Any]] = []
    active_side = 0
    active_entry_ts: Optional[pd.Timestamp] = None

    for wrapper in wrappers:
        event = wrapper.get("event")
        update = event.get("PositionUpdate") if isinstance(event, dict) else None
        if not isinstance(update, dict):
            continue
        if str(update.get("tradeable") or "").upper() != str(symbol).upper():
            continue
        if str(update.get("updateReason") or "").strip().lower() != "trade":
            continue

        ts_ms = _to_float(update.get("timestamp") or wrapper.get("timestamp"))
        old_position = _to_float(update.get("oldPosition"))
        new_position = _to_float(update.get("newPosition"))
        if ts_ms is None or old_position is None or new_position is None:
            continue
        ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
        old_side = 1 if old_position > 0 else -1 if old_position < 0 else 0
        new_side = 1 if new_position > 0 else -1 if new_position < 0 else 0

        fill_ms = _to_float(update.get("fillTime"))
        fill_ts = pd.Timestamp(fill_ms, unit="ms", tz="UTC") if fill_ms else ts
        if old_side and (
            active_side != old_side
            or active_entry_ts is None
            or fill_ts < active_entry_ts
        ):
            active_side = old_side
            active_entry_ts = fill_ts

        position_change = str(update.get("positionChange") or "").strip().lower()
        if position_change in {"close", "reverse"} and old_side:
            entry_price = _to_float(update.get("oldAverageEntryPrice"))
            exit_price = _to_float(update.get("executionPrice"))
            if entry_price is not None and entry_price > 0 and exit_price is not None:
                pnl_pct = (
                    (exit_price / entry_price - 1.0)
                    * 100.0
                    * (1.0 if old_side > 0 else -1.0)
                )
                source_ref = str(
                    update.get("executionUid")
                    or f"{int(ts_ms)}:{old_position}:{new_position}"
                )
                out.append(
                    {
                        "trade_id": f"kraken-position:{source_ref}",
                        # The first deployed version incorrectly treated this
                        # account-scoped uid as an event uid.  Carry the exact
                        # generated id privately so sync can remove that one
                        # stale row without broad historical deletion.
                        "_legacy_trade_id": (
                            f"kraken-position:{wrapper.get('uid')}"
                            if wrapper.get("uid")
                            else None
                        ),
                        "venue": "kraken",
                        "symbol": "SOL-USD",
                        "entry_ts": (active_entry_ts or ts).to_pydatetime(),
                        "exit_ts": ts.to_pydatetime(),
                        "side": "long" if old_side > 0 else "short",
                        "qty": abs(old_position),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "exit_event": f"kraken_position_{position_change}",
                        "strategy": "kraken_tv_executor",
                        "strategy_instance": "kraken_bot",
                        "config_hash": "kraken_tv_executor_v1",
                        "source_action_event_id": None,
                        "payload_json": {
                            "position_change": position_change,
                            "realized_pnl": _to_float(update.get("realizedPnL")),
                            "fee": _to_float(update.get("fee")),
                            "fee_currency": update.get("feeCurrency"),
                            "trade_type": update.get("tradeType"),
                        },
                    }
                )

        if new_side == 0:
            active_side = 0
            active_entry_ts = None
        elif new_side != old_side:
            active_side = new_side
            active_entry_ts = ts
        elif active_entry_ts is None:
            active_side = new_side
            active_entry_ts = fill_ts

    return out


def _kucoin_rows(start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    from quant.execution.kucoin_futures import KucoinFuturesBroker

    broker = KucoinFuturesBroker()
    rows: List[Dict[str, Any]] = []
    cursor: Optional[int] = None
    for _ in range(20):
        path = (
            "/api/v1/transaction-history"
            f"?startAt={start_ms}&endAt={end_ms}&maxCount=100&forward=false"
        )
        if cursor is not None:
            path += f"&offset={cursor}"
        payload = broker._req("GET", path)
        page = payload.get("dataList") or []
        rows.extend(row for row in page if isinstance(row, dict))
        if not payload.get("hasMore") or not page:
            break
        offsets = [int(row["offset"]) for row in page if row.get("offset") is not None]
        next_cursor = min(offsets) if offsets else None
        if next_cursor is None or next_cursor == cursor:
            break
        cursor = next_cursor
    return rows


def _fetch_kucoin_cashflows(
    *,
    account: str,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cursor = coverage_start
    one_day = pd.Timedelta(days=1)
    while cursor < coverage_end:
        window_end = min(cursor + one_day - pd.Timedelta(milliseconds=1), coverage_end)
        rows.extend(_kucoin_rows(int(cursor.timestamp() * 1000), int(window_end.timestamp() * 1000)))
        cursor += one_day
    return normalize_kucoin_cashflows(rows, account=account)


def _fetch_kraken_cashflows(
    *,
    account: str,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> List[Dict[str, Any]]:
    from quant.execution.kraken_futures import KrakenFuturesClient

    client = KrakenFuturesClient()
    client.timeout_s = max(client.timeout_s, 40)
    payload = client._req("GET", "/derivatives/api/v3/transfers", private=True)
    rows = payload.get("transfers") or []
    normalized = normalize_kraken_cashflows(rows, account=account)
    return [
        row
        for row in normalized
        if coverage_start <= pd.Timestamp(row["ts"]) <= coverage_end
    ]


def _fetch_kraken_closed_trades(
    *,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
) -> List[Dict[str, Any]]:
    from quant.execution.kraken_futures import KrakenFuturesClient

    client = KrakenFuturesClient()
    client.timeout_s = max(client.timeout_s, 40)
    # The authenticated history endpoint defaults to the latest 1,000 position
    # events.  That currently reaches past Fleet's configured history floor;
    # filtering is repeated locally because the legacy signer does not include
    # GET query parameters in its authentication payload.
    payload = client._req("GET", "/api/history/v3/positions", private=True)
    normalized = normalize_kraken_closed_trades(payload.get("elements") or [])
    return [
        row
        for row in normalized
        if coverage_start <= pd.Timestamp(row["exit_ts"]) <= coverage_end
    ]


def _coverage_start(now: pd.Timestamp, *, initial: bool) -> pd.Timestamp:
    if not initial:
        return now - pd.Timedelta(days=2)
    return fleet_history_start() or (now - pd.Timedelta(days=90))


def sync_once(*, venue: str, account: str, initial: bool = True) -> int:
    """Fetch and persist one account's confirmed futures-boundary cashflows."""
    venue = str(venue).strip().lower()
    account = str(account).strip()
    now = pd.Timestamp.now("UTC")
    start = _coverage_start(now, initial=initial)
    ensure_cashflow_schema()
    try:
        if venue == "kucoin":
            events = _fetch_kucoin_cashflows(
                account=account, coverage_start=start, coverage_end=now
            )
            source = "kucoin_futures_transaction_history"
        elif venue == "kraken":
            events = _fetch_kraken_cashflows(
                account=account, coverage_start=start, coverage_end=now
            )
            source = "kraken_futures_transfers"
        else:
            raise ValueError(f"unsupported cashflow venue: {venue}")
        for event in events:
            upsert_cashflow_event(event)
        if venue == "kraken":
            try:
                trades = _fetch_kraken_closed_trades(
                    coverage_start=start,
                    coverage_end=now,
                )
                legacy_ids = {
                    str(trade.pop("_legacy_trade_id"))
                    for trade in trades
                    if trade.get("_legacy_trade_id")
                }
                for trade_id in legacy_ids:
                    delete_closed_trade(trade_id=trade_id)
                for trade in trades:
                    upsert_closed_trade(trade)
                log.info(
                    "kraken closed-trade sync complete account=%s trades=%s",
                    account,
                    len(trades),
                )
            except Exception as exc:
                # Trade-history availability must not invalidate an otherwise
                # authoritative cashflow sync or leave Net Flows pending.
                log.warning(
                    "kraken closed-trade sync failed account=%s error=%s",
                    account,
                    type(exc).__name__,
                )
        upsert_cashflow_sync_state(
            venue=venue,
            account=account,
            coverage_start=start.to_pydatetime(),
            coverage_end=now.to_pydatetime(),
            source=source,
            last_error=None,
        )
        return len(events)
    except Exception as exc:
        upsert_cashflow_sync_state(
            venue=venue,
            account=account,
            coverage_start=None,
            coverage_end=None,
            source=f"{venue}_cashflow_sync",
            last_error=type(exc).__name__,
        )
        raise


def start_cashflow_sync(
    *,
    venue: str,
    account: str,
    default_enabled: bool = True,
) -> Optional[threading.Thread]:
    """Start a daemon that performs read-only ledger synchronization."""
    env = os.getenv("FLEET_CASHFLOW_SYNC_ENABLED")
    enabled = _truthy(env) if env is not None and env.strip() else default_enabled
    if not enabled:
        return None
    key = (str(venue).lower(), str(account))
    with _START_LOCK:
        if key in _STARTED:
            return None
        _STARTED.add(key)

    def _loop() -> None:
        initial = True
        while True:
            try:
                count = sync_once(venue=key[0], account=key[1], initial=initial)
                log.info(
                    "cashflow sync complete venue=%s account=%s events=%s",
                    key[0],
                    key[1],
                    count,
                )
                initial = False
            except Exception as exc:
                log.warning(
                    "cashflow sync failed venue=%s account=%s error=%s",
                    key[0],
                    key[1],
                    type(exc).__name__,
                )
            try:
                interval = max(900.0, float(os.getenv("FLEET_CASHFLOW_SYNC_SEC", "3600")))
            except Exception:
                interval = 3600.0
            time.sleep(interval)

    thread = threading.Thread(
        target=_loop,
        name=f"cashflow-sync-{key[0]}-{key[1]}",
        daemon=True,
    )
    thread.start()
    return thread
