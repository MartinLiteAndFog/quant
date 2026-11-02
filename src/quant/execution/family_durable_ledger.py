"""PostgreSQL-backed family virtual positions, filled by CONFIRMED exchange fills.

No exchange connections, API keys or order placement. Every mutation is
serialized per family through an advisory transaction lock. Never interpret
a recorded intent as evidence that an exchange order was accepted or filled.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict

from quant.execution.event_store import get_conn
from quant.execution.family_ledger import BotIntent


class DurableLedgerError(RuntimeError):
    pass


def _amount(v: object, name: str, *, zero: bool = False) -> Decimal:
    try:
        number = Decimal(str(v))
    except (TypeError, ValueError, InvalidOperation):
        raise DurableLedgerError(f"invalid {name}") from None
    if not number.is_finite() or (number < 0 if zero else number <= 0):
        raise DurableLedgerError(f"{name} must be positive or zero when allowed")
    return number


def _int(v: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or (v < 0 if allow_zero else v <= 0):
        raise DurableLedgerError(f"{name} must be a nonnegative integer contract count")
    return v


def ensure_family_ledger_schema() -> None:
    sql = (
        Path(__file__).resolve().parents[1] / "sql" / "014_family_ledger.sql"
    ).read_text(encoding="utf-8")
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)


def _family_lock(cur: Any, family: str) -> None:
    if not family or not family.strip():
        raise DurableLedgerError("family ID required")
    cur.execute("select pg_advisory_xact_lock(hashtext(%s)::bigint)", ("family:" + family,))


def _position(cur: Any, family: str, bot: str, sym: str, side: str) -> tuple:
    cur.execute(
        """select contracts, average_entry, gross_realized, fees
           from family_virtual_positions
           where family_id=%s and strategy_id=%s and symbol=%s
             and position_side=%s""",
        (family, bot, sym, side),
    )
    return cur.fetchone() or (0, Decimal(0), Decimal(0), Decimal(0))


def _pending_closes(cur: Any, family: str, bot: str, sym: str, side: str) -> int:
    cur.execute(
        """select coalesce(sum(contracts-filled_contracts), 0)
           from family_order_intents
           where family_id=%s and strategy_id=%s and symbol=%s
             and position_side=%s and operation='CLOSE'
             and state in ('prepared','submitted','partially_filled')""",
        (family, bot, sym, side),
    )
    return int(cur.fetchone()[0])


def reserve_intent(family_id: str, intent: BotIntent, *, contract_multiplier: object) -> bool:
    """Reserve once before submission. False means previously seen; DO NOT resend."""
    multiplier = _amount(contract_multiplier, "contract multiplier")
    with get_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _family_lock(cur, family_id)
                cur.execute(
                    """select strategy_id, symbol, position_side, operation,
                              contracts, contract_multiplier, state
                       from family_order_intents
                       where family_id=%s and client_oid=%s""",
                    (family_id, intent.client_oid),
                )
                prior = cur.fetchone()
                if prior is not None:
                    identical = prior[:6] == (
                        intent.bot_id, intent.symbol, intent.position_side,
                        intent.operation, intent.contracts, multiplier,
                    )
                    if not identical:
                        raise DurableLedgerError("clientOid reused with different intent")
                    return False
                if intent.operation == "CLOSE":
                    qty = int(_position(
                        cur, family_id, intent.bot_id, intent.symbol,
                        intent.position_side,
                    )[0])
                    reserved = _pending_closes(
                        cur, family_id, intent.bot_id, intent.symbol,
                        intent.position_side,
                    )
                    if qty - reserved < intent.contracts:
                        raise DurableLedgerError("close exceeds bot's unreserved lots")
                else:
                    cur.execute(
                        """select coalesce(sum(contracts),0)
                           from family_virtual_positions
                           where family_id=%s and strategy_id=%s and symbol=%s""",
                        (family_id, intent.bot_id, intent.symbol),
                    )
                    already_open = int(cur.fetchone()[0])
                    cur.execute(
                        """select count(*) from family_order_intents
                           where family_id=%s and strategy_id=%s and symbol=%s
                             and operation='OPEN'
                             and state in ('prepared','submitted','partially_filled')""",
                        (family_id, intent.bot_id, intent.symbol),
                    )
                    other_open_pending = int(cur.fetchone()[0])
                    if already_open or other_open_pending:
                        raise DurableLedgerError("bot has an open position or pending entry")
                cur.execute(
                    """insert into family_order_intents
                       (family_id, client_oid, strategy_id, symbol,
                        position_side, operation, contracts, contract_multiplier)
                       values (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (
                        family_id, intent.client_oid, intent.bot_id, intent.symbol,
                        intent.position_side, intent.operation, intent.contracts,
                        multiplier,
                    ),
                )
                return True


def mark_submitted(family_id: str, client_oid: str) -> None:
    """Only call after confirmed venue acknowledgement, not a timeout."""
    with get_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _family_lock(cur, family_id)
                cur.execute(
                    """update family_order_intents
                       set state='submitted', updated_at=now()
                       where family_id=%s and client_oid=%s and state='prepared'
                       returning client_oid""",
                    (family_id, client_oid),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        """select state from family_order_intents
                           where family_id=%s and client_oid=%s""",
                        (family_id, client_oid),
                    )
                    state = cur.fetchone()
                    if state is None or state[0] not in ("submitted", "partially_filled", "filled"):
                        raise DurableLedgerError("cannot acknowledge unknown/terminal intent")


def confirm_cancel(family_id: str, client_oid: str) -> None:
    """Only after a final exchange cancellation confirmation with all fills settled."""
    with get_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _family_lock(cur, family_id)
                cur.execute(
                    """select state from family_order_intents
                       where family_id=%s and client_oid=%s""",
                    (family_id, client_oid),
                )
                row = cur.fetchone()
                if row is None:
                    raise DurableLedgerError("unknown clientOid")
                if row[0] == "cancelled":
                    return
                if row[0] == "filled":
                    raise DurableLedgerError("fully filled orders cannot be cancelled")
                cur.execute(
                    """update family_order_intents set state='cancelled',
                       updated_at=now() where family_id=%s and client_oid=%s""",
                    (family_id, client_oid),
                )


def apply_confirmed_fill(
    family_id: str,
    client_oid: str,
    fill_id: str,
    contracts: int,
    price: object,
    fee: object = "0",
) -> bool:
    """Apply exchange-confirmed fill transactionally, exactly once per fill ID."""
    qty = _int(contracts, "contracts")
    px = _amount(price, "fill price")
    commission = _amount(fee, "fee", zero=True)
    if not fill_id:
        raise DurableLedgerError("fill_id required")
    with get_conn() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                _family_lock(cur, family_id)
                cur.execute(
                    """select client_oid, contracts, price, fee
                       from family_order_fills where family_id=%s and fill_id=%s""",
                    (family_id, fill_id),
                )
                known = cur.fetchone()
                if known is not None:
                    if known != (client_oid, qty, px, commission):
                        raise DurableLedgerError("fill_id collision with different fill")
                    return False
                cur.execute(
                    """select strategy_id, symbol, position_side, operation,
                              contracts, filled_contracts, contract_multiplier, state
                       from family_order_intents
                       where family_id=%s and client_oid=%s""",
                    (family_id, client_oid),
                )
                intent = cur.fetchone()
                if intent is None:
                    raise DurableLedgerError("unrecognized exchange fill; stop new orders")
                bot, sym, side, operation, requested, filled, multiplier, state = intent
                if state in ("cancelled", "rejected"):
                    raise DurableLedgerError("fill after terminal intent; reconcile manually")
                if int(filled) + qty > int(requested):
                    raise DurableLedgerError("venue fill exceeds intent")
                old_qty, old_avg, gross, fees = _position(cur, family_id, bot, sym, side)
                old_qty = int(old_qty)
                if operation == "OPEN":
                    new_qty = old_qty + qty
                    new_avg = (old_avg * old_qty + px * qty) / new_qty
                    new_gross = gross
                else:
                    if qty > old_qty:
                        raise DurableLedgerError("fill exceeds bot's virtual position")
                    direction = Decimal(1) if side == "LONG" else Decimal(-1)
                    new_gross = gross + (px - old_avg) * direction * qty * multiplier
                    new_qty = old_qty - qty
                    new_avg = old_avg if new_qty else Decimal(0)
                cur.execute(
                    """insert into family_order_fills
                       (family_id, fill_id, client_oid, contracts, price, fee)
                       values (%s,%s,%s,%s,%s,%s)""",
                    (family_id, fill_id, client_oid, qty, px, commission),
                )
                cur.execute(
                    """insert into family_virtual_positions
                       (family_id, strategy_id, symbol, position_side,
                        contracts, average_entry, gross_realized, fees)
                       values (%s,%s,%s,%s,%s,%s,%s,%s)
                       on conflict (family_id,strategy_id,symbol,position_side)
                       do update set
                         contracts=excluded.contracts,
                         average_entry=excluded.average_entry,
                         gross_realized=excluded.gross_realized,
                         fees=excluded.fees, updated_at=now()""",
                    (family_id, bot, sym, side, new_qty, new_avg,
                     new_gross, fees + commission),
                )
                new_filled = int(filled) + qty
                new_state = "filled" if new_filled == int(requested) else "partially_filled"
                cur.execute(
                    """update family_order_intents
                       set filled_contracts=%s, state=%s, updated_at=now()
                       where family_id=%s and client_oid=%s""",
                    (new_filled, new_state, family_id, client_oid),
                )
                return True


def get_strategy_position(family_id: str, strategy_id: str, symbol: str, side: str) -> Dict[str, Any]:
    if side not in ("LONG", "SHORT"):
        raise DurableLedgerError("invalid position side")
    with get_conn() as conn, conn.cursor() as cur:
        qty, avg, gross, fees = _position(cur, family_id, strategy_id, symbol, side)
    return {
        "family_id": family_id, "strategy_id": strategy_id, "symbol": symbol,
        "position_side": side, "contracts": int(qty),
        "average_entry": avg, "gross_realized": gross,
        "fees": fees, "net_realized": gross - fees,
    }


def reconcile_side_totals(
    family_id: str,
    symbol: str,
    *,
    venue_long: int,
    venue_short: int,
) -> Dict[str, int]:
    """Reject any discrepancy, including unknown manual/external positions."""
    _int(venue_long, "venue_long", allow_zero=True)
    _int(venue_short, "venue_short", allow_zero=True)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """select position_side, coalesce(sum(contracts),0)
               from family_virtual_positions
               where family_id=%s and symbol=%s
               group by position_side""",
            (family_id, symbol),
        )
        rows = cur.fetchall()
    expected = {"LONG": 0, "SHORT": 0}
    for side, qty in rows:
        expected[side] = int(qty)
    if expected != {"LONG": venue_long, "SHORT": venue_short}:
        raise DurableLedgerError(
            f"venue/virtual mismatch: family={family_id} symbol={symbol} "
            f"virtual={expected} venue_long={venue_long} venue_short={venue_short}"
        )
    return {"long": venue_long, "short": venue_short}
