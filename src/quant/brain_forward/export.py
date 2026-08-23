"""Read-only export of a protocol-scoped Brain-forward evidence bundle.

The observer ledger lives in Postgres; research review lives in versioned local
artifacts.  This module bridges those stores without permitting a mutation,
retune, order, or winner promotion.  Its output is intentionally suitable for
the frozen checkpoint scorer and for evidence curation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant.brain_forward.evidence import PROTOCOL, ForwardProtocol, build_checkpoint_report


def _get_conn() -> Any:
    from quant.execution.event_store import get_conn

    return get_conn()


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _records(cursor: Any) -> list[dict[str, Any]]:
    columns = [getattr(column, "name", column[0]) for column in cursor.description]
    return [
        {str(column): _json_value(value) for column, value in zip(columns, row)}
        for row in cursor.fetchall()
    ]


def export_protocol_ledger(protocol: ForwardProtocol) -> dict[str, list[dict[str, Any]]]:
    """Read the exact protocol-scoped formal ledger and required minute coverage.

    A missing or altered stored protocol is an integrity failure, never an empty
    successful export.  No DDL or DML is issued in this path.
    """

    from datetime import timedelta

    coverage_end = protocol.evidence_end + timedelta(minutes=protocol.outcome_maturity_minutes)
    with _get_conn() as conn, conn.cursor() as cursor:
        cursor.execute(
            "select protocol_sha256 from brain_forward_protocols where protocol_id=%s",
            (protocol.protocol_id,),
        )
        row = cursor.fetchone()
        if not row or str(row[0]) != protocol.protocol_sha256:
            raise RuntimeError("stored brain-forward protocol is absent or differs from the frozen protocol")

        cursor.execute(
            """
            select decision_id, event_ts, symbol, source, expected_net_bps,
                   candle_range, active_memories, payload_json, protocol_id,
                   protocol_sha256, artifact_sha256, evidence_phase
              from brain_forward_decisions
             where protocol_id=%s and protocol_sha256=%s and evidence_phase='formal'
             order by event_ts, decision_id
            """,
            (protocol.protocol_id, protocol.protocol_sha256),
        )
        decisions = _records(cursor)

        cursor.execute(
            """
            select decision_id, event_ts, entry_ts, exit_ts, entry_price,
                   exit_price, target_price, stop_price, exit_reason, gross_bps,
                   net_bps, expected_net_bps, protocol_id, protocol_sha256,
                   artifact_sha256, symbol, source, evidence_phase
              from brain_forward_trades
             where protocol_id=%s and protocol_sha256=%s and evidence_phase='formal'
             order by entry_ts, decision_id
            """,
            (protocol.protocol_id, protocol.protocol_sha256),
        )
        trades = _records(cursor)

        cursor.execute(
            """
            select ts
              from brain_forward_minute_bars
             where symbol=%s and source=%s and ts >= %s and ts < %s
             order by ts
            """,
            (protocol.symbol, protocol.source, protocol.evidence_start, coverage_end),
        )
        bar_timestamps = _records(cursor)
    return {"decisions": decisions, "trades": trades, "bar_timestamps": bar_timestamps}


def _write_json(path: Path, value: Any) -> str:
    rendered = json.dumps(_json_value(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(rendered, encoding="utf-8")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def write_evidence_bundle(
    protocol: ForwardProtocol,
    ledger: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    as_of: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Create a non-overwriting, hashed checkpoint bundle from a read-only ledger."""

    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence bundle: {output_dir}")
    output_dir.mkdir(parents=True)
    decisions = [dict(row) for row in ledger["decisions"]]
    trades = [dict(row) for row in ledger["trades"]]
    bar_timestamps = [dict(row) for row in ledger["bar_timestamps"]]
    report = build_checkpoint_report(
        protocol,
        decisions,
        trades,
        [row["ts"] for row in bar_timestamps],
        as_of=as_of,
    )
    hashes = {
        "decisions.json": _write_json(output_dir / "decisions.json", decisions),
        "trades.json": _write_json(output_dir / "trades.json", trades),
        "bar_timestamps.json": _write_json(output_dir / "bar_timestamps.json", bar_timestamps),
        "checkpoint_report.json": _write_json(output_dir / "checkpoint_report.json", report),
    }
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol.protocol_sha256,
        "artifact_sha256": protocol.artifact_sha256,
        "as_of": str(as_of),
        "files_sha256": hashes,
        "verdict": report["verdict"],
        "live_orders_permitted": False,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a read-only frozen Brain-forward evidence bundle")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--as-of", required=True, help="ISO-8601 checkpoint evaluation time")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = ForwardProtocol.load(args.protocol)
    report = write_evidence_bundle(
        protocol,
        export_protocol_ledger(protocol),
        as_of=args.as_of,
        output_dir=args.output_dir,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "verdict": report["verdict"], "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0 if report["verdict"] == "eligible_for_shadow_champion_review" else 2


if __name__ == "__main__":
    raise SystemExit(main())
