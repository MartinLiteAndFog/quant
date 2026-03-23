from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from quant.execution.event_store import get_conn


@dataclass
class RetentionResult:
    deleted_rows: int


def prune_equity_snapshots(
    *,
    keep_base_recent_hours: int = 24,
    base_bucket_minutes: int = 5,
    venue: Optional[str] = None,
    account: Optional[str] = None,
    dry_run: bool = True,
) -> RetentionResult:
    """
    Keep all event snapshots forever.
    For base snapshots older than keep_base_recent_hours:
      keep only the earliest row per (venue, account, symbol, minute-bucket),
      delete the rest.
    """
    if keep_base_recent_hours < 1:
        raise ValueError("keep_base_recent_hours must be >= 1")
    if base_bucket_minutes < 1:
        raise ValueError("base_bucket_minutes must be >= 1")

    sql = f"""
    WITH ranked AS (
        SELECT
            id,
            row_number() OVER (
                PARTITION BY
                    venue,
                    COALESCE(account, ''),
                    COALESCE(symbol, ''),
                    to_timestamp(
                        floor(extract(epoch from ts) / (%(bucket_sec)s)) * (%(bucket_sec)s)
                    )
                ORDER BY ts ASC, id ASC
            ) AS rn
        FROM equity_snapshots
        WHERE ts < now() - (%(keep_hours)s || ' hours')::interval
          AND COALESCE(payload_json->>'snapshot_kind', 'base') = 'base'
          AND (%(venue)s IS NULL OR venue = %(venue)s)
          AND (%(account)s IS NULL OR COALESCE(account, '') = %(account)s)
    )
    SELECT id
    FROM ranked
    WHERE rn > 1
    """

    params = {
        "keep_hours": int(keep_base_recent_hours),
        "bucket_sec": int(base_bucket_minutes) * 60,
        "venue": venue,
        "account": account,
    }

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        ids = [row[0] for row in cur.fetchall()]

        if dry_run or not ids:
            return RetentionResult(deleted_rows=len(ids))

        cur.execute("DELETE FROM equity_snapshots WHERE id = ANY(%s)", (ids,))
        return RetentionResult(deleted_rows=len(ids))


def main() -> None:
    p = argparse.ArgumentParser(description="Prune old base equity snapshots while keeping event snapshots.")
    p.add_argument("--keep-base-recent-hours", type=int, default=24)
    p.add_argument("--base-bucket-minutes", type=int, default=5)
    p.add_argument("--venue", type=str, default=None)
    p.add_argument("--account", type=str, default=None)
    p.add_argument("--apply", action="store_true", help="Actually delete rows. Default is dry-run.")
    args = p.parse_args()

    res = prune_equity_snapshots(
        keep_base_recent_hours=args.keep_base_recent_hours,
        base_bucket_minutes=args.base_bucket_minutes,
        venue=args.venue,
        account=args.account,
        dry_run=not args.apply,
    )

    mode = "APPLY" if args.apply else "DRY_RUN"
    print(
        f"[{mode}] equity retention candidates/deleted: {res.deleted_rows} "
        f"(keep_base_recent_hours={args.keep_base_recent_hours}, "
        f"base_bucket_minutes={args.base_bucket_minutes}, venue={args.venue}, account={args.account})"
    )


if __name__ == "__main__":
    main()