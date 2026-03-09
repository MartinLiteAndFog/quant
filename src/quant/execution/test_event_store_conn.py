from __future__ import annotations

from quant.execution.event_store import get_conn


def main() -> None:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("select 1 as ok")
        row = cur.fetchone()
        print({"ok": row[0] if row else None})


if __name__ == "__main__":
    main()
    