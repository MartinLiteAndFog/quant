"""Synthetic snapshot and adversarial tests for offline TV Renko comparison."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal as D
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from renko_reference_audit import (
    ParityInputError, compare, compare_alerts, read_alerts, read_bricks,
    read_source,
)

BOX = D("0.1")
DAY = "2026-09-15"


def write_csv(path: Path, headers: list[str], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


class ReferenceAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.tv, self.local = root / "tv.csv", root / "local.csv"
        self.source, self.tv_alerts, self.local_alerts = (
            root / "source.csv", root / "tv_alerts.csv", root / "local_alerts.csv"
        )
        self.output = root / "report.json"
        self.brick_headers = ["ts", "open", "high", "low", "close", "direction"]
        self.tv_rows = [
            (f"{DAY}T00:05:00Z", "100", "100.1", "99.98", "100.1", "1"),
            (f"{DAY}T00:05:00.001Z", "100.1", "100.2", "100.1", "100.2", "1"),
            (f"{DAY}T00:10:00Z", "100.1", "100.1", "100.0", "100.0", "-1"),
        ]
        self.local_rows = [
            (f"{DAY}T00:05:00Z", "100", "100.1", "100", "100.1", "1"),
            self.tv_rows[1], self.tv_rows[2],
        ]
        write_csv(self.tv, self.brick_headers, self.tv_rows)
        write_csv(self.local, self.brick_headers, self.local_rows)
        write_csv(self.source, ["open_at", "open", "high", "low", "close"], [
            (f"{DAY}T00:05:00Z", "100", "100.3", "99.9", "100.2"),
            (f"{DAY}T00:10:00Z", "100.2", "100.2", "99.8", "100.0"),
        ])
        alert_headers = ["reference_id", "direction", "confirmed_5m_close_at"]
        write_csv(self.tv_alerts, alert_headers, [
            ("tv-01", "1", f"{DAY}T00:10:00Z"),
            ("tv-02", "-1", f"{DAY}T00:15:00Z"),
        ])
        write_csv(self.local_alerts, alert_headers, [
            ("local-01", "1", f"{DAY}T00:10:00Z"),
            ("local-02", "-1", f"{DAY}T00:15:00Z"),
        ])

    def read(self):
        return read_bricks(self.tv, box=BOX), read_bricks(self.local, box=BOX)

    def test_snapshot_wick_difference_and_causal_available_time(self):
        r, l = self.read()
        self.assertEqual(r[0].source_open_at, datetime(2026, 9, 15, 0, 5, tzinfo=timezone.utc))
        self.assertEqual(r[0].known_at, datetime(2026, 9, 15, 0, 10, tzinfo=timezone.utc))
        self.assertEqual(r[1].sequence, 1)
        self.assertLess(r[1].display_at, r[1].known_at)
        report = compare(r, l, read_source(self.source), box=BOX)
        self.assertEqual(report["counts"], {
            "reference_bricks": 3, "local_bricks": 3,
            "brick_delta_local_minus_reference": 0,
            "reference_wick_bricks": 1, "local_wick_bricks": 0,
            "bucket_count_reference": 2, "bucket_count_local": 2,
            "different_brick_count_buckets": 0,
            "body_mismatch_bricks": 0, "direction_mismatch_bricks": 0,
            "wick_mismatch_bricks": 1,
            "display_timestamp_mismatch_bricks": 0,
            "source_range_violations": 0,
            "reference_buckets_without_source": 0,
            "reference_geometry_anomalies": 0,
        })
        self.assertEqual(report["first_mismatches"][0]["reason"], "wick_or_high_low")
        self.assertFalse(report["parity_verified"])
        self.assertFalse(report["wick_reconstruction_verified"])

    def test_missing_brick_and_anchor_delta_are_reported_without_padding(self):
        r, _ = self.read()
        write_csv(self.local, self.brick_headers, [
            (f"{DAY}T00:05:00Z", "100.1", "100.2", "100.1", "100.2", "1"),
            (f"{DAY}T00:10:00Z", "100.2", "100.2", "100.1", "100.1", "-1"),
        ])
        l = read_bricks(self.local, box=BOX)
        report = compare(r, l, box=BOX)
        self.assertEqual(report["counts"]["brick_delta_local_minus_reference"], -1)
        self.assertEqual(report["counts"]["different_brick_count_buckets"], 1)
        self.assertEqual(report["anchor_diagnostic"]["first_open_reference_minus_local"], "-0.1")
        self.assertTrue(report["anchor_diagnostic"]["whole_box_offset"])
        self.assertFalse(report["parity_verified"])

    def test_corrupt_timestamp_and_wrong_box_fail_closed(self):
        write_csv(self.tv, self.brick_headers,
                  [self.tv_rows[0], (f"{DAY}T00:05:00.002Z", *self.tv_rows[1][1:])])
        with self.assertRaisesRegex(ParityInputError, "millisecond"):
            read_bricks(self.tv, box=BOX)
        write_csv(self.tv, self.brick_headers, [
            (f"{DAY}T00:05:00Z", "100", "100.2", "100", "100.2", "1")])
        with self.assertRaisesRegex(ParityInputError, "one box"):
            read_bricks(self.tv, box=BOX)

    def test_naive_time_and_source_gaps_rejected(self):
        write_csv(self.tv, self.brick_headers,
                  [(f"{DAY}T00:05:00", *self.tv_rows[0][1:])])
        with self.assertRaisesRegex(ParityInputError, "timezone missing"):
            read_bricks(self.tv, box=BOX)
        write_csv(self.source, ["open_at", "open", "high", "low", "close"], [
            (f"{DAY}T00:05:00Z", "100", "101", "99", "100"),
            (f"{DAY}T00:15:00Z", "100", "101", "99", "100"),
        ])
        with self.assertRaisesRegex(ParityInputError, "gap"):
            read_source(self.source)

    def test_source_wick_constraint_and_reference_geometry(self):
        r, l = self.read()
        write_csv(self.source, ["open_at", "open", "high", "low", "close"], [
            (f"{DAY}T00:05:00Z", "100", "100.3", "100.0", "100.2"),
            (f"{DAY}T00:10:00Z", "100.2", "100.2", "99.8", "100.0"),
        ])
        report = compare(r, l, read_source(self.source), box=BOX)
        self.assertEqual(report["counts"]["source_range_violations"], 1)

    def test_alerts_exact_and_missing_are_not_tolerated(self):
        ref = read_alerts(self.tv_alerts)
        loc = read_alerts(self.local_alerts)
        self.assertTrue(compare_alerts(ref, loc)["matches_supplied_alerts"])
        self.assertFalse(compare_alerts(ref, loc)["parity_verified"])
        write_csv(self.local_alerts,
                  ["reference_id", "direction", "confirmed_5m_close_at"], [
                      ("local-01", "-1", f"{DAY}T00:10:00Z"),
                      ("local-02", "-1", f"{DAY}T00:20:00Z"),
                  ])
        report = compare_alerts(ref, read_alerts(self.local_alerts))
        self.assertEqual(len(report["direction_mismatch_times"]), 1)
        self.assertEqual(len(report["missing_local_times"]), 1)
        self.assertEqual(len(report["extra_local_times"]), 1)
        self.assertFalse(report["matches_supplied_alerts"])

    def test_cli_writes_hashed_comparison_but_never_executable_trades(self):
        command = [sys.executable, "research/parity/renko_reference_audit.py",
                   "--tv-bricks", str(self.tv), "--local-bricks", str(self.local),
                   "--source-5m", str(self.source),
                   "--tv-alerts", str(self.tv_alerts),
                   "--local-alerts", str(self.local_alerts),
                   "--output", str(self.output)]
        p = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        result = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(result["parity_verified"])
        self.assertEqual(result["live_orders_sent"], 0)
        self.assertEqual(result["trade_target_signals_exported"], 0)
        self.assertEqual(len(result["input_sha256"]), 5)
        self.assertTrue(result["alert_comparison"]["matches_supplied_alerts"])
        p = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("refusing overwrite", p.stderr)


if __name__ == "__main__":
    unittest.main()
