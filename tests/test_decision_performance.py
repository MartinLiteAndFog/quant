"""Tests for the decision-based equity / performance pipeline.

The decision-based builder is the single source of truth for both the
trade-mode equity chart and the Performance card on the sidebar. Every
aggregate the card exposes must be reproducible by walking the same point
list the chart plots — these tests pin that invariant down.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

import pandas as pd

from quant.execution.decision_performance import (
    build_decision_equity_curve,
    compute_performance_from_decision_points,
    build_decision_dashboard_payload,
)


def _dec(
    decision_id: str,
    ts: str,
    direction: str,
    *,
    kind: str = "entry",
    seq: int = 1,
) -> Dict[str, Any]:
    return {
        "decision_id": decision_id,
        "ts": ts,
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "decision_kind": kind,
        "direction": direction,
        "seq": seq,
        "engine_action": "enter_long" if direction == "long" else "enter_short",
        "source_action_event_id": f"src-{decision_id}",
        "payload_json": {},
    }


def _ct(
    *,
    entry_ts: str,
    exit_ts: str,
    side: str,
    entry_price: float,
    exit_price: float,
    pnl_pct: float,
    trade_id: str = "t",
) -> Dict[str, Any]:
    return {
        "trade_id": trade_id,
        "venue": "kucoin",
        "symbol": "SOL-USDT",
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "side": side,
        "qty": 1.0,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
        "exit_event": "tp_exit",
    }


class BuildDecisionEquityCurveTests(unittest.TestCase):
    def test_single_closed_decision_attributes_to_matching_close(self) -> None:
        decisions = [
            _dec("d1", "2026-05-15T14:00:00Z", "long"),
        ]
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2026-05-15T14:00:00Z",
                    exit_ts="2026-05-15T15:30:00Z",
                    side="long",
                    entry_price=91.38,
                    exit_price=89.82,
                    pnl_pct=-1.71,
                    trade_id="t1",
                ),
            ]
        )
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
            open_side=None,
        )
        pts = out["points"]
        self.assertEqual(len(pts), 1)
        p = pts[0]
        self.assertEqual(p["decision_id"], "d1")
        self.assertEqual(p["side"], "long")
        self.assertFalse(p["open"])
        self.assertAlmostEqual(p["pnl_pct"], -1.71, places=4)
        self.assertAlmostEqual(p["cum_pct"], -1.71, places=4)
        self.assertEqual(p["entry_price"], 91.38)
        self.assertEqual(p["exit_price"], 89.82)
        self.assertEqual(
            int(p["entry_time"]),
            int(pd.Timestamp("2026-05-15T14:00:00Z").timestamp()),
        )
        self.assertEqual(
            int(p["exit_time"]),
            int(pd.Timestamp("2026-05-15T15:30:00Z").timestamp()),
        )
        # Entry time MUST be the real event-based timestamp, never the
        # epoch fallback that produced "1/1/1970" in the tooltip.
        self.assertGreater(
            int(p["entry_time"]),
            int(pd.Timestamp("2020-01-01", tz="UTC").timestamp()),
        )

    def test_flip_sequence_produces_two_closed_decisions(self) -> None:
        # Flat -> long (d1) -> flip short (d2) -> exit. The flip itself
        # closes d1 at one price and opens d2 at the same fill; d2 closes
        # later. Two decisions, both closed, with their own PnL.
        decisions = [
            _dec("d1", "2026-05-15T10:00:00Z", "long", seq=1),
            _dec("d2", "2026-05-15T12:00:00Z", "short", kind="flip", seq=2),
        ]
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2026-05-15T10:00:00Z",
                    exit_ts="2026-05-15T12:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=103.0,
                    pnl_pct=3.0,
                    trade_id="t1",
                ),
                _ct(
                    entry_ts="2026-05-15T12:00:00Z",
                    exit_ts="2026-05-15T13:30:00Z",
                    side="short",
                    entry_price=103.0,
                    exit_price=101.0,
                    pnl_pct=1.94,
                    trade_id="t2",
                ),
            ]
        )
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
        )
        pts = out["points"]
        self.assertEqual(len(pts), 2)
        self.assertEqual([p["side"] for p in pts], ["long", "short"])
        self.assertFalse(any(p["open"] for p in pts))
        self.assertAlmostEqual(pts[0]["pnl_pct"], 3.0, places=4)
        self.assertAlmostEqual(pts[1]["pnl_pct"], 1.94, places=4)
        # Cumulative is additive — must equal the chart's last cum_pct.
        self.assertAlmostEqual(pts[1]["cum_pct"], 3.0 + 1.94, places=4)

    def test_open_decision_when_no_matching_close(self) -> None:
        # Latest decision has no closed_trades row yet; current live position
        # is the same side -> mark as open and exclude from wins / losses.
        decisions = [
            _dec("d1", "2026-05-15T10:00:00Z", "long"),
            _dec("d2", "2026-05-15T12:00:00Z", "short"),
        ]
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2026-05-15T10:00:00Z",
                    exit_ts="2026-05-15T12:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=102.0,
                    pnl_pct=2.0,
                    trade_id="t1",
                ),
            ]
        )
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
            open_side="short",
        )
        pts = out["points"]
        self.assertEqual(len(pts), 2)
        self.assertFalse(pts[0]["open"])
        self.assertTrue(pts[1]["open"])
        self.assertIsNone(pts[1]["pnl_pct"])
        # Cumulative does not move when an open decision is appended.
        self.assertAlmostEqual(pts[1]["cum_pct"], 2.0, places=4)

    def test_open_decision_is_not_counted_when_side_does_not_match_live_position(self) -> None:
        # Latest decision is long, but live position is flat (open_side=None
        # via detect path): the unmatched decision is still surfaced but not
        # flagged as open — the chart shouldn't promise unrealised PnL we
        # cannot verify.
        decisions = [_dec("d1", "2026-05-15T10:00:00Z", "long")]
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=pd.DataFrame(),
            open_side="",  # explicitly no live position
        )
        pts = out["points"]
        self.assertEqual(len(pts), 1)
        # When the live position is explicitly flat, an unmatched decision
        # is NOT promoted to ``open=True`` since we have no live confirmation.
        self.assertFalse(pts[0]["open"])
        self.assertIsNone(pts[0]["pnl_pct"])

    def test_garbage_entry_ts_in_closed_trades_does_not_pollute_entry_time(self) -> None:
        # The decision's own ts (from the action_event spine) is what defines
        # ``entry_time``. Even if the matched closed_trades row carries a
        # bogus 1970 entry_ts, the chart shows the decision's real entry ts.
        decisions = [_dec("d1", "2026-05-15T14:00:00Z", "long")]
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="1970-01-01T00:00:01Z",  # epoch-1 sentinel
                    exit_ts="2026-05-15T15:30:00Z",
                    side="long",
                    entry_price=91.38,
                    exit_price=89.82,
                    pnl_pct=-1.71,
                    trade_id="t1",
                ),
            ]
        )
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
        )
        p = out["points"][0]
        self.assertEqual(
            int(p["entry_time"]),
            int(pd.Timestamp("2026-05-15T14:00:00Z").timestamp()),
        )
        self.assertGreater(
            int(p["entry_time"]),
            int(pd.Timestamp("2020-01-01", tz="UTC").timestamp()),
        )

    def test_empty_decisions_with_legacy_closed_trades_flags_needs_backfill(self) -> None:
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2025-01-01T10:00:00Z",
                    exit_ts="2025-01-01T11:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=101.0,
                    pnl_pct=1.0,
                    trade_id="legacy_1",
                ),
            ]
        )
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=[],
            closed_trades_df=ct,
        )
        self.assertEqual(out["points"], [])
        self.assertTrue(out["needs_backfill"])

    def test_unsupported_venue_returns_empty(self) -> None:
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            venue="kraken",
            decisions=[_dec("d1", "2026-05-15T10:00:00Z", "long")],
            closed_trades_df=pd.DataFrame(),
        )
        self.assertEqual(out["points"], [])
        self.assertEqual(out["source"], "unsupported_venue")


class PerformanceFromDecisionPointsTests(unittest.TestCase):
    def _curve(self, decisions, closed) -> List[Dict[str, Any]]:
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=pd.DataFrame(closed),
            open_side=None,
        )
        return out["points"]

    def test_card_numbers_match_chart_cumulative(self) -> None:
        decisions = [
            _dec("d1", "2026-05-15T10:00:00Z", "long"),
            _dec("d2", "2026-05-15T12:00:00Z", "short", kind="flip"),
            _dec("d3", "2026-05-15T15:00:00Z", "long", kind="flip"),
        ]
        closed = [
            _ct(
                entry_ts="2026-05-15T10:00:00Z",
                exit_ts="2026-05-15T12:00:00Z",
                side="long",
                entry_price=100.0,
                exit_price=103.0,
                pnl_pct=3.0,
                trade_id="t1",
            ),
            _ct(
                entry_ts="2026-05-15T12:00:00Z",
                exit_ts="2026-05-15T15:00:00Z",
                side="short",
                entry_price=103.0,
                exit_price=104.0,
                pnl_pct=-0.97,
                trade_id="t2",
            ),
            _ct(
                entry_ts="2026-05-15T15:00:00Z",
                exit_ts="2026-05-15T16:30:00Z",
                side="long",
                entry_price=104.0,
                exit_price=105.0,
                pnl_pct=0.96,
                trade_id="t3",
            ),
        ]
        pts = self._curve(decisions, closed)
        perf = compute_performance_from_decision_points(
            pts,
            symbol="SOL-USDT",
            venue="kucoin",
            now=pd.Timestamp("2026-05-15T17:00:00Z"),
        )

        # The Sidebar primary invariants:
        # 1) Trade count = total decisions (open + closed)
        # 2) Wins + Losses == closed decisions (no neutral pnl=0 in this set)
        # 3) PnL % == chart's final cum_pct
        self.assertEqual(perf["trade_count"], len(pts))
        self.assertEqual(
            perf["winning_trade_count"] + perf["losing_trade_count"],
            perf["closed_decision_count"],
        )
        self.assertAlmostEqual(perf["pnl_pct"], pts[-1]["cum_pct"], places=4)

        expected_pnl = round(3.0 + (-0.97) + 0.96, 4)
        self.assertAlmostEqual(perf["pnl_pct"], expected_pnl, places=4)
        # Winrate from wins / (wins+losses): 2/3 ~= 66.6667%
        self.assertAlmostEqual(perf["winrate"], 66.6667, places=3)
        # Average trade = pnl_total / closed_decision_count
        self.assertAlmostEqual(perf["average_gain"], expected_pnl / 3, places=4)

    def test_open_decision_excluded_from_winrate_and_pnl(self) -> None:
        decisions = [
            _dec("d1", "2026-05-15T10:00:00Z", "long"),
            _dec("d2", "2026-05-15T12:00:00Z", "short"),
        ]
        closed = [
            _ct(
                entry_ts="2026-05-15T10:00:00Z",
                exit_ts="2026-05-15T12:00:00Z",
                side="long",
                entry_price=100.0,
                exit_price=105.0,
                pnl_pct=5.0,
                trade_id="t1",
            ),
        ]
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=pd.DataFrame(closed),
            open_side="short",
        )
        pts = out["points"]
        perf = compute_performance_from_decision_points(
            pts,
            symbol="SOL-USDT",
            venue="kucoin",
            now=pd.Timestamp("2026-05-15T17:00:00Z"),
        )
        self.assertEqual(perf["trade_count"], 2)
        self.assertEqual(perf["open_decision_count"], 1)
        self.assertEqual(perf["closed_decision_count"], 1)
        self.assertEqual(perf["winning_trade_count"], 1)
        self.assertEqual(perf["losing_trade_count"], 0)
        # PnL only counts the closed leg.
        self.assertAlmostEqual(perf["pnl_pct"], 5.0, places=4)

    def test_payload_helper_returns_chart_and_card_pair(self) -> None:
        decisions = [_dec("d1", "2026-05-15T10:00:00Z", "long")]
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2026-05-15T10:00:00Z",
                    exit_ts="2026-05-15T11:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=101.0,
                    pnl_pct=1.0,
                    trade_id="t1",
                ),
            ]
        )
        payload = build_decision_dashboard_payload(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
        )
        self.assertEqual(len(payload["curve"]["points"]), 1)
        self.assertEqual(payload["performance"]["trade_count"], 1)
        self.assertAlmostEqual(
            payload["performance"]["pnl_pct"],
            payload["curve"]["points"][-1]["cum_pct"],
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
