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

    def test_empty_decisions_with_legacy_closed_trades_synthesizes_history(self) -> None:
        # Spec part C: when the decision spine is smaller than the
        # ``closed_trades`` count, the chart MUST NOT silently truncate.
        # The builder synthesizes ``td_ct_synth_*`` decision points in
        # memory so historical legs stay visible, and flags
        # ``needs_backfill=True`` so the UI can offer the operator the
        # explicit ``?backfill=1`` override that persists those rows.
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
        self.assertEqual(len(out["points"]), 1)
        p = out["points"][0]
        self.assertTrue(p["decision_id"].startswith("td_ct_synth_"))
        self.assertEqual(p["side"], "long")
        self.assertAlmostEqual(p["pnl_pct"], 1.0, places=4)
        self.assertAlmostEqual(p["cum_pct"], 1.0, places=4)
        self.assertEqual(p["source"], "synth:closed_trades")
        self.assertTrue(out["needs_backfill"])
        self.assertEqual(out["synthesized_count"], 1)
        self.assertEqual(out["decision_count"], 0)
        self.assertEqual(out["closed_trade_count"], 1)

    def test_partial_spine_synthesises_remaining_closed_trades(self) -> None:
        # Spec part C: when the decision spine covers only a tiny fraction
        # of the historical ``closed_trades``, the builder fills the gap
        # in-memory with ``td_ct_synth_*`` rows so the chart and card both
        # see the full history. The card's ``pnl_pct`` must equal the
        # chart's final ``cum_pct`` after the merge.
        decisions = [
            _dec("d_recent", "2026-05-15T14:00:00Z", "long"),
        ]
        ct = pd.DataFrame(
            [
                # 1) Historical leg the spine doesn't cover.
                _ct(
                    entry_ts="2025-01-01T10:00:00Z",
                    exit_ts="2025-01-01T11:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=104.0,
                    pnl_pct=4.0,
                    trade_id="hist_1",
                ),
                # 2) Second historical leg.
                _ct(
                    entry_ts="2025-02-01T10:00:00Z",
                    exit_ts="2025-02-01T11:00:00Z",
                    side="short",
                    entry_price=120.0,
                    exit_price=118.0,
                    pnl_pct=1.5,
                    trade_id="hist_2",
                ),
                # 3) Matches the recent decision in the spine.
                _ct(
                    entry_ts="2026-05-15T14:00:00Z",
                    exit_ts="2026-05-15T15:30:00Z",
                    side="long",
                    entry_price=91.0,
                    exit_price=92.0,
                    pnl_pct=1.1,
                    trade_id="recent",
                ),
            ]
        )
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
            open_side="",
        )
        pts = out["points"]
        # 1 real decision + 2 synthesised historical legs == 3 points,
        # ordered chronologically by entry time.
        self.assertEqual(len(pts), 3)
        self.assertTrue(pts[0]["decision_id"].startswith("td_ct_synth_"))
        self.assertTrue(pts[1]["decision_id"].startswith("td_ct_synth_"))
        self.assertEqual(pts[2]["decision_id"], "d_recent")
        # Cumulative line must accumulate over the merged points so the
        # chart's final cum_pct == card's pnl_pct.
        self.assertAlmostEqual(pts[0]["cum_pct"], 4.0, places=4)
        self.assertAlmostEqual(pts[1]["cum_pct"], 5.5, places=4)
        self.assertAlmostEqual(pts[2]["cum_pct"], 6.6, places=4)
        self.assertTrue(out["needs_backfill"])
        self.assertEqual(out["synthesized_count"], 2)
        self.assertEqual(out["decision_count"], 1)
        self.assertEqual(out["closed_trade_count"], 3)

    def test_synthesized_points_round_trip_through_performance_card(self) -> None:
        # The user's clear intent: card and chart represent the SAME data.
        # When synthesis kicks in, the card's aggregates must still cover
        # the full point list — not just the spine subset.
        decisions: List[Dict[str, Any]] = []
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2025-01-01T10:00:00Z",
                    exit_ts="2025-01-01T11:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=103.0,
                    pnl_pct=3.0,
                    trade_id="hist_1",
                ),
                _ct(
                    entry_ts="2025-01-02T10:00:00Z",
                    exit_ts="2025-01-02T11:00:00Z",
                    side="short",
                    entry_price=120.0,
                    exit_price=121.0,
                    pnl_pct=-0.83,
                    trade_id="hist_2",
                ),
            ]
        )
        payload = build_decision_dashboard_payload(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
            now=pd.Timestamp("2025-02-01T00:00:00Z"),
        )
        perf = payload["performance"]
        chart_pts = payload["curve"]["points"]
        self.assertEqual(len(chart_pts), 2)
        self.assertEqual(perf["trade_count"], 2)
        # 1 winner / 1 loser when synthesized rows are honoured.
        self.assertEqual(perf["winning_trade_count"], 1)
        self.assertEqual(perf["losing_trade_count"], 1)
        self.assertAlmostEqual(perf["pnl_pct"], chart_pts[-1]["cum_pct"], places=4)
        self.assertTrue(payload["needs_backfill"])
        self.assertEqual(payload["synthesized_count"], 2)

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


class CardChartInvariantsTests(unittest.TestCase):
    """Regression guard for the spec's card-aggregate contract.

    These tests pin down the four invariants the spec calls out:

    * ``card.pnl_pct == merged_points[-1].cum_pct``
    * ``card.trade_count == |{p.decision_id : p in merged_points}|``
    * ``wins + losses + open_decisions == trade_count`` (when no
      neutral pnl=0 closures exist)
    * ``average_gain == mean(closed pnl_pct)``
    """

    def _payload_with_history(self) -> Dict[str, Any]:
        # Realistic shape: 3 historical synth legs predating the spine
        # (mix of wins/losses), then 2 spine decisions (1 closed win,
        # 1 open).
        decisions = [
            _dec("d_recent_a", "2026-05-15T10:00:00Z", "long", seq=1),
            _dec("d_recent_b", "2026-05-15T12:00:00Z", "short", seq=2),
        ]
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2025-01-10T10:00:00Z",
                    exit_ts="2025-01-10T11:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=102.0,
                    pnl_pct=2.0,
                    trade_id="hist_1",
                ),
                _ct(
                    entry_ts="2025-02-10T10:00:00Z",
                    exit_ts="2025-02-10T11:00:00Z",
                    side="short",
                    entry_price=120.0,
                    exit_price=121.0,
                    pnl_pct=-0.83,
                    trade_id="hist_2",
                ),
                _ct(
                    entry_ts="2025-03-10T10:00:00Z",
                    exit_ts="2025-03-10T11:00:00Z",
                    side="long",
                    entry_price=110.0,
                    exit_price=115.5,
                    pnl_pct=5.0,
                    trade_id="hist_3",
                ),
                _ct(
                    entry_ts="2026-05-15T10:00:00Z",
                    exit_ts="2026-05-15T11:00:00Z",
                    side="long",
                    entry_price=200.0,
                    exit_price=198.0,
                    pnl_pct=-1.0,
                    trade_id="recent_a",
                ),
            ]
        )
        return build_decision_dashboard_payload(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
            open_side="short",
            now=pd.Timestamp("2026-05-15T13:00:00Z"),
        )

    def test_card_pnl_pct_equals_merged_points_last_cum_pct(self) -> None:
        payload = self._payload_with_history()
        merged = payload["curve"]["merged_points"]
        perf = payload["performance"]
        self.assertAlmostEqual(perf["pnl_pct"], merged[-1]["cum_pct"], places=4)

    def test_card_trade_count_matches_unique_decision_ids_in_merged_list(self) -> None:
        payload = self._payload_with_history()
        merged = payload["curve"]["merged_points"]
        unique_ids = {p["decision_id"] for p in merged}
        self.assertEqual(payload["performance"]["trade_count"], len(unique_ids))

    def test_wins_plus_losses_plus_open_equals_trade_count(self) -> None:
        # Fixture intentionally avoids pnl=0 closures so the strict
        # equality holds.
        payload = self._payload_with_history()
        perf = payload["performance"]
        self.assertEqual(
            perf["winning_trade_count"]
            + perf["losing_trade_count"]
            + perf["open_decision_count"],
            perf["trade_count"],
        )

    def test_average_gain_equals_mean_of_closed_pnl(self) -> None:
        payload = self._payload_with_history()
        merged = payload["curve"]["merged_points"]
        closed_pnls = [
            float(p["pnl_pct"]) for p in merged
            if not p.get("open") and p.get("pnl_pct") is not None
        ]
        expected = sum(closed_pnls) / len(closed_pnls)
        self.assertAlmostEqual(payload["performance"]["average_gain"], expected, places=4)

    def test_monthly_growth_uses_calendar_month_window(self) -> None:
        # Three closed legs: one in April, two in May. "Now" is May 20.
        # Monthly growth must sum the May legs only — NOT a copy of
        # ``pnl_pct``.
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2026-04-20T10:00:00Z",
                    exit_ts="2026-04-20T11:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=104.0,
                    pnl_pct=4.0,
                    trade_id="april",
                ),
                _ct(
                    entry_ts="2026-05-01T00:00:00Z",
                    exit_ts="2026-05-01T01:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=99.0,
                    pnl_pct=-1.0,
                    trade_id="may_first",
                ),
                _ct(
                    entry_ts="2026-05-19T12:00:00Z",
                    exit_ts="2026-05-19T13:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=102.0,
                    pnl_pct=2.0,
                    trade_id="may_mid",
                ),
            ]
        )
        payload = build_decision_dashboard_payload(
            symbol="SOL-USDT",
            decisions=[],
            closed_trades_df=ct,
            now=pd.Timestamp("2026-05-20T08:00:00Z"),
        )
        perf = payload["performance"]
        # April leg excluded; May legs: -1 + 2 = 1.0
        self.assertAlmostEqual(perf["monthly_growth"], 1.0, places=4)
        # Total cumulative: 4 + (-1) + 2 = 5.0
        self.assertAlmostEqual(perf["pnl_pct"], 5.0, places=4)
        # Critical regression guard: monthly_growth must not silently
        # mirror pnl_pct when the spine covers earlier months.
        self.assertNotAlmostEqual(perf["monthly_growth"], perf["pnl_pct"], places=4)

    def test_monthly_growth_falls_back_to_exit_time_when_entry_time_missing(self) -> None:
        # Synth points carry an exit_time but their entry_time may be
        # ``None`` (1970/NaT-zero sentinel). The monthly bucket still
        # needs to include them via the exit_time fallback.
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="1970-01-01T00:00:00Z",  # sentinel -> entry_time None
                    exit_ts="2026-05-19T13:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=102.0,
                    pnl_pct=2.0,
                    trade_id="sentinel_may",
                ),
            ]
        )
        payload = build_decision_dashboard_payload(
            symbol="SOL-USDT",
            decisions=[],
            closed_trades_df=ct,
            now=pd.Timestamp("2026-05-20T08:00:00Z"),
        )
        perf = payload["performance"]
        # Bucketed by exit_time fallback (May 19) -> +2.0 in May.
        self.assertAlmostEqual(perf["monthly_growth"], 2.0, places=4)

    def test_winrate_is_none_when_no_closed_decisions(self) -> None:
        # Only an open decision in the spine, no matched closes ->
        # winrate must be ``None``, not 0/0 == 0%.
        decisions = [_dec("d_open", "2026-05-15T10:00:00Z", "long")]
        payload = build_decision_dashboard_payload(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=pd.DataFrame(),
            open_side="long",
        )
        perf = payload["performance"]
        self.assertEqual(perf["open_decision_count"], 1)
        self.assertEqual(perf["winning_trade_count"], 0)
        self.assertEqual(perf["losing_trade_count"], 0)
        self.assertIsNone(perf["winrate"])

    def test_card_aggregates_use_full_merged_list_not_downsampled(self) -> None:
        # 30 synth legs, max_points capped to 5 -> chart downsamples,
        # but trade_count / pnl_pct must still reflect the full 30.
        rows = []
        for i in range(30):
            day = (i % 28) + 1
            rows.append(
                _ct(
                    entry_ts=f"2025-01-{day:02d}T10:00:00Z",
                    exit_ts=f"2025-01-{day:02d}T11:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=101.0,
                    pnl_pct=1.0,
                    trade_id=f"t{i}",
                )
            )
        payload = build_decision_dashboard_payload(
            symbol="SOL-USDT",
            decisions=[],
            closed_trades_df=pd.DataFrame(rows),
            max_points=5,
        )
        curve = payload["curve"]
        perf = payload["performance"]
        self.assertEqual(len(curve["merged_points"]), 30)
        self.assertEqual(len(curve["points"]), 5)
        self.assertEqual(perf["trade_count"], 30)
        # ``pnl_pct`` still reflects the full cumulative — first and
        # last are preserved by the uniform downsample.
        self.assertAlmostEqual(perf["pnl_pct"], 30.0, places=4)
        self.assertAlmostEqual(
            curve["points"][-1]["cum_pct"], curve["merged_points"][-1]["cum_pct"], places=4
        )

    def test_chart_time_includes_full_historical_range(self) -> None:
        # Even with a 1970 sentinel ``entry_ts`` on the historical
        # synth leg, the chart's earliest plotted ``time`` must match
        # the earliest ``exit_ts`` across ``closed_trades`` — exit_ts
        # is the only reliably-historic anchor.
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="1970-01-01T00:00:00Z",  # sentinel
                    exit_ts="2025-03-01T10:00:00Z",   # real historic
                    side="long",
                    entry_price=100.0,
                    exit_price=101.0,
                    pnl_pct=1.0,
                    trade_id="hist_sentinel",
                ),
                _ct(
                    entry_ts="2026-05-15T14:00:00Z",
                    exit_ts="2026-05-15T15:30:00Z",
                    side="long",
                    entry_price=200.0,
                    exit_price=205.0,
                    pnl_pct=2.5,
                    trade_id="recent",
                ),
            ]
        )
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=[],
            closed_trades_df=ct,
        )
        pts = out["points"]
        self.assertEqual(len(pts), 2)

        earliest_exit_ts = int(pd.Timestamp("2025-03-01T10:00:00Z").timestamp())
        self.assertEqual(min(int(p["time"]) for p in pts), earliest_exit_ts)

        # The sentinel synth row keeps entry_time=None for tooltip use
        # — we do NOT silently substitute exit_time so the UI can render
        # "—" instead of a fake entry.
        sentinel_pt = next(p for p in pts if float(p["pnl_pct"]) == 1.0)
        self.assertIsNone(sentinel_pt["entry_time"])
        self.assertEqual(
            int(sentinel_pt["exit_time"]),
            int(pd.Timestamp("2025-03-01T10:00:00Z").timestamp()),
        )
        self.assertEqual(int(sentinel_pt["time"]), int(sentinel_pt["exit_time"]))

    def test_real_decision_with_unusable_entry_falls_back_to_matched_exit_for_chart_time(self) -> None:
        # A real (action-event spine) decision with an unusable ts
        # would otherwise be skipped or plot at 0. The builder must
        # derive its chart ``time`` from the matched closed leg's
        # ``exit_ts`` instead so the spine row still appears.
        decisions = [
            {
                "decision_id": "d_bad_ts",
                "ts": "1970-01-01T00:00:00Z",
                "venue": "kucoin",
                "symbol": "SOL-USDT",
                "decision_kind": "entry",
                "direction": "long",
                "seq": 1,
                "engine_action": "enter_long",
                "source_action_event_id": "src-bad",
                "payload_json": {},
            },
        ]
        ct = pd.DataFrame(
            [
                _ct(
                    entry_ts="2026-05-15T10:00:00Z",
                    exit_ts="2026-05-15T11:00:00Z",
                    side="long",
                    entry_price=100.0,
                    exit_price=101.0,
                    pnl_pct=1.0,
                    trade_id="r1",
                ),
            ]
        )
        out = build_decision_equity_curve(
            symbol="SOL-USDT",
            decisions=decisions,
            closed_trades_df=ct,
        )
        # The decision's ts is unusable, but the matched leg's exit_ts
        # is, so a single point with ``time == exit_time`` is plotted.
        # _normalize_decisions drops the bad-ts row before it reaches
        # the matcher, so the matched leg surfaces as a synth point
        # instead — same chart contract either way.
        self.assertEqual(len(out["points"]), 1)
        p = out["points"][0]
        self.assertEqual(
            int(p["time"]),
            int(pd.Timestamp("2026-05-15T11:00:00Z").timestamp()),
        )

    def test_duplicate_decision_ids_are_deduped(self) -> None:
        # Two action-event rows accidentally upserted with the same
        # decision_id must collapse to one entry in the merged list so
        # the card's ``trade_count`` doesn't inflate.
        decisions = [
            _dec("dup", "2026-05-15T10:00:00Z", "long", seq=1),
            _dec("dup", "2026-05-15T10:00:00Z", "long", seq=2),
        ]
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
        merged = payload["curve"]["merged_points"]
        self.assertEqual(len(merged), 1)
        self.assertEqual(payload["performance"]["trade_count"], 1)


if __name__ == "__main__":
    unittest.main()
