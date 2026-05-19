"""Unit + integration tests for the trade decision counter.

Classification rules (mirror the design in trade_counter.classify_action_event):

A "trade decision" is the discrete event of opening a new directional position
with its own SL/TP. Per user intent, every such decision counts once, including:

- Entering from flat (``enter_long`` / ``enter_short``).
- Flipping direction (``flip_to_long`` / ``flip_to_short``) — the new opposite
  leg is the counted decision; the close half of the flip does not double-count.
- Re-entering from flat after a previous exit (just another ``enter_*``).

The following are NOT counted as new decisions:

- Scaling buys (``scale_long`` / ``scale_short``) — same direction, no new SL/TP.
- Partial closes such as ``tp1_partial`` — reduces size, same direction.
- Full exits (``exit_long`` / ``exit_short``) — ends the trade lifecycle, but the
  lifecycle was already counted at entry/flip time.
- Holds, blocked actions, or unrelated events.
"""

from __future__ import annotations

import time
import unittest
from typing import Any, Dict, List, Optional

from quant.execution.trade_counter import (
    DECISION_ENTRY,
    DECISION_FLIP,
    TradeDecision,
    build_trade_decisions_from_action_events,
    classify_action_event,
    deterministic_decision_id,
)


def _ev(
    *,
    engine_action: str,
    action_side: Optional[str] = None,
    position_before: int = 0,
    position_after: int = 0,
    blocked: bool = False,
    reason_code: str = "test",
    ts: str = "2026-05-18T12:00:00Z",
    seq: int = 1,
    venue: str = "kucoin",
    symbol: str = "SOL-USDT",
    event_id: Optional[str] = None,
    payload_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event_id": event_id or f"act-{seq}",
        "ts": ts,
        "seq": seq,
        "venue": venue,
        "symbol": symbol,
        "engine_action": engine_action,
        "action_side": action_side,
        "position_before": position_before,
        "position_after": position_after,
        "blocked": blocked,
        "reason_code": reason_code,
        "strategy": "live_executor",
        "strategy_instance": "live_executor",
        "payload_json": payload_json or {},
    }


class ClassifyActionEventTests(unittest.TestCase):
    def test_enter_long_from_flat_counts_as_entry(self) -> None:
        d = classify_action_event(
            _ev(engine_action="enter_long", action_side="long", position_before=0, position_after=1)
        )
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.decision_kind, DECISION_ENTRY)
        self.assertEqual(d.direction, "long")

    def test_enter_short_from_flat_counts_as_entry(self) -> None:
        d = classify_action_event(
            _ev(engine_action="enter_short", action_side="short", position_before=0, position_after=-1)
        )
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.decision_kind, DECISION_ENTRY)
        self.assertEqual(d.direction, "short")

    def test_flip_to_long_counts_as_flip(self) -> None:
        d = classify_action_event(
            _ev(engine_action="flip_to_long", action_side="long", position_before=-1, position_after=1)
        )
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.decision_kind, DECISION_FLIP)
        self.assertEqual(d.direction, "long")

    def test_flip_to_short_counts_as_flip(self) -> None:
        d = classify_action_event(
            _ev(engine_action="flip_to_short", action_side="short", position_before=1, position_after=-1)
        )
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.decision_kind, DECISION_FLIP)
        self.assertEqual(d.direction, "short")

    def test_scale_long_is_ignored(self) -> None:
        # Adding to an existing long with no new SL/TP — not a new decision.
        d = classify_action_event(
            _ev(engine_action="scale_long", action_side="long", position_before=1, position_after=1)
        )
        self.assertIsNone(d)

    def test_scale_short_is_ignored(self) -> None:
        d = classify_action_event(
            _ev(engine_action="scale_short", action_side="short", position_before=-1, position_after=-1)
        )
        self.assertIsNone(d)

    def test_exit_long_is_not_a_new_decision(self) -> None:
        # The trade was already counted at entry; closing ends the lifecycle
        # but does not start a new trade.
        d = classify_action_event(
            _ev(engine_action="exit_long", action_side="flat", position_before=1, position_after=0)
        )
        self.assertIsNone(d)

    def test_exit_short_is_not_a_new_decision(self) -> None:
        d = classify_action_event(
            _ev(engine_action="exit_short", action_side="flat", position_before=-1, position_after=0)
        )
        self.assertIsNone(d)

    def test_tp1_partial_is_not_a_new_decision(self) -> None:
        # Partial close in same direction, SL stays the same.
        d = classify_action_event(
            _ev(engine_action="tp1_partial", action_side="long", position_before=1, position_after=1)
        )
        self.assertIsNone(d)

    def test_hold_is_not_a_decision(self) -> None:
        d = classify_action_event(
            _ev(engine_action="hold", action_side="long", position_before=1, position_after=1)
        )
        self.assertIsNone(d)

    def test_blocked_enter_is_not_counted(self) -> None:
        # A blocked action wasn't actually executed; no SL/TP committed.
        d = classify_action_event(
            _ev(
                engine_action="enter_long",
                action_side="long",
                position_before=0,
                position_after=0,
                blocked=True,
            )
        )
        self.assertIsNone(d)

    def test_reentry_from_flat_is_a_new_entry(self) -> None:
        # Sequence: enter -> exit -> (later) enter again.  The second enter
        # is a brand new decision.
        d1 = classify_action_event(
            _ev(engine_action="enter_long", action_side="long", position_before=0, position_after=1, seq=1)
        )
        d2 = classify_action_event(
            _ev(engine_action="exit_long", action_side="flat", position_before=1, position_after=0, seq=2)
        )
        d3 = classify_action_event(
            _ev(engine_action="enter_long", action_side="long", position_before=0, position_after=1, seq=3)
        )
        self.assertEqual([d1 is None, d2 is None, d3 is None], [False, True, False])
        assert d1 is not None and d3 is not None
        self.assertEqual(d1.decision_kind, DECISION_ENTRY)
        self.assertEqual(d3.decision_kind, DECISION_ENTRY)

    def test_unknown_action_is_ignored(self) -> None:
        d = classify_action_event(_ev(engine_action="manual_flatten_long"))
        self.assertIsNone(d)

    def test_enter_uppercase_action_is_normalized(self) -> None:
        # Defensive: action names should be matched case-insensitively.
        d = classify_action_event(
            _ev(engine_action="ENTER_LONG", action_side="long", position_before=0, position_after=1)
        )
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.decision_kind, DECISION_ENTRY)

    def test_direction_inferred_when_side_missing(self) -> None:
        # If action_side missing, derive from engine_action / position_after.
        d = classify_action_event(
            _ev(
                engine_action="enter_long",
                action_side=None,
                position_before=0,
                position_after=1,
            )
        )
        self.assertIsNotNone(d)
        assert d is not None
        self.assertEqual(d.direction, "long")

    def test_decision_id_is_deterministic(self) -> None:
        ev = _ev(
            engine_action="enter_long",
            action_side="long",
            position_before=0,
            position_after=1,
            event_id="act-deterministic-1",
            seq=42,
            ts="2026-05-18T12:00:00Z",
        )
        d1 = classify_action_event(ev)
        d2 = classify_action_event(ev)
        assert d1 is not None and d2 is not None
        self.assertEqual(d1.decision_id, d2.decision_id)
        # Also matches the public helper.
        self.assertEqual(
            d1.decision_id,
            deterministic_decision_id(
                venue=ev["venue"],
                symbol=ev["symbol"],
                source_action_event_id=ev["event_id"],
                ts=ev["ts"],
                seq=ev["seq"],
                engine_action=ev["engine_action"],
            ),
        )


class BuilderFromActionEventsTests(unittest.TestCase):
    def test_typical_lifecycle_counts_each_decision_once(self) -> None:
        # Lifecycle: enter long -> scale long -> partial close -> flip short
        #            -> exit short -> enter long
        # Expected decisions:
        #   1) entry long (the initial enter)
        #   2) flip short (the flip is a brand new opposite leg)
        #   3) entry long (re-entry after full exit)
        events = [
            _ev(engine_action="enter_long", action_side="long", position_before=0, position_after=1, seq=1),
            _ev(engine_action="scale_long", action_side="long", position_before=1, position_after=1, seq=2),
            _ev(engine_action="tp1_partial", action_side="long", position_before=1, position_after=1, seq=3),
            _ev(engine_action="flip_to_short", action_side="short", position_before=1, position_after=-1, seq=4),
            _ev(engine_action="exit_short", action_side="flat", position_before=-1, position_after=0, seq=5),
            _ev(engine_action="enter_long", action_side="long", position_before=0, position_after=1, seq=6),
        ]
        decisions = build_trade_decisions_from_action_events(events)
        self.assertEqual(len(decisions), 3)
        kinds = [d.decision_kind for d in decisions]
        dirs = [d.direction for d in decisions]
        self.assertEqual(kinds, [DECISION_ENTRY, DECISION_FLIP, DECISION_ENTRY])
        self.assertEqual(dirs, ["long", "short", "long"])

    def test_builder_is_idempotent_on_repeated_event_ids(self) -> None:
        # Running over the same events twice must yield identical decision_ids.
        events = [
            _ev(engine_action="enter_long", action_side="long",
                position_before=0, position_after=1, seq=1, event_id="a"),
            _ev(engine_action="flip_to_short", action_side="short",
                position_before=1, position_after=-1, seq=2, event_id="b"),
        ]
        first = build_trade_decisions_from_action_events(events)
        second = build_trade_decisions_from_action_events(events + events)
        # Even though we pass duplicates, the builder should dedupe by decision_id.
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 2)
        self.assertEqual([d.decision_id for d in first], [d.decision_id for d in second])

    def test_blocked_events_are_excluded(self) -> None:
        events = [
            _ev(engine_action="enter_long", action_side="long",
                position_before=0, position_after=1, seq=1, blocked=True),
            _ev(engine_action="enter_short", action_side="short",
                position_before=0, position_after=-1, seq=2, blocked=False),
        ]
        decisions = build_trade_decisions_from_action_events(events)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].direction, "short")

    def test_decisions_are_sorted_by_ts_then_seq(self) -> None:
        events = [
            _ev(engine_action="enter_long", action_side="long",
                position_before=0, position_after=1,
                seq=2, ts="2026-05-18T12:00:01Z", event_id="b"),
            _ev(engine_action="enter_short", action_side="short",
                position_before=0, position_after=-1,
                seq=1, ts="2026-05-18T12:00:00Z", event_id="a"),
        ]
        decisions = build_trade_decisions_from_action_events(events)
        self.assertEqual([d.direction for d in decisions], ["short", "long"])


class LiveExecutorIntegrationTests(unittest.TestCase):
    """Verify that live_executor._append_action_event upserts trade decisions
    for counted actions and skips ignored ones, via a stubbed event store."""

    def test_live_executor_emits_decisions_for_entry_and_flip_only(self) -> None:
        from unittest.mock import patch

        import quant.execution.live_executor as live_executor

        captured: list[Dict[str, Any]] = []

        def _capture(row: Dict[str, Any]) -> None:
            captured.append(dict(row))

        scenarios = [
            # (engine_action, action_side, position_before, position_after, expected_counted)
            ("enter_long", "long", 0, 1, True),
            ("scale_long", "long", 1, 1, False),
            ("tp1_partial", "long", 1, 1, False),
            ("flip_to_short", "short", 1, -1, True),
            ("exit_short", "flat", -1, 0, False),
            ("enter_long", "long", 0, 1, True),  # re-entry after flat
            ("hold", "long", 1, 1, False),
        ]

        with patch.object(live_executor, "insert_action_event", lambda *_a, **_k: None), \
             patch.object(live_executor, "append_event_jsonl", lambda *_a, **_k: None), \
             patch.object(live_executor, "upsert_trade_decision", side_effect=_capture):
            for i, (action, side, pb, pa, _expected) in enumerate(scenarios):
                live_executor._append_action_event(
                    strategy="live_executor",
                    symbol="SOL-USDT",
                    ts_iso=f"2026-05-18T12:00:{i:02d}Z",
                    seq=i + 1,
                    engine_action=action,
                    action_side=side,
                    reason_code="test",
                    position_before=pb,
                    position_after=pa,
                    engine_mode_before="TTP",
                    engine_mode_after="TTP",
                    blocked=False,
                )

        expected_counts = sum(1 for *_, c in scenarios if c)
        self.assertEqual(len(captured), expected_counts)
        # Verify each captured row has decision_kind in {entry, flip}.
        for row in captured:
            self.assertIn(row.get("decision_kind"), ("entry", "flip"))
            self.assertIn(row.get("direction"), ("long", "short"))
        # First and last counted decisions are entries; the middle one is a flip.
        kinds = [r["decision_kind"] for r in captured]
        self.assertEqual(kinds, ["entry", "flip", "entry"])
        dirs = [r["direction"] for r in captured]
        self.assertEqual(dirs, ["long", "short", "long"])


class TradeCountApiTests(unittest.TestCase):
    """Cover the /api/dashboard/trade_count endpoint and the trade_decision_count
    field added to /api/dashboard/performance, with the Postgres-backed store
    fully mocked out."""

    def test_trade_count_endpoint_returns_seeded_counts(self) -> None:
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        seeded = [
            TradeDecision(
                decision_id="td_1",
                ts="2026-05-18T12:00:00Z",
                venue="kucoin",
                symbol="SOL-USDT",
                strategy="live_executor",
                strategy_instance="live_executor",
                decision_kind=DECISION_ENTRY,
                direction="long",
                position_before=0,
                position_after=1,
                engine_action="enter_long",
                reason_code="entry",
                source_action_event_id="evt-1",
                seq=1,
                payload={},
            ).to_db_row(),
            TradeDecision(
                decision_id="td_2",
                ts="2026-05-18T12:01:00Z",
                venue="kucoin",
                symbol="SOL-USDT",
                strategy="live_executor",
                strategy_instance="live_executor",
                decision_kind=DECISION_FLIP,
                direction="short",
                position_before=1,
                position_after=-1,
                engine_action="flip_to_short",
                reason_code="flip",
                source_action_event_id="evt-2",
                seq=2,
                payload={},
            ).to_db_row(),
        ]

        def _count(*, venue=None, symbol=None, decision_kind=None, since_ts=None):
            rows = seeded
            if decision_kind:
                rows = [r for r in rows if r["decision_kind"] == decision_kind]
            return len(rows)

        def _list(*, venue=None, symbol=None, limit=50):
            return seeded[: int(limit)]

        ws._TRADE_COUNT_CACHE.clear()
        with patch.object(ws, "count_trade_decisions", side_effect=_count), \
             patch.object(ws, "list_recent_trade_decisions", side_effect=_list), \
             patch.object(ws, "backfill_trade_decisions_from_action_events",
                           return_value={"read_events": 2, "decisions": 2, "written": 2}):
            res = ws.api_dashboard_trade_count(
                symbol="SOL-USDT", venue="kucoin", recent_limit=10, backfill=0
            )

        self.assertTrue(res["ok"])
        self.assertEqual(res["total"], 2)
        self.assertEqual(res["entries"], 1)
        self.assertEqual(res["flips"], 1)
        self.assertEqual(len(res["recent"]), 2)

    def _fake_decision_payload(
        self,
        *,
        symbol: str = "SOL-USDT",
        venue: str = "kucoin",
        trade_count: int = 77,
        winning: int = 27,
        losing: int = 50,
        open_count: int = 0,
        pnl_pct: float = -27.9,
        winrate: float = 35.06,
        monthly_growth: float = -5.0,
        average_gain: float = -0.4,
        cum_pct: float = -27.9,
        needs_backfill: bool = False,
    ) -> Dict[str, Any]:
        closed = max(0, trade_count - open_count)
        return {
            "curve": {
                "points": [],
                "source": "postgres:trade_decisions+closed_trades",
                "needs_backfill": needs_backfill,
            },
            "performance": {
                "symbol": symbol,
                "venue": venue,
                "as_of": "2026-05-18T12:00:00Z",
                "window": "lifetime",
                "pnl_pct": pnl_pct,
                "winrate": winrate,
                "monthly_growth": monthly_growth,
                "average_gain": average_gain,
                "trade_count": trade_count,
                "closed_decision_count": closed,
                "winning_trade_count": winning,
                "losing_trade_count": losing,
                "open_decision_count": open_count,
                "cum_pct": cum_pct,
                "source": "postgres:trade_decisions+closed_trades",
            },
            "needs_backfill": needs_backfill,
        }

    def test_performance_endpoint_uses_closed_trades_for_main_metrics(self) -> None:
        """Main Performance-card metrics are closed_trades aggregates."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()
        closed_trade_perf = {
            "symbol": "SOL-USDT",
            "venue": "kucoin",
            "as_of": "2026-05-18T12:00:00Z",
            "window": "lifetime",
            "pnl_pct": -27.9,
            "winrate": 35.06,
            "monthly_growth": -5.0,
            "average_gain": -0.4,
            "trade_count": 77,
            "winning_trade_count": 27,
            "losing_trade_count": 50,
            "source": "postgres:closed_trades",
        }
        sparse_decision_payload = self._fake_decision_payload(
            trade_count=2,
            winning=1,
            losing=1,
            pnl_pct=1.0,
            winrate=50.0,
            monthly_growth=1.0,
            average_gain=0.5,
            cum_pct=1.0,
        )

        with patch.object(ws, "build_dashboard_performance", return_value=closed_trade_perf) as perf_mock, \
             patch.object(ws, "build_decision_dashboard_payload", return_value=sparse_decision_payload, create=True) as decision_mock, \
             patch.object(ws, "count_trade_decisions", return_value=2) as count_mock, \
             patch.object(ws, "_schedule_auto_backfill_trade_decisions", return_value={"scheduled": True}) as schedule_mock, \
             patch.object(ws, "_maybe_auto_backfill_trade_decisions", return_value=None) as backfill_mock:
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kucoin")

        self.assertTrue(res["ok"])
        self.assertEqual(res["trade_count"], 77)
        self.assertEqual(res["winning_trade_count"], 27)
        self.assertEqual(res["losing_trade_count"], 50)
        self.assertEqual(res["trade_decision_count"], 2)
        self.assertNotIn("open_decision_count", res)
        self.assertNotIn("closed_decision_count", res)
        self.assertNotIn("needs_backfill", res)
        self.assertNotIn("synthesized_count", res)
        perf_mock.assert_called_once()
        self.assertEqual(perf_mock.call_args.kwargs.get("venue"), "kucoin")
        decision_mock.assert_not_called()
        count_mock.assert_called_once()
        schedule_mock.assert_not_called()
        backfill_mock.assert_not_called()

    def test_performance_endpoint_includes_trade_decision_count(self) -> None:
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()
        closed_trade_perf = {
            "symbol": "SOL-USDT",
            "venue": "kucoin",
            "as_of": "2026-05-18T12:00:00Z",
            "window": "lifetime",
            "pnl_pct": 1.5,
            "winrate": 60.0,
            "monthly_growth": 3.0,
            "average_gain": 0.25,
            "trade_count": 10,
            "winning_trade_count": 6,
            "losing_trade_count": 4,
            "source": "postgres:closed_trades",
        }
        with patch.object(ws, "build_dashboard_performance", return_value=closed_trade_perf), \
             patch.object(ws, "build_decision_dashboard_payload", create=True) as decision_mock, \
             patch.object(ws, "count_trade_decisions", return_value=10):
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kucoin")

        self.assertTrue(res["ok"])
        self.assertEqual(res["trade_count"], 10)
        self.assertEqual(res["trade_decision_count"], 10)
        decision_mock.assert_not_called()

    def test_performance_endpoint_defaults_to_kucoin(self) -> None:
        """Calling /api/dashboard/performance without a venue uses KuCoin."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()

        seen_perf_kwargs: Dict[str, Any] = {}
        seen_count_kwargs: Dict[str, Any] = {}

        def _fake_perf(*args, **kwargs):
            seen_perf_kwargs.update(kwargs)
            return {
                "symbol": kwargs.get("symbol", "SOL-USDT"),
                "venue": kwargs.get("venue", "kucoin"),
                "as_of": "2026-05-18T12:00:00Z",
                "window": "lifetime",
                "pnl_pct": None,
                "winrate": None,
                "monthly_growth": None,
                "average_gain": None,
                "trade_count": 0,
                "winning_trade_count": 0,
                "losing_trade_count": 0,
                "source": "postgres:closed_trades",
            }

        def _fake_count(*, venue=None, symbol=None, decision_kind=None, since_ts=None):
            seen_count_kwargs.update(
                {"venue": venue, "symbol": symbol, "decision_kind": decision_kind}
            )
            return 0

        with patch.object(ws, "build_dashboard_performance", side_effect=_fake_perf), \
             patch.object(ws, "build_decision_dashboard_payload", create=True) as decision_mock, \
             patch.object(ws, "count_trade_decisions", side_effect=_fake_count):
            res = ws.api_dashboard_performance(symbol="SOL-USDT")

        self.assertTrue(res["ok"])
        self.assertEqual(res["venue"], "kucoin")
        self.assertEqual(seen_perf_kwargs.get("venue"), "kucoin")
        self.assertEqual(seen_count_kwargs.get("venue"), "kucoin")
        # No Kraken fallback when KuCoin numbers are missing — empty/zero is
        # the correct behaviour, never wrong-venue numbers.
        self.assertEqual(res["trade_count"], 0)
        self.assertEqual(res["trade_decision_count"], 0)
        decision_mock.assert_not_called()

    def test_performance_endpoint_respects_explicit_venue(self) -> None:
        """An explicit ``venue=kraken`` must be passed straight through."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()

        seen_perf_kwargs: Dict[str, Any] = {}
        seen_count_kwargs: Dict[str, Any] = {}

        def _fake_perf(*args, **kwargs):
            seen_perf_kwargs.update(kwargs)
            if str(kwargs.get("venue")).lower() == "kucoin":
                raise AssertionError("Unexpected fallback to kucoin")
            return {
                "symbol": kwargs.get("symbol", "SOL-USDT"),
                "venue": kwargs.get("venue", "kraken"),
                "as_of": "2026-05-18T12:00:00Z",
                "window": "lifetime",
                "pnl_pct": None,
                "winrate": None,
                "monthly_growth": None,
                "average_gain": None,
                "trade_count": 0,
                "winning_trade_count": 0,
                "losing_trade_count": 0,
                "source": "unsupported_venue",
            }

        def _fake_count(*, venue=None, symbol=None, decision_kind=None, since_ts=None):
            seen_count_kwargs.update({"venue": venue, "symbol": symbol})
            return 0

        with patch.object(ws, "build_dashboard_performance", side_effect=_fake_perf), \
             patch.object(ws, "build_decision_dashboard_payload", create=True) as decision_mock, \
             patch.object(ws, "count_trade_decisions", side_effect=_fake_count):
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kraken")

        self.assertTrue(res["ok"])
        self.assertEqual(res["venue"], "kraken")
        self.assertEqual(seen_perf_kwargs.get("venue"), "kraken")
        self.assertEqual(seen_count_kwargs.get("venue"), "kraken")
        self.assertEqual(res["trade_count"], 0)
        self.assertEqual(res["winning_trade_count"], 0)
        self.assertEqual(res["losing_trade_count"], 0)
        decision_mock.assert_not_called()

    def test_performance_endpoint_does_not_backfill_decisions(self) -> None:
        """Performance reads must not run trade_decisions backfill."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()
        closed_trade_perf = {
            "symbol": "SOL-USDT",
            "venue": "kucoin",
            "as_of": "2026-05-18T12:00:00Z",
            "window": "lifetime",
            "pnl_pct": 1.5,
            "winrate": 60.0,
            "monthly_growth": 3.0,
            "average_gain": 0.25,
            "trade_count": 10,
            "winning_trade_count": 6,
            "losing_trade_count": 4,
            "source": "postgres:closed_trades",
        }

        with patch.object(ws, "build_dashboard_performance", return_value=closed_trade_perf), \
             patch.object(ws, "build_decision_dashboard_payload", create=True) as decision_mock, \
             patch.object(ws, "count_trade_decisions", return_value=2), \
             patch.object(ws, "_maybe_auto_backfill_trade_decisions", return_value=None) as helper_mock, \
             patch.object(ws, "_schedule_auto_backfill_trade_decisions", return_value={"scheduled": True}) as schedule_mock, \
             patch.object(ws, "backfill_trade_decisions_from_action_events",
                           return_value={"read_events": 5, "decisions": 2, "written": 2}) as ae_mock, \
             patch.object(ws, "backfill_trade_decisions_from_closed_trades",
                           return_value={"read_rows": 0, "written": 0}) as ct_mock:
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kucoin", backfill=1)

        self.assertTrue(res["ok"])
        self.assertEqual(res["trade_count"], 10)
        self.assertEqual(res["trade_decision_count"], 2)
        decision_mock.assert_not_called()
        helper_mock.assert_not_called()
        schedule_mock.assert_not_called()
        ae_mock.assert_not_called()
        ct_mock.assert_not_called()

    def test_trade_count_endpoint_defaults_to_kucoin(self) -> None:
        """``/api/dashboard/trade_count`` must filter by ``kucoin`` when the
        caller omits the ``venue`` query param."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        seen_count_kwargs: Dict[str, Any] = {}
        seen_list_kwargs: Dict[str, Any] = {}

        def _fake_count(*, venue=None, symbol=None, decision_kind=None, since_ts=None):
            seen_count_kwargs.setdefault("calls", []).append(
                {"venue": venue, "symbol": symbol, "decision_kind": decision_kind}
            )
            return 0

        def _fake_list(*, venue=None, symbol=None, limit=50):
            seen_list_kwargs.update({"venue": venue, "symbol": symbol, "limit": limit})
            return []

        ws._TRADE_COUNT_CACHE.clear()
        with patch.object(ws, "count_trade_decisions", side_effect=_fake_count), \
             patch.object(ws, "list_recent_trade_decisions", side_effect=_fake_list):
            res = ws.api_dashboard_trade_count(symbol="SOL-USDT")

        self.assertTrue(res["ok"])
        self.assertEqual(res["venue"], "kucoin")
        self.assertEqual(res["total"], 0)
        self.assertEqual(res["entries"], 0)
        self.assertEqual(res["flips"], 0)
        self.assertEqual(seen_list_kwargs.get("venue"), "kucoin")
        for call in seen_count_kwargs.get("calls", []):
            self.assertEqual(call["venue"], "kucoin")

    def test_trade_count_endpoint_respects_explicit_venue(self) -> None:
        """Explicit ``venue=kraken`` filters must be honoured rather than
        silently rewritten to ``kucoin``."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        seen_venues: List[str] = []

        def _fake_count(*, venue=None, symbol=None, decision_kind=None, since_ts=None):
            seen_venues.append(str(venue))
            return 0

        def _fake_list(*, venue=None, symbol=None, limit=50):
            seen_venues.append(str(venue))
            return []

        ws._TRADE_COUNT_CACHE.clear()
        with patch.object(ws, "count_trade_decisions", side_effect=_fake_count), \
             patch.object(ws, "list_recent_trade_decisions", side_effect=_fake_list):
            res = ws.api_dashboard_trade_count(symbol="SOL-USDT", venue="kraken")

        self.assertTrue(res["ok"])
        self.assertEqual(res["venue"], "kraken")
        self.assertTrue(seen_venues, "venue must be threaded through to the store")
        for v in seen_venues:
            self.assertEqual(v, "kraken")


class BuildFromClosedTradesTests(unittest.TestCase):
    """Cover the historical-tail backfill that derives ``trade_decisions``
    directly from ``closed_trades``. This is the only path that recovers
    legs that pre-date the ``action_events`` table — the regression behind
    the "Performance card shows 2 trades but I have 77 closed legs" bug.
    """

    def _ct(
        self,
        *,
        side: str,
        entry_ts: str,
        exit_ts: str,
        trade_id: str = "t",
        venue: str = "kucoin",
        symbol: str = "SOL-USDT",
        payload_json: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "trade_id": trade_id,
            "venue": venue,
            "symbol": symbol,
            "side": side,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "strategy": "live_executor",
            "strategy_instance": "live_executor",
            "payload_json": payload_json or {},
        }

    def test_three_sequential_same_direction_closes_yield_three_decisions(self) -> None:
        # closed_trades semantics: each row is a closed leg. Three back-to-back
        # long legs are three separate decisions (the trade was re-entered each
        # time after the previous TP / SL closed it). They are NOT collapsed
        # into "one big long position".
        from quant.execution.trade_decisions_store import (
            build_trade_decisions_from_closed_trades,
        )

        rows = [
            self._ct(
                side="long",
                entry_ts="2025-03-01T10:00:00Z",
                exit_ts="2025-03-01T11:00:00Z",
                trade_id="t1",
            ),
            self._ct(
                side="long",
                entry_ts="2025-03-01T12:00:00Z",
                exit_ts="2025-03-01T13:00:00Z",
                trade_id="t2",
            ),
            self._ct(
                side="long",
                entry_ts="2025-03-01T14:00:00Z",
                exit_ts="2025-03-01T15:00:00Z",
                trade_id="t3",
            ),
        ]
        decisions, stats = build_trade_decisions_from_closed_trades(rows)
        self.assertEqual(len(decisions), 3)
        self.assertEqual([d.decision_kind for d in decisions], ["entry", "entry", "entry"])
        self.assertEqual([d.direction for d in decisions], ["long", "long", "long"])
        self.assertEqual(stats["entries"], 3)
        self.assertEqual(stats["flips"], 0)

    def test_flip_pattern_yields_entry_then_flip(self) -> None:
        from quant.execution.trade_decisions_store import (
            build_trade_decisions_from_closed_trades,
        )

        rows = [
            self._ct(
                side="long",
                entry_ts="2025-03-01T10:00:00Z",
                exit_ts="2025-03-01T12:00:00Z",
                trade_id="t1",
            ),
            # Second leg opens at the same instant the first closes — that's
            # the flip pattern the live executor would have emitted.
            self._ct(
                side="short",
                entry_ts="2025-03-01T12:00:00Z",
                exit_ts="2025-03-01T13:30:00Z",
                trade_id="t2",
            ),
        ]
        decisions, stats = build_trade_decisions_from_closed_trades(rows)
        self.assertEqual(len(decisions), 2)
        self.assertEqual([d.decision_kind for d in decisions], ["entry", "flip"])
        self.assertEqual([d.direction for d in decisions], ["long", "short"])
        # The flip's engine_action must NOT be ``enter_short`` — that would
        # break the contract the Performance card builds on top of.
        self.assertEqual(decisions[1].engine_action, "flip_to_short")
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["flips"], 1)

    def test_back_to_back_opposite_with_long_gap_is_not_a_flip(self) -> None:
        # If the gap between the previous exit and the new entry exceeds the
        # tolerance (~60s), classify as a fresh entry, not a flip.
        from quant.execution.trade_decisions_store import (
            build_trade_decisions_from_closed_trades,
        )

        rows = [
            self._ct(
                side="long",
                entry_ts="2025-03-01T10:00:00Z",
                exit_ts="2025-03-01T11:00:00Z",
                trade_id="t1",
            ),
            self._ct(
                side="short",
                entry_ts="2025-03-01T15:00:00Z",  # 4h later -> independent re-entry
                exit_ts="2025-03-01T16:00:00Z",
                trade_id="t2",
            ),
        ]
        decisions, stats = build_trade_decisions_from_closed_trades(rows)
        self.assertEqual([d.decision_kind for d in decisions], ["entry", "entry"])
        self.assertEqual(stats["flips"], 0)

    def test_skips_rows_with_pre_2000_entry_ts(self) -> None:
        # The old NaT->0 fills reconstructor left behind 1970-style sentinel
        # entry_ts values. The backfill must skip those rather than emit
        # 1970-dated decisions that pollute the chart.
        from quant.execution.trade_decisions_store import (
            build_trade_decisions_from_closed_trades,
        )

        rows = [
            self._ct(
                side="long",
                entry_ts="1970-01-01T00:00:01Z",
                exit_ts="2025-03-01T11:00:00Z",
                trade_id="bogus",
            ),
            self._ct(
                side="long",
                entry_ts="2025-03-01T12:00:00Z",
                exit_ts="2025-03-01T13:00:00Z",
                trade_id="good",
            ),
        ]
        decisions, stats = build_trade_decisions_from_closed_trades(rows)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].payload.get("closed_trade_id"), "good")
        self.assertEqual(stats["skipped_invalid_ts"], 1)

    def test_falls_back_to_payload_opened_at_when_entry_ts_is_sentinel(self) -> None:
        from quant.execution.trade_decisions_store import (
            build_trade_decisions_from_closed_trades,
        )

        rows = [
            self._ct(
                side="long",
                entry_ts="1970-01-01T00:00:00Z",
                exit_ts="2025-03-01T11:00:00Z",
                trade_id="t1",
                payload_json={"opened_at": "2025-03-01T10:30:00Z"},
            ),
        ]
        decisions, stats = build_trade_decisions_from_closed_trades(rows)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].ts, "2025-03-01T10:30:00Z")
        self.assertEqual(stats["skipped_invalid_ts"], 0)

    def test_decision_ids_are_idempotent_under_repeated_input(self) -> None:
        # Re-running the backfill over the same closed_trades rows must yield
        # the same ids — that's what makes the upsert a no-op on conflict and
        # safe to auto-trigger on every dashboard refresh.
        from quant.execution.trade_decisions_store import (
            build_trade_decisions_from_closed_trades,
        )

        rows = [
            self._ct(
                side="long",
                entry_ts="2025-03-01T10:00:00Z",
                exit_ts="2025-03-01T11:00:00Z",
                trade_id="t1",
            ),
            self._ct(
                side="short",
                entry_ts="2025-03-01T12:00:00Z",
                exit_ts="2025-03-01T13:00:00Z",
                trade_id="t2",
            ),
        ]
        first, _ = build_trade_decisions_from_closed_trades(rows)
        second, _ = build_trade_decisions_from_closed_trades(rows)
        self.assertEqual([d.decision_id for d in first], [d.decision_id for d in second])
        # ids should be namespaced so they cannot collide with the action-event
        # backfill, which emits ``td_<hash>``.
        for d in first:
            self.assertTrue(d.decision_id.startswith("td_ct_"))

    def test_rows_with_invalid_side_are_skipped(self) -> None:
        from quant.execution.trade_decisions_store import (
            build_trade_decisions_from_closed_trades,
        )

        rows = [
            self._ct(
                side="",  # invalid
                entry_ts="2025-03-01T10:00:00Z",
                exit_ts="2025-03-01T11:00:00Z",
                trade_id="t-bad",
            ),
            self._ct(
                side="long",
                entry_ts="2025-03-01T12:00:00Z",
                exit_ts="2025-03-01T13:00:00Z",
                trade_id="t-good",
            ),
        ]
        decisions, stats = build_trade_decisions_from_closed_trades(rows)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(stats["skipped_bad_side"], 1)


class AutoBackfillTriggerTests(unittest.TestCase):
    """Cover the auto-backfill helper that the dashboard endpoints invoke.

    The user-visible regression that motivated this code path: after
    switching the equity chart to the decision-based source, the chart
    and Performance card collapsed to 2 trades because ``trade_decisions``
    only had 2 rows. The auto-backfill makes the chart self-heal on the
    next request when the spine looks suspiciously thin vs.
    ``closed_trades``.
    """

    def setUp(self) -> None:
        import quant.execution.webhook_server as ws

        ws._BACKFILL_LAST_RUN.clear()
        ws._BACKFILL_IN_FLIGHT.clear()
        ws._PERFORMANCE_CACHE.clear()
        ws._TRADE_COUNT_CACHE.clear()
        ws._CHART_CACHE.clear()

    def test_helper_fires_when_decision_spine_is_thin(self) -> None:
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        with patch.object(ws, "count_trade_decisions", return_value=2), \
             patch.object(ws, "count_closed_trades", return_value=80), \
             patch.object(ws, "latest_decision_ts", return_value=None), \
             patch.object(ws, "latest_closed_trade_ts", return_value=None), \
             patch.object(ws, "backfill_trade_decisions_from_action_events",
                           return_value={"written": 0}) as ae_mock, \
             patch.object(ws, "backfill_trade_decisions_from_closed_trades",
                           return_value={"written": 78}) as ct_mock:
            out = ws._maybe_auto_backfill_trade_decisions(
                venue="kucoin", symbol="SOL-USDT"
            )

        self.assertIsNotNone(out)
        assert out is not None
        ae_mock.assert_called_once()
        ct_mock.assert_called_once()
        self.assertEqual(out["diagnosis"]["reason"], "ratio_below_threshold")

    def test_helper_skips_when_spine_is_already_complete(self) -> None:
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        with patch.object(ws, "count_trade_decisions", return_value=80), \
             patch.object(ws, "count_closed_trades", return_value=80), \
             patch.object(ws, "latest_decision_ts",
                           return_value="2026-05-19T12:00:00Z"), \
             patch.object(ws, "latest_closed_trade_ts",
                           return_value="2026-05-19T12:00:00Z"), \
             patch.object(ws, "backfill_trade_decisions_from_action_events") as ae_mock, \
             patch.object(ws, "backfill_trade_decisions_from_closed_trades") as ct_mock:
            out = ws._maybe_auto_backfill_trade_decisions(
                venue="kucoin", symbol="SOL-USDT"
            )

        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out.get("skipped"), "not_needed")
        ae_mock.assert_not_called()
        ct_mock.assert_not_called()

    def test_helper_fires_when_spine_is_stale(self) -> None:
        # Spine has plenty of rows but the newest decision is days older
        # than the newest closed_trade — newly closed legs need to be
        # rolled into the spine.
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        with patch.object(ws, "count_trade_decisions", return_value=80), \
             patch.object(ws, "count_closed_trades", return_value=80), \
             patch.object(ws, "latest_decision_ts",
                           return_value="2026-05-10T12:00:00Z"), \
             patch.object(ws, "latest_closed_trade_ts",
                           return_value="2026-05-19T12:00:00Z"), \
             patch.object(ws, "backfill_trade_decisions_from_action_events",
                           return_value={"written": 1}) as ae_mock, \
             patch.object(ws, "backfill_trade_decisions_from_closed_trades",
                           return_value={"written": 1}) as ct_mock:
            out = ws._maybe_auto_backfill_trade_decisions(
                venue="kucoin", symbol="SOL-USDT"
            )

        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["diagnosis"]["reason"], "decision_spine_stale")
        ae_mock.assert_called_once()
        ct_mock.assert_called_once()

    def test_throttle_prevents_double_run_within_window(self) -> None:
        # Two rapid back-to-back calls must result in exactly ONE invocation
        # of each backfill — the throttle is the only thing protecting
        # Postgres from chart polling.
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        with patch.object(ws, "count_trade_decisions", return_value=2), \
             patch.object(ws, "count_closed_trades", return_value=80), \
             patch.object(ws, "latest_decision_ts", return_value=None), \
             patch.object(ws, "latest_closed_trade_ts", return_value=None), \
             patch.object(ws, "backfill_trade_decisions_from_action_events",
                           return_value={"written": 0}) as ae_mock, \
             patch.object(ws, "backfill_trade_decisions_from_closed_trades",
                           return_value={"written": 78}) as ct_mock:
            first = ws._maybe_auto_backfill_trade_decisions(
                venue="kucoin", symbol="SOL-USDT"
            )
            second = ws._maybe_auto_backfill_trade_decisions(
                venue="kucoin", symbol="SOL-USDT"
            )

        self.assertEqual(ae_mock.call_count, 1)
        self.assertEqual(ct_mock.call_count, 1)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.get("skipped"), "throttled")

    def test_force_flag_bypasses_throttle_and_needed_check(self) -> None:
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        # Pretend the spine is up to date — would normally skip.
        with patch.object(ws, "count_trade_decisions", return_value=80), \
             patch.object(ws, "count_closed_trades", return_value=80), \
             patch.object(ws, "latest_decision_ts",
                           return_value="2026-05-19T12:00:00Z"), \
             patch.object(ws, "latest_closed_trade_ts",
                           return_value="2026-05-19T12:00:00Z"), \
             patch.object(ws, "backfill_trade_decisions_from_action_events",
                           return_value={"written": 0}) as ae_mock, \
             patch.object(ws, "backfill_trade_decisions_from_closed_trades",
                           return_value={"written": 0}) as ct_mock:
            # Pre-populate the throttle dict so a non-forced call would skip.
            ws._BACKFILL_LAST_RUN["kucoin:SOL-USDT"] = time.time()
            out = ws._maybe_auto_backfill_trade_decisions(
                venue="kucoin", symbol="SOL-USDT", force=True
            )

        ae_mock.assert_called_once()
        ct_mock.assert_called_once()
        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(out.get("forced"))

    def test_background_schedule_does_not_block_caller(self) -> None:
        import time
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        def _slow_execute(venue: str, symbol: str) -> Dict[str, Any]:
            time.sleep(2.0)
            return {"venue": venue, "symbol": symbol, "ts": "now"}

        with patch.object(ws, "count_trade_decisions", return_value=2), \
             patch.object(ws, "count_closed_trades", return_value=80), \
             patch.object(ws, "latest_decision_ts", return_value=None), \
             patch.object(ws, "latest_closed_trade_ts", return_value=None), \
             patch.object(ws, "_execute_trade_decision_backfill", side_effect=_slow_execute):
            t0 = time.perf_counter()
            out = ws._schedule_auto_backfill_trade_decisions(
                venue="kucoin", symbol="SOL-USDT"
            )
            elapsed = time.perf_counter() - t0
            ws._wait_trade_decision_backfill_idle(timeout=5.0)

        self.assertIsNotNone(out)
        assert out is not None
        self.assertTrue(out.get("scheduled"))
        self.assertLess(elapsed, 0.5)

    def test_helper_skips_non_kucoin_venues(self) -> None:
        # Kraken is gated out at the source — we never want synthesized
        # KuCoin decisions polluting a kraken venue spine, so the helper
        # is a no-op for anything that isn't kucoin.
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        with patch.object(ws, "backfill_trade_decisions_from_action_events") as ae_mock, \
             patch.object(ws, "backfill_trade_decisions_from_closed_trades") as ct_mock:
            out = ws._maybe_auto_backfill_trade_decisions(
                venue="kraken", symbol="ETH-USDT"
            )

        self.assertIsNone(out)
        ae_mock.assert_not_called()
        ct_mock.assert_not_called()

    def test_performance_endpoint_does_not_auto_backfill_when_spine_thin(self) -> None:
        # A sparse ``trade_decisions`` table must not affect the main
        # Performance card, which is sourced from closed_trades.
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._BACKFILL_LAST_RUN.clear()
        ws._BACKFILL_IN_FLIGHT.clear()
        ws._PERFORMANCE_CACHE.clear()

        closed_trade_perf = {
            "symbol": "SOL-USDT",
            "venue": "kucoin",
            "as_of": "2026-05-19T12:00:00Z",
            "window": "lifetime",
            "pnl_pct": 78.0,
            "winrate": 100.0,
            "monthly_growth": 0.0,
            "average_gain": 1.0,
            "trade_count": 80,
            "winning_trade_count": 80,
            "losing_trade_count": 0,
            "source": "postgres:closed_trades",
        }

        with patch.object(ws, "build_dashboard_performance", return_value=closed_trade_perf), \
             patch.object(ws, "build_decision_dashboard_payload", create=True) as decision_mock, \
             patch.object(ws, "count_trade_decisions", return_value=2), \
             patch.object(ws, "count_closed_trades", return_value=80), \
             patch.object(ws, "latest_decision_ts", return_value=None), \
             patch.object(ws, "latest_closed_trade_ts", return_value=None), \
             patch.object(ws, "_schedule_auto_backfill_trade_decisions", return_value={"scheduled": True}) as schedule_mock, \
             patch.object(ws, "backfill_trade_decisions_from_action_events",
                          return_value={"read_events": 5, "decisions": 2, "written": 2}) as ae_mock, \
             patch.object(ws, "backfill_trade_decisions_from_closed_trades",
                          return_value={"read_rows": 80, "decisions": 78, "written": 78}) as ct_mock:
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kucoin")

        self.assertTrue(res["ok"])
        self.assertEqual(res["trade_count"], 80)
        self.assertEqual(res["trade_decision_count"], 2)
        decision_mock.assert_not_called()
        schedule_mock.assert_not_called()
        ae_mock.assert_not_called()
        ct_mock.assert_not_called()


class TradeCountInvariantsTests(unittest.TestCase):
    """Pin down the cross-field invariants the Performance card relies on.

    These guard the contract the dashboard's right-sidebar reads:
    ``trade_count`` is the closed-trade aggregate count, while
    ``trade_decision_count`` is a separate diagnostic counter.
    """

    def test_endpoint_trade_count_equals_closed_trade_outcomes(self) -> None:
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()
        with patch.object(
            ws,
            "build_dashboard_performance",
            return_value={
                "symbol": "SOL-USDT",
                "venue": "kucoin",
                "as_of": "2026-05-19T12:00:00Z",
                "window": "lifetime",
                "pnl_pct": 1.0,
                "winrate": 50.0,
                "monthly_growth": 1.0,
                "average_gain": 0.25,
                "trade_count": 4,
                "winning_trade_count": 2,
                "losing_trade_count": 1,
                "source": "postgres:closed_trades",
            },
        ), patch.object(ws, "build_decision_dashboard_payload", create=True) as decision_mock, \
             patch.object(ws, "count_trade_decisions", return_value=9):
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kucoin")

        self.assertTrue(res["ok"])
        self.assertEqual(res["trade_count"], 4)
        self.assertEqual(res["winning_trade_count"], 2)
        self.assertEqual(res["losing_trade_count"], 1)
        self.assertLessEqual(
            res["winning_trade_count"] + res["losing_trade_count"],
            res["trade_count"],
        )
        self.assertEqual(res["trade_decision_count"], 9)
        self.assertNotIn("open_decision_count", res)
        self.assertNotIn("cum_pct", res)
        decision_mock.assert_not_called()


class TradeDecisionDataclassTests(unittest.TestCase):
    def test_to_dict_round_trip(self) -> None:
        d = TradeDecision(
            decision_id="abc",
            ts="2026-05-18T12:00:00Z",
            venue="kucoin",
            symbol="SOL-USDT",
            strategy="live_executor",
            strategy_instance="live_executor",
            decision_kind=DECISION_ENTRY,
            direction="long",
            position_before=0,
            position_after=1,
            engine_action="enter_long",
            reason_code="entry",
            source_action_event_id="act-1",
            seq=1,
            payload={"foo": "bar"},
        )
        row = d.to_db_row()
        self.assertEqual(row["decision_id"], "abc")
        self.assertEqual(row["decision_kind"], DECISION_ENTRY)
        self.assertEqual(row["direction"], "long")
        self.assertEqual(row["payload_json"]["foo"], "bar")


if __name__ == "__main__":
    unittest.main()
