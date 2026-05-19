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

    def test_performance_endpoint_keeps_counts_internally_consistent(self) -> None:
        """The Performance card and the equity chart must share one source.

        With the decision-based pipeline, ``trade_count`` is the number of
        decisions (entries + flips, open + closed). ``closed_decision_count``
        equals ``winning + losing`` plus neutral pnl=0 closures. The chart's
        final ``cum_pct`` is what the card's ``pnl_pct`` shows."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()

        seen_payload_kwargs: Dict[str, Any] = {}
        seen_count_kwargs: Dict[str, Any] = {}

        def _fake_payload(*args, **kwargs):
            seen_payload_kwargs.update(kwargs)
            return self._fake_decision_payload()

        def _fake_count(*, venue=None, symbol=None, decision_kind=None, since_ts=None):
            seen_count_kwargs.update({"venue": venue, "symbol": symbol})
            return 77

        with patch.object(ws, "build_decision_dashboard_payload", side_effect=_fake_payload), \
             patch.object(ws, "count_trade_decisions", side_effect=_fake_count):
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kucoin")

        self.assertTrue(res["ok"])
        # ``trade_count`` now counts decisions, not closed_trades aggregates.
        self.assertEqual(res["trade_count"], 77)
        self.assertEqual(res["winning_trade_count"], 27)
        self.assertEqual(res["losing_trade_count"], 50)
        # closed_decision_count >= wins + losses (no neutrals in this fixture)
        self.assertEqual(res["closed_decision_count"], 77)
        self.assertGreaterEqual(
            res["closed_decision_count"],
            res["winning_trade_count"] + res["losing_trade_count"],
        )
        # The card's pnl_pct == the chart's final cum_pct.
        self.assertEqual(res["pnl_pct"], res["cum_pct"])
        # Every count must originate from the kucoin venue end-to-end.
        self.assertEqual(seen_payload_kwargs.get("venue"), "kucoin")
        self.assertEqual(seen_count_kwargs.get("venue"), "kucoin")

    def test_performance_endpoint_includes_trade_decision_count(self) -> None:
        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()
        payload = self._fake_decision_payload(
            trade_count=10, winning=6, losing=4, pnl_pct=1.5, winrate=60.0,
            monthly_growth=3.0, average_gain=0.25, cum_pct=1.5,
        )
        with patch.object(ws, "build_decision_dashboard_payload", return_value=payload), \
             patch.object(ws, "count_trade_decisions", return_value=10):
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kucoin")

        self.assertTrue(res["ok"])
        self.assertEqual(res["trade_count"], 10)
        # Back-compat alias for the old field name; same value as trade_count
        # in the decision world (the legacy frontend reads this).
        self.assertEqual(res["trade_decision_count"], 10)

    def test_performance_endpoint_defaults_to_kucoin(self) -> None:
        """Calling /api/dashboard/performance without a venue must filter by
        ``kucoin`` end-to-end (decision builder and trade-decision count)."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()

        seen_payload_kwargs: Dict[str, Any] = {}
        seen_count_kwargs: Dict[str, Any] = {}

        def _fake_payload(*args, **kwargs):
            seen_payload_kwargs.update(kwargs)
            return self._fake_decision_payload(
                symbol=kwargs.get("symbol", "SOL-USDT"),
                venue=kwargs.get("venue", "kucoin"),
                trade_count=0, winning=0, losing=0,
                pnl_pct=None, winrate=None, monthly_growth=None,  # type: ignore[arg-type]
                average_gain=None, cum_pct=None,  # type: ignore[arg-type]
            )

        def _fake_count(*, venue=None, symbol=None, decision_kind=None, since_ts=None):
            seen_count_kwargs.update(
                {"venue": venue, "symbol": symbol, "decision_kind": decision_kind}
            )
            return 0

        with patch.object(ws, "build_decision_dashboard_payload", side_effect=_fake_payload), \
             patch.object(ws, "count_trade_decisions", side_effect=_fake_count):
            res = ws.api_dashboard_performance(symbol="SOL-USDT")

        self.assertTrue(res["ok"])
        self.assertEqual(res["venue"], "kucoin")
        self.assertEqual(seen_payload_kwargs.get("venue"), "kucoin")
        self.assertEqual(seen_count_kwargs.get("venue"), "kucoin")
        # No Kraken fallback when KuCoin numbers are missing — empty/zero is
        # the correct behaviour, never wrong-venue numbers.
        self.assertEqual(res["trade_count"], 0)
        self.assertEqual(res["trade_decision_count"], 0)

    def test_performance_endpoint_respects_explicit_venue(self) -> None:
        """An explicit ``venue=kraken`` must be passed straight through to the
        loaders. The decision builder short-circuits non-KuCoin venues to an
        empty result, so the endpoint should respond with zeros instead of
        silently returning KuCoin data."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()

        seen_payload_kwargs: Dict[str, Any] = {}
        seen_count_kwargs: Dict[str, Any] = {}

        def _fake_payload(*args, **kwargs):
            seen_payload_kwargs.update(kwargs)
            if str(kwargs.get("venue")).lower() != "kucoin":
                return self._fake_decision_payload(
                    symbol=kwargs.get("symbol", "SOL-USDT"),
                    venue=kwargs.get("venue", "kraken"),
                    trade_count=0, winning=0, losing=0,
                    pnl_pct=None, winrate=None, monthly_growth=None,  # type: ignore[arg-type]
                    average_gain=None, cum_pct=None,  # type: ignore[arg-type]
                )
            raise AssertionError("Unexpected fallback to kucoin")

        def _fake_count(*, venue=None, symbol=None, decision_kind=None, since_ts=None):
            seen_count_kwargs.update({"venue": venue, "symbol": symbol})
            return 0

        with patch.object(ws, "build_decision_dashboard_payload", side_effect=_fake_payload), \
             patch.object(ws, "count_trade_decisions", side_effect=_fake_count):
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kraken")

        self.assertTrue(res["ok"])
        self.assertEqual(res["venue"], "kraken")
        self.assertEqual(seen_payload_kwargs.get("venue"), "kraken")
        self.assertEqual(seen_count_kwargs.get("venue"), "kraken")
        self.assertEqual(res["trade_count"], 0)
        self.assertEqual(res["winning_trade_count"], 0)
        self.assertEqual(res["losing_trade_count"], 0)

    def test_performance_endpoint_runs_backfill_when_requested(self) -> None:
        """``backfill=1`` must call the idempotent action-event backfill before
        rebuilding the decision-based payload — that's how the operator
        refreshes the spine on Railway without redeploying."""

        from unittest.mock import patch

        import quant.execution.webhook_server as ws

        ws._PERFORMANCE_CACHE.clear()
        backfill_seen: Dict[str, Any] = {}

        def _fake_backfill(*, venue=None, symbol=None, since_ts=None):
            backfill_seen.update({"venue": venue, "symbol": symbol})
            return {"read_events": 5, "decisions": 2, "written": 2}

        with patch.object(ws, "build_decision_dashboard_payload",
                          return_value=self._fake_decision_payload()), \
             patch.object(ws, "count_trade_decisions", return_value=77), \
             patch.object(ws, "backfill_trade_decisions_from_action_events",
                           side_effect=_fake_backfill) as bf_mock:
            res = ws.api_dashboard_performance(symbol="SOL-USDT", venue="kucoin", backfill=1)

        self.assertTrue(res["ok"])
        bf_mock.assert_called_once()
        self.assertEqual(backfill_seen.get("venue"), "kucoin")

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
