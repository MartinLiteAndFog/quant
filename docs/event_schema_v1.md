# Event Schema v1

## 1. Goals

- Signals, engine decisions, and executions must be stored separately.
- The schema must support live debugging, post-mortem analysis, attribution, and later PostgreSQL storage.
- Dashboard equity should later be derivable from stored account/equity events, not only from ad-hoc runtime state.
- The first version should stay minimal, explicit, and backward-compatible with current JSONL-based flows.

## 2. Event families
We use three event families:

1. `signal_events`
   - Alpha or strategy statements such as long, short, flat, trend flip, re-arm.
   - Example source: IMBA on Renko.

2. `action_events`
   - Engine decisions derived from signals and state.
   - Example actions: enter, flip, tp1_exit, tp2_exit, ttp_exit, sl_exit, be_exit, regime_exit, hold, ignore, blocked.

3. `execution_events`
   - Venue or OMS-level execution facts.
   - Example: order submitted, order filled, order canceled, position synced, rejection, fallback execution.

## 3. Minimal fields
All event families should aim to include these minimal fields where applicable:

- `event_id`
- `event_family`
- `strategy`
- `symbol`
- `venue`
- `ts`
- `seq`
- `position_before`
- `position_after`
- `engine_mode_before`
- `engine_mode_after`
- `reason_code`
- `source_event_id`
- `source_signal_event_id`
- `blocked`
- `block_reason`

### Field meanings

- `event_id`: unique id for the event.
- `event_family`: one of `signal_events`, `action_events`, `execution_events`.
- `strategy`: strategy name, for example `imba_countertrend`.
- `symbol`: canonical symbol, for example `SOLUSDT`.
- `venue`: execution or source venue, for example `kucoin`, `kraken`, `internal`.
- `ts`: event timestamp in UTC ISO format.
- `seq`: monotonically increasing sequence within the producing component.
- `position_before`: signed position state before the event.
- `position_after`: signed position state after the event.
- `engine_mode_before`: mode before processing, for example `WAIT`, `TTP`, `FLAT`.
- `engine_mode_after`: mode after processing.
- `reason_code`: compact standardized reason.
- `source_event_id`: upstream originating event id.
- `source_signal_event_id`: upstream signal event id if applicable.
- `blocked`: whether the intended action was blocked.
- `block_reason`: standardized block reason if blocked.

## 4. Standard reason codes

Initial standardized `reason_code` / `block_reason` values:

### Signal reasons
- `imba_long`
- `imba_short`
- `trend_flip`
- `rearm`
- `flat_signal`

### Action reasons
- `enter_signal`
- `opposite_imba`
- `same_dir_ignored`
- `tp1_hit`
- `tp2_hit`
- `trailing_tp_hit`
- `stop_loss_hit`
- `break_even_hit`
- `regime_off_exit`
- `manual_action`
- `fallback_enter`
- `position_sync`

### Block reasons
- `cooldown_block`
- `stale_block`
- `duplicate_block`
- `same_dir_block`
- `regime_off_block`
- `not_confirmed`
- `confirm_timeout`
- `risk_block`
- `size_block`

## 5. Canonical event shapes

### signal_event

```json
{
  "event_id": "signal:imba:SOLUSDT:2026-03-06T11:23:00Z:77",
  "event_family": "signal_events",
  "strategy": "imba_countertrend",
  "symbol": "SOLUSDT",
  "venue": "internal",
  "ts": "2026-03-06T11:23:00Z",
  "seq": 77,
  "signal": -1,
  "signal_side": "short",
  "signal_family": "imba",
  "signal_kind": "trend_flip",
  "reason_code": "imba_short",
  "source_event_id": "renko:SOLUSDT:2026-03-06T11:23:00Z:123",
  "position_before": 1,
  "position_after": 1,
  "engine_mode_before": "TTP",
  "engine_mode_after": "TTP",
  "blocked": false,
  "block_reason": null
}

{
  "event_id": "action:imba_countertrend:SOLUSDT:2026-03-06T11:23:00Z:455",
  "event_family": "action_events",
  "strategy": "imba_countertrend",
  "symbol": "SOLUSDT",
  "venue": "internal",
  "ts": "2026-03-06T11:23:00Z",
  "seq": 455,
  "engine_action": "flip",
  "action_side": "short",
  "reason_code": "opposite_imba",
  "source_event_id": "renko:SOLUSDT:2026-03-06T11:23:00Z:123",
  "source_signal_event_id": "signal:imba:SOLUSDT:2026-03-06T11:23:00Z:77",
  "position_before": 1,
  "position_after": -1,
  "engine_mode_before": "TTP",
  "engine_mode_after": "TTP",
  "blocked": false,
  "block_reason": null
}

{
  "event_id": "execution:kucoin:SOLUSDT:2026-03-06T11:23:01Z:991",
  "event_family": "execution_events",
  "strategy": "imba_countertrend",
  "symbol": "SOLUSDT",
  "venue": "kucoin",
  "ts": "2026-03-06T11:23:01Z",
  "seq": 991,
  "execution_kind": "fill",
  "order_action": "sell",
  "reason_code": "position_sync",
  "source_signal_event_id": "signal:imba:SOLUSDT:2026-03-06T11:23:00Z:77",
  "position_before": 1,
  "position_after": -1,
  "blocked": false,
  "block_reason": null
}

## 6. Storage direction

Phase 1:
- Keep current JSONL flow alive.
- Emit richer events in parallel.
- Do not break existing readers yet.

Phase 2:
- Add PostgreSQL as the durable event store.
- Store `signal_events`, `action_events`, `execution_events`, and later `equity_events`.
- Dashboard and analytics should progressively read from durable stored events instead of ad-hoc runtime files.

Phase 3:
- Add derived views for strategy attribution, realized pnl, live equity, and multi-bot aggregation.