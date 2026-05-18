# Event Schema v1

## 1. Purpose

This document describes the current event model for the Quant system.

The event model exists to support:

- live debugging
- post-mortem / forensic reconstruction
- attribution from signal to realized outcome
- migration from ad-hoc runtime traces to durable Postgres-backed analysis

The practical target chain is:

**Signal → Action → Execution → Closed Trade → Equity**

This schema document focuses on the event families themselves.  
For broader system structure, see `docs/ARCHITECTURE.md`.

---

## 2. Current status

The event model is no longer only aspirational.

### Already active in practice
- `action_events`
- `execution_events`
- `closed_trades`
- `equity_snapshots`
- `trade_decisions` (derived from `action_events`)

### Partially active / not yet fully live-wired
- `signal_events`

### Persistence model
Current persistence is hybrid:

- JSONL remains as append-only local trace
- Postgres is becoming the durable forensic source of truth

This is intentional during the migration period.

---

## 3. Event families

## 3.1 `signal_events`

Signals are strategy or alpha statements.

Examples:
- IMBA long
- IMBA short
- trend flip
- re-arm
- flat signal

Typical source:
- IMBA on Renko
- later potentially other alpha / gating / model outputs

Current status:
- concept and schema exist
- not yet fully live-wired end-to-end as authoritative upstream records

---

## 3.2 `action_events`

Actions are engine decisions derived from signals plus current state.

Examples:
- `enter_long`
- `enter_short`
- `flip_to_long`
- `flip_to_short`
- `exit_long`
- `exit_short`
- `scale_long`
- `scale_short`
- `hold`
- blocked / ignored decisions

Typical producer:
- execution engine / live executor / bot logic

Current status:
- Kraken: active
- KuCoin/live_executor: active
- persisted to JSONL and Postgres

---

## 3.3 `execution_events`

Executions are OMS / venue facts.

Examples:
- order fill
- cancel
- rejection
- fallback execution
- position sync
- submitted order
- partial fill

Current status:
- Kraken: active
- KuCoin/live_executor: initial writer integrated for successful OMS execution paths
- currently still coarser than a full order lifecycle model

Important:
An `execution_event` should represent something that actually happened at the OMS / venue layer, not just an intended action.

---

## 3.4 Related durable objects outside the core event trio

These are not part of the original minimal trio, but are already important in the current architecture.

### `closed_trades`
Represents realized trade outcomes.

Used for:
- trading diary
- trade markers
- post-trade analysis

(Note: entry→exit trade-connector segments are no longer rendered on the
dashboard.)

### `equity_snapshots`
Represents durable equity/account snapshots.

Used for:
- dashboard equity curves
- venue/account equity history
- later attribution from event chain to account evolution

### `trade_decisions` (derived)

Counts discrete directional position-opening decisions. Derived from
`action_events`, so rebuilding the table from history always yields the same
rows (every `decision_id` is deterministic).

Schema: `src/quant/sql/002_trade_decisions.sql`.
Classifier: `src/quant/execution/trade_counter.py`.
Store/backfill: `src/quant/execution/trade_decisions_store.py`.

Classification rules (authoritative):

| `engine_action` | counted? | `decision_kind` |
|-----------------|----------|-----------------|
| `enter_long`    | yes      | `entry`         |
| `enter_short`   | yes      | `entry`         |
| `flip_to_long`  | yes      | `flip`          |
| `flip_to_short` | yes      | `flip`          |
| `scale_long` / `scale_short` | no | — (same-direction add, no new SL/TP) |
| `tp1_partial` and any partial close | no | — (size reduction, keeps SL) |
| `exit_long` / `exit_short` | no | — (ends lifecycle; trade already counted at entry/flip) |
| `hold` | no | — |
| any unrecognised label | no | — |
| any row with `blocked = true` | no | — (SL/TP was never committed) |

Idempotency: `decision_id = "td_" + sha1(source_action_event_id)[:16]`
(falls back to a hash over `(venue, symbol, ts, seq, engine_action)` when
there is no source event id). All write paths upsert by `decision_id`.

API:
- `GET /api/dashboard/trade_count?symbol=&venue=&recent_limit=&backfill=`
- `GET /api/dashboard/performance` → adds `trade_decision_count`

Backfill:
- `python scripts/backfill_trade_decisions.py --venue kucoin --symbol SOL-USDT`
- or one-shot via the API with `?backfill=1`

---

## 4. Common minimal fields

All event families should include these fields where applicable:

- `event_id`
- `event_family`
- `strategy`
- `symbol`
- `venue`
- `ts`
- `seq`
- `position_before`
- `position_after`
- `reason_code`
- `source_event_id`
- `source_signal_event_id`
- `blocked`
- `block_reason`

Additional fields vary by event family.

---

## 5. Field meanings

- `event_id`: unique identifier for the event
- `event_family`: one of `signal_events`, `action_events`, `execution_events`
- `strategy`: producing strategy or engine name
- `symbol`: canonical symbol, for example `SOLUSDT`
- `venue`: source or execution venue, for example `internal`, `kucoin`, `kraken`
- `ts`: UTC timestamp in ISO format
- `seq`: monotonically increasing sequence within the producing component
- `position_before`: signed position state before the event
- `position_after`: signed position state after the event
- `reason_code`: compact normalized reason
- `source_event_id`: upstream event if applicable
- `source_signal_event_id`: upstream signal event id if applicable
- `blocked`: whether the intended action was blocked
- `block_reason`: normalized reason for blocking

### Additional action-only fields
- `engine_action`
- `action_side`
- `engine_mode_before`
- `engine_mode_after`

### Additional execution-only fields
- `execution_kind`
- `order_action`
- `order_id`
- `client_oid`
- `side`
- `qty`
- `price`
- `reduce_only`
- `status`
- `reject_reason`

---

## 6. Linking rules

## 6.1 Intended linkage direction

The intended linkage is:

- `signal_events` link upstream strategy statements
- `action_events` link to `source_signal_event_id`
- `execution_events` link to `source_action_event_id` and/or `source_signal_event_id`
- `closed_trades` later link back to upstream actions/executions where useful
- `equity_snapshots` remain time-series state, not necessarily one-to-one linked

---

## 6.2 Current practical rule

Do **not** force foreign-key-style linkage until the upstream producer is genuinely live and reliable.

That means:

- `source_signal_event_id` may legitimately be `None`
- `source_action_event_id` may legitimately be `None`
- this is preferable to fake linkage or broken references

This is a temporary but deliberate migration rule.

---

## 7. Standardized reason and block codes

These are target-normalized values.  
Not all current producers are fully aligned yet.

## 7.1 Signal reasons
- `imba_long`
- `imba_short`
- `trend_flip`
- `rearm`
- `flat_signal`

## 7.2 Action reasons
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

## 7.3 Block reasons
- `cooldown_block`
- `stale_block`
- `duplicate_block`
- `same_dir_block`
- `regime_off_block`
- `not_confirmed`
- `confirm_timeout`
- `risk_block`
- `size_block`

Important:
Current production data may still contain coarse or placeholder values such as `none`.  
Reason-code standardization is still in progress.

---

## 8. Canonical event examples

## 8.1 Example `signal_event`

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
  "blocked": false,
  "block_reason": null
}

##8.2 example action_event
{
  "event_id": "action:live_executor:SOLUSDT:2026-03-06T11:23:00Z:455",
  "event_family": "action_events",
  "strategy": "live_executor",
  "strategy_instance": "live_executor",
  "config_hash": "live_executor_v1",
  "symbol": "SOLUSDT",
  "venue": "kucoin",
  "ts": "2026-03-06T11:23:00Z",
  "seq": 455,
  "engine_action": "flip_to_short",
  "action_side": "short",
  "reason_code": "opposite_imba",
  "source_event_id": null,
  "source_signal_event_id": null,
  "position_before": 1,
  "position_after": -1,
  "engine_mode_before": "TTP",
  "engine_mode_after": "TTP",
  "blocked": false,
  "block_reason": null
}

#8.3 Example execution_event
{
  "event_id": "execution:live_executor:SOLUSDT:2026-03-06T11:23:01Z:991:kucoin",
  "event_family": "execution_events",
  "strategy": "live_executor",
  "strategy_instance": "live_executor",
  "config_hash": "live_executor_v1",
  "symbol": "SOLUSDT",
  "venue": "kucoin",
  "ts": "2026-03-06T11:23:01Z",
  "seq": 991,
  "execution_kind": "fill",
  "order_action": "sell",
  "reason_code": "opposite_imba",
  "source_event_id": null,
  "source_signal_event_id": null,
  "position_before": 1,
  "position_after": -1,
  "blocked": false,
  "block_reason": null,
  "client_oid": "tp_flip:PO:1741682581000",
  "order_id": "1234567890",
  "side": "sell",
  "qty": 54,
  "price": 86.2,
  "reduce_only": true,
  "status": "fill",
  "reject_reason": null,
  "payload_json": {}
}
 8.4 example closed_trade
 {
  "trade_id": "trade:kucoin:SOLUSDT:2026-03-06T10:00:00Z:2026-03-06T11:23:01Z:1",
  "venue": "kucoin",
  "symbol": "SOLUSDT",
  "entry_ts": "2026-03-06T10:00:00Z",
  "exit_ts": "2026-03-06T11:23:01Z",
  "side": "long",
  "qty": 54,
  "entry_price": 84.7,
  "exit_price": 86.2,
  "pnl_pct": 0.0177,
  "exit_event": "flip_exit",
  "strategy": "live_executor",
  "strategy_instance": "live_executor",
  "config_hash": "live_executor_v1"
}
8.5 Example equity_snapshot

{
  "ts": "2026-03-06T11:23:01Z",
  "venue": "kucoin",
  "account": "main",
  "symbol": null,
  "equity": 176.04,
  "currency": "USD",
  "source": "dashboard_state.load_real_equity_history",
  "payload_json": {}
}

9. Storage direction

Phase 1

keep JSONL flow alive

emit richer events in parallel

do not break existing readers

Phase 2

persist action_events, execution_events, closed_trades, equity_snapshots

progressively move dashboard and analytics to Postgres-first reads

Phase 3

fully live-wire signal_events

add reliable upstream linkage

add derived SQL views for attribution and forensic timelines

10. Practical migration rules

Prefer incremental migration over big rewrites

Writers first

Readers second

Linkage third

Documentation fourth

Cleanup after coverage is real

And:

Postgres should become the forensic truth

JSONL should remain a secondary local trace

placeholder linkage is better than fake linkage

reason-code normalization is still an active task