# Glossary

## IMBA
Impulse-based signal indicator used as a core strategy signal source.

In the current system, IMBA-derived signals are used in both:
- countertrend execution
- trend-following / ImbaTrend execution

---

## Renko
A brick-based price representation used instead of time bars.

Renko is central to:
- IMBA signal generation
- backtest parity
- flip-engine logic
- parts of the live execution decision layer

---

## Gate
A regime filter that determines which strategy family is active.

In the current routing logic, the gate decides whether the system behaves in:
- countertrend mode
- trend-following mode

---

## ON regime
The regime in which the countertrend flip strategy is active.

Typical behavior:
- IMBA-driven countertrend entry
- trailing take-profit
- stop-loss
- opposite-signal flip logic

---

## OFF regime
The regime in which the trend-following / ImbaTrend strategy is active.

Typical behavior:
- IMBA entry
- TP1 / TP2 structure
- SL clamp
- different execution behavior from ON regime

---

## OMS
Order Management System.

The OMS is responsible for translating intended actions into actual execution behavior.

In the current stack, this includes:
- entry ladder logic
- maker-first logic
- flatten-first flip handling
- fallback execution behavior

Main file:
- `src/quant/execution/oms.py`

---

## TTP
Trailing Take Profit.

A trailing profit-taking mechanism used in the execution logic.

In some strategy/execution contexts it is part of:
- exit logic
- flip preparation
- active level display in the dashboard

---

## Signal
A strategy or alpha statement.

Examples:
- long
- short
- trend flip
- re-arm
- flat signal

Signals are conceptually upstream of actions and executions.

---

## Action
An engine decision derived from signal + state.

Examples:
- `enter_long`
- `enter_short`
- `flip_to_long`
- `flip_to_short`
- `exit_long`
- `exit_short`
- `hold`
- `scale_long`
- `scale_short`

Actions are persisted in `action_events`.

---

## Execution
A venue / OMS fact.

Examples:
- order fill
- cancel
- rejection
- fallback execution
- position sync

Executions are persisted in `execution_events`.

---

## Closed trade
A realized trade outcome with entry/exit timestamps and realized PnL information.

Closed trades are used by the dashboard for:
- trading diary
- trade markers
- post-trade analysis

(Trade-connector segments between entry and exit markers were removed from
the dashboard; only the markers themselves and the trading diary remain.)

---

## Trade decision
A discrete directional position-opening event that carries its own SL/TP
commitment.

Counted:
- entry from flat (`enter_long`, `enter_short`) — `decision_kind = entry`
- flip to opposite direction (`flip_to_long`, `flip_to_short`) — `decision_kind = flip`

Not counted:
- `scale_long` / `scale_short` (same-direction add, no new SL/TP)
- `tp1_partial` and other partial closes
- `exit_long` / `exit_short` (the trade was already counted at entry / flip)
- `hold`
- any `blocked = true` action

Persisted in `trade_decisions` (derived from `action_events`); classified by
`src/quant/execution/trade_counter.py`. Exposed by
`/api/dashboard/trade_count` and as `trade_decision_count` on
`/api/dashboard/performance`.

---

## Equity snapshot
A durable record of equity/account state at a point in time.

Equity snapshots are used for:
- dashboard equity history
- account/venue equity analysis
- later forensic reconstruction

---

## Postgres-first
The current architecture direction where Postgres becomes the preferred forensic source of truth.

This means:
- query Postgres before reconstructing from logs/files
- JSONL remains useful as secondary trace
- runtime files are increasingly operational only, not authoritative

---

## Forensic truth
The preferred durable source used to reconstruct what actually happened.

The intended chain is:

**Signal → Action → Execution → Closed Trade → Equity**

The system is being migrated so this chain becomes queryable without relying on log archaeology.

---

## JSONL trace
An append-only local event/file trace kept for debugging and compatibility.

Important:
JSONL is still useful, but it is no longer the desired final truth layer where Postgres already contains the relevant information.

---

## Live executor
The KuCoin-side live execution component.

Responsibilities include:
- reading live signals
- deriving desired action
- calling OMS
- persisting `action_events`
- persisting `execution_events` on successful OMS execution paths

Main file:
- `src/quant/execution/live_executor.py`

---

## Kraken bot
The Kraken-side live execution loop.

Two entrypoints exist:
- `src/quant/execution/live_executor_2.py` — current, stop-order-native
- `src/quant/execution/kraken_bot.py` — legacy

Responsibilities include:
- live bot state management
- execution
- event persistence (gated)
- equity persistence (gated)
- reconciliation with venue state

**Kraken persistence is gated.** Equity snapshots, Kraken
`action_events` / `execution_events` and per-bot metrics/equity files are only
written when `KRAKEN_TRADE_TRACKING_ENABLED=1`. With the flag unset (the
default), the decision loops still run; only the persistence side-effects are
skipped.