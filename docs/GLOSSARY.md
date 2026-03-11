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
- trade segments
- post-trade analysis

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

Responsibilities include:
- live bot state management
- execution
- event persistence
- equity persistence
- reconciliation with venue state

Main file:
- `src/quant/execution/kraken_bot.py`