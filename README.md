# quant

Quant is the working repository for live trading, research, execution, dashboarding, and forensic event persistence.

The current architecture direction is:

**Postgres-first for forensic truth**, with JSONL and runtime files kept as transitional operational traces.

The intended chain is:

**Signal → Action → Execution → Closed Trade → Equity**

---

## Core areas

### Live execution
Live trading and venue execution components.

Primary live target:
- **KuCoin Futures** — `src/quant/execution/live_executor.py` (the dashboard and equity/performance views are KuCoin-focused).

Secondary / experimental:
- **Kraken Futures** — `src/quant/execution/live_executor_2.py` (stop-order-native executor) and legacy `src/quant/execution/kraken_bot.py`. Kraken trade/equity/event tracking persistence is gated behind `KRAKEN_TRADE_TRACKING_ENABLED=1` and defaults **OFF**.

Other supporting files:
- `src/quant/execution/oms.py`
- `src/quant/execution/kucoin_futures.py`
- `src/quant/execution/kraken_futures.py`

### Strategy logic
Main live/backtest strategy logic includes:
- countertrend flip strategy
- ImbaTrend / trend-following regime
- IMBA-based signal generation
- Renko-based state and execution logic

Main files:
- `src/quant/strategies/flip_engine.py`
- `src/quant/backtest/renko_runner.py`

### Dashboard / API
Dashboard, state views, fills, markers, and runtime APIs.

The dashboard chart, equity curve, and performance views are KuCoin-focused.
Kraken-only compatibility fields in chart payloads (`equity_kraken`,
`equity_combined`, `kraken_metrics`) are kept for backwards compatibility but
are now empty, and Kraken loaders have been removed from `dashboard_state.py`.

Trade-connector segments (the lines between entry and exit markers) are no
longer produced or rendered. The chart payload still includes
`segments: []` for backwards compatibility, but neither the React
(`frontend/src/components/charts/PriceChart.tsx`,
`frontend/src/components/layout/Dashboard.tsx`) nor the Svelte
(`dashboard/src/components/PriceChart.svelte`) dashboards render them.

> **In progress:** main dashboard performance/loading improvements. Behavior
> may shift while that work lands; this README will be updated once it is
> verified.

Main files:
- `src/quant/execution/webhook_server.py`
- `src/quant/execution/dashboard_state.py`

### Trade decision counter
Counts discrete directional position-opening decisions (each entry or flip
gets its own SL/TP, so each is counted independently). Source of truth is
derived from `action_events`.

Main files:
- `src/quant/execution/trade_counter.py` — classification rules
- `src/quant/execution/trade_decisions_store.py` — Postgres upsert / count / list / backfill
- `src/quant/sql/002_trade_decisions.sql` — idempotent `trade_decisions` table
- `scripts/backfill_trade_decisions.py` — one-shot backfill CLI

API:
- `GET /api/dashboard/trade_count?symbol=&venue=&recent_limit=&backfill=`
- `GET /api/dashboard/performance` also returns `trade_decision_count`

See `docs/ARCHITECTURE.md` and `docs/event_schema_v1.md` for the full
classification rules and schema.

### Event persistence
Durable event and forensic persistence.

Main files:
- `src/quant/execution/event_builders.py`
- `src/quant/execution/event_store.py`
- `src/quant/execution/event_types.py`

---

## Main documents

### Architecture and event model
- `docs/ARCHITECTURE.md`
- `docs/event_schema_v1.md`

### Deployment and operations
- `docs/LIVE_DEPLOY.md`
- `docs/RAILWAY.md`
- `docs/RAILWAY_RUNBOOK.md`

### Debugging and terminology
- `docs/DEBUGGING.md`
- `docs/GLOSSARY.md`

---

## Current persistence direction

The system is in an incremental migration phase.

### Already important in Postgres
- `action_events`
- `execution_events`
- `closed_trades`
- `equity_snapshots`
- `trade_decisions` (derived from `action_events`; see
  `src/quant/sql/002_trade_decisions.sql`)

### Not yet fully live-wired end-to-end
- `signal_events`

### Practical rule
Use this source priority when debugging:

1. Postgres
2. current deployed code behavior
3. JSONL/runtime traces
4. logs

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies using the project's current dependency method in the repo.
If package/import issues appear, prefer running commands with:

```bash
PYTHONPATH=src
```

Example:

```bash
PYTHONPATH=src python3 -m quant.execution.live_executor --once
```

## Typical workflows

Local compile check:

```bash
PYTHONPATH=src python3 -m py_compile src/quant/execution/live_executor.py
```

Run one live-executor cycle:

```bash
PYTHONPATH=src python3 -m quant.execution.live_executor --once
```

Run one Kraken executor cycle (legacy `kraken_bot.py` or newer `live_executor_2.py`).
Trade/equity/event persistence is gated — set `KRAKEN_TRADE_TRACKING_ENABLED=1` if
you want those side effects in this run:

```bash
KRAKEN_TRADE_TRACKING_ENABLED=1 PYTHONPATH=src \
  python3 -m quant.execution.kraken_bot --once
```

Backfill the `trade_decisions` table from existing `action_events` (idempotent):

```bash
PYTHONPATH=src python3 scripts/backfill_trade_decisions.py \
  --venue kucoin --symbol SOL-USDT
```

Or trigger the same backfill via the dashboard API (one-shot):

```bash
curl '<host>/api/dashboard/trade_count?symbol=SOL-USDT&venue=kucoin&backfill=1'
```

## Working rules for live changes

Preferred workflow:

1. fix in repo
2. compile/check locally
3. commit
4. push
5. deploy
6. verify live
7. verify Postgres persistence

Avoid hotpatching running containers unless absolutely necessary.

## State space

The repository also contains state-space / predictive-coding related research components.

For that area, see the corresponding scripts and docs under:

- `docs/plans/`
- `docs/visual/`

Older state-space work remains relevant, but the repo now also includes substantial live execution and forensic infrastructure beyond that earlier scope.