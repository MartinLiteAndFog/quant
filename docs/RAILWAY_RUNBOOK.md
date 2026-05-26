# Railway Runbook

This is the operational runbook for the current Railway deployment.

Use it for:

- service overview
- deployment/runtime checks
- incident response
- live debugging
- handover

For broader architecture, see:

- `docs/ARCHITECTURE.md`
- `docs/event_schema_v1.md`
- `docs/LIVE_DEPLOY.md`

---

## 1. Current operational model

The Railway setup currently consists of a small set of services around:

- dashboard / API
- live signal generation
- live execution
- venue access
- Postgres-backed forensic persistence

The main direction is now:

**Postgres-first for forensic truth**, while still keeping JSONL and runtime files as transitional operational traces.

---

## 2. Main services

## `quant` service

Primary public-facing web/API service.

Responsibilities:
- serves `/dashboard`
- serves `/api/*`
- dashboard/state APIs
- chart/state-space/fills/trade views
- reads Postgres-first where implemented
- may still use runtime files as fallback for some operational views

Typical public endpoints:
- root domain
- `/dashboard`
- `/api/status`
- `/api/position`
- `/api/dashboard/chart`
- `/api/dashboard/statespace`
- `/api/gate/*`
- `/api/signals/latest/*`

---

## signal / execution worker service

Current practical responsibilities:
- `live_signal_worker`
- `live_executor`

The exact Railway service name may vary, but operationally this worker side does the following:

### `live_signal_worker`
- generates live routed signals
- writes active signal stream
- writes strategy-specific substreams

### `live_executor`
- reads routed live signal stream
- derives engine action
- calls OMS / venue adapter
- writes `action_events`
- writes `execution_events` on successful OMS execution paths
- writes dashboard-active levels / runtime state

---

## Kraken execution path

If Kraken is deployed separately in Railway or another environment, the relevant live entrypoints are:

- `quant.execution.live_executor_2` (preferred, stop-order-native)
- `quant.execution.kraken_bot` (legacy)

Responsibilities:
- Kraken live execution loop
- event persistence — gated behind `KRAKEN_TRADE_TRACKING_ENABLED=1` (default OFF)
- equity persistence — gated behind `KRAKEN_TRADE_TRACKING_ENABLED=1`
- state reconciliation with venue

With `KRAKEN_TRADE_TRACKING_ENABLED` unset, the decision loops still run; only
the equity / `action_events` / `execution_events` / metrics writes are
skipped. This is the supported "Kraken shadow / observe-only" mode.

Memory note:
The main dashboard service does not load Kraken dashboard state. If Railway
reports OOM on a separate Kraken service while Kraken trading is not intended
to run, disable that Railway service or change its start command away from
`quant.execution.live_executor_2` / `quant.execution.kraken_bot`. Leaving
`KRAKEN_TRADE_TRACKING_ENABLED=0` stops persistence side effects but does not
stop the Kraken decision loop itself.

---

## 3. Current truth model

When operational evidence conflicts, prefer this order:

1. Postgres
2. current deployed code behavior
3. JSONL runtime traces
4. local runtime files
5. logs alone

Reason:
runtime files and logs are useful for operations, but Postgres is becoming the durable forensic source of truth.

---

## 4. Active durable persistence

The following durable objects are already relevant in current operations:

- `action_events`
- `execution_events`
- `closed_trades`
- `equity_snapshots`
- `trade_decisions` (derived from `action_events`; schema in
  `src/quant/sql/002_trade_decisions.sql`)

### Verified current state

#### KuCoin / live_executor
- `action_events`: live and verified in Postgres
- `execution_events`: writer deployed; awaiting confirmation on next real execution-triggered event

#### Kraken
- `execution_events`: active
- `equity_snapshots`: active

#### Dashboard
Several reads are already Postgres-first, especially around:
- equity history
- trade markers
- trading diary
- closed trades
- trade decision counts (`trade_decisions`; exposed via
  `/api/dashboard/trade_count` and as `trade_decision_count` on
  `/api/dashboard/performance`)

Notes:
- The dashboard is KuCoin-focused. Kraken-only chart payload fields
  (`equity_kraken`, `equity_combined`, `kraken_metrics`, `equity_live`,
  `equity_realized`) and the `segments` field are kept for backwards
  compatibility but are now empty / inert.
- Main dashboard performance/loading improvements are **in progress** and
  will land in a separate change.

---

## 5. Runtime files still in use

These are still operationally useful and may live on persistent volume paths such as `/data/live` or `/data/events`.

Examples:
- signal JSONL streams
- event JSONL traces
- active levels JSON
- Renko parquet cache
- metrics JSON / CSV
- local bot/executor state JSON

Important:
These are still valid operational traces, but they should no longer be treated as the preferred long-term forensic source if Postgres already contains the answer.

---

## 6. Environment variables

Set Railway variables per service as needed.

## Core venue/auth variables
- `KUCOIN_FUTURES_API_KEY`
- `KUCOIN_FUTURES_API_SECRET`
- `KUCOIN_FUTURES_PASSPHRASE`
- `PYTHONUNBUFFERED=1`

Add Kraken credentials where the Kraken bot is deployed.

## Kraken trade-tracking gate

- `KRAKEN_TRADE_TRACKING_ENABLED` (default `0`) — when `1`, the Kraken
  executors (`live_executor_2.py`, legacy `kraken_bot.py`) persist equity
  snapshots, Kraken `action_events` / `execution_events` and per-bot
  metrics/equity files. Leave unset (or `0`) to run Kraken in
  shadow / observe-only mode without writing those side effects.

---

## Core runtime paths
Use persistent, absolute paths where applicable.

Examples:
- `SIGNALS_DIR=/data/live/signals`
- `LIVE_EXECUTOR_STATE=/data/live/live_executor_state.json`
- `LIVE_SIGNAL_STATE=/data/live/live_signal_state.json`

Some historical path variables may still exist in the deployment, but operationally we should prefer the current code’s actual readers/writers over old path assumptions in legacy docs.

---

## Live safety controls
Recommended safety-related variables include:

- `LIVE_TRADING_ENABLED=0` for hard-off
- `LIVE_EXECUTOR_DRY_RUN=1` for simulation
- symbol allowlists
- leverage limits
- logging throttle controls

Never go live by changing multiple safety variables blindly at once.

Preferred go-live order:
1. dry-run stable
2. check logs + expected behavior
3. enable trading permission
4. disable dry-run
5. keep size conservative first

Rollback:
- set `LIVE_EXECUTOR_DRY_RUN=1`
- and/or set `LIVE_TRADING_ENABLED=0`

---

## 7. Current signal routing

Current live routing logic is regime-based.

Typical routing idea:
- `gate_on=1` → countertrend stream
- `gate_on=0` → trend-following stream

Signal files typically include:
- active routed stream
- strategy-specific history subdirectories

Treat signal files as operational traces; do not assume they are yet the final authoritative `signal_events` chain.

---

## 8. Standard start commands

## Web/API service

Typical form:

```bash
uvicorn quant.execution.webhook_server:app --host 0.0.0.0 --port $PORT

Signal + executor worker

A combined command may still be used operationally, for example:

bash -lc "python -u -m quant.execution.live_signal_worker --symbol SOLUSDT --signals-dir /data/live/signals & python -u -m quant.execution.live_executor --symbol SOLUSDT --signals-dir /data/live/signals; wait"

Operational preference going forward:
separate services are cleaner than one chained shell process, unless there is a strong reason to keep them combined.

## Cronjob service

Purpose:
- build the durable daily CHOP/ADX/ER gate from Renko OHLC
- persist daily gate history in Postgres
- mirror the latest gate snapshot to Redis
- prune old base equity snapshots

Recommended Railway start command:

```bash
bash -lc "python -u scripts/build_live_daily_gate_artifacts.py --symbol SOL-USDT && python -u -m quant.execution.equity_retention --apply"
```

Required variables for the cron service:
- `POSTGRES_URL` or `DATABASE_URL`
- `REDIS_URL`

Usually set as well:
- `LIVE_GATE_SYMBOL=SOL-USDT`
- `LIVE_GATE_PRIMARY=on` to keep `gate_on = countertrend`
- `PYTHONUNBUFFERED=1`

Important runtime rule:
- the web/API service and both live executors now read the gate from `get_live_gate_state()` with priority `Postgres -> Redis -> forced countertrend`
- service-local gate CSVs are now optional debug artifacts only and are not the live source of truth
- the producer now follows the winner lineage `CHOP + ADX + ER` with `qch=0.4`, `qadx=0.6`, `qer=0.3`
- cron reads Renko from Postgres first, so it no longer requires a shared Renko parquet volume

Renko updater requirement:
- the service that runs `renko_cache_updater` still writes the parquet cache for existing readers, but it must also have `POSTGRES_URL` so it can mirror bricks into `live_renko_bricks`
- that service may still use `DASHBOARD_RENKO_PARQUET` locally for dashboard/chart readers

9. Health checks
Web/API checks

Useful checks from a local machine:

dashboard loads

chart endpoint returns data

state-space endpoint returns data

gate endpoint responds

latest signal endpoint responds

Worker checks

Useful checks in Railway SSH:

signal JSONL for fresh writes

runtime state file timestamps

latest action_events in Postgres

latest execution_events in Postgres

current venue/account position

Cron checks

Useful checks after a cron run:

- verify the cron logs show `persisted daily gate history rows=...`
- verify `GET /api/gate/solusd` returns `source=postgres_daily_gate` in the healthy case
- verify the returned `ts`, `gate_on_ts`, and `gate_off_ts` match the intended trading day
- if Postgres is temporarily unavailable, verify the endpoint falls back to `source=redis`
- if both Postgres and Redis are unavailable, verify the endpoint falls back to `source=forced_countertrend`

10. Recommended live-debug flow

When something looks wrong, use this order:

A. Check if the system is actually acting

current signal

current gate

current position

last action event

B. Check whether the action became an execution

latest execution_events

venue-side position / fills

OMS result logs

C. Check realized outcome

closed_trades

equity_snapshots

D. Only then use runtime files/logs for secondary evidence

signal JSONL

/data/events/*.jsonl

active level files

metrics JSON/CSV

11. Practical DB checks

Typical questions to ask in incidents:

Latest live executor actions

query action_events filtered by strategy='live_executor'

Latest executions

query execution_events

Latest realized trades

query closed_trades

Latest equity

query equity_snapshots

The goal is to avoid reconstructing incidents from logs alone if Postgres already contains the answer.

12. Known current caveats

Railway shells may be minimal and lack common tools like ps, rg, or curl

here-doc copy/paste into Railway SSH can sometimes render oddly

not every legacy runtime file is fully authoritative anymore

signal_events are not yet fully live-wired end-to-end

source_signal_event_id / source_action_event_id may still legitimately be None

KuCoin execution_events are deployed but still need confirmation on the next real fresh execution-triggered event

13. Operational rules

verify code in repo first

push to Git before live verification

avoid hotpatching running containers unless absolutely necessary

prefer Postgres checks over log archaeology

prefer minimal, verifiable fixes over speculative refactors

document architectural changes after the event chain is real

14. Current priority checklist
High priority

confirm first real KuCoin execution_event after deploy

continue wiring signal_events

improve action → execution linkage

Medium priority

standardize reason codes

reduce dependency on legacy runtime files

align runbooks with actual service topology

Cleanup

remove stale backup files

prune old docs assumptions

reduce duplicate fallback paths