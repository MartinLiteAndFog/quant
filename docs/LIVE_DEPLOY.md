Ja — `LIVE_DEPLOY.md` ist an mehreren Stellen veraltet.

Die größten Probleme darin:

* es spricht noch von **Postgres später**, obwohl wir dafür schon mitten in der Migration sind
* es beschreibt die Persistenz noch zu stark als `data/live`- und JSONL-/Parquet-Welt
* die neue **Postgres-first-Forensik** fehlt
* `live_executor`-Event-Persistenz und Dashboard-Postgres-first-Leser fehlen
* Railway-/Service-Bild ist zu grob für den aktuellen Stand

Wir sollten es neu schreiben, aber **pragmatisch** und nicht als Monster-Dokument.

Erster Schritt: Ersetze `docs/LIVE_DEPLOY.md` komplett durch diesen Inhalt.

````md
# Live Deploy

## Purpose

This document describes how the live Quant stack is currently deployed and operated.

It focuses on:

- live services
- environment variables
- runtime persistence
- dashboard/runtime data flow
- current deployment reality during the Postgres migration

For system structure, see:

- `docs/ARCHITECTURE.md`
- `docs/event_schema_v1.md`
- `docs/RAILWAY_RUNBOOK.md`

---

## Current deployment model

The live stack is currently a small set of long-running services around:

- live signal generation
- live execution
- dashboard / API
- venue adapters
- Postgres-backed forensic persistence

Main execution venues currently in scope:

- KuCoin Futures
- Kraken Futures

---

## Core live services

## 1. Dashboard / API service

Main entrypoint:
- `quant.execution.webhook_server:app`

Responsibilities:
- dashboard UI
- dashboard APIs
- regime endpoints
- fills / trade views
- state-space and chart endpoints
- some runtime refresh helpers

Typical start command:

```bash
uvicorn quant.execution.webhook_server:app --host 0.0.0.0 --port $PORT
````

---

## 2. Live signal worker

Responsibilities:

* generate live strategy signals
* write active signal stream
* maintain strategy-specific signal streams

Typical output:

* `SIGNALS_DIR/<SYMBOL>/<day>.jsonl`
* strategy substreams such as:

  * `countertrend`
  * `trendfollower`

---

## 3. Live executor

Main entrypoint:

* `quant.execution.live_executor`

Responsibilities:

* read live signals
* align signals with current Renko/backtest state
* derive engine action
* call OMS / venue broker
* write `action_events`
* write `execution_events` for successful OMS execution paths
* write active levels for dashboard display

Current practical status:

* KuCoin `action_events` are live and verified in Postgres
* KuCoin `execution_events` are wired in and deployed, awaiting confirmation on the next real execution-triggered event

---

## 4. Kraken bot

Main entrypoint:

* `quant.execution.kraken_bot`

Responsibilities:

* live Kraken execution path
* event persistence
* equity persistence
* bot state reconciliation with venue

Current note:

* Kraken event/equity persistence is already integrated into the Postgres migration path

---

## Market data and price source

## KuCoin price source

For KuCoin live execution, bid/ask and live order placement come from the KuCoin Futures API through:

* `KucoinFuturesBroker`
* `get_best_bid_ask("SOL-USDT")`

This means live price checks and execution use the same exchange source.

---

## Secrets and credentials

Never place credentials in code.

Use environment variables only.

### KuCoin Futures

* `KUCOIN_FUTURES_API_KEY`
* `KUCOIN_FUTURES_API_SECRET`
* `KUCOIN_FUTURES_PASSPHRASE`

### Kraken

Use the corresponding Kraken environment variables configured by the Kraken client/bot deployment.

### Webhook / service auth

Use environment variables for any webhook token or service secret.

---

## Local vs cloud secret handling

## Local

* keep secrets in a local `.env` or shell environment
* never commit them

## Cloud / Railway

* set secrets in Railway Variables
* do not rely on committed `.env` files

---

## Persistence model

## Current practical rule

The live system currently uses a hybrid persistence model:

* JSONL remains as local append-only trace
* some runtime files remain operational state
* Postgres is becoming the durable forensic source of truth

This is intentional during migration.

---

## Runtime files still in use

Examples of runtime/local artifacts still used operationally:

* signal JSONL streams
* active level JSON / execution state files
* Renko cache parquet files
* metrics JSON / CSV
* expected trade traces
* local event JSONL files under `/data/events`

These are still useful for:

* operational inspection
* fallback compatibility
* local debugging

But they are no longer the desired long-term forensic truth layer.

---

## Postgres-first direction

The current architecture is moving toward:

**Signal → Action → Execution → Closed Trade → Equity**

with Postgres as the main queryable store.

Already active or partly active in Postgres:

* `action_events`
* `execution_events`
* `closed_trades`
* `equity_snapshots`

Not yet fully live-wired:

* `signal_events`

Important:
JSONL is still emitted in parallel and should currently be treated as a secondary trace, not the preferred truth source.

---

## Dashboard data flow

Main dashboard logic:

* `src/quant/execution/dashboard_state.py`
* `src/quant/execution/webhook_server.py`

Current direction:

* dashboard reads Postgres first where available
* legacy files remain as fallback/compatibility inputs

Already migrated toward Postgres-first:

* real equity history
* Kraken equity history
* closed-trade backed trade markers
* trading diary
* trade segments

---

## Worker / process layout

A common practical deployment layout is:

### Service A: dashboard/API

Runs:

* `webhook_server`

### Service B: signal generation

Runs:

* `live_signal_worker`

### Service C: execution

Runs:

* `live_executor`
* and/or Kraken bot, depending on deployment split

This separation is preferred over one giant mixed process because it makes:

* logs cleaner
* restart behavior safer
* debugging easier
* resource use more understandable

---

## Railway deployment

Railway is currently a reasonable deployment target for the live stack.

Typical Railway responsibilities:

* host long-running services
* store environment variables
* expose public dashboard/API endpoints
* attach persistent storage where needed

Environment variables should be managed in Railway, not in repo files.

For Railway operational details, see:

* `docs/RAILWAY.md`
* `docs/RAILWAY_RUNBOOK.md`

---

## Persistence locations and volumes

Any runtime state that must survive restart should live on persistent storage.

Typical examples:

* `/data/live/...`
* `/data/events/...`

These may include:

* signals
* event JSONL traces
* dashboard runtime state
* Renko cache artifacts

However, durable forensic reconstruction should increasingly come from Postgres rather than these files.

---

## Live safety defaults

Recommended safe defaults before go-live:

* `LIVE_TRADING_ENABLED=0`
* `LIVE_EXECUTOR_DRY_RUN=1`

Also keep conservative values for:

* leverage
* size limits
* allowlists
* throttling
* fallback behavior

The practical go-live flow should be:

1. dry-run only
2. inspect logs and expected actions
3. verify venue/account reads
4. enable live trading
5. disable dry-run

Never flip directly into full live mode without an observed dry-run phase.

---

## Example execution start

A typical execution command may look like:

```bash
PYTHONPATH=src python3 -m quant.execution.live_executor --once
```

In deployed services, use the service-specific production command and environment.

For signal generation + executor combinations, prefer separate services over shell background chaining unless there is a strong reason not to.

---

## Dashboard and runtime APIs

The dashboard exposes various operational endpoints such as:

* status
* position
* chart/state-space
* signal/gate views
* fills/trade views

Exact endpoint surface may evolve, but the key rule is:

* operational APIs may still read mixed runtime state
* forensic and historical views should progressively move to Postgres-backed reads

---

## Current migration note

There are still documents and codepaths that reflect the older file-first worldview.

The current live deployment reality is already beyond that:

* event persistence is active
* dashboard reads are partly Postgres-first
* forensic truth is shifting away from logs and ad-hoc runtime files

So when deployment and debugging instructions conflict, prefer:

1. current code behavior
2. `ARCHITECTURE.md`
3. event schema docs
4. older operational notes

---

## Operational verification pattern

For live verification, the preferred pattern is:

1. verify code in repo
2. push to Git
3. deploy
4. run targeted live check
5. inspect Postgres tables
6. inspect runtime logs only as secondary evidence

This is especially important for event-chain work.

---

## Open deployment tasks

* confirm first real KuCoin `execution_event` after deploy
* fully wire `signal_events`
* improve event linkage (`source_action_event_id`, `source_signal_event_id`)
* reduce legacy runtime-file dependence
* keep runbooks aligned with actual service topology

