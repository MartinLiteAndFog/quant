# Railway Deployment

This document explains how to deploy the Quant stack on Railway at the current stage of the project.

It is focused on:

- connecting the repo
- using Dockerfile-based deployment
- setting variables
- understanding service roles
- understanding the current persistence model

For operational procedures, see:
- `docs/RAILWAY_RUNBOOK.md`

For system structure, see:
- `docs/ARCHITECTURE.md`

---

## 1. Deployment model

Railway should be treated as a host for a small set of cooperating services, not just a single web app.

Typical current roles are:

- dashboard / API service
- signal worker
- execution worker
- optionally Kraken-specific execution path

The exact service naming can vary, but the architectural roles should stay clear.

---

## 2. Connect the repository

In Railway:

1. **New Project**
2. **Deploy from GitHub repo**
3. choose the `quant` repository
4. deploy from the desired branch, usually `main`

If the repo contains a `Dockerfile`, prefer Dockerfile-based deployment over ad-hoc build guessing.

---

## 3. Build and start

## Use the Dockerfile

If Railway detects the `Dockerfile`, build and runtime use the same environment and dependency set.

That is preferred because it avoids classically broken situations such as:
- missing `uvicorn`
- different Python environments between build and runtime
- wrong package resolution

In Railway:
- keep Build Command / Start Command empty if Dockerfile is used
- verify that Dockerfile is selected as the build method

Root directory should normally stay at repo root.

---

## 4. Service split

## A. Web / dashboard service

Typical responsibility:
- serve `/dashboard`
- serve `/api/*`
- expose public endpoints
- provide operational visibility

Typical start pattern:

```bash
uvicorn quant.execution.webhook_server:app --host 0.0.0.0 --port $PORT


B. Signal worker

Typical responsibility:

generate live signals

write routed signal streams

C. Execution worker

Typical responsibility:

run live_executor

place venue orders through OMS/broker

write events and runtime state

D. Kraken bot service (if deployed separately)

Typical responsibility:

run kraken_bot

handle Kraken-side live execution and persistence

Operational preference:
separate services are usually better than one giant mixed process because they improve:

restart safety

log clarity

debugging

operational reasoning

### DATABOT service (data pipeline)

Typical responsibility:
- Produce raw market data (Renko bricks) for all trading bots
- Write to Redis (`renko:{SYM}:latest`, `renko:{SYM}:events` stream)
- Write to Postgres (`live_renko_bricks` table)
- Provide health endpoint for monitoring

Typical start command:

```bash
python -m databot.main
```

Uses `Dockerfile.databot` instead of the main `Dockerfile`.

Required environment variables:
- `KUCOIN_FUTURES_API_KEY`
- `KUCOIN_FUTURES_API_SECRET`
- `KUCOIN_FUTURES_PASSPHRASE`
- `REDIS_URL`
- `POSTGRES_URL` or `DATABASE_URL`
- `DATABOT_SYMBOLS` (default: `SOL-USDT`, comma-separated for multi-symbol)
- `DATABOT_RENKO_BOX` (default: `0.1`)
- `DATABOT_RENKO_DAYS_BACK` (default: `14`)
- `DATABOT_POLL_SEC` (default: `60`)

Notes:
- DATABOT runs alongside the existing quant service during migration
- Once verified, disable the embedded Renko updater in the quant service by setting `ENABLE_DASHBOARD_RENKO_UPDATER=0`
- DATABOT does NOT compute gates or signals — those remain with strategy services
- `kraken_bot.py` is deprecated; use `live_executor_2.py` for Kraken execution

5. Environment variables

Set secrets and runtime variables in Railway Variables, not in committed files.

Core KuCoin variables

KUCOIN_FUTURES_API_KEY

KUCOIN_FUTURES_API_SECRET

KUCOIN_FUTURES_PASSPHRASE

Use Futures permissions, not Spot-only credentials.

Optional auth/security

WEBHOOK_TOKEN

Common runtime variables

Examples:

PYTHONUNBUFFERED=1

LIVE_TRADING_ENABLED

LIVE_EXECUTOR_DRY_RUN

LIVE_EXECUTOR_LEVERAGE

LIVE_EXECUTOR_SYMBOL_ALLOWLIST

SIGNALS_DIR

LIVE_EXECUTOR_STATE

The exact set depends on the service.

Daily gate runtime variables

For the cron service that builds the durable daily gate:

- `POSTGRES_URL` or `DATABASE_URL`
- `REDIS_URL`
- `LIVE_GATE_SYMBOL`
- `LIVE_GATE_PRIMARY`

Recommended cron start command:

```bash
bash -lc "python -u scripts/build_live_daily_gate_artifacts.py --symbol SOL-USDT && python -u -m quant.execution.equity_retention --apply"
```

Operational note:
- the durable source of truth is now Postgres `daily_gate_history`
- the durable Renko input for gate creation is now Postgres `live_renko_bricks`
- Redis carries the latest cached snapshot for fast reads
- the producer computes the winner-lineage `CHOP + ADX + ER` daily 2-of-3 gate from Renko OHLC
- local gate CSV paths are optional debug outputs only, not the live reader input anymore
- recommended setting: `LIVE_GATE_PRIMARY=on` so `gate_on` continues to mean countertrend

Renko updater note:
- **DATABOT** is the preferred Renko producer going forward (see [DATABOT service (data pipeline)](#databot-service-data-pipeline) above)
- the service running the Renko updater should still keep its local `DASHBOARD_RENKO_PARQUET` cache for existing readers
- but it must also have `POSTGRES_URL` so it can mirror the Renko bricks into `live_renko_bricks` for the cron gate builder

6. Public domain

In Railway:

go to service settings

enable/generate public domain for the web-facing service

Typical public endpoints may include:

/dashboard

/health

/api/status

/api/dashboard/chart

/api/gate/...

/api/signals/latest/...

Only the web/API service should normally need public networking.

7. Persistence on Railway

Railway containers are ephemeral.

That means local container filesystem state is not durable unless backed by:

a Railway volume

or an external durable store

Important practical rule

There are currently two persistence layers:

A. Runtime / operational traces

Examples:

signal JSONL

event JSONL

active levels JSON

Renko cache parquet

state JSON files

These may need a persistent volume if you want them to survive redeploys.

B. Durable forensic truth

This is increasingly moving to Postgres:

action_events

execution_events

closed_trades

equity_snapshots

This is the preferred long-term truth layer.

8. Current persistence direction

Historically, the system relied heavily on:

local files

JSONL traces

runtime state

dashboard-specific caches

Current direction is now:

Postgres-first for forensic truth

This means:

runtime files still matter operationally

but incident reconstruction should increasingly be done from Postgres

JSONL is still emitted as transitional trace

not every document that speaks file-first is still fully current

9. Live execution safety

Before enabling real orders:

Recommended safe settings:

LIVE_TRADING_ENABLED=0

LIVE_EXECUTOR_DRY_RUN=1

Go-live sequence:

deploy dry-run

verify logs and behavior

verify venue/account reads

set LIVE_TRADING_ENABLED=1

set LIVE_EXECUTOR_DRY_RUN=0

Rollback:

set LIVE_EXECUTOR_DRY_RUN=1

and/or set LIVE_TRADING_ENABLED=0

Do not flip to full live execution without a dry-run validation phase.

10. Railway-specific operational caveats

Railway shells can be minimal.

You may not always have:

ps

rg

curl

In those cases:

use Python one-liners

use /proc-based inspection if needed

prefer explicit DB queries over shell-heavy debugging

Also note:
copy/paste of here-doc blocks in Railway SSH can render oddly in some terminals.
If that becomes annoying, prefer Python -c one-liners.

11. Verification pattern

The preferred verification workflow is:

fix in repo

compile/check locally

commit

push

let Railway deploy

verify in live service

verify Postgres persistence

only then inspect logs as secondary evidence

Avoid hotpatching running containers unless absolutely necessary.

12. Current known state

At the current stage:

KuCoin action_events from live_executor are live and verified in Postgres

KuCoin execution_events are wired and deployed, but still await confirmation on the next real execution-triggered event

Kraken event/equity persistence is active

several dashboard readers are already Postgres-first

signal_events are not yet fully live-wired end-to-end

13. Minimal deployment checklist

 repo connected in Railway

 Dockerfile used for build/runtime

 required venue credentials set

 web service domain generated

 dashboard reachable

 API endpoints reachable

 dry-run execution variables set safely

 persistent volume attached if runtime files must survive redeploy

 Postgres connectivity confirmed where required

