# Railway Multi-Bot Deployment (4 KuCoin Sub-Accounts)

This guide sets up **four parallel live bots**, each on its own KuCoin Futures sub-account, running different strategy profile variants.

You provide four API key triples (one per sub-account). Everything else is configured per Railway service.

---

## Strategy profiles

| Service name (suggested) | `BOT_PROFILE` | `BOT_INSTANCE_ID` | What it does |
|---|---|---|---|
| `bot-canonical` | `canonical` | `canonical` | Production dual-regime: daily gate toggles flip ↔ TP2 |
| `bot-countertrend` | `countertrend` | `countertrend` | Always countertrend/flip; SL exits flat in WAIT |
| `bot-countertrend-sl-reverse` | `countertrend_sl_reverse` | `countertrend-sl-reverse` | Always countertrend/flip; SL reverses position in WAIT |
| `bot-pc3axis` | `pc3axis` | `pc3axis` | PC 3-axis state-space gate (strict 3-of-3) |

All four trade the same symbol (typically `SOL-USDT`) but on **separate KuCoin sub-accounts**. Isolated state paths do **not** isolate exchange positions — separate API keys are required.

---

## Shared infrastructure (unchanged)

Keep these existing Railway services running:

| Service | Purpose |
|---|---|
| **quant** (web/API) | Dashboard only — disable embedded `ENABLE_LIVE_EXECUTOR` / `ENABLE_LIVE_SIGNAL_WORKER` when bots run separately |
| **DATABOT** | Shared Renko → Redis + Postgres |
| **gate-cron** | Daily CHOP/ADX/ER gate → Postgres + Redis (used by `canonical` profile) |

---

## Create four bot services

For **each** bot, create a new Railway service from the same GitHub repo/branch.

### Start command

```bash
python -u -m quant.execution.railway_bot
```

Or reference the Procfile process name if Railway supports it (`bot-canonical`, etc.).

### Mount a volume

Attach a Railway volume at `/data` so bot state survives restarts:

- `/data/live/bots/<BOT_INSTANCE_ID>/` — signals, executor state, events

---

## Environment variables (per bot service)

Copy this checklist for each service. Replace placeholders with your sub-account credentials.

### Identity (required — different per service)

```bash
BOT_PROFILE=canonical                    # or countertrend | countertrend_sl_reverse | pc3axis
BOT_INSTANCE_ID=canonical                  # must match table above; used in Postgres as strategy_instance
LIVE_SYMBOL=SOL-USDT
```

### KuCoin sub-account (required — unique per service)

```bash
KUCOIN_FUTURES_API_KEY=<your-subaccount-key>
KUCOIN_FUTURES_API_SECRET=<your-subaccount-secret>
KUCOIN_FUTURES_PASSPHRASE=<your-subaccount-passphrase>
```

### Shared database / cache

```bash
POSTGRES_URL=<same-as-other-services>
REDIS_URL=<same-as-other-services>
```

### Live safety (start dry, then go live)

```bash
LIVE_TRADING_ENABLED=0          # set 1 when ready
LIVE_EXECUTOR_DRY_RUN=1         # set 0 for real orders
LIVE_EXECUTOR_SYMBOL_ALLOWLIST=SOL-USDT
LIVE_EXECUTOR_POS_PCT=0.90      # adjust per account size
LIVE_EXECUTOR_LEVERAGE=1
KUCOIN_FUTURES_ORDER_LEVERAGE=6
PYTHONUNBUFFERED=1
```

### Optional overrides

Countertrend profiles get these backtest defaults automatically (override if needed):

```bash
LIVE_IMBA_LOOKBACK=150
LIVE_FLIP_TTP_TRAIL_PCT=0.0025
LIVE_FLIP_MIN_SL_PCT=0.010
LIVE_FLIP_MAX_SL_PCT=0.080
LIVE_FLIP_SWING_LOOKBACK=180
```

PC 3-axis profile only:

```bash
PC3AXIS_STATE_SPACE_PATH=/data/live/state_space_latest.parquet
PC3AXIS_DRIFT_ABS_Q=0.55
PC3AXIS_ELASTICITY_Q=0.25
PC3AXIS_INSTABILITY_Q=0.35
PC3AXIS_LOOKBACK_ROWS=4000
```

---

## Rollout procedure

### Phase 1 — Deploy dry-run (all four bots)

1. Deploy all four services with `LIVE_EXECUTOR_DRY_RUN=1`.
2. Confirm each service starts: logs should show `starting railway bot instance=... profile=...`.
3. Verify isolated paths on volume: `/data/live/bots/<instance>/`.
4. Check Postgres `action_events` — each bot should write with distinct `strategy_instance` (`canonical`, `countertrend`, etc.).

### Phase 2 — Fund sub-accounts

Transfer a small amount to each KuCoin Futures sub-account. Keep sizing conservative initially (`LIVE_EXECUTOR_POS_PCT`).

### Phase 3 — Go live (one bot at a time)

For each bot, in order of risk preference:

1. Set `LIVE_TRADING_ENABLED=1` and `LIVE_EXECUTOR_DRY_RUN=0`.
2. Watch logs and Postgres for the first `execution_event`.
3. Confirm position on KuCoin sub-account matches executor state.
4. Repeat for next bot.

### Rollback

Per service:

```bash
LIVE_EXECUTOR_DRY_RUN=1
LIVE_TRADING_ENABLED=0
```

---

## Monitoring in Postgres

Each bot tags events with `strategy_instance = BOT_INSTANCE_ID`:

```sql
SELECT strategy_instance, count(*)
FROM action_events
WHERE venue = 'kucoin'
  AND ts > now() - interval '24 hours'
GROUP BY 1
ORDER BY 1;
```

```sql
SELECT strategy_instance, symbol, engine_action, blocked, ts
FROM action_events
WHERE strategy_instance = 'countertrend'
ORDER BY ts DESC
LIMIT 20;
```

---

## Disable embedded workers on dashboard service

On the main `quant` web service, prevent duplicate signal/execution loops:

```bash
ENABLE_LIVE_SIGNAL_WORKER=0
ENABLE_LIVE_EXECUTOR=0
```

The dashboard continues to read Postgres and shared Renko/gate data.

---

## What you need to send

When ready, provide four KuCoin Futures API credential sets. Map each to a service:

| Sub-account | Service | `BOT_PROFILE` |
|---|---|---|
| Sub-account 1 | `bot-canonical` | `canonical` |
| Sub-account 2 | `bot-countertrend` | `countertrend` |
| Sub-account 3 | `bot-countertrend-sl-reverse` | `countertrend_sl_reverse` |
| Sub-account 4 | `bot-pc3axis` | `pc3axis` |

Do **not** commit API keys to the repo. Set them only in Railway service variables.
