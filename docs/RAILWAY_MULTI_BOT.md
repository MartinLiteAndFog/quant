# Railway Multi-Bot Deployment (KuCoin Sub-Accounts)

This guide sets up **parallel live bots**, each on its own KuCoin Futures sub-account,
running different strategy profile variants.

You provide one API key triple per sub-account. Everything else is configured per Railway service.

---

## Live services (production as of 2026-07)

| Railway service | Display | `BOT_PROFILE` | `BOT_INSTANCE_ID` (Postgres tag) | Mode |
|---|---|---|---|---|
| `sol-pilot-canonical` | Imba Runner | `canonical` | `sol-pilot-canonical` | TV webhook (`TV_WEBHOOK_ENABLED=1`) |
| `sol-pilot-IMBA5TTP` | Pure ImbaTP | `pc3axis` | `sol-pilot-pc3axis` | TV webhook |
| `sol-pilot-countertrend` | Countervariante | `countertrend` | `sol-pilot-countertrend` | TV webhook |
| `sol-pilot-countertrend-sl-reverse` | Counter SL Reverse | `countertrend_sl_reverse` | `sol-pilot-countertrend-sl-reverse` | dry / not live |
| `Kraken` | Kraken Legacy | — | `kraken_bot` | separate stack |
| `quant` | Dashboard + `/api/fleet/*` | — | should **not** execute | API only |

> Docs historically used short IDs (`canonical`, `countertrend`). **Live bots use
> `sol-pilot-*` IDs.** Fleet Desktop and `/api/fleet/*` key off the live IDs.

All KuCoin pilots trade `SOL-USDT` on **separate sub-accounts**. Isolated state paths do
**not** isolate exchange positions — separate API keys are required.

---

## Tracking / Postgres spine (important)

Each bot must tag rows with `strategy_instance = BOT_INSTANCE_ID`.

| Stream | Writer | TV-webhook pilots today |
|---|---|---|
| `action_events` | `tv_signal_executor` / `live_executor` | sparse but tagged |
| `execution_events` | same | sparse but tagged |
| `closed_trades` | **`live_executor` historically**; **`tv_signal_executor` as of fleet fix** | was **missing** while `TV_WEBHOOK_ENABLED=1` (live_executor not started) — now written on exit/flip/sl/tp2 |
| `trade_decisions` | decision backfill / live_executor | mostly flooded by `quant` if `ENABLE_LIVE_EXECUTOR=1` |

### Audit query

```sql
SELECT strategy_instance, count(*) AS n, max(exit_ts) AS last_exit
FROM closed_trades
WHERE exit_ts > now() - interval '30 days'
GROUP BY 1
ORDER BY n DESC;
```

```sql
SELECT strategy_instance, count(*) AS n, max(ts) AS last_ts
FROM action_events
WHERE ts > now() - interval '7 days'
GROUP BY 1
ORDER BY n DESC;
```

### Dashboard service hygiene

On `quant` (dashboard), keep workers off so they do not spam `strategy_instance='quant'`:

```bash
ENABLE_LIVE_SIGNAL_WORKER=0
ENABLE_LIVE_EXECUTOR=0
ENABLE_LIVE_FLIP_WORKER=0
LIVE_TRADING_ENABLED=0
```

Fleet Desktop reads curves from `/api/fleet/performance` on the `quant` host
(`https://quant-production-5533.up.railway.app`).

---

## Shared infrastructure

| Service | Purpose |
|---|---|
| **quant** | Dashboard + fleet API |
| **DATABOT** | Shared Renko → Redis + Postgres |
| **gate-cron** | Daily CHOP/ADX/ER gate (canonical profile) |
| **Postgres** | Shared `DATABASE_URL` |

---

## Create bot services

### Start command

```bash
python -u -m quant.execution.railway_bot
```

### Volume

Attach a Railway volume at `/data`:

- `/data/live/bots/<BOT_INSTANCE_ID>/` — signals, executor state, events

### Identity (required — different per service)

```bash
BOT_PROFILE=canonical                    # or countertrend | countertrend_sl_reverse | pc3axis
BOT_INSTANCE_ID=sol-pilot-canonical      # must match live table above
BOT_DISPLAY_NAME=Imba Runner             # optional friendly label
LIVE_SYMBOL=SOL-USDT
TV_WEBHOOK_ENABLED=1                     # TradingView-driven pilots
```

### KuCoin sub-account (required — unique per service)

```bash
KUCOIN_FUTURES_API_KEY=...
KUCOIN_FUTURES_API_SECRET=...
KUCOIN_FUTURES_PASSPHRASE=...
```

### Shared database / cache

```bash
DATABASE_URL=<same-as-other-services>
REDIS_URL=<same-as-other-services>
```

### Live safety

```bash
LIVE_TRADING_ENABLED=1
LIVE_EXECUTOR_DRY_RUN=0
LIVE_EXECUTOR_SYMBOL_ALLOWLIST=SOL-USDT
LIVE_EXECUTOR_POS_PCT=0.90
LIVE_EXECUTOR_LEVERAGE=10
PYTHONUNBUFFERED=1
```

---

## Rollout procedure

1. Deploy dry-run (`LIVE_EXECUTOR_DRY_RUN=1`) and confirm `/health` shows `instance` + `executor_ready`.
2. Confirm Postgres tags: `action_events.strategy_instance` matches `BOT_INSTANCE_ID`.
3. Fund sub-accounts; go live one bot at a time.
4. After each closed trade, confirm a `closed_trades` row for that instance (fleet curves).

### Rollback

```bash
LIVE_EXECUTOR_DRY_RUN=1
LIVE_TRADING_ENABLED=0
```

---

## What you need to send

| Sub-account | Service | `BOT_PROFILE` |
|---|---|---|
| Sub-account 1 | `sol-pilot-canonical` | `canonical` |
| Sub-account 2 | `sol-pilot-countertrend` | `countertrend` |
| Sub-account 3 | `sol-pilot-countertrend-sl-reverse` | `countertrend_sl_reverse` |
| Sub-account 4 | `sol-pilot-IMBA5TTP` | `pc3axis` |

Do **not** commit API keys. Set them only in Railway service variables.
