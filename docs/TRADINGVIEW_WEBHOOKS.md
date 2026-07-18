# TradingView Webhooks — Per-Bot Addresses

Each pilot bot runs its own webhook receiver on its own Railway public domain,
bound to its own KuCoin Futures sub-account. TradingView is the **sole** source
of buy/sell in this mode — `TV_WEBHOOK_ENABLED=1` stops `railway_bot` from
starting the internal Renko `live_signal_worker`, so there is exactly one
controller per sub-account.

---

## Webhook addresses

| Bot | `BOT_PROFILE` | Webhook URL |
|---|---|---|
| Countertrend | `countertrend` | `https://sol-pilot-countertrend-production.up.railway.app/webhook/tv-execute` |
| Countertrend SL-reverse | `countertrend_sl_reverse` | `https://sol-pilot-countertrend-sl-reverse-production.up.railway.app/webhook/tv-execute` |
| PC 3-axis | `pc3axis` | `https://sol-pilot-pc3axis-production.up.railway.app/webhook/tv-execute` |
| Canonical | `canonical` | `https://sol-pilot-canonical-production.up.railway.app/webhook/tv-execute` |

Health check (no auth): `GET https://<domain>/health` — returns profile,
instance, whether the executor is ready, and whether it is in dry-run.

---

## Auth

TradingView cannot send custom headers, so the token goes **in the JSON body**
as `"token"`. Each bot has its own token in its Railway variable
`BOT_WEBHOOK_TOKEN` — read it from the Railway dashboard for that service.

A request with a missing or wrong token is rejected with `401`. If
`BOT_WEBHOOK_TOKEN` is unset the endpoint returns `503` rather than accepting
unauthenticated orders (fail-closed).

---

## Alert message format

Paste into the TradingView alert's **Message** box, and put the bot's URL in
the **Webhook URL** box.

Open long:

```json
{"token": "<THIS_BOT_TOKEN>", "action": "entry", "side": "buy"}
```

Open short:

```json
{"token": "<THIS_BOT_TOKEN>", "action": "entry", "side": "sell"}
```

Close the position:

```json
{"token": "<THIS_BOT_TOKEN>", "action": "exit"}
```

Reverse (close and open the other way):

```json
{"token": "<THIS_BOT_TOKEN>", "action": "flip", "side": "buy"}
```

Valid `action` values: `entry`, `exit`, `flip`, `tp1`, `tp2`, `sl`.
`side` (`buy` | `sell`) is required for `entry` and `flip` only.

`symbol` is optional — it defaults to the service's `LIVE_SYMBOL`
(`SOL-USDT`). Only send it if the alert trades something else.

### Optional: guard against cross-firing

If you ever reuse one alert template across bots, add a `bot` field. A bot
ignores any alert addressed to a different instance and replies
`{"skipped": "bot_mismatch"}`:

```json
{"token": "<THIS_BOT_TOKEN>", "action": "entry", "side": "buy", "bot": "sol-pilot-pc3axis"}
```

Matches on either `BOT_INSTANCE_ID` or `BOT_PROFILE`.

---

## Contract sizes — the 10x Kraken/KuCoin difference

Verified live from both exchange APIs:

| Venue | Instrument | 1 contract | Max leverage |
|---|---|---|---|
| KuCoin | `SOLUSDTM` | **0.1 SOL** | **75x** |
| Kraken | `PF_SOLUSD` | **1 SOL** | — |

The same contract count is **10x more exposure on Kraken than on KuCoin**.
This is a real venue difference, not a bug — but the code made it into one:
`KucoinFuturesBroker.get_contract_multiplier` silently returned `1.0` (the
Kraken convention) whenever the contracts API call failed, which oversized
every KuCoin order by exactly 10x.

It now raises instead, so the caller falls back to the correct configured
default `LIVE_EXECUTOR_CONTRACT_MULTIPLIER=0.1`. Fallbacks here must always
err **low** — undersizing costs opportunity, oversizing costs the account.
Regression tests: `tests/test_contract_multiplier_sizing.py`.

## Position sizing

Pilot accounts are funded with **$15** and trade **90% of equity**.

| Leverage | SOL $100 | SOL $150 | SOL $200 | SOL $250 |
|---|---|---|---|---|
| 3x | 4 contracts (0.4 SOL) | 2 (0.2) | 2 (0.2) | 1 (0.1) |
| 10x | 13 (1.3) | 9 (0.9) | 6 (0.6) | 5 (0.5) |

Margin used stays at or below $13.50 (90% of $15) in every case.

The webhook path uses the same caps as the internal executor
(`_live_order_qty`), so `LIVE_EXECUTOR_MAX_MARGIN_USDT` and
`LIVE_EXECUTOR_MAX_CONTRACTS` bind on TradingView-driven orders too.
`railway_bot` derives `TV_EXEC_POS_PCT` / `TV_EXEC_LEVERAGE` from
`LIVE_EXECUTOR_POS_PCT` / `LIVE_EXECUTOR_LEVERAGE` at startup, instead of their
own unrelated defaults (`0.50` / `10x`).

## Leverage — why 10x never worked

**3x was never an exchange limit.** KuCoin allows 75x on SOL-USDT. The blocker
was a hard-coded check in `railway_bot.py` that raised
`"micro pilot leverage cap must not exceed 3"` whenever
`LIVE_EXECUTOR_MAX_LEVERAGE > 3`.

Those bounds are now configurable ceilings, defaulting to:

| Ceiling | Env | Default |
|---|---|---|
| Leverage | `MICRO_PILOT_LEVERAGE_CEILING` | 10 |
| Margin | `MICRO_PILOT_MARGIN_CEILING_USDT` | 20 |
| Contracts | `MICRO_PILOT_CONTRACTS_CEILING` | 50 |

Services currently run **3x**. To move a bot to 10x, set all three together —
startup fails if they disagree:

```bash
LIVE_EXECUTOR_LEVERAGE=10
LIVE_EXECUTOR_MAX_LEVERAGE=10
KUCOIN_FUTURES_ORDER_LEVERAGE=10
```

Going above 10x additionally requires raising `MICRO_PILOT_LEVERAGE_CEILING`.

## Flips

A flip is executed as **two legs, not one net-off order**:

1. Close the existing position (`reduce_only`).
2. Wait `TV_EXEC_FLIP_DELAY_SEC` (default **2.0s**).
3. Re-check the position — if anything is still open, **abort without
   reversing** and log `close_leg_incomplete`.
4. Open the opposite side.

Each leg writes its own execution event (`tv_flip_close`, `tv_flip_open`), so
both fills are separately analysable.

---

## Go-live sequence

All four services are currently **fail-closed**:

```bash
LIVE_TRADING_ENABLED=0
LIVE_EXECUTOR_DRY_RUN=1
TV_EXEC_DRY_RUN=1
```

1. Paste the KuCoin sub-account credentials into each service
   (`KUCOIN_FUTURES_API_KEY`, `KUCOIN_FUTURES_API_SECRET`,
   `KUCOIN_FUTURES_PASSPHRASE` — currently `PASTE_ME`).
2. Deploy and confirm `GET /health` returns `executor_ready: true`.
3. Fire a test alert from TradingView; confirm a `200` and a logged decision
   while still in dry-run.
4. Go live **one bot at a time**: set `LIVE_TRADING_ENABLED=1`,
   `LIVE_EXECUTOR_DRY_RUN=0`, `TV_EXEC_DRY_RUN=0`.
5. Confirm the position on that KuCoin sub-account before starting the next.

Rollback per service: set `TV_EXEC_DRY_RUN=1` and `LIVE_TRADING_ENABLED=0`.

---

## Monitoring and fill analysis

Every bot tags both `action_events` and `execution_events` with
`strategy_instance = BOT_INSTANCE_ID` and `config_hash = <instance>_<profile>_v1`,
so the four sub-accounts stay separable.

> Two fixes were required to make this work. The TV executor previously
> hard-coded `strategy_instance = "tv_executor"`, which would have merged all
> four bots into one indistinguishable stream. And `insert_execution_event`
> referenced `%(price)s` / `%(reject_reason)s` that the TV executor never
> supplied, so **every** TradingView execution event failed to bind and was
> swallowed by an except-and-warn handler — nothing reached Postgres at all.
> Migration `003_execution_events_instance.sql` adds the columns and a
> `execution_fill_analysis` view.

Activity per bot:

```sql
SELECT strategy_instance, count(*)
FROM action_events
WHERE venue = 'kucoin' AND ts > now() - interval '24 hours'
GROUP BY 1 ORDER BY 1;
```

Fill quality / slippage per bot — every execution event records the mid, bid and
ask at decision time:

```sql
SELECT strategy_instance,
       execution_stage,
       side,
       count(*)                              AS legs,
       round(avg(qty), 2)                    AS avg_qty,
       round(avg(ask - bid), 4)              AS avg_spread,
       round(avg(reference_price), 3)        AS avg_ref_price
FROM execution_fill_analysis
WHERE ts > now() - interval '7 days'
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3;
```

Flip legs paired up (close vs open), useful for measuring the cost of the 2s gap:

```sql
SELECT strategy_instance, execution_stage, count(*), round(avg(reference_price), 3)
FROM execution_fill_analysis
WHERE execution_stage = 'market_fill'
  AND (payload_json ->> 'reason_code') IN ('tv_flip_close', 'tv_flip_open')
GROUP BY 1, 2;
```
