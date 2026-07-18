# SOL KuCoin Micro-Pilot

## Objective

Run three deliberately small SOL-USDT Futures strategy variants on three separate KuCoin subaccounts. Each account is treated as a live execution experiment, not as evidence that the strategy is validated.

## Frozen pilot variants

| Railway service | Profile | Hypothesis |
| --- | --- | --- |
| `sol-pilot-canonical` | `canonical` | The confirmed daily CHOP/ADX/ER gate can choose between countertrend flip and trend-following TP2 more reliably than a permanently forced mode. |
| `sol-pilot-countertrend` | `countertrend` | The original IMBA/Renko countertrend behavior can be measured with real fills, even though the strict-fill historical variant was rejected. |
| `sol-pilot-pc3axis` | `pc3axis` | A strict 3-of-3 state-space gate can reduce weak countertrend exposure without tuning on the live window. |

Do not enable `countertrend_sl_reverse` in this three-account pilot. It adds another reversal rule without enough independent evidence and would make attribution harder.

## Hard account limits

Every service must set `MICRO_PILOT_MODE=1`. The launcher then fails closed unless all of these conditions hold:

- SOL-USDT only.
- Isolated margin only; cross-margin fallback is disabled.
- Executor leverage and KuCoin order leverage match and do not exceed 3x.
- At most 5 USDT of account equity is used for sizing.
- At most one SOLUSDTM contract is ordered.
- Live trading remains disabled by default.

At the 2026-07-18 contract specification, one SOLUSDTM contract represents 0.1 SOL. The live contract endpoint remains the source of truth; the executor reads the multiplier from KuCoin.

## Shared Railway variables

Set these on all three services:

```text
RAILWAY_PROCESS=bot
MICRO_PILOT_MODE=1
LIVE_SYMBOL=SOL-USDT
LIVE_EXECUTOR_SYMBOL_ALLOWLIST=SOL-USDT
LIVE_TRADING_ENABLED=0
LIVE_EXECUTOR_DRY_RUN=1
LIVE_EXECUTOR_LEVERAGE=3
KUCOIN_FUTURES_ORDER_LEVERAGE=3
KUCOIN_FUTURES_MARGIN_MODE=isolated
KUCOIN_FUTURES_STRICT_MARGIN_MODE=1
LIVE_EXECUTOR_MAX_MARGIN_USDT=5
LIVE_EXECUTOR_MAX_CONTRACTS=1
LIVE_EXECUTOR_MAX_LEVERAGE=3
LIVE_EXECUTOR_POS_PCT=1.0
PYTHONUNBUFFERED=1
```

Each service also needs its own `BOT_PROFILE`, `BOT_INSTANCE_ID`, and unique KuCoin Futures API key, secret, and passphrase. API secrets belong only in Railway variables, never in Git or chat.

Use the same Postgres and Redis services for all three bots. The executor already writes `strategy_instance` into action and execution events, so the dashboard can compare the profiles without merging their account state.

## Rollout gates

1. Deploy all three services in dry-run mode.
2. Verify that each service receives Renko data, emits a distinct `strategy_instance`, and proposes no more than one contract.
3. Confirm through the KuCoin API that each subaccount uses isolated margin and has no unrelated open orders or positions.
4. Enable one account at a time by changing both `LIVE_TRADING_ENABLED=1` and `LIVE_EXECUTOR_DRY_RUN=0`.
5. After the first real entry and exit, reconcile order, fill, fee, position, and Postgres events before enabling the next account.

## Stop conditions

Return the affected service to dry-run immediately if any of these occurs:

- an order exceeds one contract;
- leverage is not exactly 3x;
- margin mode is not isolated;
- live position and executor state disagree after the verification delay;
- a duplicate entry is submitted for the same decision;
- a native protective stop is absent after an entry;
- cumulative realised loss reaches the funded micro-account amount;
- data is stale or a source gap prevents a causal signal decision.

## Railway and Kraken analysis

Railway/Postgres is the common observation layer. Compare each `strategy_instance` on:

- decision-to-submit, submit-to-acknowledge, and submit-to-fill time;
- signal price, bid/ask at submission, average fill, and slippage in basis points;
- actual maker/taker fee and funding;
- missed or blocked actions;
- realised PnL, drawdown, and position/state mismatches.

Kraken stays a comparison venue during the pilot. Mirror the same strategy decisions into Kraken's existing dry-run/calibration path and compare executable quotes and estimated fills. Do not place a second live Kraken position until the KuCoin micro-pilot has produced reconciled evidence and a separate funding decision is made.

## Interpretation rule

Three tiny live accounts can reveal execution and integration defects. They cannot establish profitability. No parameters may be tuned from the opened live window; changes require a new versioned pilot.
