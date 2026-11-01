# Pilot account selection: Blaupunkt (2026-11-01)

## Operator choice

The first Strategy Family pilot is the existing KuCoin Futures Blaupunkt
subaccount. The user selected this account explicitly. Keep TradingView as
the signal source until independent live-signal parity is demonstrated.

## Railway inventory (read-only, November 1 2026)

- Project QUANTUM: a7c1a403-f106-4853-86d7-b9faa30a65cf
- Service Blaupunkt: f98cba4d-7e6f-4511-8528-cff4af6b502b
- Environment production: aa9c2cd9-2384-4088-a2ed-754510249b7d
- Repo and source: MartinLiteAndFog/quant, branch main.
- Start command: python -u -m quant.execution.bot_webhook
- Health URL: https://blaupunkt-production.up.railway.app/health
- Attached persistent volume: /data
- Latest deployment successful on Nov 1, commit e9a1b28.
- Configured variable NAMES include KuCoin credentials, BOT_INSTANCE_ID,
  BOT_PROFILE, TV_WEBHOOK_ENABLED, LIVE_TRADING_ENABLED, TV_EXEC_DRY_RUN
  and margin settings. VALUES ARE REDACTED. We have not confirmed whether
  live trading is enabled or which margin or position mode is active.
- Deploy logs show frequent successful health polls. Health polls are NOT
  evidence that there are no open positions or conditional orders.
- The current bot_webhook module exposes token-guarded read-only KuCoin
  diagnostic endpoints. Those need user authorization to query; no token
  has been requested or disclosed.

## Initial mapping

family_id: blaupunkt
parent_instance: blaupunkt (confirm against actual health/diag)
venue: kucoin_futures
venue_symbol: SOLUSDTM
signal_source: tradingview_webhook
account_binding: existing Blaupunkt KuCoin Futures subaccount
child_trading_enabled: false
family_hedge_mode_switch_enabled: false

## Requirements before any live mode switch

1. Authenticated read-only query of all current positions and open orders,
   including conditional orders. Verify whether any other trader or bot
   can submit orders to the same subaccount.
2. Confirm venue position mode, margin mode and isolated/cross behavior.
   Never switch account mode with an open position or pending order.
3. Persist bot intents, fills, fee attribution and virtual positions to
   PostgreSQL with transactional idempotency and crash-safe replay.
4. Integrate a hedge-aware KuCoin API adapter and single order-writing owner
   for this subaccount. Existing signed net-position readers are NOT
   hedge-aware and cannot be used for multi-bot execution.
5. Implement and test per-bot stop/TP lifecycle, overclose prevention,
   venue-vs-virtual side-total reconciliation, and family risk limits.
6. Feed the TradingView alerts into the new family manager in no-order
   rehearsal mode; verify every event belongs to the right bot and family.
7. Separate approval for first live child and the required maintenance
   window to change position mode. No automatic flattening.
8. Validate one real minimal entry and its independent safe exit under
   explicit authorization before adding more children.

## Deployment status

NOT DEPLOYED. Draft PR #22 is a code prototype and plans only. This change
must not modify Blaupunkt live variables, its TradingView alarms, exchange
orders, open position, balances, or Railway deployment.
