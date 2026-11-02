# Blaupunkt multi-strategy hedge mode (opt-in)

## Accounting

- Actual, independent strategy key: webhook JSON `bot` (`Blaupunkt`, `ElliotP1`, ...).
- **Trading pool** = current KuCoin Futures account equity × 90%. **Reserve** = 10%.
- N = registered strategy IDs, including idle bots.
- **Target margin per bot** = trading pool / N (e.g. two = 45% total equity each).
- Market orders additionally constrained by current exchange **available margin minus the single shared reserve**. No repeated 90% reduction for each bot.
- Leverage multiplies **notional**, not the budgeted margin.
- New bots auto-register on their first authenticated webhook. No forced resizing of already open positions when N changes.
- Side/quantity attribution is stored in the persistent service volume, independently of exchange net same-side legs.

## Existing TradingView payloads

```json
{"bot":"Blaupunkt","action":"flip","side":"buy","symbol":"XBT-USDT"}
```

```json
{"bot":"ElliotP1","action":"flip","side":"sell","symbol":"XBT-USDT"}
```

Supported actions: `entry`, `flip`, `exit`, `tp1`, `tp2`, `sl`.
For `entry` or `flip`, include `side=buy|sell`. Position ownership is never inferred from the exchange net position alone.

## Activation preflight — DO NOT SKIP

The old executor uses one signed net position. Never activate hedge fleet while old executor (or another Railway replica) is acting on the same KuCoin sub-account.

1. Verify the **correct KuCoin Futures sub-account** and that there are no legacy/unattributed XBTUSDTM positions or stop orders. Reconcile/remove these manually — never automatically flatten a live account.
2. In KuCoin Futures, select **Hedge Position Mode** (account-wide `positionMode=1`). It generally cannot be changed while incompatible positions/orders exist. This code deliberately refuses to switch account mode automatically.
3. Keep exactly one Blaupunkt Railway replica with a persistent `/data` mount.
4. Use **authenticated webhooks** (no anonymous bypass). Put `BOT_WEBHOOK_TOKEN` in the TradingView webhook URL query (`?token=...`) or JSON body `token`. Keep `BOT_WEBHOOK_ALLOW_ANON=0`.
5. Configure only the Blaupunkt Railway service:
   ```env
   HEDGE_MULTI_BOT_ENABLED=1
   HEDGE_BOT_IDS=Blaupunkt,ElliotP1
   HEDGE_AUTO_REGISTER=1
   HEDGE_AUTO_SET_POSITION_MODE=1
   HEDGE_SYMBOL=XBT-USDT
   HEDGE_BOT_CAPITAL_USAGE=0.90
   HEDGE_STATE_FILE=/data/blaupunkt-hedge.json
   HEDGE_WEBHOOK_QUEUE_FILE=/data/blaupunkt-webhooks.sqlite3
   BOT_WEBHOOK_ALLOW_ANON=0
   TV_EXEC_DRY_RUN=1
   LIVE_TRADING_ENABLED=0
   ```
   Keep the existing KuCoin API keys, margin mode, order leverage and health-check.
6. Verify `GET /hedge/status?token=...` shows pool 90%, reserve 10%, both IDs and the proper 45% total-equity allocation each.
7. Run dry-run webhook tests with both IDs, then check exchange mode & **actual** isolated-hedge order acceptance with a controlled minimal size before changing `TV_EXEC_DRY_RUN=0` and `LIVE_TRADING_ENABLED=1`.
8. Review alerts, position reconciliation and stop orders after the first filled orders. Never force-clear the pending-order or ownership ledger while live exposure exists.

Hedge mode allows LONG and SHORT legs simultaneously, **not** a separate exchange position per bot. The durable virtual ledger tracks two same-side strategy allocations inside the combined exchange side, and reconciles sums against the exchange before every action. A mismatch fails closed. Each strategy cancels its **own** recorded stop order, not every stop on the contract.

No migration of pre-existing XBTUSDTM exposure is attempted. Protect other running bots by restricting these environment changes to Blaupunkt only.

## Automated checks

`python -m pytest -q tests/test_blaupunkt_hedge.py`

Current tests use a fake broker; they do **not** confirm real KuCoin hedge exchange behavior, API credentials, funding/margin effects, or network execution. CI workflow: `.github/workflows/blaupunkt-hedge.yml`.

## Webhook acknowledgments / retry behavior

The authenticated webhook returns `{"ok":true,"queued":true,"event_id":123}` immediately after its local SQLite commit. This means **received and persisted**, *not* an executed trade. A single worker processes signals in order and checks KuCoin before every order. See `GET /hedge/events?token=...` to distinguish `executed`, `rejected` (including insufficient funds), and `failed` outcomes. Signals older than 120 seconds are rejected and never traded late. An interrupted in-flight order is quarantined and **never replayed automatically** until reconciled.

Do not expose the webhook token in public screenshots or logs. Never activate anonymous unauthenticated trading.
