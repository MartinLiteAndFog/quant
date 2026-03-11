# Debug Playbook

This file contains the fastest recurring checks for live trading bugs.

Use these queries and checks before digging through logs.

---

# 1. Latest actions

Use this when you want to know what the engine decided.

```sql
select ts, strategy, venue, engine_action, reason_code, position_before, position_after
from action_events
order by ts desc
limit 20;

Questions this answers:

did an action happen?

what did the engine want to do?

was it enter / exit / flip / hold?

2. Latest executions

Use this when you want to know what actually happened at OMS / venue level.

select ts, venue, execution_stage, side, qty, price, status, reject_reason
from execution_events
order by ts desc
limit 20;

Questions this answers:

did anything execute?

was it a fill, reject, cancel, or fallback?

did the venue/OMS produce a real outcome?

3. Latest closed trades

Use this when you want realized outcome.

select exit_ts, venue, symbol, side, qty, entry_price, exit_price, pnl_pct, exit_event
from closed_trades
order by exit_ts desc
limit 20;

Questions this answers:

was a trade actually realized?

what was the PnL?

what exit type closed it?

4. Latest equity

Use this when the dashboard equity looks wrong.

select ts, venue, account, symbol, equity, source
from equity_snapshots
order by ts desc
limit 20;

Questions this answers:

is equity being written?

from which source?

is the dashboard using stale runtime state or real snapshots?

5. Action → execution gap

Use this when it feels like the engine decided something but no trade happened.

select ts, strategy, venue, engine_action, reason_code, position_before, position_after
from action_events
where strategy in ('live_executor', 'kraken_bot')
order by ts desc
limit 20;

Then compare with:

select ts, venue, execution_stage, side, qty, price, status
from execution_events
order by ts desc
limit 20;

Questions this answers:

was there an action without an execution?

is the problem in strategy/engine or in OMS/venue?

6. KuCoin live executor actions only
select ts, strategy, venue, engine_action, reason_code, position_before, position_after
from action_events
where strategy = 'live_executor'
order by ts desc
limit 20;

Use for:

KuCoin executor debugging

hold/exit/flip history

checking whether the executor is actually doing anything

7. KuCoin execution events only
select ts, venue, execution_stage, side, qty, price, status
from execution_events
where payload_json->>'strategy' = 'live_executor'
   or payload_json->>'strategy_instance' = 'live_executor'
order by ts desc
limit 20;

Use for:

confirming whether KuCoin execution persistence actually fired

distinguishing “writer not triggered” vs “writer broken”

8. Kraken execution events only
select ts, venue, execution_stage, side, qty, price, status
from execution_events
where venue = 'kraken'
order by ts desc
limit 20;

Use for:

Kraken bot forensic checks

confirming fill/reject activity on Kraken side

9. Preferred debugging order

Always debug in this order:

action_events

execution_events

closed_trades

equity_snapshots

only then logs / JSONL / runtime files

10. Quick interpretation guide
If action exists but execution does not

Likely issue in:

OMS

venue adapter

order placement

execution writer not triggered

dry-run / safety flags

If execution exists but closed trade does not

Likely issue in:

trade persistence

close recognition

dashboard/trade reader

trade attribution

If closed trade exists but equity looks wrong

Likely issue in:

equity snapshot writer

dashboard reader source priority

stale runtime fallback

aggregation / display logic

If nothing exists

Check:

signal path

gate path

service actually running

credentials

dry-run vs live flags