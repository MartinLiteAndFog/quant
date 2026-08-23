# Frozen Brain Forward Observer

This is a paper-only Railway service for `cost_aware_thousand_brains_five_minute_v1`.
It has no broker integration and refuses to start when `LIVE_TRADING_ENABLED` is enabled.

## Why it is separate from the Renko builder

The current Renko builder obtains KuCoin OHLC data and persists transformed Renko bricks. The frozen Brain requires native one-minute OHLCV **and taker-buy base volume** to calculate `flow_imbalance`. Renko bricks cannot reconstruct that information.

The observer therefore shares Railway Postgres with DATABOT but consumes closed Binance spot SOLUSDT klines, whose `taker buy base asset volume` is the exact field used by the research input. It does not write to Redis, signal workers, executors, exchange APIs, or live order tables.

## Railway service

Create a dedicated service from the same repository using `Dockerfile.brain-forward` and schedule it as a Railway Cron job:

```text
*/5 * * * *
```

Set only these variables:

```text
POSTGRES_URL=<the existing shared Railway Postgres URL>
LIVE_TRADING_ENABLED=0
LIVE_EXECUTOR_DRY_RUN=1
BRAIN_FORWARD_SYMBOL=SOL-USDT
LOG_LEVEL=INFO
```

The image also contains the immutable `forward_protocol.json`. The service verifies
the symbol, source, frozen artifact SHA-256 and byte hashes of the runtime, service
and store modules against it before writing evidence.
The preregistered schedule is:

- operational warmup: 19 July through 1 August 2026 UTC;
- formal locked evidence: 2 August through 30 October 2026 UTC;
- five-minute outcome-maturity tail and deterministic checkpoint at
  `2026-10-31T00:10:00Z`.

Warmup observations are labeled and excluded from the formal checkpoint. Missing
coverage, fewer than six formal trades, non-positive 95% lower confidence bound,
non-positive 22-bps stress expectancy, excessive drawdown, or any ledger identity
error fails closed. A pass permits shadow-champion review only and never enables
live orders.

Do not set exchange API credentials on this service. Do not reuse an existing executor service for it.

## v2 evidence restart

The original v1 epoch is not independent evidence: its warmup continuity cannot
be established. It must remain immutable and must not be extended. The separate
`brain-forward-v2` entrypoint uses
`forward_protocol_v2_20260824.json`, which pins its own entrypoint and checkpoint
code in addition to the frozen runtime, service and store.

Deploy v2 only as the same paper-only five-minute cron, before the fixed
`2026-08-24T00:00:00Z` warmup begins. Its warmup ends on 7 September 2026, its
formal epoch ends on 6 December 2026 and its checkpoint is fixed at
`2026-12-06T00:10:00Z`. It refuses a live-trading environment and must never be
attached to an exchange credential.

The v2 observer evaluates one frozen signal stream through three parallel,
strictly hypothetical entry policies:

- `immediate`: the original next-minute-open entry and comparison baseline;
- `stop_cooldown_3m`: the baseline entry, except entries during the three
  minutes following that variant's own stop are suppressed;
- `previous_high_confirmation`: wait up to five closed one-minute bars for a
  bar to trade above the immediately preceding bar's high. An opening gap is
  filled at the open; an intrabar break is filled at the prior high.

All variants retain the same five-bar conservative target/stop simulation and
14-bps round-trip cost. The confirmation bar is included in its five-bar exit
path, so same-bar target/stop ambiguity still resolves to the stop.

At or after that checkpoint, create one immutable local evidence bundle:

```text
brain-forward-export \
  --protocol src/quant/brain_forward/forward_protocol_v2_20260824.json \
  --as-of 2026-12-06T00:10:00Z \
  --output-dir research/forward_evidence/brain-forward-sol-5m-v2-20260824
```

The export is read-only against Postgres, verifies the stored protocol identity,
writes protocol-scoped decisions, trades, minute timestamps, a checkpoint report
and SHA-256 manifest, and refuses to overwrite an existing bundle. A nonzero exit
is expected for a locked or failed gate; it never changes an order or winner.

## Stored tables

- `brain_forward_minute_bars`: closed Binance minute bars plus volume and taker-buy base.
- `brain_forward_decisions`: every qualifying frozen Brain event, with the causal features used.
- `brain_forward_trades`: completed hypothetical five-minute trades only.
- `brain_forward_protocols`: immutable protocol JSON and canonical SHA-256.
- `brain_forward_variant_events`: every triggered, suppressed, pending or
  confirmed candidate for each v2 entry policy.
- `brain_forward_variant_trades`: completed hypothetical outcomes keyed by
  protocol, variant and originating event.

All writes are idempotent. Every decision and trade stores its protocol ID, protocol
hash, artifact hash and evidence phase. A restart reloads recent closed candles and
recreates the same records. The signal enters at the next minute open; target and
stop are one event-candle range; simultaneous target/stop bars resolve to stop; the
latest exit is the fifth close; round-trip costs are fixed at 14 bps.

## Start criteria

Before the official forward period, allow the observer to run for two weeks and verify:

1. five-minute polling and timestamp freshness;
2. no gaps in `brain_forward_minute_bars` beyond an explicit outage record;
3. every decision and trade is reproducible from its stored bars;
4. the service remains hard-disabled for live trading.

The artifact, dates and gates are already frozen in the protocol. If the observer is
not healthy throughout warmup, the fixed formal epoch is invalidated rather than
silently shifted or extended; a new protocol ID must be preregistered.

After the checkpoint time, export the protocol-scoped decision rows, trade rows and
minute timestamps as JSON lists or JSONL, then run:

```text
brain-forward-checkpoint \
  --decisions decisions.jsonl \
  --trades trades.jsonl \
  --bar-timestamps bar_timestamps.jsonl \
  --as-of 2026-10-31T00:10:00Z \
  --output checkpoint_report.json
```

The command exits successfully only for a fully eligible shadow-review report. A
locked or failed gate returns a nonzero status and still writes the complete hashed
audit report.
