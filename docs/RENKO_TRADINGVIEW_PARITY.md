# Renko / TradingView parity note

## 2026-10-07 correction

Historical backtests that used the previous `renko_from_close()` implementation should **not** be treated as TradingView-parity backtests.

The old builder had two material differences from the fresh TradingView Traditional Renko export:

1. It reversed direction after a one-box move. TradingView Traditional Renko requires a **two-box move** to reverse direction.
2. It assigned the same timestamp to every brick emitted from one source bar. Downstream code de-duplicates timestamps, so intermediate bricks could be silently lost. TradingView emits same-source-bar Renko bricks with consecutive **1 ms** timestamp offsets.

The corrected builder now reproduces those two structural behaviors.

### Evidence from the fresh TradingView export

Reference file inspected on 2026-10-07: Kraken SOLUSD.PM, 5-minute Traditional Renko, box size 0.1.

- 5,322 Renko bricks in the fresh export.
- 984 direction reversals.
- All 984 reversals follow the two-box reversal geometry exactly.
- 1,264 five-minute source buckets contain more than one Renko brick.
- In every such bucket, TradingView orders those bricks with consecutive 1 ms timestamps.

### Backtest consequence

Any result generated with the old one-box-reversal / duplicate-timestamp builder can differ materially over long histories and should be re-run before being used for strategy decisions.

### Remaining parity caveat

`renko_from_close()` is still a close-driven reconstruction. Exact parity additionally depends on using the same venue/feed, source timeframe, source-bar history and TradingView's own historical reconstruction behavior. For validation, compare rebuilt signal order and timing against real TradingView webhook alerts before trusting long-horizon results.


## Source timeframe is mandatory

For the SOL TradingView setup validated on 2026-10-07, the Renko chart source interval is **5 minutes**.
Raw 1-minute OHLCV must therefore be resampled to 5-minute OHLC **before** Renko construction.

Validation on the fresh TradingView export (2026-09-15 to 2026-10-07):
- TradingView 5-minute Renko: 5,322 visible bricks and 99 IMBA direction changes (lookback 50).
- Corrected Renko run directly on 1-minute source data: 7,859 bricks and 144 IMBA direction changes.
- Corrected Renko run on 5-minute-resampled source data: 5,279 bricks and 105 IMBA direction changes.

So using 1-minute input for a 5-minute TradingView Renko configuration creates about 48% more bricks and 45 extra IMBA direction changes in this validation window. Backtests made with that source mismatch are invalid for TradingView parity.

The fresh TradingView export also shows that Renko wicks matter to IMBA because the strategy uses Renko high/low over the lookback. Wicks occur on the first brick emitted by a 5-minute source bucket and are opposite the brick direction. Exact wick parity still depends on the original venue/feed OHLC.
