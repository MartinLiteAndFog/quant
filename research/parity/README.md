# Agent C: independent TradingView Renko / IMBA parity diagnosis

**Scope: research only.** No changes to production strategy/execution,
archive, ECO-19b candidate signal logic, venue credentials, alerts or orders.

Compare **independently sourced** TradingView Renko bricks against local
bricks by UTC **5-minute source bucket**, ordered intrabucket brick sequence,
body, direction and wicks. Separately compare independently established
confirmed source-close IMBA alerts. The output parity_verified flag is
**always false**; even a perfect match to supplied data is not proof of
end-to-end TV / broker parity.

## Git evidence already documented, not re-verified by this change

- docs/RENKO_TRADINGVIEW_PARITY.md: Kraken SOLUSD.PM, Traditional Renko
  box 0.1, 5m source, 2026-09-15 through 2026-10-07: **5,322** TV bricks
  / **99** IMBA flips (lookback 50), versus **5,279** rebuilt 5m bricks /
  **105** IMBA changes. A wrong 1m input produced **7,859 / 144**.
- src/quant/features/renko.py implements two-box reversals and synthetic
  1ms timestamps but uses close only, not TV historical wicks.
- src/quant/strategies/imba.py suppresses its initialization signal, while
  tradingview/sol_imba_family.pine defaults to emitInitialSignal=true.
  Bootstrap-signal policy MUST be normalized before comparing alerts.
  This alone does not explain the observed six-signal difference.
- The inspected Git snapshot contains **no original TV brick/alert CSV and
  no confirmed Kraken SOLUSD.PM 5m OHLCV source**. The Cloudgarden SOL-USDT
  Parquet path in PR #62 is not verified same-instrument evidence.

## Strict input contract

Normalize exports **outside** this tool, preserving original raw files
and hashes in a private research archive. UTC offsets (Z / +00:00) are
mandatory; do not infer the timezone of TradingView's display.

1. TV and local brick CSVs: columns
   ts,open,high,low,close and optional direction (+1/-1). ts is the
   exact **5m source open timestamp** plus 0,1,2,... milliseconds
   per synthetic brick in that bucket; every body spans one box.
   Keep TV wicks; local close-only bricks use body extrema for high/low.
2. Source 5m CSV (optional): open_at,open,high,low,close,
   gap-free continuous 5m UTC same-instrument bars. A TV wick/body outside
   that source candle's range is a diagnostic warning, not conclusive
   proof of feed corruption (price gaps or synthetic bridging are possible).
3. Both TV and local alerts (optional together): columns
   reference_id,direction,confirmed_5m_close_at. Timestamps must be
   independently known source-bar closes, NOT webhook receipt times
   and NOT synthetic Renko 1ms stamps. Never force them into buckets
   using a guessed tolerance.

    python research/parity/renko_reference_audit.py \
      --tv-bricks /private/tv_5m_normalized.csv \
      --local-bricks /private/local_5m_normalized.csv \
      --source-5m /private/kraken_solusd_pm_5m.csv \
      --tv-alerts /private/tv_confirmed.csv \
      --local-alerts /private/local_confirmed.csv \
      --box 0.1 --output /private/parity-audit.json

Output: SHA-256 hashes of inputs; first 30 exact mismatch locations
with causal source_open_at and known_at=source_open_at+5m; brick
counts, direction/body/wick/timestamp deviations, source-range feasibility,
reference reversal geometry checks, first-brick whole-box anchor
**hypothesis**, and exact reference/local alert pairing. No trading
signals or orders, synthetic fills, paper NAV or live transactions are
produced. Refuses to overwrite an existing audit output.

## Unresolved engineering / evidence

**Wicks:** A 5m high/low constrains possible wick prices, but does NOT
identify the intrabar price path or exact TradingView historical
assignment. Never silently choose optimistic OHLC traversal.

**Anchors:** The first-brick box-grid delta is not proof that an anchor
shift explains all following differences. Validate the reference prefix,
feed start and entire warmup.

**Multi-bricks:** 0/1/2ms are synthetic display sequences within the same
unclosed 5m bar; **all** are causally available only at source close.
They are never valid independent same-bar order timestamps.

**IMBA:** Confirm precise Pine version, sensitivity x10/explicit lookback,
initial bootstrap policy, wick highs/lows, start time, direction and
exact confirmed-5m-close reference alerts before considering any signal
parity status.

**TP2/Flip/TTP:** Not implemented here because the source-price and
signal parity evidence is absent. A later isolated paper module must
latch exit engine per leg (gate changes apply to NEW legs only), enforce
TP1 partial, BE/SL, TP2, opposite-IMBA and confirmed-flat before reverse;
fills must use subsequent available real prices and instrument fees.
The legacy Python backtest is not broker parity.

### Acceptance

Provide the original TV Renko export with high/low/wicks, independent TV
alerts, same-instrument Kraken SOLUSD.PM 5m OHLCV, provenance/timezone,
full-window hashes and exact known-at timestamps. Only then replay
5,322 vs 5,279 and 99 vs 105, identify first divergent source bucket
and explain every remaining mismatch. Until then machine-readable
parity_verified=false. No live deployment or real orders.

Run:

    python -m unittest discover -s research/parity -p 'test_renko_reference_audit.py' -v
