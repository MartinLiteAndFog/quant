# Research Journal

## 2026-04-02

Today focused on non-SOL tradfi backtesting using Yahoo-sourced `2m` data for `EURUSD`, `NQ_NASDAQ`, and `USDJPY`. Verified that Yahoo only gives short intraday retention, so `2m` is usable for research but not sufficient for year-scale walk-forward work. Cached local `2m` parquet datasets for the three assets and used those as the consistent base for all later runs.

The main strategy work was around Renko sizing, IMBA lookback, and CHOP/ADX/ER gate behavior. The original tradfi defaults were far too sparse. Scaling boxes from SOL and keeping long IMBA lookbacks suppressed signal generation, especially in FX. Smaller boxes fixed that immediately. This confirmed that the Renko box size materially changes IMBA behavior because the transformed price path determines swing structure and flip opportunities.

For the `2m` tradfi experiments, the strict `CHOP & ADX & ER` gate and the earlier `2-of-3` gate behaved very differently. On the strict runner, smaller TTP values expressed in Renko-brick units were required before the flip branch produced closed trades at all. A useful working convention emerged:
- express TTP in brick units, not raw percent
- retune gates on the exact same runner used for the backtests
- treat long IMBA lookbacks as a first-order throttle on signal cadence

NQ remained the strongest asset through most of the day. After fixing TTP and recalibrating the gate on the same cached `2m` runner, the best NQ region clustered around:
- box near `15.0` to `17.5`
- lookback near `50`
- TTP near `3` to `4` bricks

A focused NQ confirmation sweep gave the best local candidate at:
- `box=17.5`
- `lookback=50`
- `ttp=3` bricks
- gate ON about `34%`
- combined return about `+10.49%`
- max drawdown about `-0.57%`

EUR/USD became much more interesting once the gate was removed from the routing problem. With always-ON routing, a local sweep around the earlier winner found a better configuration than the first pass:
- `box=0.00009`
- `lookback=90`
- `ttp=10` bricks
- about `2.85` signals/day
- return about `+14.77%`
- max drawdown about `-0.87%`

This was materially better than the nearby `box=0.0001`, `lookback=100`, `ttp=12` version. It also showed that for EUR/USD the trailing stop is sensitive: widening TTP too far, for example to `25` bricks, degraded performance sharply.

Additional EUR/USD routing tests showed:
- `lookback=300` starves the strategy even with small Renko boxes
- reducing lookback to `30` or `100` reactivates the system
- for the tested EUR/USD setup, the strict gate reduced returns relative to always-ON routing
- a near-`50%` gate ON split was worse than the original gate, and both were worse than always-ON for this local sample

Kelly sizing was examined on the EUR/USD always-ON winner. The repo's log-growth Kelly search hit the configured upper search bound, which means the raw Kelly number is not trustworthy as a literal leverage recommendation. A `5x` leveraged equity curve was still computed for visualization purposes and produced approximately:
- unlevered return `+14.77%`
- `5x` levered return `+97.67%`
- `5x` max drawdown `-4.35%`

Important caveat: the current equity calculation is still likely a bit skewed. The leveraged equity curves are useful for comparison and visualization, but should not yet be treated as final capital curves for production sizing decisions.

Created supporting artifacts:
- Databento downloader for year-scale futures history: `scripts/download_databento_ohlcv.py`
- EUR/USD chart renderer with price, equity overlay, and entry markers: `scripts/plot_eurusd_equity_markers.py`

Generated plot:
- `data/research/tradfi_2m/eur_always_on_5x/eurusd_price_equity_markers_5x.png`

Next research priorities:
1. Pull year-scale `1m` NQ history via Databento and build a proper walk-forward dataset.
2. Add a long-history FX source, likely Polygon or Dukascopy, into the same parquet format.
3. Recompute equity and leverage handling carefully before trusting any Kelly-derived sizing.
