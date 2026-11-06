# QUANT 0.36 — Intraday Research dashboard / sweep storage audit
Date: 2026-11-06
Status: research prototype, **not production parity**.

## Offline application
Local release artifact: `QUANT_IMBA_Dashboard_0.36_Intraday_Research_Offline.html` (self-contained ~24 MB) and `QUANT_IMBA_Dashboard_0.36_Intraday_Research_Offline.zip` (~15 MB). The binary/HTML release is **not included in this commit**: the GitHub connector only supports small UTF-8 file writes. Do not interpret this document as evidence the HTML artifact is committed. Keep the local release ZIP as canonical until uploaded via a normal git push/release.

## Version progression
0.31 performance inspector and historical model; 0.32 sweep objective (return/profit/drawdown); 0.33 risk laboratory (portfolio trailing protection, pause, profit vault); 0.34 reusable local OHLCV/CSV.GZ pipeline, monthly cache, QDP export; 0.35 Renko controls and precomputed USD/%/candle variant sweep; 0.36 intraday research lab for selected IMBA strategies.

## 0.36 functionality
- Local SOL OHLCV minute-data loading through 0.34 pipeline.
- Traditional fixed Renko with two-box reversal, reconstructed IMBA with sensitivity x10 lookback.
- Research executions on next minute open, minute high/low based stops and TP, selected bots up to 12, optional 1m/5m signal resolution.
- Comparison chart, stop/TP counts, CSV export.
- This is a research simulator, **not** a validated TradingView parity engine, nor a full multi-bot portfolio replay with the historical cash reserve and 14-day rebalance.
- OHLC bars do not identify high/low touch order. Explicitly specify pessimistic stop-first convention if TP and SL both touch.

## Existing GitHub implementation and discrepancies
`src/quant/features/renko.py`: close-driven traditional Renko, 2-box reversal, same-source-bar bricks offset by milliseconds.
`src/quant/strategies/imba.py`: sticky trend, no initial signal, opposite flip only.
`tradingview/sol_imba_strategy.pine`: optional initial bootstrap signal and different live strategy/exit semantics. Parity must explicitly align these assumptions.
`docs/research/imba_top6_reproduction_audit_2026-11-05.md`: archived portfolio results are not reproduced by replacement engines. Never force calibration to historical 8k/9k numbers.

## Sweep persistence audit (from offline 0.36 JS)
`quantRunSweep()` builds a Cartesian product:
- topCount [4,5,6]
- rank unique([7,10,14,30,90,current])
- guard unique([0,20,30,40,current])
- stopScope individual/global (global omitted when guard=0)
- quantTakeCandidates()
- allocation [0.25,0.5,0.75,1] when allocation sweep enabled, otherwise current
- quantRiskCandidates()
- other fixed config: capitalModel, execution, takeCustom, selected candidate pool
Each config is evaluated via `simulatePortfolio({...conf,pool:currentPool,endDay:bounds.end,rankedCache})`; only `quantSweepEvaluate` output metrics plus config are stored in `quantSweepRanking` (in-memory array). `quantSweepInvalidate` clears this. No durable sweep-result IndexedDB object store currently exists; IndexedDB in 0.34/0.36 is used for market data pipeline. A new sweep recomputes results even when configs overlap. Changing objective return/profit/drawdown only re-sorts existing rows; no recompute required.

## Proposed deterministic cache
Persistent IndexedDB stores: `sweep_runs`, `sweep_cases`, `equity_series` (optional), `market_manifest`, `engine_versions`. Use a canonical SHA-256 cache key of (engine version + source dataset hash + candidate pool sorted IDs and backtest version + year + start/end/cutoff + complete normalized config + execution granularity + fee/slippage and stop priority + signal/renko engine version). Include exact training cutoffs and prohibit future leakage. Each unique case is reused across runs. Keep run manifest separately to preserve what was compared and when; never use stale cases after engine or dataset changes. Store numeric results with Float64/Float32 arrays only when needed; do not store 32k full equity curves by default. Keep summary records and selected/top-N curves; allow user export/import as portable QDP/ZIP. Add quota display, LRU purge and pinning, checksum, schema migrations.

## Next milestone 0.37
1. Import local SOL 1m and validate feed/timestamps/gaps.
2. Reconcile Renko and IMBA signal parity against known TradingView alerts and fills.
3. Event-driven portfolio ledger with per-bot orders, fees, leverage, capital reserve, vault, stop and rebalancing.
4. Crash-day diagnostic replay (Jan 29, Jul 17, Oct 23 2025), compare exact causal sequences and stop-first uncertainty.
5. Implement persistent sweep case cache; verify cache hits/misses and cross-run reuse; multi-year walk-forward OOS.
6. Browser/CI parity regression tests before using any research result live.

## Storage sizing
32,000 summary rows at an illustrative 0.5–1 KB each = 16–32 MB raw (estimate, not measured); compressed size and IndexedDB overhead vary. A complete 365-day Float64 NAV array is 2,920 bytes per case before metadata, ~93.4 MB for 32,000 cases. Minute-resolution equity curves are vastly larger; persist selectively.
