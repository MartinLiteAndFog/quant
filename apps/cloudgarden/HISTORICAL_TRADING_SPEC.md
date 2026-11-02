# Historical SOL trading and inherited source modules

The default playable mode must execute funded plants when the user presses Play or advances days. A completed UTC day must update NAV from real archive prices, show fills and assumed costs, and retain one reconciled 90/10 ledger. Ecology-only steps remain explicitly named demo views.

## Source identity

- Immutable local input: `research/solusdt_robust_growth_v1/immutable/SOL-USDT_1m_20260711T092900Z.parquet`.
- SHA256: `fa3e689ddbc42b35b89f66db4240521a330896e2d89fc2d2df8624bc749fda64`.
- 3,109,186 actual minute OHLCV rows, 2020-08-11 06:00 UTC to 2026-07-11 09:29 UTC. Ten gaps / 1,424 missing minutes; no duplicates.
- `scripts/export_sol_history.py` preserves source values as decimal strings without silently rounding and produces daily arrays plus SHA256 per file. The browser verifies exact bytes before normalization. Raw market assets are ignored by Git and copied into the local Desktop build.
- Venue, source market type and license are unverified. These prices do not establish KuCoin/Binance futures parity. The model is a hypothetical linear long/short position; unavailable funding is excluded.

## Strategy material

IMBA channel rules derive from `tradingview/imba_renko_signal.pine`, `tradingview/sol_imba_strategy.pine`, and the existing `src/quant/strategies/imba.py` port: rolling highs/lows, Fibonacci channel .236/.786, sticky direction. The countertrend parent is an inverted-channel research variant, not a reproduction of the full Pine CT/TF order manager.

EW material derives from `EW_v3_2_MTF_AUDIT.pine` and the existing `research/ew_mtf_parity_20261018/ew_replay.py` port. The MPL-2.0 source notice is preserved. Heritable donor modules include confirmed pivots, ATR minimum movement, .382–.786 W3-style retracement filter and six-pivot impulse exit. Pivots become usable only after their right-hand confirmation candles close. No historical HTF mapping with future dependencies is used.

Downloaded Tradingade Strategy Builder result workbooks contain settings/results rather than complete executable source. Neither exact Tradingade nor full EW/Renko/MTF parity is claimed. The existing parity research has unresolved causality and order-management gates; these modules are clearly labeled candle research variants.

Source genomes are separate from habitat/physiology genes. Local offspring blend both local parents; Hive draws from the central source library; Hybrid combines both. Random offspring are independent of local parents. Every mode keeps a bounded compulsory stop trigger; arbitrary code is never executed.

## Daily causal execution

1. Validate the historical session and archive identity. Load the requested actual UTC day plus the previous day's warmup, checking SHA256 and exact day equality between signal and execution inputs.
2. Stop before a partial/gapped day; never fabricate missing prices or carry a signal across missing observations. Previous-day indicators are warmup only; orders originate within the requested day.
3. Size a target from half the plant allocation and the known first session open. No future price participates in sizing. Long/short target exposure remains within the 1x allocation ceiling; oversized orders reject.
4. A decision uses completed candles; execution uses the first observed open strictly later than availability. Apply spread/slippage in fill price and debit each fill's fee once. Default assumptions: fee 6 bps, full spread 2 bps, slippage 1 bp.
5. Stops are completed-close triggers from the decision reference price, not guaranteed fill prices or exact TradingView stops. Reserve the last three bars for flat liquidation and require every plant to finish flat; this daily-flat variant differs from overnight trading.
6. Book all intraday results into the same ledger before end-of-day ecology and predation. Advance ecology exactly once. Transactions are immutable; a failed day is not partially committed. Reserve and deployable balances use complementary rounding, preventing double rounding from creating equity.
7. Conceptions use existing single-use entitlements. Offspring receive deterministic source genomes before their first next-day trade. Display per-plant booked PnL/fees/fills/rejections and up to 400 recent fills. Historical and legacy demo saves are separate; invalid historical saves are protected until explicit restart.

## Verification and evidence boundary

Tests cover source-module prefix causality, delayed pivots, inherited modules/compulsory stops, explicit historical opt-in, real-price fills, accounting under repeated fine fractional PnL, warmup/source mismatch rejection, day gaps, multi-day offspring, and restored accounting.

The full HTTP integration check over the frozen archive ran 140 days (2026-01-01–2026-05-20), executed 28,078 fills, consumed all 20 seedling entitlements, and ended at NAV 31.3791 from 100 with 45.9937 modeled fees. This variant loses capital; it has not been optimized or promoted. Benchmark: cash/no trade. Verdict: **not_comparable** for empirical market performance because provenance, venue calibration, funding, full source parity and frozen OOS evaluation remain unresolved.

## Independent reproduction boundary

The reference figures above remain developer-reported unless verified on an authorized private runner. See [SOURCE_PROVENANCE_AUDIT.md](SOURCE_PROVENANCE_AUDIT.md). Passing the same 140-day integration outputs does not prove vendor identity, contract terms, source license, actual funding or strategy parity.
