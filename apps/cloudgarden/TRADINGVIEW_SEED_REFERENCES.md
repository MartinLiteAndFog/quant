# Quant / Cloudgarden — nine TradingView seed references

**Status: reference-only first plants, Pine v3.2 CTF logic partially reproduced (147/147 signals with TradingView-exported MTF state); execution and independently reconstructed MTF/RSI still pending.**

## Outcome

Nine Tradingade Strategy Builder Excel exports (17–18 October 2026) have been parsed, hashed and preserved in Quant. There are 1,193 *closed reference trades* and 4 *unclosed* open trade entries across the snapshots. These are NOT 1,193 independent market opportunities: trade timelines overlap heavily between configurations. Every closed trade's displayed entry/exit price, size, commission and net PnL reconciles within 0.032 USDT (display precision; acceptance tolerance 0.25). No unclosed trade is booked into PnL.

The nine candidate seed recipes have real SL bps, TTP deviation bps, 90% sizing, leverage, indicator-source identity and filter toggles. Most use 0.9% stop, 0.8% trailing deviation, 10x leverage; KUCOIN SOL uses 1.1% stop; two KUCOIN BTC exports use 0.5% trailing deviation and 20x/40x leverage. The user supplied Elliott Wave Detector Pro v3.2 Pine (1405 lines), which is now partially ported and signal-audited; the `HH Lookback=20` in Properties belongs to inactive SL/TP variants and is NOT known to be the Elliott-Wave entry lookback. Therefore entry-lookback and pivot-length genes are intentionally null.

## Parity gates

1. **Renko OHLC/time:** offline Binance SOL snapshot has 10,219 / 10,221 exact matches after historical H1→5m source resampling; a documented 12 Oct event differs. This does not by itself validate strategy signals.
2. **Reference trade tape:** 1,193 closed trades + 4 open entries extracted and structurally/PnL checked. This proves correct import, not causal trade generation.
3. **Independent strategy signals:** CTF Elliott Wave verified on TradingView Renko (147/147 long/short after warm-up) **using TV-exported MTF/HTF state**. Still blocked on independently generated MTF/RSI Renko semantics and full Tradingade Builder logic. Existing Cloudgarden IMBA/EW research module is *not* to be substituted as if identical.
4. **Execution parity:** not proven. TradingView Renko strategy engine's order fills are synthetic. Replay should compare signal times and orders on causal executable market candles with fees/slippage separately.
5. **Capital & vitality:** do NOT book reference PnL into Cloudgarden's NAV or apply an extra `inactiveDays` tick before independent execution parity and a complete flat session are validated.

## Stored artifacts

- `quant_tv_first_plants_corrected_utc_v2.json`: sanitized corrected v2 nine-variant source manifest and **trade tape** with full SHA-256 identity.
- `quant_tv_seed_import.py`: deterministic importer based on `artifact_tool` (set `TV_SEED_SOURCE_DIR` to the local XLSX folder; `TV_SEED_OUTPUT_PATH` optionally changes the destination); skips the full TradingView ALERTS section (this contained potentially sensitive webhook content). Credentials must never be checked into GitHub.
- `tradingViewSeedCatalog.ts` and `.test.ts`: bounded read-only Cloudgarden seed catalog / trade-tape validator.
- `_audit_tv_seed_reference.mjs`: local audit entrypoint for the full manifest.

**Privacy:** no webhook payloads, tokens or private source XLSX are to be committed to the GitHub PR. The complete user XLSX originals stay in the user-controlled `Quant` Library project. Original manifest schema v1 falsely tagged Berlin local wall times as UTC; never use v1 for timestamp or signal audit. v2 converts Europe/Berlin to genuine UTC. Treat any token in the originals as a secret and rotate it if it remains active.

## Validation

- `node --experimental-strip-types --test tradingViewSeedCatalog.test.ts`: 4 passed.
- `node --experimental-strip-types _audit_tv_seed_reference.mjs`: passed, 9 seed variants, 1,193 closed trades, 4 unbooked open positions, max PnL residual 0.0319005 USDT.
- Live trading: unchanged. Unreal: postponed.

## Causal EW signal and trade-tape verification, 2 November 2026

See `TRADINGVIEW_EW_SIGNAL_PARITY.md` for exact methods, source SHA checksums, 422/422 confirmed primary pivots, 147/147 recovered EW long/short plot signals after warm-up (with TV MTF oracle), corrected Europe/Berlin import semantics, 41/41 matching KuCoin SOL entry reference rows and RSI gate limitations. The Pine indicator is **not** the Tradingade Strategy Builder execution engine. Never book these reference fills into simulated NAV as causal profits.

## Offline independent Binance Renko-MTF RSI/market-price audit (2026-11-02)

See [`TRADINGVIEW_MTF_RSI_AND_REAL_PRICE_AUDIT.md`](TRADINGVIEW_MTF_RSI_AND_REAL_PRICE_AUDIT.md).
Independent source-close Traditional Renko from Binance 1m reproduced **10,217/10,225**
historical plotted 30m RSI values to <1e-6, excluding incomplete source
coverage. A strict **completed-30m-source** timing scenario accepted only
353/726 TV reference entry timestamps with RSI enabled, versus 726/726 when
looking up retrospectively plotted RSI values. This is a stress test, not
proof of original TradingView indicator lookahead. The protected RSI source
and original Tradingade exit engine remain unverified.

At 746 closed BINANCE:SOLUSDT reference trades with both future-next-open
prices available, the mean absolute discrepancy was 13.58 bp at entry and
18.15 bp at exit. Those prices were derived from the *same original entry/exit
timestamps*, not independently generated orders. This is NOT a market PnL
backtest, fitness, alpha, or a causal comparison of generated strategies.
Reference trade fills and PnL remain barred from Cloudgarden capital/vigor.
