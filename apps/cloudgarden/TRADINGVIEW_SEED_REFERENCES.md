# Quant / Cloudgarden — nine TradingView seed references

**Status: read-only source trade fixtures, not independently reproduced trading strategies.**

## Outcome

Nine Tradingade Strategy Builder Excel exports (17–18 October 2026) have been parsed, hashed and preserved in Quant. There are 1,193 *closed reference trades* and 4 *unclosed* open trade entries across the snapshots. These are NOT 1,193 independent market opportunities: trade timelines overlap heavily between configurations. Every closed trade's displayed entry/exit price, size, commission and net PnL reconciles within 0.032 USDT (display precision; acceptance tolerance 0.25). No unclosed trade is booked into PnL.

The nine candidate seed recipes have real SL bps, TTP deviation bps, 90% sizing, leverage, indicator-source identity and filter toggles. Most use 0.9% stop, 0.8% trailing deviation, 10x leverage; KUCOIN SOL uses 1.1% stop; two KUCOIN BTC exports use 0.5% trailing deviation and 20x/40x leverage. The third-party indicator formulas are opaque; the `HH Lookback=20` in Properties belongs to inactive SL/TP variants and is NOT known to be the Elliott-Wave entry lookback. Therefore entry-lookback and pivot-length genes are intentionally null.

## Parity gates

1. **Renko OHLC/time:** offline Binance SOL snapshot has 10,219 / 10,221 exact matches after historical H1→5m source resampling; a documented 12 Oct event differs. This does not by itself validate strategy signals.
2. **Reference trade tape:** 1,193 closed trades + 4 open entries extracted and structurally/PnL checked. This proves correct import, not causal trade generation.
3. **Independent strategy signals:** BLOCKED on proprietary/missing Elliott Wave Detector Pro + Multi Time Frame RSI source formulas, full Tradingade Builder logic and exact indicators/inputs. Existing Cloudgarden IMBA/EW research module is *not* to be substituted as if identical.
4. **Execution parity:** not proven. TradingView Renko strategy engine's order fills are synthetic. Replay should compare signal times and orders on causal executable market candles with fees/slippage separately.
5. **Capital & vitality:** do NOT book reference PnL into Cloudgarden's NAV or apply an extra `inactiveDays` tick before independent execution parity and a complete flat session are validated.

## Stored artifacts

- `quant_tradingview_first_plants_reference_2026-11-02.json`: sanitized complete nine-variant source manifest and **trade tape** with full SHA-256 identity.
- `quant_tv_seed_import.py`: deterministic importer based on `artifact_tool`; skips the full TradingView ALERTS section (this contained potentially sensitive webhook content). Credentials must never be checked into GitHub.
- `tradingViewSeedCatalog.ts` and `.test.ts`: bounded read-only Cloudgarden seed catalog / trade-tape validator.
- `_audit_tv_seed_reference.mjs`: local audit entrypoint for the full manifest.

**Privacy:** no webhook payloads, tokens or private source XLSX are to be committed to the GitHub PR. The complete user XLSX originals stay in the user-controlled `Quant` Library project. Treat any token in the originals as a secret and rotate it if it remains active.

## Validation

- `node --experimental-strip-types --test tradingViewSeedCatalog.test.ts`: 3 passed.
- `node --experimental-strip-types _audit_tv_seed_reference.mjs`: passed, 9 seed variants, 1,193 closed trades, 4 unbooked open positions, max PnL residual 0.0319005 USDT.
- Live trading: unchanged. Unreal: postponed.