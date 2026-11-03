# Cloudgarden

Cloudgarden is a local research game for evolving trading strategy parameters over real historical SOL and BTC minute archives. The native Unreal Engine 5.8 target lives in `../cloudgarden-unreal`; this React application is the reference implementation and replay engine, not the final native game.

The two gardens maintain independent populations, accounts and validated save states. Each starts with 100 USDT and receives an external contribution of 100 EUR at each subsequent simulated month transition. The disclosed fixed simulation conversion is 1 EUR = 1 USDT, not a historical FX series. Deposits are excluded from trading PnL. Shared controls run both gardens; individual steps inspect one garden.

## Local development

From this directory:

```sh
npm install
npm run dev
npm test
npm run build
```

Export the already downloaded datasets from the repository root:

```sh
.venv/bin/python apps/cloudgarden/scripts/export_sol_history.py
.venv/bin/python apps/cloudgarden/scripts/export_btc_history.py
```

SOL contains 3,109,186 observed minute candles, 2020-08-11 through 2026-07-11. BTC contains 3,084,497 observed minute candles, 2020-08-12 through 2026-06-24. Exporters preserve exact decimal prices, validate chronology/OHLCV and write daily SHA256-checked assets. Partial dates and gaps remain actual observations; missing minutes are not fabricated. Raw assets in `public/market/` and `public/market-btc/` remain ignored and local.

The native bridge runs locally using the development Node/tsx runtime:

```sh
node --import tsx scripts/unreal_bridge.mjs ../cloudgarden-unreal/Saved
```

See `../cloudgarden-unreal/README.md`, `INTEGRATION.md` and `STATUS.md` for native build and runtime evidence. A successful reference build does not prove a packaged native game works offline. Native packaging and visual verification have their own gates.

## Strategies, genes and evolution

Five named Kraken/IMBA templates include the two November 17.2 variants: effective lookback 172 signal-Renko bars, 0.5% TTP, 2.8% partial TP and distinct WAIT-stop close/reverse behavior. These are source-derived, confirmed-five-minute-close Renko proxies; full TradingView intrabrick/MTF parity is not established. SOL templates are transferred unchanged to BTC as an explicit experiment, not BTC-optimized strategies.

A strategy combines atomic genes. The library contains 12 atomics: IMBA signal, confirmed EW swing, EW confirmation, TTP, mandatory stop-loss, partial TP, capital limit, AND, OR, NOT, threshold and cooldown. Seven recipes describe combinations separately from the atoms. Named compiled profiles execute through tested code; free-form graph execution is not implemented. Custom snippets are inert notes, never executable code. Every gene exposes input history, confirmation delay and availability semantics; a historical pivot marker is not a tradable timestamp.

Each child receives bounded parameters frozen at birth, including actual Kraken parameters and confirmed EW filter/exit genes. Parent selection, sizing weights and performance-derived physiology use prior completed records. Today's result from an earlier array element cannot size another plant's earlier trade. Prefix and plant-order invariance tests cover this distinction.

Births require free capital rather than fixed planting slots. The safety ceiling is 200 living plants; ecological predation and cleanup regulate the population below it. Historical worlds retain population, accounts and genomes across archive wraps. The old Meadow/Understory synthetic demos retain their original 20 one-use entitlements and are explicitly separate fixtures.

Kraken positions and pending orders persist across dates. Signals use completed observations and subsequent actual minute opens; modeled costs and a mandatory hard stop apply under a 1× notional cap. Allocation-cap reductions are real modeled fills and are labeled separately from strategy exits. A predeclared final 60-minute archive window closes positions before repeating the source, avoiding a fabricated price jump between its last and first dates. Outstanding exposure fails closed rather than silently resetting capital.

## State and evidence

Reference worlds use IndexedDB with separate symbol keys and protected legacy-save loading. The gene collection uses a validated localStorage format with import/export. Native bridge sessions are local validated JSON. There is no account sync, exchange connection or real-order path.

Run short actual-data checks with `npm run verify:history -- 7` and `node --import tsx scripts/verify_btc.mjs 7`. Full repeat verification uses `node --import tsx scripts/verify_loop.mjs 2 /tmp/cloudgarden-loop-evidence.json`. Inspect each verifier's output and assumptions; a unit test or short run cannot establish full-archive completion.

The current directly observed seven-day BTC check executed 4,800 modeled fills and ended at NAV 73.4391 from 100, with no deposits (net trading loss 26.5609 USDT). This losing transfer experiment is integration evidence only. Venue/instrument provenance, funding and execution calibration remain incomplete. Benchmark: cash/no trade. Research verdict: `not_comparable` for independent market performance.

Repeated archive passes are training on the same observations, not additional independent decades. Causal execution prevents one class of leakage but does not prevent selection overfitting. A trustworthy champion still requires frozen out-of-sample/forward evaluation, costs and explicit promotion gates. No profitability claim follows from garden survival or contribution-funded NAV growth.
