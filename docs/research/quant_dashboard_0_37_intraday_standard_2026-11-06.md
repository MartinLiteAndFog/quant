# QUANT 0.37 — Intraday as primary research ledger
Date: 2026-11-06

Status: **integration prototype**. The separately tested 0.36 intraday simulation has been refactored into a single minute-bar portfolio ledger. **Not yet complete multi-year/TradingView parity**. No live-trading change.

## Offline release
- `QUANT_IMBA_Dashboard_0.37_Intraday_Standard_Offline.html` SHA-256 `5b1d2595334df610da76f87f85315446a34300d11b0be74ad66093ff2554fb36`.
- `QUANT_IMBA_Dashboard_0.37_Intraday_Standard_Offline.zip` SHA-256 `c8aec36a22c88c1f076777077bd3288efeea9d3046fdae60958dfdb9462061f9`.
- ZIP has the complete offline HTML, separate `quant_intraday_core_037.js`, `quant_intraday_ui_037.js`, CSS, readme and a dependency-free Node core test.
- **ZIP/HTML NOT committed to GitHub**: available as the original conversation download. These documents are the GitHub-side release manifest only.

## Engine
- Source: locally imported 1m OHLCV with 90–120-day warmup; 1m/5m confirmed close -> traditional 2-box Renko/IMBA sensitivity x10 -> signal submitted for next 1m open.
- One shared 1000 EUR portfolio ledger: each selected bot gets `portfolio_budget / selected_count`; 10x applies on each bot sleeve without duplicating principal. Each sleeve has its own cash/quantity, fees, stop/TP; portfolio reserve/vault included exactly once.
- Fixed cohort or causal Top-N (prior completed days only); 7/10/14/30/90 day ranking; rebalancing every 14 days or at calendar month boundary.
- Minute high/low protective exits with pessimistic SL-first convention if multiple thresholds touch; additional individual/global account stops; depot trailing/pause, profit locks, archive-inspired and ladder take profits. Signals and minute-data are reused for compact 12-case risk sweeps, whose summaries and signal-event lists can be persisted to IndexedDB keyed by dataset, period, engine and gene set.
- Main portfolio HUD defaults to intraday status and shows new minute NAV only after execution. After preparation, the 3D chart shows the reconstructed 1x per-gene return for the prepared cohort. The archived daily charts and former sweep/OOS/WF remain available only as visibly labeled Tagesarchiv.

## Reproducibility
- Chromium browser on 36,000 genuine SOL 1m bars / six bots: new 1m portfolio execution and 12-case risk sweep ran without JS page errors.
- Under matched configurations (no reserve changes, stops, portfolio interventions or profit transfers), new one-depot equity matches to floating point rounding the average of six original independent v0.36 account simulations because equally sized subaccounts were used.
- Node invariants: 0% budget=flat NAV; cloned equal bots do not duplicate NAV; 50% budget halves 1x gain/loss in linear fee-free case; depot stop triggers at most once per cycle.

## Critical remaining blockers
- Historical 32k-case sweep, full manual WF player, multi-year historical portfolio and 3D/individual bot daily histories are **not** all migrated. Daily archive is deliberately marked rather than mixed with 1m results. Full candidate universe >600 is blocked in the browser until a separate worker/persistent compute server exists.
- Minute OHLC does not reveal tick sequence or real gap/slippage; missing TV Renko-feed parity and exact TP/TTP reversal state machine, exchange margin/funding, fees and broker fills. Some portfolio profit plans remain approximations and cannot be described as exact archive reproduction.
- For a real 0.38 conversion: event-stream and signal cache by gene + data checksum; a worker-driven full 2022–2026 parameter matrix; unified causal WF and all 32k sweep configs; cross-venue feed/TV alert parity tests; independent year OOS checks; persistent sweep case cache and explicit cache invalidation; deterministic strategy benchmark fixtures.

**Never deploy this research model as an execution bot until these are verified.**
