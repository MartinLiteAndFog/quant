# Cloudgarden

Cloudgarden is a standalone evolution and historical paper-trading game under `apps/cloudgarden`. The default **SOL Trading** view executes source-derived IMBA/EW candle strategies over the locally downloaded SOL minute archive. **Play** and the day/week/season buttons execute all financed plants, reconcile fills and costs, update the single NAV ledger, then advance ecology. The Meadow/Understory demo tabs retain the earlier synthetic fixtures. There is no account, exchange, or real-order connection.

## Play the garden

```sh
npm install
npm run dev
```

The historical view needs a local data export before the first build. From the repository root, run `.venv/bin/python apps/cloudgarden/scripts/export_sol_history.py`. This reads the existing immutable SOL parquet archive into ignored `public/market/` day assets with SHA256 checksums. The verified local export contains 3,109,186 actual minute candles over 2020-08-11 through 2026-07-11, with 10 gaps / 1,424 missing minutes. The initial and final dates are partial days; playback requires complete days and stops at gaps. `npm run build` includes the exported data in the Desktop package. Raw archive assets stay local and are not committed.

The three historical parent templates are IMBA Trend, IMBA Countertrend, and IMBA × EW. Their bounded heritable modules include channel lookback, confirmed pivot length, ATR movement threshold, EW retracement filter, impulse exit, and compulsory stop trigger. Local inheritance blends actual parents; Hive uses the central source library; Hybrid combines both. These are source-derived candle research variants, **not exact TradingView/Renko/MTF or Tradingade Strategy Builder replicas**. Downloaded Tradingade result workbooks do not contain full executable source. See `HISTORICAL_TRADING_SPEC.md` for source identity and execution assumptions.

Historical worlds use a separate validated browser-local save. The UI shows the next UTC market date, NAV to four decimals, net PnL, fees, fills, rejected orders, inherited source genes and recent actual-price paper executions. Reload restores the same world; changing the start date and selecting **Start from this date** creates a new historical world. The historical price archive remains on disk; this is a local exported snapshot rather than a live database or market connection.

The app starts at `http://127.0.0.1:5184`. Step one day, a week, or a 140-day season, or let the world advance automatically. The same world seed and actions produce the same climate, births, predation, and archive events.

Cloudgarden automatically saves its versioned world snapshot, selected view, research inputs, and latest closed replay in this browser. Reloading resumes the garden and restores the replay with the plant and signal genes used for that run. Existing first-version saves migrate to the current format without losing the garden or replay. Save data stays in local browser storage; there is no account sync. Use **Reset world** to start over on this device.

The initial ledger has 100 synthetic USDT of NAV, with a hard 90 USDT allocation ceiling and a protected 10 USDT reserve. Three parent plants begin with 20 USDT each. The remaining 30 USDT is held against 20 single-use seedling entitlements of 1.50 USDT each. Conception transfers that entitlement into deployed capital; it does not create equity. Predation returns an active plant's allocation to free capital only after it is archived. It never restores a spent seed entitlement.

Plant forms (Grass, Bush, and Cactus) are derived from root and leaf genes. Those genes share a bounded daily growth budget. Daily habitat fit measures only the plant's match to the current synthetic climate; vigor is its accumulated state, while synthetic evidence reflects simulated observation and is not market evidence. Dormancy is a reversible climate response: dormant plants remain visible and capitalized, recover when conditions improve, and are protected from predators. The predator has a minimum observation gate, 70/80 vulnerability thresholds, and a stronger bark-like age protection after 120 observation days. Archived plants leave the visible meadow and remain only as research records. The recent event window is capped at 2,000 records during long autoplay.

The Understory view aggregates archive and seed-bank activity and can run matched Random, Local, Hive, and Hybrid cohorts over the same world seeds and day horizon. Reports include synthetic daily habitat fit and vigor alongside population outcomes. The experiment compares only Cloudgarden's synthetic rules. Its result is marked `not_comparable` to market performance and must not be treated as evidence for a trading strategy.

Each plant carries heritable trading traits for lookback, entry, and exit thresholds alongside its habitat genes. Local bees blend and mutate parental traits; Hive and Hybrid bees bring in deterministic hive traits; Random bees sample independently. Matched cohort reports also show descriptive means for these three inherited traits, averaging active plants within each seed and then weighting each seed equally; these means describe genomes, not trading performance. The selected plant's traits drive its 7-, 30-, or 90-day signal-to-paper fixture over generated UTC minute bars. Each range is deterministic for its garden seed and starting week, and later garden cycles receive a distinct synthetic fixture. The versioned synthetic Momentum v1 recipe reads completed closes, reports executable opportunity observations before the reserved forced-flat close, and emits targets for the causal replay. Strictly-next-open fills, bounded delay, gap/zero-volume rejection, spread/slippage embedded in price, one-time fees, and signed funding remain enforced. A versioned evidence report reconciles PnL, costs, cash, daily drawdown, generated targets, fills, and rejected signals. All synthetic horizons are marked `insufficient_sample`; independent opportunity count is unavailable. A closed result books into the same garden NAV and advances the daily clock by the replay's exact horizon. A no-opportunity run is labeled `no_trade` without inventing a trading loss. Executable opportunity observations use the selected plant's synthetic entry threshold and are not independent market opportunities. The signal fixture is not alpha or market evidence and does not change habitat fit, vigor, or evidence.

## Scope and evidence

Ecology and the demo tabs are generated fixtures; the SOL Trading price series is historical. Archive venue, market type and license remain unverified. The historical engine models a 1x linear long/short paper position, uses assumed fees/spread/slippage, excludes unavailable funding, and flattens daily. Stop thresholds trigger from completed closes and do not guarantee the subsequent fill price. This is an explicitly labeled research variant rather than proven strategy parity or profitability. No account, exchange, order, Fleet API, promotion or real capital is available to any plant.

## Verify

```sh
npm test
npm run build
```

To verify actual exported data end-to-end, run `npm run verify:history -- 140` (or add the local server URL as a third command argument). The frozen integration run from 2026-01-01 through 2026-05-20 executed 28,078 fills, conceived all 20 funded offspring, and ended at **31.3791 paper USDT** after **45.9937 USDT** of modeled fees. This is a losing research variant; it is not a profitability result. Benchmark: cash/no trade. Verdict: `not_comparable` for market performance. The run establishes working data-to-signal-to-execution-to-ledger plumbing, not market viability.
