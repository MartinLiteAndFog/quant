# QUANT 0.38 — Preserving the Tagesarchiv and whole-depot protection

Date: 2026-11-06. Research prototype, not live deployed.

## Decision: retain two independent model families

**Intraday (primary research)**: 1m SOL OHLCV → confirmed 1m/5m source bar → traditional two-box Renko → IMBA gene signal → next-minute-open execution → shared portfolio cash/reserve/vault → minute-level protection. Still *not tick or TradingView parity*.

**Tagesarchiv (preserved, not deprecated)**: all historic 3D surfaces, imported bot-performance series, Top-N/handelsregeln sweeps, manual Walk-Forward and saved presets remain selectable in the UI. Top toolbar explicitly switches between Intraday and Tagesarchiv and remembers last browser choice. Old daily curves must never be silently shown as newly computed minute returns. Keep the legacy calculation pathways and stored parameter choices for future side-by-side parity research.

The old results are not necessarily false, but their fill sequence, signal generator, frequency, capital ledger and portfolio rules differ from the current minute model; their live/TV accuracy is unverified. Preserve research outcomes and configuration snapshots instead of discarding them when adopting a newer data resolution.

### Historical dashboard examples to retain with exact model labels

| Version/model | 2025/2026 results from earlier dashboard | Warning |
|---|---|---|
| 2026 daily historical allocation case, 462 standard IMBAs, Top 6, 14-day ranking, 20% individual stop, monthly reserve release, 10x | €9,367.05 end equity from €1,000 initial | Earlier dashboard reproduced this model result; **not independently TV-parity validated** |
| 2025 daily 462-bot family, Top 6, 7-day ranking, 5% individual stop, 100% account budget, 10x | €5,297.23 from €1,000; -90.8% max equity drawdown | Different rule set from 2026 reference, therefore not a direct year-on-year return comparison |
| 2026 Jan 29–Feb 11, v0.36 six manually marked bots, minutely signal model | €2,597.91 without added price stop; €2,865.77 with 5% bot price stop | The two €1,000-base portfolio figures come from a **different signal and capital model**, not the 2026 daily selection ledger; not evidence 186% was reliably tradable |

Related conflicting archival reproduction: `docs/research/imba_top6_reproduction_audit_2026-11-05.md` records another independent backtest with different original cohort and missing parity. Do not merge its outcomes into the rows above.

User mentioned other presets downloaded locally; those files were not available to this GitHub save, and additional performance numbers must not be invented. Preserve import/export compatibility to recover their exact metadata and returns later.

## 0.38 whole-depot risk semantics

- **Fixed portfolio floor**: `initial_equity * (1 - depotFloor/100)` (20% on €1,000 → €800). Includes uninvested cash, reserve, vault and open positions; it is NOT a percentage stop for each bot.
- **Portfolio trailing floor**: `max_completed_minute_NAV_so_far * (1 - depotTrail/100)` (30% after NAV of €2,500 → €1,750). Highest prior *confirmed minute-close* equity persists across 14-day/monthly rebalances; no reset each cycle, no same-bar future-high assumption. Active protection uses whichever configured floor is higher.
- **Trigger**: evaluate current minute-open and minute OHLC range with total portfolio NAV as a linear function of the common SOL price. When a range breaches the threshold, interpolate the SOL price corresponding to the NAV threshold; if the minute opens beyond the line, execute at the worse open. Apply configured fee and slippage. The whole-depot stop is processed before bot stops/TPs when intra-minute order is unknown; this priority is an assumption and not independently justified by tick data.
- **Action**: `close` closes 100% of existing exposure, `half` closes 50% and leaves 50% invested. It is crucial that **a half-close does not guarantee the €800/€1750 floor** once residual exposure trades on.
- **Pause**: new bot entries disabled until next rebalance, 14 days, 30 days or year end. For half-close the remaining exposure may keep losing (individual price stops may still act). Past portfolio stops and timestamps are recorded.
- **Visual evidence**: red dashed constant floor line, amber dotted trailing line, red event markers, sortable trade-event table, status and performance comparison with matching run minus the two new whole-depot protections.
- **Research sweep**: 16 configurations over additional individual price stop, fixed depot floor, trailing depot floor and close/half actions (not the legacy 32,000+ combination sweep). IndexedDB case cache is version isolated.

## Explicit non-parity / modeling limitations

Minute candles do not resolve high/low sequence for mixed event types. Modelling an equity threshold crossing from a single underlying SOL price is tractable for equal-symbol linear long/short notional but cannot be assumed valid for a BTC+ETH cross-asset basket without an event-time merged price feed. Funding, liquidation fees, genuine order-book slippage, network latency, actual exchange fills and TradingView synthetic-Renko parity are not modeled. No guaranteed stop fills or protected minimum NAV, especially with gaps or partial closing. A replayed model equity high is not evidence that a live exchange stop could have been submitted/fill at that exact price.

## Reproducible checks

Test in `test_quant_038_core.js` (packaged in offline ZIP):
- €1,000 portfolio, no fees/slip, 10x, 20% fixed floor: triggered and filled at €800, corresponding to SOL price 98 in toy sequence.
- €2,500 confirmed portfolio peak, 30% trailing: triggered at €1,750.
- 50% partial exposure cut after triggering €800: continued adverse move yields €650; no false guarantee of floor.
- 50% capital budget preserves reserve; equal long+short SOL neutral portfolio does not spuriously trigger the depot guard.

Chromium browser smoke test: toggle Archive/Intraday both ways; 4 bots, 28,800 authentic SOL minute bars, fixed + trailing controls, minute portfolio execution, Plotly stop-lines, 16-case sweep. No JavaScript page errors. These are **unit and smoke checks, not historical signal-parity verification**.

## Local release 0.38 (source not uploaded to repo)

- `QUANT_IMBA_Dashboard_0.38_Portfolio_Schutz_Offline.html`: SHA-256 `f3e87a943e03d829aa9ce4c9cf9297747875cc8327ed578e2a8b049fc2a9a57d`
- `QUANT_IMBA_Dashboard_0.38_Portfolio_Schutz_Offline.zip`: SHA-256 `a496cafb575a4a2d8d5a07d3a2b96e9f4e3a9ac50f8c528de01875336a2fc252`

These files are **available in the ChatGPT output, not as files committed on GitHub**. The GitHub connector cannot read a local sandbox binary, and the runtime cannot directly connect to github.com. Upload ZIP as a separate GitHub Release from an authenticated local git/browser environment when possible. This documentation is NOT a substitute for the complete release artifact.

## Next

Regression parity among daily vs reconstructed-minute signals under the exact same bot IDs, periods and ledger; continuous portfolio-high persistence across rebalance, OOS years 2022–2026, multi-asset market synchronization, realistic gap/fee and liquidation scenarios, persistent full sweep case store, and archived preset import verification. Do not activate live execution from this prototype.