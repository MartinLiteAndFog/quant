# QUANT 0.42 — UX re-layout and Walk-Forward section diagnostic

Date: 2026-11-07. Offline prototype; no live executor or Railway changes.

## User-requested workflow

The original 3D genome/parameter heatmap remains the primary research display. Existing algorithms for daily archive, intraday signal construction, 14-day/monthly Walk-Forward, sweep and portfolio accounting remain unchanged.

1. The existing `researchDetails` panel has been moved immediately below the 3D chart, above the Walk-Forward portfolio.
2. The old standalone `familyInfoBar` (factual bot selection) has been rehomed as a GENPOOL indicator within the 3D plotting area. It renders the actual selected `poolMode` label, counts and selected dataset/mode: Original 462, 2022 Top 100 Raw, etc. It distinguishes archive preview from genuine prepared intraday genes. The original `pointHint` DOM element is retained for older code compatibility, but not displayed as a separate box.
3. The existing `quantRenkoDeck` and nested `q40Toolbar` have been moved *after* the Walk-Forward portfolio and individual-bot curve view, grouping Intraday mode, diagnostic/trades, USD/% Renko, Box-Sweep and minute archive processing into one compact area.
4. The Walk-Forward ledger has clickable rows, previous/next section buttons, and chart-point selection. Open **Diagnose & Trades dieses Abschnitts** to inspect the chosen completed section.

## Diagnosis: crucial semantics

The section view **does not rerun the bot strategy**. It clips the executed `QUANT40_WF.curve` and `QUANT40_WF.fills` to the selected decision's [start 00:00 UTC, end+1d 00:00 UTC) interval. Displayed starting/ending NAV comes from the actual decision's `before`/`after` values. Actual fill markers and table rows are based on the same ledger events as the Walk-Forward chart. The summary shows realized exit-fill net separately from the remainder (open PnL, fees, transfers and other ledger effects). Original selected bot IDs are read from the decision, rather than the possibly modified pins intended for the *next* section.

The pre-existing standalone diagnostic inputs are synchronized to the viewed section's dataset, end date, exact inclusive day length (14 or calendar-month length), selected bot IDs, source signal bar, selection mode, leverage and allocation where available. Clicking **Minutenportfolio berechnen** remains an independent replay that does **not** automatically carry open positions and cash from earlier Walk-Forward sections. The UI warns about this distinction.

The section-navigation selection is an inspector state, **not** a change to the future Walk-Forward bot group, nor a historical strategy reoptimization.

## Browser checks

- Live DOM relocation order: 3D chart → Research → WF Portfolio → Individual bot curves → combined Renko/Intraday workstation; verified in Chromium.
- GENPOOL badge reports current chosen pool and count; tested with Original 462 and 2022 Top 100 Raw.
- Synthetic two-section WF ledger: select Jan 1–14 vs Jan 15–28; dialog dates and durations follow selection; selected IDs differ appropriately; each shows its own original fill events and minute curve; no JS errors.
- **Real** minute worker: 36,000 SOL-USDT 1m test rows, two IMBA bots, Jan 1–14 2024 WF. End NAV €966.3164102735412 from €1,000, four actual fills, 1,345 NAV records. Opening the view synchronizes Jan 14/14 days and shows those same two bot IDs and four fill rows. No JS errors.
- ZIP CRC and JavaScript syntax checks passed. This does *not* establish TradingView/tick/fill parity, nor validate all multi-year sweep combinations.
- A large-pool browser-switch stress test was slow in a headless environment; it has not been presented as passed.

## Files (available as ChatGPT downloads, not committed as binaries to GitHub)

- `QUANT_IMBA_Dashboard_0.42_WF_Synchronisierung_Offline.html`, SHA256 `a1a52e66542c5f6a81ff53e25994cdd949e4423db0a4a23d11979b77e0bd9c14`
- `QUANT_IMBA_Dashboard_0.42_WF_Synchronisierung_Offline.zip`, SHA256 `fb064a3023174bb473c0186b66c4a48a30be62bde78741e9e2ca764a75dab05e`

The ZIP contains the standalone HTML, `quant_042_layout_wf_diagnostics.js` and `quant_042_layout.css`, and source/tests inherited from 0.41. GitHub's connector cannot directly ingest the full local ZIP as a release asset here. This commit records the implementation, tests and hashes, not the ZIP binary.

## Still outstanding

- Intraday multi-year large parameter/OOS sweep stress tests, historical/day-intraday signal parity, exact exchange fills, minute high/low ordering, cost and liquidation verification.
- App release upload to GitHub. No live deployment performed.
