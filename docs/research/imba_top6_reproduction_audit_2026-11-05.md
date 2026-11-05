# IMBA Top-6 portfolio reproduction status — 2026-11-05

## Source of truth (archived, **not** reproduced)
The saved `portfolio_results_2023_2026.csv`, case `main_10x_100_50`, records isolated starts with €1,000 and **zero external deposits**:
- 2023: €6.118467
- 2024: €498.526612
- 2025: €8,305.644975
- 2026 through October 16: €14,514.312436

The original portfolio script was generated in a previous ChatGPT session and packaged into `portfolio_2023_2026_analysis.zip`, present in the user's Quant Library. The current session cannot materialize its raw bytes. This repository did not contain that original backtest as of this audit. Existing replacement engines are **not exact parity**.

## Known defects in former reconstruction
- External €100 monthly funding was mistakenly added.
- Later code implicitly returned active trading capital to the bank when changing selected six bots.
- Its bot-level drawdown stops did not represent the original portfolio-cycle pause.
- A profit withdrawal decreased trading balance while leaving the old trading-equity high-water mark, thereby falsely triggering a 20% drawdown and stopping on a bookkeeping transfer.
- Five strategies were represented twice in the combined 333-row registry; canonical unique count is 328.
- The daily underlying bot signals are not equal to the archived ones: example S120/box USD0.12 had archival 2025 1x return +151.46%; current 5m Renko reconstruction approximates -16% instead. Do not tune the 10x portfolio ledger to force a match.

## Corrected, reproducible counterfactual (not exact parity)
- €1,000 start, no outside funds; each calendar month transfer `min(freeReserve, max(100, 0.2 * freeReserve))` into an ongoing trading pot.
- Refresh active 6 every 14 days using only prior 14 complete days' 1x returns; bots not selected are dormant.
- 10x factor exactly once on the average six-bot 1x daily return.
- Stop for remainder of the cycle after a 20% *market-performance* drawdown from the rolling cycle peak; use unitized NAV, invariant to deposits/withdrawals.
- At trading-pot equity doubling relative to the current principal reference, bank half the gain. Transfers are internal and cannot trigger the drawdown stop.
- No weather filter; current synthetic daily model still ignores intraday liquidation and is not TV-parity verified.
- Original 60 = archived 2025 quality-selected cohort; broad pools 283 and 328 unique candidates. The 328 pool incorporates historical and newly discovered bots retroactively, and is therefore a research counterfactual, not a tradable 2023–2025 strategy.

| Independent year | Original 60 | Expanded 283 | Unique 328 |
|---|---:|---:|---:|
| 2023 | €4.90 | €10.28 | €10.80 |
| 2024 | €17.46 | €166.68 | €380.13 |
| 2025 | €557.79 | €11.49 | €14.78 |
| 2026 to Oct 16 | €934.43 | €661.70 | €704.61 |

All combined models deplete the uninterrupted 2023–2026 trading account under these assumptions. **These results do not invalidate the archived €8.3k/€14.5k results because the engines and source daily bot returns differ.** The missing original code and original 1x bot curves are a blocking parity dependency.

The full runnable research script, input 1x returns (NPZ), selection ledger, cash-flow audit, charts, and CSVs were exported together as `imba_portfolio_reproduction_v2_bundle.zip` in this conversation. Keep that bundle together to reproduce the counterfactual without needing to rerun the multi-million-row Binance Renko construction. Do not activate or deploy these experiment outputs to live execution.
