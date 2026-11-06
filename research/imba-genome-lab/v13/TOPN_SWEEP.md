# Trading rules: Top-N, stop scope, automatic sweep

**Research-only feature on `research/imba-topn-stop-sweep-2026-11-06`.** Main / Railway / venue executors stay unchanged.

## New controls in the current Genome v13 dashboard

- **Top N** slider: choose 1, 2, 3, 4, 5 or 6 strategies in each regular 14-day rotation.
- **Stop scope** switch: `strategy` (legacy individual trailing peak stop, default) or `portfolio` (trading sleeve high-water drawdown).
- **Existing stop slider** is reused for the threshold in either scope. Its off setting remains off.
- **Automatic sweep**: evaluate every combination of available ranking windows (read from the dashboard's Ranking dropdown), stop thresholds (existing guard slider, at least 5-percentage-point grid plus current value), Top N = 1–6, and both stop scopes.
- **Selected strategy universe stays fixed**: year, candidate pool, price basis, box filter, exit families, correlation cap and fallback. The sweep does not invent new backtested strategies, modify the 3D surface, or change other filters.
- **Results**: five best historical combinations by terminal wealth, each with overall portfolio max drawdown and an **Übernehmen** button. Abort/cancel and progress indicators included.
- **Causal selection** remains based only on fully completed past daily returns. **Choosing sweep parameters from the whole historical year is in-sample optimization** and therefore does **not** imply out-of-sample profitability.

## Portfolio semantics

The initial protected reserve is **€1,000**, with **no external cash additions**. Monthly transfers remain `min(reserve, max(100, 20% × reserve))`, constrained by available reserve. The existing profit-transfer/payout logic remains unchanged. Six arrays are retained internally as a maximum number of slots, but only the selected Top N slots may be assigned capital.

The **portfolio stop** checks drawdown on the *trading sleeve*, cash plus currently trading positions, rather than the protected reserve. Its multiplicative performance index is unaffected by monthly transfers from the reserve or by payouts to the reserve. A global stop liquidates/parks every selected bot, and the sleeve is not re-armed until the next scheduled 14-day rotation. Individual stops keep their previous per-bot behavior.

**Defaults preserve v13:** Top N = 6, stop scope = per strategy. Applying one of the original 2026 screenshot presets also resets these controls to the historical baseline; saving a JSON preset writes both new properties. Old v7–v13 JSON presets load with the same default behavior.

## Build or upgrade

The research branch has no committed large standalone HTML snapshot (its historical gzip-embedded 7,326 daily curves were generated as a separate v13 artifact). There are two ways to produce the updated standalone HTML using the **existing real v13 HTML** and without recalculating all strategy histories:

```bash
python3 research/imba-genome-lab/v13/upgrade_trade_rules.py /path/to/imba_genome_parameter_lab_v13_presets_charts.html
```

This creates `..._topn_sweep.html` next to the source, preserves all the original compressed market data and evidence charts, and swaps in the patched `app_v13.js`. The existing v13 full-build script `build_v13.py` now also reads the patched app when it is present in the same directory.

## Verification performed

Using the original v13 ledger and new ledger with the **same deterministic synthetic daily factor array**, code checks established:

- JavaScript syntax parsed successfully;
- baseline Top-6 / strategy-stop replay gave **max absolute equity delta = 0** across 50 days and identical payout and stop counters;
- Top 1 and Top 2 selected exactly one and two distinct candidate IDs, respectively;
- portfolio stop liquidated both trading positions on the constructed downside event while strategy stop left the unaffected second position running;
- cached pre-ranked sweep evaluation and normal portfolio simulation gave **identical terminal wealth and max drawdown**.

The actual full v13 standalone HTML / 7,326-factor payload was not available in this workspace, so no claim is made that a full 2026 Chromium run or screenshot-preset reproduction has been executed on the modified artifact.
