# Independent Elliott Wave / MTF reconstruction — progress 2026-10-18

**Research verdict: `not_comparable` for executable strategy performance.**
Benchmark: frozen TradingView BINANCE:SOLUSDT exports `(4)` and `(5)` dated
October 2026. No champion promotion or profitability claim is made.

## Verified milestone

`independent_run.py` generates chart Renko, HTF Renko, raw pivots, ATR, wave
states and signals from Binance spot OHLC minutes. Reference chart values and
signals are loaded only after predictions have been generated, for evaluation.
The historical chart uses 1h source bars before September 7 and 5m afterward,
with a 0.1 box and two-box reversal. The transition was inferred from the export;
it is export-specific metadata, not a universal TradingView rule.

In the fixed September 7 through October 17 export window, audit export `(5)`:

| Signal | Reference | Matched | Missing | Extra |
|---|---:|---:|---:|---:|
| Long entry | 84 | 84 | 0 | 0 |
| Short entry | 57 | 57 | 0 | 0 |
| Bullish Wave-5 exit | 18 | 18 | 0 | 0 |
| Bearish Wave-5 exit | 16 | 16 | 0 | 0 |

These are indicator signals, not the Strategy Builder's complete order history.
This is in-sample historical reconstruction, not independent out-of-sample proof.
The full export includes a cold-start interval outside this comparison window.

The matching configuration has primary/secondary lengths 13/5 and effective
1h/4h pivot lengths **7/5**. The supplied Pine defaults specify **8/7**. Both
configurations are reported; with the defaults only 117 of 141 entries match,
with 24 missing and 38 extra. The configuration/source identity remains unresolved.

## What changed

- Ported pivot confirmation, wave-pattern selection, HTF trend/phase/confidence,
  alignment and MTF confidence weighting to Python.
- Implemented Pine's comparison precision separately from arithmetic precision.
- Reconstructed Renko opposite wicks and pending-price reset behavior; this
  removed HTF ATR differences in the tested raw-field window.
- Kept market-source completion times separate from synthetic Renko timestamps.
- Compared auto/manual configurations and a long-history bootstrap rather than
  treating a single favorable run as sufficient evidence.

## Data provenance

The supplied `Downloads/SOLUSDT (3).csv.gz` contains 3,250,370 minute rows from
2020-08-11 to 2026-10-17 10:33 UTC. SHA-256:
`c4f9b9f85d00072f8e367a4bd23978ec92c3bce0d326fe3d17b2c26b5f3d1c9c`.
The retained July-onward slice has 156,154 contiguous rows. The filename alone
does not establish spot/futures provenance.

A separate public Binance spot download contains 157,440 contiguous minutes
from July 1 to October 18 08:00 UTC (exclusive). The common interval has 71
minutes with different OHLC values versus the supplied file. For example,
October 12 19:29 UTC closes at 108.10 in the supplied file and 108.20 in the
Binance response. The table above uses the official spot download.

Raw downloads, user chart exports, workbooks and full per-bar outputs remain
local and are excluded from Git. Public response hashes and compact comparison
reports are included. New downloads may differ if the provider revises history.

## Unresolved gates

1. **Causality:** historical matching uses Renko/security mapping whose source
   bars can complete later than the displayed chart time. On the earlier
   supplied-data window with 140 entries, 92 had later HTF1 dependencies and
   119 had later HTF2 dependencies (maximum 55 and 230 minutes respectively).
   This flags future dependencies, not proof that every flagged signal changes.
   A completed-source-only alternative matched 116/140 entries, with 24 missing
   and 23 extra. That alternative is not yet proven equivalent to realtime Pine.
2. **Original export `(4)`:** 11/5 manual 4h/day with 8/8 HTF lengths matches
   143/153 entries and all 44 Wave-5 exits in the July-initialized run. Extending
   initialization to 2020 changes the result substantially: 129/153 entries
   match, 24 are missing and 34 extra. Warmup/source identity needs resolution.
3. **Strategy Builder:** complete conditions, RSI inputs, entries, stop/trailing
   exits and position state have not been independently matched.
4. **Execution:** synthetic prices do not prove executable prices. Fees,
   slippage, latency and intrabar ordering still require evaluation.

## Reproduction and checks

Python dependencies: `numpy`, `pandas`, `openpyxl`.

Run from this directory:

```sh
python -m unittest test_replay.PineSemanticsTests
python fetch_binance_minutes.py
python independent_run.py
```

The last command additionally needs the original chart exports named
`inputs/BINANCE_SOLUSDT, 5 (4).csv` and `(5).csv`; they are deliberately not
published. Create `results/` if starting without the committed reports.
`audit.py` and `prepare_market.py` are local acquisition helpers with explicit
original Downloads paths; adjust those paths on another machine.

All nine `test_replay.py` tests passed locally with the frozen audit input.
Five test numeric/pivot/pattern semantics; four use private reference exports.
The diagnostic prefix test verifies the downstream replay conditional on its
provided HTF inputs; it does **not** establish raw-input or execution causality.
The independent four-configuration replay was rerun before this publication.
See `results/market_only_signal_report.json` for missing/extra event timestamps.
Older raw-model and causal diagnostics are distinct experiments, not the final
official-source run.

## Source and interpretation

The numerical port derives from `EW_v3_2_MTF_AUDIT.pine`, whose notice reads
“© Elliott Wave Advanced Multi-Degree Analyzer — v3.2 MTF Edition” and specifies
[Mozilla Public License 2.0](https://mozilla.org/MPL/2.0/). The Python port is
published with that notice and license designation.

TradingView documents [float comparison precision](https://www.tradingview.com/pine-script-docs/language/type-system/),
[historical Renko resolution changes](https://www.tradingview.com/support/solutions/43000480330-the-renko-chart-shows-incorrect-values-and-or-is-constantly-being-recalculated/)
and [non-standard chart data](https://www.tradingview.com/pine-script-docs/concepts/non-standard-charts-data/).
Exact reconstruction of this export is not evidence that synthetic historical
signals were available at their displayed time in live trading.
