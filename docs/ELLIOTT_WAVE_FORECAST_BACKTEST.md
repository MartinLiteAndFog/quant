# Elliott Wave Detector Pro v3.2 — SOL Forecast / Renko backtest

Status: **research specification / implementation pending** (2026-10-17). This document records the agreed test, not a completed TradingView-parity or profitable backtest. Do not infer live-trading readiness from this note.

## Objective

Evaluate the **unmodified** Pine Script EW Detector Pro v3.2 grades and its native `generateForecast()` action, target zone and stop. Compare:
1. Ordinary SOL/USDT **5-minute OHLC** candles.
2. **TradingView Traditional Renko**, fixed **0.10 USDT box** on a **5-minute source interval** (user's KuCoin SOLUSDT chart).

Keep scenario generation separate from order fills: all entry, target and stop execution for both variants is priced from **real, timestamp-aligned SOL/USDT market OHLC**, preferably 1-minute with conservative same-minute ambiguity handling. Never fill trades at synthetic Renko open/close. No parameter optimization before the baseline is audited.

## Existing Renko implementation — reuse, do not reinvent

- **Canonical builder:** `src/quant/features/renko.py::renko_from_close()`.
- **Validated parity investigation:** `docs/RENKO_TRADINGVIEW_PARITY.md` (2026-10-07).
- Continuation = 1 box; established-trend reversal = **2 boxes measured from prior brick close**; first reversal brick opens 1 box beyond prior close. Multiple bricks from one source bar receive successive **1 ms** timestamps to prevent deduplication.
- **Mandatory:** resample 1-minute OHLCV to the same **5-minute source candles** before constructing a 5m TradingView Renko series.
- The parity document compared **Kraken SOLUSD.PM** TradingView export, whereas the screenshot for this EW study is **KuCoin SOLUSDT**. Feed-specific differences may remain. `renko_from_close()` is close-driven and explicitly does not guarantee exact TradingView history/wick/alert parity. Match source venue, interval, historical coverage and signal timestamps where possible.
- Existing October 7 comparison: TV = **5,322 bricks / 99 IMBA flips**; corrected 5m-sourced reconstruction = **5,279 bricks / 105 flips** for the documented window. This is a major improvement, **not an exact byte-for-byte match**. Source-wick parity not established.

## Pine Script forecast rules (must be ported literally)

Source: user-supplied Pine Script text, `generateForecast()`, lines ~1035–1124 of the v3.2 file.

Only evaluate a **confirmed, gated** pattern: `confirmedPattern != none`, confidence >= 0.35, and `endIdx == lastP.idx`. In the non-gated case, Pine shows **reference levels only, no stop**: **no trade**. Do not treat forecast labels painted on pivot bars as knowable before their confirmation.

- **Completed motive impulse/diagonal**: forecast Wave A correction, direction opposite the motive trend; projected zone between the last primary pivot and `pivot ± 0.618 × lastWaveLen`, stop at `pivot ± 0.1 × lastWaveLen`; action `TAKE PROFIT / SHORT` or `TAKE PROFIT / LONG`. Treat these separately from unequivocal entries because the action also indicates closing a pre-existing trend trade.
- **Completed corrective pattern** (zigzag, flat, triangle): direction **opposite correction direction**, `BUY ZONE` or `SELL ZONE`; target bounds include `pivot ± 1.618 × lastWaveLen` in the new trend direction with the secondary bound at `pivot ∓ 0.1 × lastWaveLen` for bulls or `pivot ± 0.5 × lastWaveLen` for bears, and stop `pivot ∓ 0.382 × lastWaveLen`. For comparisons, separately measure **first favorable target-zone boundary** vs **far projected target** if applicable and clearly label which TP is used.
- Base grade `round(confidence × 5)` after HTF score adjustment; if completed correction and HTF1 trend is known, HTF1 bias can raise/lower the **displayed grade by one additional star**. The final grade for grouping must be the **displayed forecast grade**, not raw rule score.
- MTF automatic selection on **5m** = **1h and 4h**. Use only higher-timeframe information confirmed by signal time.

## Critical distinction: forecast versus signals (2026-10-17 code review)

In the original Pine v3.2, `generateForecast()` is called **only inside `if barstate.islast`** (roughly lines 1247–1268). Consequently, forecast grade/action/target/stop are **drawn for the last/current bar only** and are not emitted as historical alert events, and there is **no forecast-entry order mechanism** in Pine. To backtest "forecast appears" causally, calculate a historical snapshot on **each confirmed bar** and log the first transition into a distinct confirmed, gated forecast; do not read a completed chart's last-bar display backwards.

The actual `sigLong`/`sigShort`/`sigW5Exit*` alerts are separate (roughly lines 998–1029, 1396–1405): evaluated when `primaryPushed` and optional W3/WC/W5 logic fires. Some forecast appearances have **no accompanying** entry alert and some W3 signals exist without a gated forecast. A completed correction `BUY ZONE`/`SELL ZONE` is a **forecast action label**, not a concrete order/limit-entry band: its `targetHigh`/`targetLow` are forecast price **target zone** bounds. The visually extended `projectionBars = 30` only determines the drawn forward horizon; **not** an entry window, exit after 30 bricks, expiry or mandatory holding time. The chart draws a new forecast snapshot as pattern/HTF state changes, but must not count every redraw as another trade.

Therefore test **two distinct, predeclared strategies**, both on realistic real-market prices:
A. **Forecast-as-signal**: enter hypothetically at next eligible real bar after the **first confirmed new** BUY ZONE / SELL ZONE forecast (deduplicate same pattern/pivot; account for grade changes). Exit at native stop or directional projected target.
B. **Forecast-plus-alert — primary filtered-entry strategy**: forecast defines direction/stop/target and remains valid in a documented state machine; enter **only** on a subsequent original Pine W3 or wave-C entry alert (`sigLong`/`sigShort`) if **all** conditions hold at its confirmed timestamp: (i) a previously confirmed **active, gated BUY ZONE for LONG or SELL ZONE for SHORT** exists; (ii) the alert direction **matches** that forecast; (iii) the forecast has **not** been superseded, invalidated at its native stop, or already completed at its native target; (iv) the next real-market execution price still gives a valid stop/target ordering. All standalone, opposite-direction, and expired-forecast alerts are **excluded**, with rejection reason counted. Use the exact native stop and target from the matched forecast snapshot (record forecast/alert IDs and delay). An alert occurring on the same confirmation bar as a newly established forecast may qualify only if both are fully confirmed at bar close and entry is strictly afterwards. Report overlap/delay and actual filtered trade returns; never treat an unrelated historical entry alert as a forecast trade.

Report setup/alert overlap matrix, delays from forecast onset to signal, percentage of forecast-only setups, realized P&L for A vs B. This is a **research extension** to the original baseline, not an assertion that Pine trades the forecast by itself.

## Trade/event conventions for initial baseline

- Preserve a unique confirmed forecast **event id**, signal-generation bar, **actual confirmation timestamp**, pattern, raw rule score, MTF inputs, final grade, action, direction, targetLow/targetHigh, stop.
- A `BUY ZONE` or `SELL ZONE` is **not automatically a market-entry alert**. To quantify its actionable potential, baseline **entry at first available real-price candle open AFTER confirmed forecast publication**; tag as **hypothetical forecast-entry assumption**, not original Pine buy/sell execution. Provide separate subset for explicit W3 and wave-C signal markers, and optionally zone revisit entry later.
- For synthetic Renko, the builder's source-bar timestamp labels **must not imply brick knowledge at the source bar open**. For a source 5m bar, mark resulting bricks as available **no earlier than its close**, including multiple same-bar bricks; signal confirmation and MTF must be causal. Entry at first following real 1m open or later (plus modeled latency).
- Validate TP/SL direction/ordering and discard stale, missing, reversed, already-breached stop/target or nontradeable setups with logged rejection reasons.
- Execute TP / SL against original **1m OHLC**, not Renko bars; when both touch inside one 1m bar and path is unknown, count **stop first** conservatively (and separately flag the ambiguous case). Account for gaps, taker fees and slippage at entry/exit; explicitly state assumed fees.
- No overlapping positions in a single equity curve unless portfolio policy is separately modeled; do not count repeated render updates of the same forecast as fresh trades. Exit upon **first real-price TP or SL**, with a declared max holding cutoff to avoid indefinite positions (report cutoffs separately; evaluate alternatives only after baseline).
- Calculate realized trade count, win %, average/median net %, profit factor, realized equity, max drawdown, exposure, long vs short, pattern, grade, calendar split, and out-of-sample performance. Keep unrealized MFE/MAE and true realized P&L distinct.

## Inputs and known prior work

- Historical SOL minute file (user Library): `SOLUSDT (2).csv.gz`, reported **3,235,682 1-minute bars from 2020-08-11 to 2026-10-07**, with **1,424 missing minutes**. Verify provenance and venue before declaring KuCoin parity; investigate gaps.
- Earlier local baseline artifacts (not committed; no confidential market file to GitHub without request): `elliott_sol_5m/grade_events.csv`, `grade_events_with_targets.csv`, `grade_summary.csv`, `fixed_tp_sl_grid.csv`, `fib_target_stop_results.csv`, `data_and_run_metadata.json`.
- Earlier **ordinary-candle Python approximation**: **17,733** confirmed pattern refresh events, including **1,667** labeled 5-star. These are **not** validated TradingView outputs, are **not Renko** and are **not 1,667 proven trade entries**. Earlier fixed TP/SL experiments do **not** resolve this study.
- Screenshot example dated 2026-10-17 (outside historical data): KuCoin SOLUSDT 5m Traditional Renko 0.1; **expanded flat**, rule score 73/100, partial MTF 0.33, `SELL ZONE`, **five stars**, target **101.27**, stop **106.13**; *illustrative only, not a backtested fill*.

## Next concrete actions

1. Reuse the repository's parity-corrected `renko_from_close()`, on 5m-source-candle close; independently reconstruct ordinary 5m candles.
2. Port full Pine swing/ATR, primary and secondary pivots, pattern selection, 1h/4h MTF, gated `generateForecast()` and grade corrections for **both** series, bar by bar, with event publication time.
3. Compare a sample of forecast patterns/grades/levels/timestamps to actual TradingView exports; be explicit about any remaining Renko feed/wick parity deviation.
4. Backtest next-real-1m-open forecast entries with **native** target and stop. Produce separate Renko vs candle, grade 3/4/5 and pattern-type tables with fees, realistic prices and maximum drawdown.
5. Treat this baseline as frozen before exploring trend filters, zone-revisit entries, multiple profit targets or grade recalibration.

No live bot, TradingView alert or existing strategy has been modified by recording this research specification.

## Provisional 2025–2026 cleaned findings (2026-10-17)

**Research only; NOT TradingView parity verified.** Underlying local, user-supplied SOL/USDT 1m price file, 2025-01-01–2026-10-07 test window (2024-06 warm-up), corrected 5m-source Renko two-box reversal, original native Forecast Fibonacci TP/SL formulas, NEXT REAL-minute-open fills, 0.10% assumed fee **plus** 0.02% slippage each side (0.24% roundtrip), one non-overlapping 100%-notional position per strategy:

| Five-star only | Valid nonoverlapping trades | Win % | Average **net** return/trade | Profit factor | Compounded 1× return | Max equity drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary 5m, A forecast entry | 361 | 39.89% | -0.2677% | 0.733 | -64.97% | -73.12% |
| **Ordinary 5m, B active-forecast + matching entry alert** | **331** | **39.27%** | **-0.2294%** | **0.765** | **-56.55%** | **-65.02%** |
| Renko 0.10, A forecast entry | 473 | 42.07% | -0.1731% | 0.764 | -58.44% | -60.44% |
| **Renko 0.10, B active-forecast + matching entry alert** | **469** | **42.00%** | **-0.1730%** | **0.764** | **-58.11%** | **-60.12%** |

Forecast counts before execution screening (BUY/SELL only): 5-star ordinary **502**, Renko **565**. Earlier **1,667** event count refers to different all-history candle analysis and should not be compared directly.

**Major discovery:** among matched forecast alerts, **2,523/2,528 (ordinary)** and **3,687/3,691 (Renko)** coincide with the same confirmed 5m source bar as the forecast. Of 5-star matched forecasts, **423/424** ordinary and **550/550** Renko have same-bar alerts. Therefore, this original W3/Wave-C indicator **almost never produces a later entry alert** during an already-present forecast; forecast-vs-alert is primarily a filtering decision. Do not present its original WC alert as a subsequent independent entry confirmation.

Exclusion now enforced: no matching entry after its forecast's stop/target had already been reached by real 1m prices; only direction-matched BUY/SELL forecast entries; ignore displaced forecast after a newer one; conservative SL-first path if both levels touched within one minute. Historical 2025-26 minute source segment had no missing 1m bars.

Gross B five-star mean returns **before** the assumed 0.24% roundtrip costs: +0.0106% ordinary, +0.0670% Renko. Both turn negative net. These are **provisional Python-port** outputs, not TradingView-verified broker fills, and no model selection should treat the observed pattern-specific subgroups as a validated edge.

**Remaining required validation:** exact KuCoin/TV 5-minute Renko pivots, native historical wicks, original TradingView v3.2 forecast + W3/WC signal-by-signal parity, on-Renko 1h/4h MTF semantics, and non-gated primary pattern changes which may terminate an active forecast earlier. Local outputs (not uploaded/committed) `/mnt/data/elliott_sol_5m/forecast_ab/cleaned_summary.csv`, `cleaned_trade_candidates.csv`, `cleaned_match_audit.csv`, `ERGEBNISSE_VORLAEUFIG.md`, and executable reproducibility script `clean_backtest.py`. Do not claim the entire 2020–26 period was rerun.

