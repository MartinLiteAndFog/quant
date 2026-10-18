# Elliott Wave v3.2 parity investigation

Goal: reproduce TradingView alarms independently from causal market inputs, match
downloaded Strategy Builder entries AND exits, then calculate executable prices
and cost-adjusted backtests. No production promotion until every layer passes.

User clarification (2026-10-18): independent HTF/MTF raw-value reconstruction is
the immediate priority. Reference-fed replay is only a diagnostic and must not
be substituted for this work or presented as completion.

Benchmark: frozen BINANCE:SOLUSDT TradingView exports dated October 17–18, 2026.
Each export/configuration is a separate benchmark. Research verdict remains
`not_comparable` until a full independent run is possible; partial diagnostic
matches do not prove alarm, execution, or profitability parity.

Plan and gates:
1. Inventory/hash sources, preserve duplicate property names and millisecond Renko
   ordering, identify chart timeframe, actual HTFs and indicator configuration.
2. Reproduce Pine pivot, pattern, MTF alignment and signal logic. First isolate
   downstream logic using exported HTF raw inputs, clearly marked diagnostic.
3. Independently construct HTF Renko/request.security inputs and MTF RSI; compare
   every intermediate field against exports. Validate prefix invariance and
   availability times. Never use reference signals/states to generate candidates.
4. Reproduce Strategy Builder conditions, position/order state and all entry/exit
   events for each configuration. Report missing, extra and mismatched events.
5. Price orders from real market data at/after causal availability; quantify fees,
   latency and slippage, unresolved intrabar paths and data gaps. Historical
   synthetic Renko timestamps are not automatically execution timestamps.
6. Freeze evaluator, run tests and independent holdout/realtime checks, then
   publish evidence and exact limitations. Complete only with all gates proven.

Initial verified facts:
- Audit CSV has 10,264 rows, 92 long and 59 short signals (151 total).
- Its new-period fields correspond to 1h and 4h, not 4h and daily. Each has just
  the expected first-row mismatch against a locally initialized period detector.
- Pine defaults are 13/5 swing lengths; prior chat mentions 11/5. Configuration
  identity must be established before judging mismatches.
- Previous Python files are absent from searched project/Downloads locations;
  shared-chat artifact links have no downloadable href. User location requested.
- Existing unrelated working-tree modifications must be preserved.

Verification: source-anchored pattern tests, pivot tie tests, deterministic replay,
prefix invariance, per-bar intermediate and final comparisons, full trade-event
comparison, time and price provenance checks. A reference-fed diagnostic replay
must never be labelled an independent backtest.
