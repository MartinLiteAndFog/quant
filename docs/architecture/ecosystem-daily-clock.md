# ECO-03: reproducible UTC day clock for paper ecology

**Scope:** `src/quant/ecosystem/sim_clock/clock.py` is a pure deterministic tick scheduler. There are no database, KuCoin, Kraken, Railway, TradingView, network or wall-clock calls.

- `new_simulation(simulation_id, equity_decimal_string, first_day_utc, seed)` starts the tick state on the preceding UTC date.
- `advance_day(state, day, intraday_events, reducers)` advances exactly one day or raises without mutating the old state.
- `advance_days` runs a batch; `SimulationState.to_json/from_json` is a deterministic canonical checkpoint/restart path.
- `IntradayEvent` requires unique source `event_id`, valid timezone-aware `occurred_at <= available_at`, explicitly typed incremental signed NAV amounts. `signal` always has zero amount. `fee` is nonpositive. No lookahead/future events.
- **Only paper execution supplies PnL.** Amounts labeled `pnl_delta` must be incremental NAV change since previous marking. Do NOT pass cumulative unrealized PnL and realized PnL for the same economic event; no historical fills are invented.
- `execution -> metrics -> capital -> ecology -> breeding -> lifecycle -> snapshot` phases are fixed. The clock makes exact Decimal 90/10 reserve calculation and audit logs; other modules add phase reducers. A reducer receives only current-known, copied daily data plus a **local seeded RNG** keyed by simulation ID/day/phase. It returns a replacement JSON-compatible ecology/biological state, never real orders. Reducers must be pure (no I/O), stateless and versioned. No-op reducers are allowed; the snapshot marks which phases have custom logic.
- Pause/resume: serialize snapshot and continue; multi-run deterministic for the same input series+seed+reducer version. Replaying an identical day or event ID is refused, not double-counted.
- Fees, funding, external cashflows are distinguished from trading PnL for later cashflow-adjusted TWR; the clock does NOT calculate Sharpe/fitness or model fills. ECO-02 supplies causal execution, ECO-07 supplies metrics, ECO-05 supplies per-account Groundwater budgets, and ECO-06 supplies ecology reducer.
- Global negative equity freezes **new** distributable budget and does not automatically flatten positions. The clock does not know real margin or account reconciliations: production executor remains entirely separate.

## Tests and intended integration

`python -m unittest discover -s tests/ecosystem -p test_sim_clock.py -v` runs synthetic tests for midnight/year rollovers, canonical snapshots, deterministic seeded RNG, crash/restart, fees/funding, external deposits, negative NAV, duplicate IDs and lookahead. A separate GitHub workflow runs without any venue credentials.

**Contract compatibility:** ECO-01 `SimulationEvent` names/phases/schema align conceptually; a validated interchange adapter is still required once branch `codex/ecosystem-contracts` lands. This module is not a claim that full five-year SOL trades or biological outcomes have been simulated.
