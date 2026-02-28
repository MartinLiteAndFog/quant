# Regime-Aware Dashboard and Independent Gate Store Design

**Date:** 2026-02-23  
**Status:** Approved (brainstorming phase)  
**Scope:** Infra and dashboard design for regime storage, observability, and binary gate execution (single account), with forward compatibility for confidence-driven scaling.

---

## 1) Goals

- Make regime logic independent from TradingView as a source of truth.
- Persist regime decisions and inputs for improvement and post-analysis.
- Expose live regime state and execution context in the dashboard.
- Keep current execution binary (`gate_on` true/false) for one-account constraints.
- Prepare data model for future confidence-based up/down allocation across multiple accounts.

## 2) Current Reality (as of this design)

- Backtest regime logic is strong (hysteresis, walk-forward, gate CSV integration).
- Live dashboard currently covers API status, ticker, and position.
- Live signal worker currently emits IMBA signals but is not yet wired to a persistent regime store.
- Historical-data depth and feature persistence for regime decisions are not yet productionized in live infra.

## 3) Architecture Decision

**Chosen direction:** phased local-first architecture.

- Phase 1: SQLite as the regime source of truth (single writer/low ops).
- Phase 2: migrate to Postgres with same schema contract and repository layer.
- External sources (including TradingView) may be used for comparison or enrichment, but not as authoritative truth.

## 4) Regime Model and Decision Policy

### 4.1 Operating policy now

- Execution remains binary (`gate_on`) due to one-account limitation.
- Regime decision should be sticky and multi-day (flip-resistant).
- Fast intraday risk guard can block entries, but does not redefine structural regime.

### 4.2 Forward-compatible signal model

Persist these even if execution initially consumes only `gate_on`:

- `regime_state` (`trend`, `countertrend`, optional `neutral`)
- `gate_on` (`0/1`)
- `regime_score` (`-1..+1`, optional at first)
- `confidence` (`0..1`)
- `reason_code` (transition or hold reason)
- `model_version`
- `threshold_snapshot_id`

### 4.3 Anti-flip controls

- Hysteresis (different enter/exit thresholds)
- Confirmation windows
- Minimum dwell time
- Debounce/grace on gate-off transitions

## 5) Storage Design

### 5.1 Core tables (SQLite first, Postgres-ready)

- `regime_state_ts`
  - `ts`, `symbol`, `gate_on`, `regime_state`, `regime_score`, `confidence`
  - `reason_code`, `model_version`, `threshold_snapshot_id`
  - `feature_values_json`, `created_at`

- `regime_threshold_snapshots`
  - `snapshot_id`, `fitted_at`, `symbol`
  - `train_window_start`, `train_window_end`, `dead_days`
  - quantiles/bands used, feature-set signature, `model_version`

- `regime_transitions`
  - `ts`, `symbol`, `prev_state`, `new_state`
  - `trigger`, `confirmation_bars`, `hold_time_prev_state`, `debounce_applied`

- `regime_data_quality`
  - `ts`, `symbol`, `missing_bars`, `stale_age_sec`, `source_name`, `fallback_used`

### 5.2 Data handling requirements

- Every decision row must be reproducible from stored thresholds + features.
- Store enough metadata to explain regime changes on chart and in logs.
- Maintain UTC timestamps throughout.

## 6) Dashboard Design (first version)

### 6.1 Main chart

- Live Renko chart as primary panel.
- Scrollable history (not fixed short window only).

### 6.2 Regime visualization

- Background shading overlays:
  - Gate ON: green
  - Gate OFF: blue
- Confidence drives transparency/intensity:
  - stronger confidence -> stronger color intensity
  - lower confidence -> lighter shading

### 6.3 Trade and risk overlays

- Trade markers (entries/exits, long/short distinctions).
- Active levels rendered on chart:
  - `SL`
  - `TTP` (when armed)
  - `TP1`
  - `TP2`

### 6.4 Right-side readout

- Current confidence score shown at right side of chart.
- Also show: current regime state, gate status, reason code, last update age.

## 7) API / Service Contracts

- Add regime status endpoint(s), e.g.:
  - latest state for symbol
  - recent transition log
  - recent chart overlays payload (gate spans, levels, trades)
- Execution components read regime state from store/API (not ad hoc files as final design).
- Keep fallback to last-known-good state if update job fails.

## 8) Testing Strategy

- Unit tests for regime decision persistence and transition correctness.
- Integration tests for:
  - feature -> threshold snapshot -> decision row pipeline
  - dashboard payload assembly (gate spans, confidence, levels, trades)
- Regression tests ensuring no lookahead bias in historical recomputes.
- Smoke tests for stale/missing data behavior and fallback handling.

## 9) Risks and Mitigations

- **Insufficient historical depth:** implement backfill + quality flags before trusting decisions.
- **Flip instability:** enforce dwell/hysteresis and track transition churn metrics.
- **Schema drift on migration:** isolate DB access behind repository contract from day one.
- **Operator trust:** expose reason codes and threshold snapshot IDs directly in dashboard/API.

## 10) Phased Delivery

### Phase 1 (now)

- SQLite regime store
- Persist state/snapshots/transitions/quality
- Dashboard regime overlays + confidence shading + levels/trades overlays
- Binary execution policy (`gate_on`)

### Phase 2 (later)

- Postgres migration
- Stronger scheduler/recompute infra
- Multi-account allocation consuming `regime_score` and `confidence`

### Phase 3 (future)

- Confidence-driven phased in/out orchestration
- Portfolio-level regime coordination across accounts/strategies

---

## Decision Summary

- Build independent regime infra now.
- Execute with binary gate now.
- Persist confidence and score now for future smooth allocation.
- Use dashboard as both operations panel and analysis instrument.
