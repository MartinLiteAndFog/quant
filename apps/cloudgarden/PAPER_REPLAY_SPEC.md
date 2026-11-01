# Cloudgarden paper replay v1

## Goal

Add a reproducible, isolated intraday paper-execution vertical slice to the
standalone Cloudgarden prototype. It demonstrates causal fills and cost-aware
closed-position PnL using a generated one-week fixture, then books the replay's
net result into the existing 100-USDT garden ledger. The fixture is explicitly
synthetic; it is not a strategy test and cannot produce empirical trading
evidence.

## Data contract and eligibility

- Bars use UTC timestamps that identify the bar open, a fixed interval, and
  finite positive OHLC prices plus non-negative volume.
- The dataset manifest requires source, license, symbol, market type, interval,
  contract multiplier, timezone, timestamp convention, and a synthetic flag.
- Normalization sorts deterministically, rejects duplicates, off-grid bars, and
  invalid OHLCV, and reports gaps without filling or interpolating them.
- This v1 app path accepts generated synthetic data only. The existing local
  SOL file is not imported because its source, venue/contract, and license are
  not verified. No UI setting can promote unverified data to empirical.

## Signal and execution semantics

- Signals are externally supplied target quantities. The first generated stream
  uses a versioned `TradingSignalGenome` separate from the ecological genome;
  its close-to-close momentum threshold is a synthetic fixture, not strategy
  alpha. No signal is inferred from future or current-bar close values.
- Signal, availability, and receipt timestamps are distinct and timezone-aware.
- A market order fills at the first observed bar open strictly after both
  availability and receipt. The delay is bounded from availability; missing or
  stale bars and zero-volume bars reject rather than fabricate a fill.
- Signal reports count observations, pre-close executable opportunity
  observations, and emitted targets separately. The final eligible observation
  is reserved to close positions and cannot open a new one. These opportunities
  are not independent market events; a genome-specific threshold is not a fair
  denominator for comparing strategies. A week with no eligible move is
  `no_trade`; it has no invented trading loss. Strategy output remains
  `not_comparable` to market performance and does not alter habitat fit, vigor,
  or plant evidence.
- Spread and slippage are adverse adjustments embedded in execution price.
  Fees are charged once on filled notional. Slippage cost is diagnostic and is
  never debited a second time.
- Linear-futures funding is a separate signed cashflow, at most once per unique
  settlement, over the holding interval `(open, close]`. Every supplied
  settlement timestamp must fall inside the covered dataset window; out-of-range
  events reject instead of silently disappearing from the report.
- v1 accepts only replays that finish flat. An open position is never silently
  liquidated at the end of a fixture.

## Capital and integration

- Paper positions are bounded by the selected active plant's existing capital
  entitlement at 1x leverage; no separate 100-USDT account is created.
- A closed replay updates the existing garden NAV by price PnL minus fees plus
  funding. The 10% reserve and 90% allocation ceiling are recalculated from
  that NAV; the capital ledger must reconcile after the result is booked.
- A loss that cannot be reconciled within the selected plant's entitlement
  fails closed. An order that exceeds the plant's available allocation is
  rejected; quantities are never silently clipped.
- Replay processing advances the ecology by exactly the replay's whole-day
  horizon. Minute fills and daily plant updates share the same starting world
  snapshot and the resulting NAV is carried forward. World-mutating controls
  are locked during replay so a stale snapshot cannot overwrite a step, policy
  change, or reset.
- A versioned synthetic evidence report reconciles price PnL, fees, funding,
  net PnL, ending cash, and daily marked drawdown, then lists observations,
  executable opportunity observations, targets, fills, and rejected signals. Seven synthetic
  days always remain `insufficient_sample`; independent opportunity count,
  confidence, fitness, Sharpe, CVaR, and OOS claims are unavailable. Drawdown
  is descriptive for this fixture only.

## Verification

Tests must cover deterministic synthetic bars and replay, strict next-open
timing, late and missing-bar rejection, zero-volume rejection, no duplicate
cost charging, funding signs/boundaries/deduplication, flat-at-end enforcement,
allocation limits, net PnL booking, 90/10 reconciliation, and replay/ecology
day alignment. The UI and reports must say synthetic and `not_comparable` to
market performance.

## Deferred gates

This slice does not claim historical-data provenance, venue-calibrated costs,
TradingView parity, strategy alpha, multiple independently funded open
positions, arbitrary genome-to-signal execution, exchange behavior, or live
capital eligibility. Those require separate validated evidence and controlled
integration work.
