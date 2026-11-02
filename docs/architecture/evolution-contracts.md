# Ecosystem contract v1.0 (ECO-01)

The canonical implementation is `quant.ecosystem.contracts`. The Fleet wire interfaces
are `ecosystemContracts.ts`; JSON examples and a generated JSON Schema live in
`tests/fixtures/ecosystem/`. All examples are synthetic paper worlds, never trading
instructions. Existing live, legacy monthly, and family pilot behavior is unchanged.

## Boundary and version policy

Use `load_snapshot` to validate/migrate and `dump_snapshot` for canonical sorted JSON.
Schema `1.0` rejects unknown fields and versions. The only supported migration is
explicit `0.1`: rename `replay_seed` to `seed`; ambiguous documents fail closed.
This is an interchange migration fixture, not an assertion that v0.1 was deployed.
JSON Schema describes shape; Python validators additionally enforce accounting,
protected genes, causality and event ordering. TS interfaces are compile-time wire
types, not a runtime trust boundary. Incoming UI data must be backend validated.

All financial amounts, prices, quantities and leverage are exact `Decimal` internally
and decimal strings on the wire. Float inputs are rejected. All timestamps include
an offset and normalize to UTC. Field names remain snake_case in both languages.
No FX conversion is inferred. Every AccountNAV in one snapshot must share its currency
and reconciled timestamp. Accounts and strategy entitlements have unique identities.

## Capital is one balance sheet

`equity = sum(unique account equity)`; `reserve = equity * .10`;
`distributable = equity - reserve`; `sum(entitlement.amount) <= distributable`.
Free account margin is an observed constraint, not additional capital. Entitlement
is a right; realizable margin is capacity bounded by that right; used margin and
reserved planned-order margin are distinct commitments. Notional is separate from
margin. Future available capacity is `max(0, realizable-used-reserved)` and cannot
collectively exceed observed free account margin. Per-account claims cannot exceed
that account's NAV, even where aggregate portfolio equity would allow it.

Loss can put already committed margin above a reduced entitlement. Preserve that
commitment and require both `risk_breach` and `allocations_frozen`, rather than invent
a position close or relax the reserve. The execution/allocation implementation must
honor the freeze. These records do not authorize liquidation, transfers or orders.
Negative NAV remains signed and requires a frozen risk breach; reserve and
distributable use max(0, NAV). Recovery/reconciliation requires separate operator handling. Predation releases money only with confirmed closed positions.

## Genes and metrics

Every gene has exactly one policy. `protective_stop=true` is immutable;
`max_leverage`, `max_margin_fraction`, and `reproduction_budget` are required and have
explicit bounds. Fractions use 0–1, not percentages. Optional stop tuning never
replaces the protective-stop requirement. Mutation code must enforce immutable
parent-to-child values: schema validation alone cannot establish lineage history.
Default complexity penalty is exactly zero. Module versions are identifiers, never
executable user source.

Fitness (`MetricWindow`), current `vitality` (0–1), and `EvidenceState` are separate.
Missing fitness is null; zero observations do not imply credible performance. Metric
window end cannot exceed plant time. RootNetwork is priority/topology, not a wallet.
ResearchCandidate always names its benchmark and one of beats/does_not_beat/
not_comparable. A verdict is a recorded claim; promotion gates must verify evidence.

## Deterministic day protocol

1. `execution`: consume only signals known at execution; intraday fills/costs/funding.
2. `metrics`: NAV marking, opportunity, risk and historical windows.
3. `capital`: reconcile NAV, reserve, entitlements, free margin and water target.
4. `ecology`: growth, roots, leaves, dormancy and risk modifiers.
5. `breeding`: seeded candidates and internally transferred reproduction budgets.
6. `lifecycle`: eligibility, closing/archive and confirmed capital release.
7. `snapshot`: persist immutable audit records and UI snapshot.

Events have globally unique IDs and sequences within a snapshot, day_index, phase,
occurrence and knowledge timestamps. Their ordering is day/phase/sequence; future
or foreign simulation events are rejected. The engine owns a specified seeded RNG;
`seed` and version-specific opaque `rng_state` support pause/resume. Snapshot
as_of is a complete observation boundary, not a license to inspect later bars.
MarketSnapshot records observation and availability separately. ExposureIntent and
TradeFill require signal <= available <= execution. The fill engine must additionally
apply bar-open/close semantics, gaps, stale-data limits and minimum execution latency.
Contract validation never claims TradingView signal parity.

## Fixtures and verification

Grass, bush and cactus share 100 USDT NAV, 10 reserve and at most 90 entitlement;
their root/leaf values differ without hardcoded strategy algorithms. Fixed seed 42
and seven ordered events permit repeatable roundtrips. Tests reject duplicate NAV,
negative/overallocated budgets, removed safety genes, out-of-bound genes, unzoned
and future data, reordered/duplicate events, and unexplained committed-margin
breaches. Distressed snapshots preserve actual margin. These tests validate the
wire/accounting contract; they do not demonstrate trading profitability.
