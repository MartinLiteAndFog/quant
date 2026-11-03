# ECO-01 ↔ Cloudgarden read-only Wire Bridge v1

**Status:** research contract / Draft-PR integration only. This is not a live-trading
account API, a TradingView-parity verifier, or an OOS-performance validator.

## Compatibility and writers

- Canonical accounting model: ECO-01 \`GroundwaterState\` **1.0**
  (Python \`src/quant/ecosystem/contracts.py\`).
- Exchange envelope: \`eco-cloudgarden/1.0\`. Dedicated version because
  Cloudgarden's \`SimulationWorld\` is **not** an ECO-01
  \`SimulationSnapshot\`. No structural casts or synthetic-evidence promotion.
- The exporter \`apps/cloudgarden/src/ecoWireBridge.ts\` reads a world or
  \`HistoricalSession\`, validates it, and emits a JSON snapshot. The Python
  \`cloudgarden_bridge.py\` validates the same envelope *through* ECO-01
  \`GroundwaterState\`. Neither side books into the other's ledger.
- \`capital_writer="cloudgarden-paper"\` and \`access="read_only"\` are mandatory.
  \`account_id="cloudgarden-paper:<simulation_id>"\` is a namespace, not a real
  exchange/account identifier. The observer journal refuses another account,
  source hash, nonmonotonic day, or changed duplicate \`event_id\`.
- \`event_id=<simulation_id>:d<day_index>:wire-snapshot\` is deterministic.
  \`known_at\` and \`as_of\` are explicit UTC, with \`known_at <= as_of\`.
  This is a day-close snapshot known at the following UTC midnight, **not**
  fabricated intraday order-event timestamps.
- \`dataset_fingerprint\` currently uses Cloudgarden's **reported source archive
  SHA256** for historical sessions, **not** the per-day normalized-bar digest.
  The underlying source provenance/venue/rights remain unverified. A
  \`"000…000"\` fingerprint is permitted **only** for synthetic research.

## Financial invariants

All wire amounts are JSON **decimal strings**, never floating-point JSON
numbers. Cloudgarden's current source ledger stores USD/USDT as JavaScript
numbers rounded to 0.0001; export rejects non-four-decimal state and unsafe
integer precision.

- ECO-01 \`equity\` = sum(account NAV), once. \`reserve = 10% *
  max(NAV, 0)\`, \`distributable = max(NAV, 0) − reserve\`, with **exact decimal
  arithmetic**. Reserve/distributable can have a fifth decimal place.
- \`deployed_allocation\` = sum(active plants' claims).
  \`seed_reserved_allocation\` = sum(available *single-use* seedbank claims).
  A used seedbank claim disappears and the offspring plant claim takes its
  place; the paper account has **no new deposit** and total available
  entitlements never exceed distributable.
- \`free_allocation\` is the uncommitted research **budget**; it is **not**
  \`free_margin\`. \`free_margin\` is the reconciled, currently flat paper
  account's available **margin**. \`used_margin\` and
  \`reserved_order_margin\` are separate, and at each **flat day-close**
  projection they are zero. A future intraday adapter must source actual
  margin/order state; never infer it from plant budgets.
- Cloudgarden rounds reserve/ceiling to four decimals, while ECO-01 uses
  exact 90/10. The bridge holds back at most 0.00005 USDT of *otherwise free*
  capital as \`rounding_holdback\`, instead of promising it twice. If the
  already-committed claims alone exceed the exact ceiling, export **fails
  closed** and requires a separate fix to Cloudgarden's authorititative
  ledger. It does not rebook or silently shrink the plant claims.
- \`booked\` includes opening NAV, explicit external cashflow, cumulative
  realized price PnL, cumulative fees/funding, cumulative net PnL and a
  **separate exact rounding reconciliation delta**. Equation:
  \`opening_nav + cashflow + net_pnl = equity\`;
  \`price_pnl − fees + funding + reconciliation_delta = net_pnl\`.
  A delta is transparent bookkeeping, not alpha or unrealized profit.
- Negative NAV is a reporting-only possibility for downstream ECO-01:
  \`distributable=0\`, \`reserve=0\`, \`risk_breach=true\` and
  \`allocations_frozen=true\`. Locked past margin may remain reported.
  Cloudgarden's *current* simulation rejects nonpositive NAV, so the negative
  scenario is a cross-language golden adversarial test, **not** a successful
  Cloudgarden market replay.

## DNA and evidence — no semantic promotion

For each plant: source-family/strategy-code version, the unmodified validated
\`SourceGenome\` (or null for synthetic demos), visual genes, trading traits,
lineage, active/archived state and exact allocation. Original
Cloudgarden \`fitness\`, \`vigor\`, \`evidence\`, \`evidenceDays\` are placed
**only** under \`habitat\`. The \`empirical\` field MUST be JSON null in v1.
No TWR, drawdown, statistically qualified vitality, or OOS credibility
can be synthesized from habitat or elapsed time. That requires Agent D/E's
causal, independently measured experimental pipeline and a new contract.

The source genome's \`stopLossBps\` stays compulsory. It is a **close-trigger
research gene**, not an exchange protective stop guarantee.

## How to check

With both unmerged draft dependencies present in an authorized research branch
(Cloudgarden PR #62 and ECO-01 PR #57):

\`\`\`bash
cd apps/cloudgarden
# PR #62 currently has an invalid package-lock; do not change it from Agent B.
npm install --no-package-lock --no-audit --no-fund
npm install --no-save --no-package-lock --no-audit --no-fund '@types/node@^22'
npm test
npm run build
cd ../..
PYTHONPATH=src python -m pytest -q \
  tests/ecosystem/test_contracts.py tests/ecosystem/test_cloudgarden_bridge.py
\`\`\`

The Python test feeds the Python-validated golden JSON through
\`node --import tsx apps/cloudgarden/src/ecoWireBridge.cli.ts\`,
then parses it back through ECO-01. The isolated
\`.github/workflows/eco-cloudgarden-wire.yml\` requires these tests in CI
without live keys or historical private data.

**Explicitly excluded:** any change to live Webhook/API/KuCoin/Kraken
executors, orders, transfers, merges to \`main\`, Railway deployment, or
automatic promotion of research candidates.
