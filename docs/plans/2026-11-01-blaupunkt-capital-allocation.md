# Blaupunkt capital allocation — confirmed 2026-11-01

Scope: one KuCoin Futures subaccount ("Blaupunkt") hosting multiple
independent parent strategies and their daughter bots, receiving distinct
TradingView webhook IDs but sending venue orders through a SINGLE family
order manager. TradingView stays the signal source.

## Two-level allocation

Given CURRENT KuCoin account equity E, the combined parent-family
margin budget is 90% of E. The other 10% is reserved.

If N parent strategies are active, each family receives:

    B_family = E * 0.90 / N

The initial distribution is EQUAL by ACTIVE PARENT COUNT, not proportional
to recent score. Children DO NOT increase N. They are funded entirely
from the budget of their own parent family.

Within one family's budget, each newly created daughter initially has
2% of the family budget as described in the user's earlier evolution rule;
the parent keeps the remainder. Six daughters therefore receive 12%
combined and the parent receives 88%. Future rating-based weights can
change WITHIN that family's budget without affecting another family.

Examples with E = 1000 USDT:
- N=1: family A receives 900, reserve=100.
- N=3: families A/B/C each receive 300, reserve=100.
- If family A has six 2%-funded daughters: each daughter 6;
  A parent itself has 264; families B/C remain 300 each.

For active parent's effective BUY/SELL order quantity, convert its own
budget into allowed contracts using up-to-date price, leverage, contract
multiplier, true free margin and venue minimums. Never round UP if it
would exceed budget. Zero-contract child is registered but cannot trade.

## Constraints

- 90% is a TOTAL margin budgeting ceiling, not a 90%-of-equity LOSS
  allowance. With e.g. 10x leverage nominal exposure can be much larger
  than equity. Per-position/family loss caps are also required.
- When N changes, an existing family's target capital changes. That
  DOES NOT authorize immediate partial liquidation of open positions.
  Reallocate available margin for future entries; perform any explicit
  rebalance through a separate, audited rule.
- The budget uses real free margin in addition to equity; outstanding
  entries/stops, funding, fees, maintenance margin and volatility
  must be respected.
- Inactive parents receive no new entry budget. Existing live positions
  require careful lifecycle/ownership, not instant zeroing.
- The Fishing/Angel Pool is accounted separately; it must not become an
  extra evolution parent by accident. If it shares the physical exchange
  account, its funding must be reserved BEFORE computing the 90% of the
  explicitly defined evolution pool equity. Do not double-count capital.
- Hedge Mode allows venue side aggregation. The virtual ledger must
  attribute each fill and close to its individual strategy, not touch
  unrelated family members.
- Equal allocations and daughter weights are separate from strategy rating.
  The user may later choose rating-based weighting WITHIN the families.

## Code

Pure, side-effect-free allocator:

  quant.execution.family_capital.allocate_account

Dry-only registry and alert ingestion (NOT mounted on production app):

  quant.execution.family_ingress

No live trading executor or Railway variable is changed by this PR.
