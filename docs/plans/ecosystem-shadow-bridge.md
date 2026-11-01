# ECO-13: Research candidate to Family Shadow

This adapter prepares a versioned, read-only configuration snapshot from an
ECO-01 ResearchCandidate. It accepts only a candidate at
"eligible_for_sowing" whose frozen benchmark verdict is "beats", whose evidence
is explicitly out of sample, and whose genome retains its immutable protective
stop and bounded financial policies. The snapshot preserves candidate/genome
IDs, genome version, strategy-code version, parent genome IDs, benchmark ID,
and evidence. It is data only: no registry API is called, no arbitrary genome
code is loaded, and no allocation or capitalization is made.

The generated strategy ID is deterministic for a candidate and uses the
existing Family ID character/length constraints. A human must review the
snapshot, map any parent strategy ID, and register it through a separately
authorized workflow. Re-running the adapter yields the same ID and config
inputs, which supports review and replay.

## Pilot and research allocation are separate

The existing pilot remains three parents with three daughters each. Its current
daughter seed amount is 2% of its parent's family budget; the 90/10 rule keeps
at least 10% of reconciled NAV reserved and caps distributable budgets at 90%.
ECO-13 never computes or changes these allocations.

Future experimental Groundwater allocation belongs to ECO-05. It must conserve
one reconciled equity total, keep entitlements within the 90% distributable
ceiling, and distinguish entitlement from used/reserved/free margin. Nothing in
this adapter grants an entitlement, initiates a transfer, or promotes a
candidate to capitalized.

## Manual TradingView and Family Shadow path

prepare_manual_alert returns a secret-free message draft for a human to
configure. The operator must separately obtain the shared
FAMILY_WEBHOOK_TOKEN through the approved secret-management workflow and
create the TradingView alert manually. The draft contains only a placeholder;
this module neither reads a secret nor contacts TradingView.

The existing Family ingress accepts a shared family token, a registered
strategy_id, and a unique event_id; its current response records the signal
with recorded_no_execution and order_sent=false. ECO-13 mirrors the
(strategy_id, event_id) idempotency key in an in-memory replay fixture.
Same-key/same-signal repeats are duplicates; same-key/different-signal input is
rejected. The adapter does not call the ingress itself. Virtual PnL, fees,
funding, cashflow-adjusted TWR, and risk metrics remain downstream shadow
measurements and must keep their own provenance; ECO-13 does not infer
profitability from alert registration.

Candidate progression stays explicit:
proposed -> screened -> backtested -> out_of_sample_verified -> paper_forward
-> eligible_for_sowing -> capitalized. No automatic stage transition or
cutover is implemented.

## Manual cutover and Go-live blockers

This document and code prepare review artifacts only. Before any production
cutover, each gate below requires independently recorded evidence and explicit
operator approval:

- KuCoin Hedge Mode verified with a read-only venue check.
- One authoritative, hedge-aware order writer with virtual-position
  attribution; no competing writer.
- Per-bot protective TP/SL orders with reduce-only behavior.
- Confirmed-fill reconciliation, idempotent fill processing, and crash replay.
- Fees, funding, realized/unrealized PnL, and cashflow-adjusted TWR included in
  the audit ledger.
- Legacy cancel-all behavior removed or proven unable to cancel another bot's
  protective orders.
- SOLUSDTM minimum lot of 0.1 verified against current venue constraints and
  every bot's sizing rules.
- Global stops, drawdown caps, margin limits, and 90/10 reserve enforced
  outside mutable genome values.
- Registry lineage, immutable genome/strategy-code versions, and shadow event
  IDs reconciled end to end.
- Separate operator approval for any registration, TradingView alert creation,
  production route toggle, deployment, position-mode change, transfer, or
  order. None of these approvals is granted by ECO-13.

ECO-13 is not a go-live authorization. A positive research verdict, a successful
shadow alert, or a high simulated PnL alone cannot satisfy these gates.
