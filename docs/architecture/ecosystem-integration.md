# ECO-16: Integrated synthetic paper ecology (stacked research preview)

This prototype composes the already separately implemented **ECO-03 clock**, **ECO-05 capital accounting**, and **ECO-06 daily plant physiology** in an isolated read-only / paper world. Code is based on snapshots of their Draft-PRs; after those PRs are reviewed/merged the integration branch must be rebased to remove duplicate dependency copies.

It is **not** a trading strategy backtest, a real exchange reconciliation, an executed order simulator, nor an optimization result. ALL gain/loss intraday events in the CLI are deliberately **synthetic** and explicitly sourced, not computed from SOL market history.

## Quick start

From the repository root:

```bash
PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_research_world.py -v
PYTHONPATH=src python scripts/ecosystem_paper_demo.py --days 30 --equity 100 --seed 2026
PYTHONPATH=src python scripts/ecosystem_paper_demo.py --days 365 --equity 100 --seed 2026
```

`start_world` creates a unique initial 3-parent/20-child population. Realistic capital book logic supplies 90% of opening NAV as *budget entitlement*, keeps 10% reserve, and funds 20 newborns by moving tiny 2%-of-current-parent-budget slices to the children (no new money). It does **not** register TradingView alerts, place exchange orders or change real subaccounts.

`step_world` applies the existing UTC clock's explicitly dated incremental paper PnL, fees, funding and external flows. A **capital reducer** then creates same-currency, account-owned per-plant entitlements and net free-margin bounds; an **ecology reducer** then updates plant roots, biomass, leaves, water eligibility and sunshine-dependent *intended* margin and leverage. The final margin *intent* is additionally clamped by the account's observed free margin via the ECO-05 allocator. It is never an executable trading order. If equity is negative, no new entries are proposed, and no existing positions are forced closed.

The entire world can be canonical-JSON checkpointed by the ECO-03 SimulationState and restored later. A registry/config hash prevents silently changing genes or removing old plants while replaying the same world. Inputs and state have stable IDs so duplicate days/events fail closed. Events must have actual known timestamp and unique ID. No future market data are accessible through this adapter.

**Known scope limits and next agents:**
- Current world is a **single virtual USDT paper account**, intentionally not a multiaccount equity/FX reconciliation. Genuine multi-asset/FX/commodity trading requires contract specifications, FX conversion and verified independent market history.
- All 23 plants are pre-registered at simulation start; future births/predation, ownership migrations, real BEE crossover and death/archival await ECO-08/09. Initial seed funding is demonstrated, not dynamically triggered by profitable live plants.
- End-of-day sunlight modifies next-day ecological intents. It must NOT be used retroactively to size intraday fills earlier in the same day (causal constraint).
- The paper event producer supplies *incremental* mark-to-market PnL. It must not supply both cumulative unrealized gains and realized gains for the same period. ECO-02 real-price fills and ECO-07 proper strategy-level metrics/evidence must replace handcrafted events and externally specified `evidence_qualified`.
- Capital allocations may change after a day. Real orders/positions need explicit fills, reserved orders, per-venue lot sizes and reconciled available margin before ever being made live; no hedge-aware writer is included.
- ECO-15 VisualDNA is **purely aesthetic** and does not affect this Python return/Equity math. TS UI/snapshot adapter to visualize actual Python daily state is deferred until ECO-01 authoritative snapshot contracts land, rather than inventing duplicate incompatible schemas.
- The research-world state uses a versioned small internal JSON record under ECO-03 `state_json`; it is NOT an ECO-01 `SimulationSnapshot` and needs a validated bridge before app/backend integration.

## Acceptance / tests

`test_research_world.py` checks:
1. Three roots and 20 child genomes, all initial entitlements from one 90% NAV pool;
2. 30-day pure reproducibility and canonical pause/resume checkpoints, no double-booking;
3. Portfolio reserve exactly 10% of net NAV each day and future margin intents bounded by free margin;
4. One-leaf photosynthetic green-rind cactus with lower own margin and growing deep roots;
5. External cashflows not falsely treated as strategy profit;
6. Negative-equity freezes new allocations without forced flattening;
7. Unknown strategy PnL, future-known weather and mutated population fail closed.

The CI workflow runs the 30-day integration tests and a synthetic 365-day smoke report with no credentials or network dependencies.
