# Evolution Meadow — 3-D parameter-space visualization and mutation proposal

Status: **preview / simulation**, November 1 2026. No real orders, funding,
TradingView alarm creation, or production deployment.

## Mental model

Every strategy is represented by a flower rooted at one parameter pair:

- Ground X: IMBA sensitivity (initial display 12–25).
- Ground Y: TTP trailing distance in percent (initial display 0.25–1.65).
- Vertical Z: cashflow-adjusted net monthly return (positive upward).
  Losing strategies sit on the soil as faded/wilted sprouts.
- Flower size: risk and confidence-adjusted fitness score, not raw revenue.
- Flower color: ancestor root family (3 roots in the initial 3x3 pilot).
- Lines in the latest generation: seed travel from old flower canopy to the
  exact child position on the parameter plane.

As a user rotates the 3D projection, clusters of profitable and unsuccessful
parameter regions become visible. Selecting a plant shows its strategy ID,
parent lineage, sensitivity, TTP, net return, drawdown, trade count, score,
and how many seeds its rating would permit. The demo highlights up to five
estimated fitness-cluster centroids on the ground, grouped from nearby
credible/high-score strategies using normalized Euclidean distances.

**Projection caveat:** this 2D ground plane is meaningful only when the
remaining strategy inputs, symbol, timeframe, leverage conventions and data
feed are the same (or explicitly filtered). Otherwise an apparent thriving
cluster may be caused by hidden third/fourth input differences, not TTP or
sensitivity. The live view must allow filtering or faceting by the full
immutable strategy configuration.

## Mathematics

Normalize each parameter onto its own range:

  u = (sensitivity - sensitivity_min) / (sensitivity_max - sensitivity_min)
  v = (ttp_pct - ttp_min) / (ttp_max - ttp_min)

Candidate seed j of N lies at:

  theta_j = 2*pi*(j+0.5+epsilon_j)/N
  r_j = R*sqrt(U_j)
  u_j = u_parent + r_j*cos(theta_j)
  v_j = v_parent + r_j*sin(theta_j)

where epsilon is a bounded small angular jitter, U is uniform [0,1],
and R increases with evidence-adjusted profit. Points are clipped to
the allowed parameter ranges and rounded to valid parameter increments;
duplicates are retried.

The angular stratification ensures different directions are explored;
sqrt(U) ensures equal sampling density per unit disk AREA, not an
overconcentration at the center.

The provisional reproduction formula for future approval is:

  eligible = monthly_net_twr > 5%, >= 30 observed days,
             >= 15 real closed trades, verified cashflows + fees + funding
  fitness = 100 * (monthly_net_twr * n/(n+20) - 0.35*drawdown)
  strength = clamp((fitness-1)/20, 0, 1)
  seeds = 5..15 as strength grows
  R = 0.036 + 0.115*strength (normalized plane units)

The variable 5–15 seed policy is a PREVIEW design and does NOT yet replace
the previously planned six-daughter live trading policy. The first three
parents and three daughters each are manually configured in TradingView;
subsequent automatically generated candidates will still require a working
TradingView setup/alert, unless the signal parity problem is eventually solved.

The parent ROOT allocation rule is unchanged: active root families split
90% of account equity evenly and 10% stays in reserve. Monthly ratings
may redistribute capital *within* a family, and a new daughter can inherit
a 2% family subbudget. Angel/Fishing strategies are evaluated separately
and must not distort the strategy meadow/parent count.

## Implemented files

- apps/fleet-desktop/src/components/EvolutionMeadow.tsx
  Interactive rotatable projection in place of the Fleet hero equity chart,
  via the new "Evolution 🌱" header button. Includes generation step, reset,
  family selector, seed-arc toggle, flower selection, detailed annotations.
- apps/fleet-desktop/src/lib/evolutionMeadow.ts
  Deterministic local simulation with three root and nine daughter plants.
  **ALL DEMO RETURNS/TRADES ARE SYNTHETIC AND LABELED.**
- src/quant/execution/family_mutation_meadow.py
  Deterministic candidate parameter proposals; no exchange or TV API calls.
- tests/test_family_mutation_meadow.py
  Risk/sample eligibility, within-bounds, distribution across angles,
  deterministic replay, no duplicate configuration and 5–15 seed scaling.
- .github/workflows/fleet-evolution-meadow.yml
  TypeScript typechecking and Vite build on the draft PR.

## Gene library, pollinators and slow biomass dynamics (added)

The Fleet desktop Evolution Meadow now has a **local, editable Gene Library**:
numeric strategy parameters with min/max/step/default or Boolean on/off
modules. Example genes: sensitivity, TTP, Stop Loss %, RSI period, RSI entry
threshold, ADX gate and ADX period. The two visible ground-axis genes are
immutable. User-added optional genes are persisted to the browser's
localStorage, not to the production strategy registry or Pine script.

Every simulated plant now carries its own typed, independently inherited
parameter genome. Adding a gene to the library does NOT force it into all
plants. Bees may inherit newly introduced genes from one parent or the
collector breed can introduce an as-yet-absent gene from the library.

Five bee algorithms are selectable in the UI:

1. **Mixer**: average of numerical parental genes; occasional 5% optional
   gene deletion.
2. **Mosaic**: per-gene allele choice from one parent.
3. **Explorer**: widened numerical mutation for parameter-space exploration.
4. **Collector**: high probability of incorporating a gene present in only
   one parent and 10% chance to introduce a gene absent in both.
5. **Pruner**: deletes optional genes with 10% probability to reduce bloat.
   Required axis genes cannot be deleted.

All current bee output is deterministic for repeatable simulations (given
the demo RNG) and does not modify any real TradingView settings.

**Gradual monthly plant growth:**

    biomass_next = max(0.04, biomass_now * (1 + clamp(net_return, -0.25, +0.36)))
    screen_height ~ log(1 + biomass)

A +36% net month yields 36% biomass growth at most, and a poor month shrinks
biomass progressively rather than deleting the plant. Shrinkage capped
at -25% per month avoids visually wiping out strategies in two months.
The logarithmic scale keeps unusually successful plants visible without
dominating the entire scene. A young seed starts small (biomass 0.10)
and remains on the board as a growing plant. Months continue to age all
existing flowers even when the visual maximum number of displayed flowers
is reached. That visual guard has nothing to do with limits on live bots.

**Important distinction:** Numerical gene definitions and Boolean modules
are parameter schemas, not arbitrary executable strategy code. Before new
modules have trading effects, each must be implemented in a versioned,
sandboxed strategy interpreter or compiled Pine/engine code with semantic
validation. Arbitrary user-pasted code must NEVER be directly executed by
the production order manager.

The user also suggested fungi/mycelium: future candidate role is a
read-only knowledge network for horizontal transfer of useful gene patterns
and past failed configurations. It must not transfer account equity by
itself; all capital changes remain ledger events with conservation checks.

## Next integration work

1. Store immutable strategy parameter snapshots on each strategy ID. Family
   registry currently stores only user-supplied metadata, not checked inputs.
2. Compute monthly TWR from per-bot virtual equity (INCLUDING open PnL,
   executed fees, funding, deposits/withdrawals and per-bot margin use).
   Refuse to rate if data gaps exist.
3. Persist month-end ratings, score confidence, period, generation, cluster,
   capital target and candidate seed lineage atomically and idempotently.
4. Expose verified parameter+month-return records to Fleet API. Replace
   synthetic flower coordinates and profits only once source is truly ready.
5. Add human review workflow: click a proposed seed -> copy TradingView ID,
   token-protected shared URL and correct entry/exit alert JSON template.
6. Configure manual TradingView alert and verify message delivery to the
   Blaupunkt shadow inbox. Do not execute two order writers on one KuCoin
   subaccount until Hedge Mode reconciliation and protective stops are proven.

## Safety

This is visualization/analysis ONLY. No production services were changed.
The meadow's maximum 450 displayed flowers is a browser-rendering guard in
the demo, NOT a trading allocation or lifetime population cap.
The simulated fitness surface deliberately has peaks for visual testing;
it must never be presented as empirical SOL trading performance.
