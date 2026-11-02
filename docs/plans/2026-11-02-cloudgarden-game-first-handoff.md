# Cloudgarden: game-first mechanistic evolution handoff
**Date:** 2026-11-02  
**Scope:** Research-only Cloudgarden game, stacked on Draft PR #62. No live-bot, exchange, Railway, main-branch or production change.

## Product decision (confirmed in conversation)
- The available **one historical SOL dataset is sufficient to run the game**. Reuse it as often as needed. It is not required to obtain independent datasets or a statistically significant result to let plants trade, breed, grow, become dormant, be eaten, or be archived.
- Evolution is a **deterministic mechanical world**, not an optimizer centrally choosing a champion. Its outcomes emerge from individual trading, capital/resource accounting, ecology and breeding rules. An analyst's `beats`, `does_not_beat`, `not_comparable`, p-value, confidence interval, OOS review or data-license verification status must **never** become an ecological trigger, historical play lock or breeding eligibility criterion.
- Later add other markets (BTC, FX, etc.) as distinct inputs/regions, with explicit market and execution assumptions. Do not force cross-market comparability as a prerequisite to play.
- Scientific experiments, confidence levels and robustness reports are **observers**, performed separately and after game functionality. They may inspect frozen game snapshots, but may not feed verdicts back into the canonical world. Game safety guards for absent/invalid candles, risk, conservation, or live routing are separate from statistical evidence gating.
- Do not secretly add statistical smoothing, winsorization, outlier filters, champions, or selection bias to gameplay. Possible ecology mechanisms should be designed and documented as game rules later, not smuggled in as significance fixes.

## Code audit: what exists today (on Draft PR #62)
1. `apps/cloudgarden/src/historicalGarden.ts` advances a local historical SOL day, generates signals for funded plants, accounts for paper fills/costs and updates the same NAV ledger **before** one deterministic ecology tick. The world is local and reproducible; no live venue route.
2. `simulation.ts` implements three founding parents, a 90/10 ledger, 20 **single-use** funded seedling entitlements, scheduled conception every seven days, synthetic light/water/biomass/vigor/dormancy, a fifth-day predator patrol and capital release on archive.
3. `experiments.ts` runs separate controlled **synthetic** cohort comparisons; `src/quant/ecosystem/experiments/` is also a separate research harness (Draft #59 with evidence fix Draft #64). **Neither is imported into the historical trading day nor the core ecology tick.** The old historical UI nonetheless displayed the research verdict inline with game controls; this follow-up removes that conflation.
4. Historical records contain per-plant cumulative net PnL, fills, costs and opportunities. However **historical profit is not yet the direct driver of growth, mate selection or predation**. Current ecology is primarily synthetic climate-based. Scheduled births use retained founder-parent records (even after archive), not a competition among all economically successful living plants.
5. `HISTORICAL_TRADING_SPEC.md` documents an actual local archived SOL export, but the private raw candles are not shipped with GitHub. Agent A audits the dataset/CI separately. A missing or corrupt candle day stops the clock to preserve determinism; that is **data integrity**, not a statistical significance gate.
6. The existing historical engine intentionally differs from full TradingView Renko and actual venue rules; strategy signal parity and provenance work remain with Agents A/C. Do not present any historical profit as validated live performance.

## Implemented in this follow-up
- Historical **game screen** no longer displays the research `not_comparable` verdict as part of its trading controls. It explicitly states that external statistical comparisons do not control Play, breeding, growth or predation.
- Regression test: a separate experiment can return `not_comparable`, yet the identical historical day advances, books transactions and conserves accounting **unchanged**. It requires only synthetic fixture candles; no access to the user's private local dataset.
- No change to climate, PnL arithmetic, predation thresholds, mutation, birth timing, live trading or research significance calculations.

## G1a implemented on the stacked market-growth branch
The first direct economic feedback is now wired into the playable historical SOL mode:
- After each full paper-trading day has booked all plants to the conserved 90/10 ledger, each active plant receives only its **own** daily net PnL and its allocation measured **at the beginning of that day**.
- The explicit purely mechanical growth rule is `factor = max(0, 1 + plant_booked_day_net_pnl / plant_start_allocation)`. The historical plant's `biomass` and `vigor` multiply by this factor. They use the existing game's anatomical bounds (biomass 5..100, vigor 0..100) and four-decimal internal rounding; **the PnL ledger itself is not clipped, censored or modified** by these visual/physiology limits.
- No-trade/no-PnL day: factor exactly 1, neither trading-derived body mass nor vigor changes. Artificial sunshine, water and daily habitat-fit/evidence do not add or remove economic biomass/vigor in historical mode; they remain a separate, still synthetic environment/physiology layer.
- Legacy synthetic demo mode retains its original climate-growth code path. The historical tick now logs `marketDrivenBiomass=true` as an audit flag; game rules and the entire 90/10 ledger remain deterministic. The selected plant's economic body mass and vigor are shown directly in the meadow.
- These market-driven measurements **do not implement scientific significance**, they are not superiority ratings, and no central winner is chosen. Initial synthetic habitat evidence and the existing 80-day/5-day predator mechanism have **not** been replaced; so this is G1a, **not** completed ecological survival selection.
- Tests added for isolated parent +10%/-5%/0% outcomes, unchanged no-trade days, rejection of corrupt/unowned outcomes, historical price-to-PnL-to-body linkage, and deterministic replay without a significance gate.
- The historical save is now **schema v2** with the separate `cloudgarden.historical.v2` key. The old v1 save is left intact and is never silently replayed using changed rules. The game UI discloses the preserved older save when starting a new v2 world.
- The historical data export still stays private. CI uses generated fixture candles. Keep this stacked draft unmerged until CI passes and Agent B's canonical adapter is reconciled.

### What G1b still needs
1. Connect physical resource competition and survival to real market-derived resource availability, without statistical ranking and without inventing new NAV or economically meaningless biomass bonuses.
2. Revisit the game's anatomical caps and minimum size as explicit physical ecology rules, distinct from uncensored financial PnL. Any change must be versioned for reproducibility.
3. Ensure dormancy, trading eligibility, mortality and reproduction refer to the same well-defined plant lifecycle before introducing more market regions.

## G2a / inactivity genetics implemented on the next stacked game branch

**Core change:** Adult descendants now actually enter the bee's breeding pool. There is no global strategy winner or statistical gate. Only plants with `status="active"`, a real funded allocation and `evidenceDays >= 7` are potential parents. The mother is visited deterministically in stable insertion/age order and the father is a distinct living plant chosen by the existing seeded PRNG. `local`/`hybrid` blend their actual genes and source modules; `hive`/`random` retain their existing mutation modes, but the physical ancestry of every child is now recorded in `parents`. Archiving is permanent for future reproduction but never deletes historic parent records.

**New ecological gene:** `genes.patience` is seeded separately for each founder and inherited/mutated by all bee modes, clamped to **0.05–0.95**. It is a visual/ecological trait, **not** a TradingView/Pine strategy parameter, risk override, cash balance or statistical confidence coefficient. `plant.inactiveDays` persists the consecutive number of completed historical UTC days **without any actual paper fill**; fabricated/generated signals do not count as trades. With one or more fills, even if net PnL is 0, the streak resets to 0.

**Very small vitality effect:** First **14 consecutive idle days** have zero inactivity penalty. From the fifteenth idle day forward, only `vigor` receives the extra daily multiplier
`1 - 0.0003 × (1.2 - genes.patience)`.
The inherited patience range means about **0.0075% to 0.0345%** extra daily vigor decay once the grace period expires. PnL, NAV, stop policies and plant allocation **never change** due to inactivity. Existing PnL-based growth still applies independently. No-trade body mass remains unchanged; no-trade vigor now loses a tiny amount after the grace period (this intentionally supersedes the G1a statement that *all* no-trade days leave vigor unchanged). Dormancy/cactus plants remain allowed to wait with high genetic patience.

**Game-state version:** This changes the world mechanics, so the historical save becomes `schemaVersion=3`, `cloudgarden.historical.v3`. Existing v1/v2 browser records are left intact and are **not silently replayed under new rules**; the UI explains that the previous version is preserved. This version requires re-running from a chosen date, not writing over or migrating a v2 run with altered physiology.

**Boundaries:** 20 one-time seed entitlements still cap the existing game; capital recycling after that is **G3**, not claimed as done here. The existing 80-day protection, environmental dormancy and synthetic predator/evidence tests are not redesigned in this PR. A genuinely market-driven survival/predation mechanism should be designed later as **an explicit ecology rule**, never introduced as an unverified significance/fitness gate.

**Tests:** 14-day grace, exact tiny genotypic decay, execution resets, negative/malformed fill-count rejection, 28-day direct descendant-to-descendant breeding, altered parent-patience changes child, archived parents excluded but genealogy retained, version 3 save rejection of v1/v2, and real price-replay results correctly resetting inactivity. All use generated deterministic fixtures, with no private SOL file. GitHub CI must pass before integration.

## Practical game-development order (not scientific review)
**G1 — Complete economic feedback and life-cycle contract.** Define simple, explicit deterministic links between each *plant's own* net trading outcome and its accessible resources/growth. Decide what performance/history a bee or predator observes. Keep market PnL separate from cosmetic climate and do not invent arbitrary statistical corrections. Record the exact rule and add no-trade, profit, loss and replay-determinism tests.

**G2 — Make reproduction actually evolutionary.** Current offspring are created every seven days with ancestry based on founder records. Move toward a mechanically funded process in which eligible living plants and their heritable strategy genomes can be parents. Archive must remove economic reproduction eligibility without erasing genealogy. No centralized winner.

**G3 — Long-running resource loop.** The present 20 one-time entitlements cap births after 140 days; ensure depleted and reclaimed resources have explicit rules for continued generations, while preserving 90/10 accounting and preventing capital creation. Stress test 1,000+ days.

**G4 — Visual feedback and controls.** Surface deterministic causes directly in the world: plant profit/loss, resource level, lineage, age, dormancy, mutation, predation and capital transfers. Preserve save/load and Play/Pause at identical state hashes. Keep synthetic demo and historical SOL views clearly distinct.

**G5 — Integration and scale.** Reconcile Cloudgarden/Python/Fleet genome, ledger and world-state contracts after Agent B, then improve performance without changing per-tick outcomes. New markets may enter as additional clearly labeled habitats after their own data integrity checks.

**Last — Optional scientific evaluation.** On frozen output snapshots, compare search policies with statistical rigor appropriate to the available evidence. The presence or absence of such analysis never changes historical replay or ecological outcomes.

## Invariants and acceptance checks
- Same historical data, start date, world seed, game rules and user inputs => same plants, fills, records, NAV and events.
- An external verdict (including `not_comparable`) must not be accessed by the game engine or used as a stop condition. One market dataset is fine for gameplay.
- Real missing candles, malformed snapshots, invalid prices, nonconserving capital or unsettled exposure can still stop/fail a tick; no fabricated fills or repair by statistical inference.
- All actions remain paper-only, no writes to Kraken, KuCoin, Railway, live bots or `main`.
- Do not merge overlapping A/B/C work until interfaces are reconciled. This PR intentionally modifies only the isolated historical UI, its regression test and this handoff.

**Next recommended coding task:** G1b: physical survival/resource loop, then G2 population-wide parent eligibility. Maintain an explicit mechanical rule, preserve the shared NAV and reconcile Agent B's ABI before merge.
