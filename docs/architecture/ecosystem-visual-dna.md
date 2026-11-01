# ECO-15 — Seeded inheritable visual DNA and functional morphology demo

Every plant's appearance is genetically individual and reproducibly inherited. Appearance is persisted in a VisualDNA record, not assigned randomly at paint time.

## Two deliberately distinct genomic classes

**VisualDNA** (purely aesthetic, version 1): hue, saturation, petal count, petal roundness, bloom twist, stem curve, ribbing, spine density and leaf shape. Every field is bounded; random initialization requires a caller-injected seeded RNG. Offspring recombine maternal/paternal values and apply bounded probabilistic mutations. These fields cannot modify trading signals, profit, leverage, risk protection, margin, fitness or budget.

**FunctionalMorphology** (temporary simulated bridge; source of truth becomes ECO-06): branching, leaf_count, leaf_size, photosynthetic_rind, rind_efficiency, bark_strength. High branching and leaf count increase biologically modeled photosynthetic area in ECO-06 and are visible as extra branching and foliage in the Fleet Meadow. The demo breeds and mutates these bounded values independently from protected trading/financial genes. Leaf count has a hard lower limit of 1. At exactly leaf_count==1 && photosynthetic_rind, a cactus exposes a green stem as a photosynthetic leaf; we do not draw a crown of individual leaves. Bark armor is independent and cannot increase leverage as a direct reward.

Grass / flower / shrub / cactus presets are only *initial sampling distributions* for the demo. The actual morphology is always derived from traits in classifyMorphotype(), never hardcoded to a trading strategy or asset. Three root families receive deterministic but distinct initial distributions; offspring inherit both gene layers.

## Render integration and testing

EvolutionMeadow.tsx uses visual DNA for petal count, blossom shape and rotation, hue, stem curve, ribs, spine patterns. Branching/leaf count change twig count/foliage. A one-leaf green-stem cactus gets a green ribbed trunk instead of a normal blossom. Family colors can still distinguish ancestry and seed lines. Genparameter coordinates remain sensitivity/TTP: optical DNA cannot move a plant in the optimization map.

Run: node --experimental-strip-types tests/plantAppearance.smoke.mjs. The test checks seeded deterministic generation, 500 crosses, minimum leaf count, cactus green rind with zero leaf size, independent bark protection and invalid genes. CI additionally runs original gene smoke tests and npm run build.

**Limits:** This is appearance/physiology research and a synthetic Fleet demo, not a real trading system. ECO-06 Python is canonical for daily sunlight/entitlement modeling; an explicit adapter will replace interim JS functional inputs when actual ecosystem snapshots connect. ECO-04 protected safety genes are not changed.
