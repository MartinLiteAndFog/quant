# Agent F — Gene ↔ Unreal procedural parameters ↔ physics ↔ evolution (2026-11-02)

**State: bridge design + executable synthetic reference fixtures; NO Unreal project yet.** Agent B remains owner of the authoritative source/ledger exchange contract, Agent D of fitness/evolution approval, and Agent C of historical trading parity. Agent F only defines a read-only or explicitly gated downstream integration. No live orders.

## Why this is different from drawing plants

An organism can convert inherited **numeric genes** into growth/launch/collision parameters, and the environment can convert physical outcomes back into **typed numeric observations**. Those observations can affect **future ecological placement and child research hypotheses**.

Two useful loops coexist:

A. **Visual/physiological phenotype loop:** canonical genome → validated bindings → Unreal PCG/Blueprint mesh, height, leaf count, color, branching, green rind, pollen, seed count and spawn settings → visual effects. No return signal required.

B. **Ecological feedback loop:** canonical ecology genes + scene environment → authorized seed release → Chaos trajectory/contact/settled collision → recorded impact location/material/time → validated observation → canonical ecology owner accepts a deduplicated event → habitat competition/germination → next day/next generation state. The returned measurement is *not* a market signal or realized profit.

C. **Optional offspring gene creation from geography:** If a seed lands in a bounded 2-axis strategy meadow, its **landing x/y** maps back to a proposed child genome's two strategy parameters. This is a *future-only experimental candidate*, not an adjustment to the parent's trading strategy or existing orders. Other child genes still need approved inheritance/mutation, and only fresh out-of-sample research can assess trading merit.

## Current source of genes: do not conflate

- Existing Cloudgarden #62 has a **synthetic ecology genome** root/leaf/water/sun in simulation.ts, **tradingTraits** and historical **SourceGenome** fields (channelLookback and stopLossBps among others). These are distinct models.
- Existing ECO-15 Draft #45 has **VisualDNA** (hue, saturation, petals, curves, ribbing, spines etc.) and **FunctionalMorphology** (branching, leaf_count, leaf_size, photosynthetic_rind, rind_efficiency, bark_strength). It is NOT yet unified with Cloudgarden's SourceGenome.
- Existing fleet gene library (#41/#45) defines sensitivity, ttp, stop-loss, RSI, ADX etc. It is a distinct synthetic research prototype; names are not silently equated to SourceGenome fields.
- **New proposed ecology genes** for this bridge: seed_release_height_m, seed_azimuth_deg, seed_elevation_deg, seed_launch_speed_mps, seed_spread_deg and seed_count. They do not yet exist in the canonical data model. Each must be approved before it becomes a mutable inheritable field.
- The complete registry is contracts/parameter_registry_v1.json, including origin, type, ranges, default, role, units, proposed Unreal exposed field and conversion multiplier.

### Example typed bindings

| Source value | Example | Target (proposed UE asset property) | Conversion | Can Unreal write source? |
|---|---:|---|---|---|
| visual.hue_deg | 126° | BP_CloudPlant.HueDegrees | identity | No |
| morph.branching_ratio | 0.62 | PCG_CloudPlant.Branching | identity | No |
| morph.leaf_count | 22 | PCG_CloudPlant.LeafCount | identity | No |
| morph.photosynthetic_rind | false | BP_CloudPlant.PhotosyntheticRind | bool identity | No |
| eco.seed_release_height_m | 1.7 m | BP_SeedEmitter.ReleaseHeightCm | ×100 | No |
| eco.seed_azimuth_deg | 35° | BP_SeedEmitter.AzimuthDeg | identity | No |
| eco.seed_elevation_deg | 25° | BP_SeedEmitter.ElevationDeg | identity | No |
| eco.seed_launch_speed_mps | 2.2 m/s | BP_SeedEmitter.LaunchSpeedCmps | ×100 | No |
| eco.seed_spread_deg | 6° | BP_SeedEmitter.YawSpreadDeg | identity | No |
| eco.seed_count | 3 | BP_SeedEmitter.SeedCount | identity | No |
| trade.channel_lookback | 50 | WBP_PlantInspector.ChannelLookback | display only | No |
| trade.stop_loss_bps | 250 | WBP_PlantInspector.StopLossBps | display only | No |

Targets are planned class/property names, NOT claims that the Blueprints exist. Registry versioning and a static allowlist are mandatory: do not accept arbitrary object paths or executable expressions from remote data. Unit conversion is explicit: normalized source coordinates in **meters**, UE defaults in **centimeters**, physical gravity sample 9.81 m/s² (980/981 cm/s² as applicable), angles in degrees; do not accidentally mix degrees/radians, cm/m or collision momentum with price momentum.

## Seed example: one reproducible end-to-end roundtrip

- Plant plant-a, at simulation day 7, has proposed ecology genes release height=1.7m; azimuth=35°; elevation=25°; speed=2.2m/s; yaw spread ±6°; count=3.
- Release-event ID seed-release:plant-a:7:001, RNG xorshift32_v1 initial seed 20261102, gravity 9.81m/s², no wind/drag, flat loam at z=0. The analytical reference solver varies **yaw** once per seed and solves ballistic ground-intersection. This is a controlled fixture, **not a Chaos simulation**.
- In the synthetic fixture the three seeds land about 1.378m from the emitter; impact coordinates and flight times are recorded in seed_landing_v1.json and checked by script. Units and random stream are not free-floating labels.
- Raw physics results are recorded as **observations**, including impact position, collider/physical material, tick, source request ID, seed ID, RNG seed, engine/physics/geometry version hashes. For Chaos, distinguish impact contact from **terminal settled landing**: multiple OnHit contacts/bounces must not create multiple offspring.
- The event owner validates allowed scene revision, unmodified request genes, reasonable bounds, event uniqueness, material from the authoritative map, simulation day and seed membership. An accepted observation becomes a **future ecology input** no earlier than day 8. A screenshot or particle simulation is not an authoritative receipt.
- In this demo, seed displacement is local to the parent. The parent’s trading gene location maps to tile meters using X=channelLookback 20…500 and Y=stopLossBps 50…500 on a 10m×10m meadow. Its parent coordinate is (0.625m, 4.444444m). Adding seed impact offsets produces landing positions around (1.77–1.81m, 5.15–5.22m).
- Inverse mapping from landing position to axis values yields *proposed* child parameter pairs, e.g. (channelLookback=107, stopLossBps=282) for seed 1. Proposals are generated by scripts/propose_offspring_genes.py and checked against fixtures/offspring_proposals_v1.json. Nothing is booked, and no current plant is rewritten.

### Inverse mapping — explicit, reversible and constrained

For each tile axis, with world range L..U meters, gene range min..max and step q:
  normalized = (landing_m - L) / (U - L)
  raw_gene = min + normalized * (max - min)
  proposed_gene = min + round_half_up((raw_gene - min) / q) * q

Reject outside tile range and classify as **unassigned** (never force/clamp to a valid strategy). If terrain hills/3D geometry exist, invert **horizontal x/y in a versioned local tile coordinate frame**, not the seed's vertical z or a screenshot-pixel coordinate. Keep per-tile rotation, origin and scale in the authoritative map once multiple tiles exist. Landing coordinates may create a *child strategy candidate*, but must not modify parent traits or other genes. A later, separately approved crossover/risk policy composes complete source genomes.

**Important statistical consequence:** A candidate created from where a seed lands is a search/mutation proposal. It does not prove trading skill. Require proper training/test splits, realistic fees/slippage, known-at-time evaluation, drawdown/risk gates, and corrections for repeated candidate search. Habitat quality or collision luck must never be labeled as empirical alpha.

## How physical hits are allowed to become numbers

- **Producer:** UE5 Chaos runtime, after the local agent has built the scene and configured a supported collision component, may emit an untrusted candidate (contracts/chaos_hit_candidate_v1.schema.json). Hit notifications require collision hit event generation; do not equate overlap, initial impact and final rest. A bounded fixed/substep configuration helps stability but **does not guarantee bit-identical Chaos outcomes across builds, GPU/CPU or frames**.
- **Observer:** The adapter records launch request ID, seed ID, source plant, scene day/revision, UE version, solver physics config hash, collision geometry hash, deterministic seed, tick, vector position/velocity in cm, contact normal/material, final rest. Convert cm→m **once** at the boundary and preserve raw bytes for audit.
- **Validator:** Reject unknown object/seed, missing parent, stale source snapshot, impossible units or coordinates, unapproved collider, duplicated or out-of-order terminal events, bounds violations, unsolicited events, and any request to write strategy/ledger/order state. Do not accept a client-provided soil fertility score as truth: resolve cell/material against the canonical terrain version.
- **Owner/commit:** Only an approved ecology owner can emit an idempotent accepted event at **future day** with a durable receipt/digest. A replay consumes the same accepted receipt, not a fresh Chaos run, so Unreal/CPU changes cannot re-roll historical evolution. Untrusted observations are *not yet* canonical.
- **Metrics:** derived dispersal distance, landed tile/cell, substrate, neighbor distance/competition and candidate germination suitability. We have **not** defined/approved the growth, germination probability, and competition formula; they remain placeholders for Agent D, not hardcoded game economics.
- **Game clock:** transition Atlas↔Explore or camera FPS has zero influence on the canonical day tick, event ordering, gene values, allocations, cash, exits, or market history.

## Integration phases / approval gates

1. **Prepared now:** synthetic source snapshots, a typed versioned registry and conversion allowlist, reference ballistics, inverse gene mapping, strict semantic validations and tamper examples. This requires no SOL market archive, UE editor, MCP or broker.
2. **Local Unreal agent:** create actual BP_CloudPlant / PCG_CloudPlant / BP_SeedEmitter / seed actor, implement parameter importer, collision/terminal-settlement recorder, same-world dual cameras; run a Chaos drop and export a candidate receipt. Compare with reference only under matching simple flat-world settings; never claim numerical parity of collision solver from reference analytic code alone.
3. **Ecology owner:** agree provenance, terrain material map, exact germination/competition rule, receipt signing/trust, and whether accepted physics creates placement only or also makes proposed child axes.
4. **Research owner:** introduce candidate genomes via *new generations only*, run causal forward paper evaluation and assess overfitting. No live trading/price source can be mutated by physics before separate approval.
5. **Promote to production only** with authorized, versioned canonical ABI from Agent B, controlled CI, UE packaged tests, replay-consistent event ingestion and explicit user approval.

## Official UE5 references (implementation guidance, not proof of working integration)

- PCG graph params/attributes: https://dev.epicgames.com/documentation/en-us/unreal-engine/procedural-content-generation-overview
- Data Assets: https://dev.epicgames.com/documentation/en-us/unreal-engine/data-assets-in-unreal-engine
- OnComponentHit: https://dev.epicgames.com/documentation/en-us/unreal-engine/BlueprintAPI/Collision/OnComponentHit
- Physics substepping: https://dev.epicgames.com/documentation/en-us/unreal-engine/physics-sub-stepping-in-unreal-engine
- Python Editor Scripting (experimental, editor-only): https://dev.epicgames.com/documentation/en-us/unreal-engine/scripting-the-unreal-editor-using-python
- Remote Control (Beta; bind loopback and use restricted allowlist): https://dev.epicgames.com/documentation/en-us/unreal-engine/remote-control-for-unreal-engine
