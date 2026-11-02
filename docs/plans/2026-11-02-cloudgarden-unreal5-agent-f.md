# Agent F — Cloudgarden / Unreal Engine 5 preparation (2026-11-02)

Status: ARCHITECTURE PROPOSAL, research-only. No Unreal project has been compiled, run, or benchmarked. This ADR does NOT supersede ECO-01's exchange/ledger contract, Agent B's pending adapter, or Agent C's price/TV parity work.

## Decision proposal

Adopt **Unreal Engine 5 as the long-term interactive 3D visualization/client** for Cloudgarden, while retaining deterministic trading, research/evolution state and capital as one authoritative non-UE simulation. Do not migrate price analytics, risk gates, book accounting or order placement to Blueprints/C++.

Initial camera: **zoomable oblique/isometric strategy atlas (god view)** with close-up inspection and orbit. Optional later first-person flythrough; do NOT build first-person locomotion first. Each meadow is a 2D parameter coordinate plane inside a beautiful 3D habitat. The two axes must be real strategy genes, not arbitrary decor. Market/asset regions are visual zones, not independent money pools.

Read-only architecture:

    historical source / deterministic Python or Cloudgarden research engine
                              |
                     authoritative day snapshots
                       + causal event stream
                              |
                  versioned read-only scene adapter
                     (final ABI owned by Agent B)
                              |
                   UE5 client — world + camera + UI
                       (no accounting writes)

The React prototype remains a useful fast research/test harness. Avoid a costly feature-by-feature port: retain it until the UE proof-of-concept validates usability and performance. No two ledgers and no independent UE market engine.

## Grounding in current code

- Cloudgarden #62: apps/cloudgarden/src/simulation.ts contains GardenPlant (root/leaf/water/sun, phenotype, climate, biomass, habitat fitness, vigor, synthetic evidence), and CapitalLedger; historicalGarden.ts adds per-plant SourceGenome + research fills.
- Cloudgarden SourceGenome v1: channelLookback, pivotLength, atrLength, ewMinMoveAtr, ewRetraceMin/max, ewFilter, ewExit, obligatory stopLossBps. Initial source-genome families imba_trend, imba_countertrend, imba_ew are explicitly candle research variants, not TV parity.
- ECO-14 #41: 2D parameter tiles (correct real gene axes), separate asset regions, read-only synthetic demo.
- ECO-15 #45: heritable visual DNA (hue, saturation, petals, rotation, stem curve, ribbing, spine, branching/leaf/rind features) and cactus green-rind physiology.
- ECO-11 #61: prototype landscape / underground aggregate; existing max ~1200 SVG scene objects is a prototype cap, NOT a UE benchmark.
- Agent B owns the canonical JSON/wire ABI. UE renderer consumes a downstream projection of that ABI; this design must NOT publish a competing ledger contract.

## Visual mapping (truth and labels remain explicit)

1. **Meadow location:** chosen strategy family/tile declares two axis-gene IDs and their finite minimum and maximum. x/y are normalized from each plant's *actual versioned strategy genome*. Missing/out-of-range/mismatched family => "unassigned" display area, not fabricated coordinates. Never substitute habitat genes (root, leaf) for trading axes.
2. **Identity and genealogy:** immutable plant_id, lineage parent IDs, birth/archive effective day and family. Visual DNA derives only from protected, versioned appearance data; no future offspring shown before birth. Retain archived plant metadata in a separate archive genealogy layer.
3. **Visual biology:** shape, branching, petal count, hue, root distribution, chlorophyll rind, climate response and bee/predator animation. This is a *visual* encoding. Distinguish synthetic habitat fitness and biomass from empirical out-of-sample confidence and net TWR: use separate labels, palettes and tooltips; do not color profitable plants green based on unverified paper scores.
4. **Groundwater, sun, hive/mycelium:** single world shared capital display; water/soil metaphor does not create any new balance. Hive/genomic exchange conveys research information, never a currency transfer.
5. **Trading:** display only signed, known-at-or-before-day metrics and per-plant execution evidence. Where actual source/venue/parity is unverified, badge/notices in plain language "historical paper / venue unverified / not_comparable". Never imply exact exchange fills or financial alpha from game health.
6. **Research Underworld:** aggregate counts and evidence distributions per cluster/asset/family. Millions of candidates are never independently materialized into meshes or actors.

## Rendering and interaction plan

- UE5 PCG: tile terrain, local plant scatter, soil props. For changing individual plants, adapt material parameters/transforms rather than rerunning entire landscape every day.
- ISM / HISM / Nanite as benchmarks justify: instance repeated stems/foliage/grass. Do not create a UObject/Actor for every leaf, seed, candidate or fungus. Use per-instance custom material data for health/height/hue. Nanite and HISM choices require profiling on target hardware, not an unconditional promise.
- Niagara: bee flight paths, pollen, seed distribution and limited soil particles. Bee motion is cinematic interpolation of **already-authorized** causal research/birth events, not a mutation engine.
- UMG: evidence inspector, strategy genome and real two-axis coordinates, day control (play/pause/seek), selected lineage, accounting display (read-only). UI may send *simulation control requests* to an owner service only in a future approved local research adapter; no direct balance or trading mutation.
- Camera: overview -> market region -> two-gene tile -> individual plant. Zoom/LOD switches to aggregate cluster impostors at distance, detailed branches/roots only for selected/nearby objects.
- Underground: separate toggle/camera/layer showing aggregate root, water and mycelium networks, with labels explaining metaphor versus real capital. Keep all rendered objects bounded by camera and performance budget.
- World Partition: optional after measured need for landscape streaming. For the initial single-tile vertical slice, a simple level is easier and more reproducible.
- All times UTC, one authoritative day_index/event ordering. Render at arbitrary frame rate while simulation steps only when canonical snapshots arrive. Recover after reconnect from full snapshot, then ordered event IDs; do not replay old births twice.
- Data-bound animation is one-way: an FPS drop must never alter trade count, NAV, observed fitness, mutation sequence or timing.

## Scene adapter contract — NOT the Agent B wire contract

This view contract is provisional; integration must wait for Agent B's ECO-01/Cloudgarden reconciliation. The checked-in fixture apps/cloudgarden-ue5/fixtures/scene_demo_v0.json demonstrates **visual fields only**, schemaVersion 0 and synthetic_visual_fixture_only. It is NOT a validated live or empirical source and has no real cash booking. Names, IDs, family category, axis definition, genome value and visible appearance are illustrative, with camera- and level-relative units. Do NOT introduce a standalone "UE NAV" or serialized capital ledger.

Later read-only projection must reference the canonical source snapshot with:
- schema version, source dataset fingerprint, strategy code version, source clock/day index, event IDs and known_at;
- canonical plant IDs/parents, statuses and two actual per-tile gene values, verified lineage;
- distinct synthetic_habitat vs paper_performance vs empirical_OOS evidence status;
- existing money amounts as *read-only exact decimal strings* from Agent B, not recreated or re-rounded in UE;
- scene presentation versions and stable hashing for deterministic visual regeneration.
- fail closed on missing/mismatched values, stale timestamps, unsupported schema or unverified category.

## Suggested first UE vertical slice (no historical market data required)

1. Create blank UE5 C++/Blueprint desktop project in a **separate directory** after choosing supported host machine and pinning UE5 version. Do not commit DerivedDataCache, Binaries, Intermediate, Saved, proprietary Marketplace assets or private OHLCV. No current UE editor run is claimed.
2. Load the generated example JSON using a tiny read-only importer. Three starting plants and one illustrative child on a single named tile, with at least one missing-axis placement visible in "unassigned". Use simple primitives and placeholder materials, not expensive final foliage.
3. Orthographic/oblique zoom, plant click to show real channelLookback vs stopLossBps axes, archived parent genealogy; sunshine, simple roots and cactus green rind. Demo label must persist in screenshots.
4. Advance local *display-only* events from a short baked timeline. Birth, death, dormancy, pollination and animation do not change the fixture's trading results or create balances.
5. Benchmark 100, 1,000 and 10,000 visible instances and record editor FPS, packaged desktop FPS, frame time, VRAM/RAM and GPU class. Select PCG/ISM/Nanite/World Partition optimizations based on these measured values.
6. After Agent B's ABI is stable, build one read-only adapter to actual published research snapshots and compare serialized plant IDs, axes, evidence labels, event counts and capital display 1:1 with canonical source output. Agent D will later decide **how empirical trading fitness affects reproduction** — not the renderer.

## Explicit acceptance tests

- The same initial fixture/seed yields identical world placement and visual DNA on repeated loads. A camera move/LOD change never changes source metrics.
- Each meadow tile has exactly two strategy gene axes, printed with min/max units and verified against the available genome; incompatible or missing genes never get arbitrary placements.
- One archived plant disappears from active population but remains in lineage; one child is shown only after its authorized birth event.
- No synthetic habitat visual is presented as empirical net performance or OOS confidence.
- No market/order/account/wallet/venue credential or executable endpoint can be reached by Unreal.
- Delayed, repeated, missing, or out-of-order events fail closed to source snapshot reconciliation rather than producing fake evolution.
- No data-dependent Unreal scene requires the local SOL Parquet file or a real broker.
- Benchmark results and visual UX must justify the extra cost vs the existing React/SVG prototype before a complete frontend migration.

## Unresolved product decisions to discuss

1. **View:** strategic isometric garden with free zoom (recommendation) vs immersive first-person exploration.
2. **Style:** high-quality stylized/semi-realistic botanical world vs photorealistic vegetation.
3. **Platforms:** native desktop first (recommendation); browser/WebGPU access is not something to assume from an Unreal editor build.
4. **When to migrate:** small proof-of-concept first, no unconditional full port.
5. **Long-term use:** research cockpit only vs optional game-like public product; must never silently expose private exchange histories or trading credentials.

## Official engine references

- PCG framework: https://dev.epicgames.com/documentation/en-us/unreal-engine/procedural-content-generation-framework-in-unreal-engine
- Instanced Static Mesh: https://dev.epicgames.com/documentation/en-us/unreal-engine/instanced-static-mesh-component-in-unreal-engine
- PCG generation modes: https://dev.epicgames.com/documentation/en-us/unreal-engine/using-pcg-generation-modes-in-unreal-engine
- World Partition: https://dev.epicgames.com/documentation/en-us/unreal-engine/world-partition-in-unreal-engine

No merger, Railway deployment, trading alerts, live capital or exchange orders are authorized by this proposal.
