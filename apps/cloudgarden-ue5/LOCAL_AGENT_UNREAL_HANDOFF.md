# Local Unreal Engine 5 Agent Handoff — Agent F (2026-11-02)

## Mission

Turn an already specified **bidirectional data contract** into a small running UE5 vertical slice:
(1) typed numeric source genes control prozedural plant geometry and seed release;
(2) Chaos collision/settling becomes a typed candidate observation;
(3) a read-only adapter derives future ecological/offspring proposals;
(4) Atlas and Explore move seamlessly inside one 3D scene.
Do not open broker endpoints, write to the live source ledger, or silently change the strategy rules.

## Inputs already in this PR

- docs/plans/2026-11-02-cloudgarden-unreal5-agent-f.md — equal Atlas/Explore camera/world contract.
- docs/plans/2026-11-02-cloudgarden-ue5-gene-physics-bridge.md — exact authority, source genes and feedback workflow.
- contracts/parameter_registry_v1.json — 13 typed definitions, ranges, source references, target property names and unit conversions.
- contracts/gene_meadow_inverse_mapping_v1.json — signed two-axis mapping from landing m to proposed child strategy parameters; tile 10×10m.
- contracts/chaos_hit_candidate_v1.schema.json — **UNTRUSTED** physics observation shape.
- fixtures/scene_demo_v0.json — synthetic plants/lineage.
- fixtures/parameter_snapshot_v1.json — synthetic plant-a gene values and expected target property values.
- fixtures/seed_release_v1.json — three seeded launch requests, analytic flat-terrain flight.
- fixtures/seed_landing_v1.json — reference trajectory receipts and numeric dispersion.
- fixtures/offspring_proposals_v1.json — proposed candidate genes from coordinates.
- fixtures/camera_journey_v0.json — same-world camera views.

Run from repository root first:
    python3 apps/cloudgarden-ue5/scripts/validate_fixture.py
    python3 apps/cloudgarden-ue5/scripts/validate_gene_physics.py
    python3 apps/cloudgarden-ue5/scripts/project_unreal_parameters.py > ue-props.json
    python3 apps/cloudgarden-ue5/scripts/propose_offspring_genes.py > proposed-children.json

These are synthetic, not local market data tests. Golden fixtures are non-empirical.

## Preflight on the real local Unreal machine

1. Record OS, GPU, Unreal **exact engine version**, project path, plugin versions. The GitHub runner has not opened the Unreal Editor. Pin the version before generating assets; use legally licensed local assets only.
2. Create a new dedicated project in a chosen checked-in source directory (exact path selected locally); ignore DerivedDataCache, Intermediate, Saved, Binaries, large cooked assets and private OHLCV. Do not alter Quant live executors or Railway.
3. Enable PCG and Niagara; add Python Editor Script and Editor Scripting Utilities for **editor-time automation only**. Runtime collision/evolution must be C++/Blueprint, **not Python Editor Scripts**. Remote Control is Beta and optional.
4. Set basic test terrain: a 10m by 10m plane with named loam collider and one Plant-a pivot located at (0.625m, 4.444444m). UE world units are cm; reference synthetic ballistics are m. Explicitly verify handedness and +/- axes in the adapter.
5. Build one persistent level with both Atlas and Explore modes, smooth camera transition, and Underworld surface visibility. No separate scene loads or duplicate simulation instances.

## Planned engine classes/assets (names are proposed, not preexisting)

- CGGeneRegistryDataAsset: versioned, allowlisted parameter metadata. Expose approved Editor/UI properties, no arbitrary script payload. Use a Data Asset or structured Data Table.
- BP_CloudPlant: stable canonical plant ID, source snapshot reference, appearance (hue/saturation) and ecology visuals, selected/archived state.
- PCG_CloudPlant: PCG graph parameters for branching and leaf_count, plus instancing and procedural leaf/stem shapes. Regenerate only affected organism assets, not the entire world.
- BP_SeedEmitter: release-height, yaw, elevation, yaw-spread, speed and count; seeded launch; owner-issued release request. IDs created before spawning.
- BP_CloudSeed: physical collision shape and hit events; enable Simulation Generates Hit Events where appropriate. Accumulate contacts, bounces and sleep state; emit **one terminal landing candidate per seed**, never a hit-per-contact lineage.
- CGPhysicsObservationSubsystem (C++ or Blueprint): immutable request-to-seed correlation, cm→m conversion at boundary, known event/scene revision, stable deduplication and export; record engine/build/physics/geometry hashes.
- CGCameraDirector: one shared view and selected-plant/day state; same-world Atlas/Explore/Underworld as defined by separate camera contract. Same UMG inspector in both views.

## Optional MCP connector (do not assume UE offers this natively)

Recommended: a *local, allowlisted* MCP wrapper around restricted Unreal Editor Python utilities, or UE Remote Control API reachable only on 127.0.0.1. The agent may use these **logical tools**, implemented/tested locally:

| Tool name (proposed) | Allowed action |
|---|---|
| unreal.project_info | Read engine, plugins, project metadata and current level |
| unreal.list_cloudgarden_assets | Read only within /Game/Cloudgarden |
| unreal.import_gene_registry | Validate version + create/update allowlisted Data Asset |
| unreal.apply_parameter_snapshot | Only set registered non-financial UI/PCG/seed emitter properties; report diff |
| unreal.build_seed_preview | Create a temporary preview actor from an approved synthetic request |
| unreal.run_seed_test | Run fixed-input preview in editor/PIE and capture untrusted hit candidates |
| unreal.export_observations | Emit schema-validated candidate JSON to restricted local output |
| unreal.capture_preview | Screenshot or report scene for review |
| unreal.run_cloudgarden_tests | Run read-only data and smoke tests, surface diffs/errors |

Never expose arbitrary Python code execution, world destruction, trading/account calls, broad filesystem access, app credentials or public listener from this MCP. Require explicit opt-in for asset mutations; logs/redact paths/tokens. Version the functions and require scene snapshot/version IDs and allowlist checking before applying any write.

**If MCP is unavailable:** load the registry/parameter-snapshot JSON via Python Editor Script and Data Assets; expose selected properties via Editor tooling or locally bound Remote Control Preset. Epic's Python Editor integration is experimental and *not* a packaged gameplay runtime; Remote Control remains Beta. Test the exact engine version.

## Suggested implementation order for the local agent

1. Parse and validate all synthetic fixtures (no secrets, source SHA unaffected).
2. Generate Data Asset/Gene descriptor classes and mapping UI; enforce type/ranges, cm↔m, bool vs int; render plant-a hue=126°, branching=0.62, leaf count=22.
3. Configure yaw/elevation/speed/count 35°/25°/2.2m/s/3 from the source snapshot. Trace to the loam plane and emit raw collision contacts; aggregate terminal landing once per seed, don't mistake first impact for stable rest.
4. Implement flat ballistic control in reference script and compare *qualitatively* with Chaos, while exporting exact Chaos outputs and settings; do not assume bit-identical landing positions due to solver/collider choices.
5. Using inverse mapping, demonstrate landing position → proposed child channelLookback/stopLossBps on the tile; render ghost seedlings as **unapproved research proposals**. Do not make trades, commit capital or use the trading bot's current state.
6. Add Atlas/Explore seamless camera changes (same plant and day), Underworld, and end-to-end smoke captures.
7. Record all test commands, screenshots, engine build, performance (100/1k/10k), known problems and exact output JSONs in a new Draft PR. No merge to main or live deployment until specifically requested.

## Review/acceptance rules

- The source snapshot controls all exposed properties; incorrect fields or units are rejected before any UE changes.
- A collision yields a numeric candidate with source event ID, seed ID, material, revision and settled status; repeated hits cannot spawn two children.
- Source physics observations are untrusted until the canonical ecology owner approves and seals them.
- A seed landing outside tile is unassigned, never silently clamped into a valid trading gene.
- Derived child gene values apply only to future research generations after approval and independent trading evaluation.
- Identical captured accepted receipts replay to identical ecology; rebuilding the visual scene does not re-roll the past.
- Editor/PIE failure, missing plugins or lack of local market files is reported as such; never invent UE test results.

## Dry-run machine contracts

Before enabling any editor automation, consume the generator outputs as JSON:

- scripts/project_unreal_parameters.py produces a signed-by-file-digest *preview* of 13 typed, allowlisted BP/PCG/WBP property assignments (not actual Unreal asset paths).
- scripts/propose_offspring_genes.py produces JSON only on stdout and PASS/debug notices on stderr. It maps 3 reference landings to forward-only child gene proposals.
- contracts/local_mcp_tools_v1.json specifies the planned local MCP tool names, allowlisted paths and side-effect classes. It is a design contract, not an implemented MCP server.

The agent must implement and test each tool rather than assuming a native Unreal MCP. Neither JSON output contains private price data, account tokens or broker actions.

## Additional mandatory task: multi-tile genomic projections

Follow docs/plans/2026-11-02-cloudgarden-ue5-multi-parameter-tiles.md.
Build TileCatalogDataAsset from contracts/parameter_tiles_v1.json and a read-only
projection subsystem using scripts/project_parameter_tiles.py as semantic oracle.
The same plant must appear four times in four compatible tile coordinate frames
with ONE immutable plantId, ONE source genome and ONE inspector/selection.
Use the fixture tile_genomes_v1.json as the temporary sparse gene source.
A field absent from that plant means NO on every tile requiring that field.
Do not automatically fill unlisted Sensitivity or TTP with display defaults:
these are synthetic research fields and not approved SourceGenome trade genes.

Implement render instance IDs as tileId::plantId and rebuild Unreal ISM index
lookups after instancing changes. Clicking any representation selects the same
plant, highlights every other eligible tile instance, and does not alter a bot,
account or fitness. Atlas/Explore share this source map without cloning events.
New decision: EVERY eligible tile appearance can run an independently keyed
seed physics RESEARCH experiment, even when the same plant appears four times.
This does not multiply source trading strategies, account funds or canonical
births: every release is tile/plant/scene/revision/seed keyed and requires owner
approval before acceptance. Per-tile capital off/paper/live-request switches
are independent of evolution. Live is rejected pending trusted OMS/risk gate.
Before editor work, pass:

    python3 apps/cloudgarden-ue5/scripts/project_parameter_tiles.py > /tmp/cloudgarden_tile_instances.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_parameter_tiles.py' -v

Report the actual UE asset paths, version and screenshots separately; the
prepared JSON and Python tests do not constitute an installed Unreal project.

## Mandatory local build: tile experiments and financial mode controls

Read docs/plans/2026-11-02-cloudgarden-ue5-tile-experiments-and-capital.md
and contracts/tile_research_activation_v1.json. Use synthetic profiles from
fixtures/tile_seed_profiles_v1.json. The per-tile plan oracle is
scripts/plan_tile_experiments.py with tests/test_tile_experiments.py.
Expected fixture: 4 canonical source plants, 6 tile instances, 6 separately
keyed releases, 16 synthetic seed landings and 60.00 USDT of virtual paper
budget requested (NO cash booked). Render two independent switches per tile:
evolution on/off and capital off/paper (future live request locked). Only an
independent trusted backend may approve future per-tile paper/live funding.
In Unreal Chaos, every tile can have its own seed emitter and scene transform;
record terminal settled receipt ID, tile ID, canonical plant ID and experiment
ID. Inverse-map each impact point only onto that tile's TWO defined gene axes.
No direct order/API calls or automatic canonical offspring creation.

## Next stage: event journal + mode-request preview

Before implementing persistent gameplay events, run:

    python3 apps/cloudgarden-ue5/scripts/tile_research_journal.py > /tmp/journal.json
    python3 apps/cloudgarden-ue5/scripts/preview_tile_mode.py > /tmp/mode_preview.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_tile_journal.py' -v

The journal is the semantic oracle for tile-scoped seed IDs, release→landing→
next-day child candidate causality and repeated replay. It is hash-chained for
change detection against a source plan, but NOT trusted by itself or signed.
Real Chaos hits need a separately validated terminal-rest observation, persisted
by an authorized ecology backend before they affect future source state.

In Unreal add a per-tile event/seed inspection timeline and a mode-change
PREVIEW (not a direct setting mutator). Preview must use CAS expected config
hash, show budget/paper/evolution state, reject stale inputs and any live mode,
and await future trusted backend approval. The preview is not accepted cash.
See docs/plans/2026-11-02-cloudgarden-ue5-research-journal-and-mode-preview.md.

## Preferred entrypoint for the local agent: single bootstrap JSON

Instead of manually reconciling every contract, start from:

    python3 apps/cloudgarden-ue5/scripts/export_unreal_bootstrap.py > /tmp/cloudgarden-unreal-bootstrap.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_unreal_bootstrap.py' -v

Implement a read-only JSON/UStruct importer with explicit schema and source
SHA-256 checking. Consume its canonical sourcePlants once, tileVisualInstances
many times, referenceEventJournal and tileCapitalAndEvolutionRequests separately.
Do NOT deserialize world instances into independent canonical plants, copy
ledger balances from visual objects or treat reference seed outcomes as real
Chaos events. Reject unsupported schema versions, missing gene axes, stale
source revision and any attempt to turn the bootstrap into live trading.

The bundle's digest is not authentication; local agent must validate the known
repository source and use only explicitly allowlisted Editor assets. Create
seamless Atlas and Explore views plus one selection inspector shared across
four parameter regions. On a capital-mode toggle use the mode preview module,
then wait for external backend acceptance before representing the change as
active or financially booked. The current bootstrap is generated, not an
installed or packaged UE project.

## HUD: next local agent implementation milestone

Use docs/plans/2026-11-02-cloudgarden-ue5-hud.md and the starter in
apps/cloudgarden-ue5/ue5-starter/Source/CloudgardenUE5/.

As a visual and input reference generate the offline browser HUD:

    python3 apps/cloudgarden-ue5/scripts/export_hud_preview.py > /tmp/cloudgarden-hud-preview.html

Read the default missing-finance and --mock-finance snapshots separately.
Create WBP_CloudgardenHUD child of UCloudgardenHUDWidget and bind source banner,
Portfolio NAV as GESAMTBIOMASSE, return/net PnL, drawdown, winrate, risk unknown,
unique canonical plant Top 5 winner/loser tabs, 4 tile rows, active/archived
population, seed releases/proposed offspring, event ticker and view controls.
Make Atlas information-dense and Explore compact, same selected plant and day.

The C++ code is a starter: compile it in the local pinned UE5 project; there
are no checked-in UMG Blueprint assets or Unreal Editor validation yet. HUD
ApplySnapshotJson is READ ONLY and rejects unsupported live or funded claims.
Capital tile button MUST use a pending backend request preview, not a direct
client-side investment mutation. Never derive a portfolio valuation from 6
visual instances, synthetic seed score or requested paper budget.
