# Cloudgarden UE5 — preparation only

This folder is NOT a compiled Unreal Engine 5 project. It contains only a
**synthetic visual scene fixture** for the planned read-only game client.
Nothing here connects to the trading engine, private SOL history or any venue.

Architecture / implementation decision record:
../../docs/plans/2026-11-02-cloudgarden-unreal5-agent-f.md

Fixture:
- fixtures/scene_demo_v0.json — fictional plants, two real-named *example*
  SourceGenome visual axes, simulated appearance and genealogy.
- fixtures/camera_journey_v0.json — synthetic, same-world Atlas/Explore navigation path.
- scripts/validate_fixture.py — stdlib validation including camera continuity, zero market data dependencies.

Local fixture verification from repository root:

    python3 apps/cloudgarden-ue5/scripts/validate_fixture.py

No competing ECO-01/Cloudgarden capital ledger or final exchange ABI is defined.
Agent B owns the canonical source contract. UE5 will consume a read-only
rendering projection after that contract is settled, not write to it.

**Decided 2026-11-02:** equal, seamless Atlas and Explore modes in ONE world.
Zoomable strategy-gene meadows and immersive ground/underworld exploration use
one perspective camera rig, one population, one selected plant, one research
day and one authoritative ledger. No level loading or second simulation occurs
when switching. See fixtures/camera_journey_v0.json for synthetic state checks.
Do not port trading/PnL, strategy, risk or bot execution code into Unreal.

## Gene → procedural parameters → physics → numeric feedback (Agent F)

See ../../docs/plans/2026-11-02-cloudgarden-ue5-gene-physics-bridge.md and
LOCAL_AGENT_UNREAL_HANDOFF.md. Typed registry, inverse 2-gene meadow map,
reference seed falls, provisional Chaos hit schema and all synthetic fixtures
are checked into contracts/ and fixtures/. Existing visual/morph genes are
reused by explicit mapping; seed release genes are NEW PROPOSALS, not yet in
the authoritative ecology genome. Seed landing can create a future-only **child
research genome proposal**, never alter current trading parameters or balances.

Run from repo root:

    python3 apps/cloudgarden-ue5/scripts/validate_gene_physics.py
    python3 apps/cloudgarden-ue5/scripts/project_unreal_parameters.py > ue-props.json
    python3 apps/cloudgarden-ue5/scripts/propose_offspring_genes.py > proposed-children.json

The reference solver is analytic, NOT actual Chaos; the latter requires a
local Unreal editor and approved event ingestion.

The two generated JSON files above are dry-run outputs; targets are proposed
Unreal fields, not writable UE assets until a local agent creates the matching
Blueprints/Data Assets. They are not live API calls. See contracts/local_mcp_tools_v1.json
for proposed restricted automation tool signatures (not an installed MCP server).

## Pairwise parameter tiles and one plant in multiple world locations

Four SOL example tiles are registered in contracts/parameter_tiles_v1.json:
Lookback × Stop-Loss, Lookback × Sensitivity, Sensitivity × TTP, and Lookback × TTP.
Optional per-plant genes reside in fixtures/tile_genomes_v1.json; historical
Cloudgarden SourceGenome DOES NOT yet include approved Sensitivity/TTP fields.
This is an explicitly synthetic research-sidecar demo, not historical evidence.

    python3 apps/cloudgarden-ue5/scripts/project_parameter_tiles.py > ue-tiles.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_parameter_tiles.py' -v

The projections contain a yes/no reason for each (plant, tile) pair and a
stable tileId::plantId visual instance ID. Same plantId across tiles means ONE
canonical genome/lineage/account, not multiple strategies or capital allocations.
Missing genes are not defaulted, out-of-range values are not clamped, and
archived plants remain genealogy-only. **Updated decision:** every eligible tile representation may run independently
keyed seed experiments with its own research offspring candidates. Plant genome
and account remain singular. Evolution and financial mode are separate per tile,
and no source cash/strategy trade is changed by a visual representation.
See ../../docs/plans/2026-11-02-cloudgarden-ue5-multi-parameter-tiles.md.

## Independent per-tile ecology and paper-capital modes

See contracts/tile_research_activation_v1.json, fixtures/tile_seed_profiles_v1.json,
and docs/plans/2026-11-02-cloudgarden-ue5-tile-experiments-and-capital.md.

    python3 apps/cloudgarden-ue5/scripts/plan_tile_experiments.py > tile-experiments.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_tile_experiments.py' -v

Every compatible tile may emit separate synthetic seed experiments. One tile
may have evolution on and financial mode off; another can request virtual paper
capital. The fixture requests 60.00 USDT virtual paper budget, none booked or
funded. Live trading is hard-rejected pending trusted external authorization.
The reference physics is analytic and is NOT Unreal Chaos.

## Synthetic event journal and per-tile mode request previews

The source-plan-verified, SHA256-chained research timeline and the guarded
capital-mode preview can be generated without Unreal or price data:

    python3 apps/cloudgarden-ue5/scripts/tile_research_journal.py > tile-research-journal.json
    python3 apps/cloudgarden-ue5/scripts/preview_tile_mode.py > tile-mode-preview.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_tile_journal.py' -v

A synthetic journal contains 4 unfunded tile mode declarations, 6 seed
releases, 16 terminal reference landings, and 16 next-day hypothetical child
patches or unassigned results. The hash chain is NOT a signature, nor is this
a Chaos result or canonical accepted event. Mode preview uses old-config hash
compare-and-swap and preserves evolution when paper mode is switched off; no
account/ledger/exchange writes occur. Actual approvals and persisted receipts
must come from a future trusted backend.

Full protocol: ../../docs/plans/2026-11-02-cloudgarden-ue5-research-journal-and-mode-preview.md

## One-command local UE agent bootstrap

Generate the complete, source-fingerprinted, strictly synthetic scene bundle:

    python3 apps/cloudgarden-ue5/scripts/export_unreal_bootstrap.py > cloudgarden-unreal-bootstrap.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_unreal_bootstrap.py' -v

The JSON contains one source plant index, sparse genes, all registered axis/tile
regions, 6 view instances, 6 independent seed research plans, a 42-event
synthetic reference journal, camera itinerary, planned BP/PCG values and
per-tile unfunded mode requests. The file digest is integrity-only, not a
signature. Importing it is an explicit local Editor action, not a running MCP
service or an authorized ecology/market/finance state. For actual Chaos physics
and real data an independently trusted source and approval are required.

## Game HUD, Top 5 winner/loser and portfolio biomass

Specification: ../../docs/plans/2026-11-02-cloudgarden-ue5-hud.md
Unreal C++ UMG code starter: ue5-starter/Source/CloudgardenUE5/
Offline interactive browser layout template: hud/hud_preview_template.html

Run from repo root:

    python3 apps/cloudgarden-ue5/scripts/build_hud_snapshot.py > hud-missing-finance.json
    python3 apps/cloudgarden-ue5/scripts/build_hud_snapshot.py --mock-finance > hud-fictional-finance.json
    python3 apps/cloudgarden-ue5/scripts/export_hud_preview.py > cloudgarden-hud-preview.html
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_hud*.py' -v

The locally generated HTML can be opened in a browser: Atlas / Explore, Top 5
winners/losers, tile selection, and *fictional finance / no finance* toggles are
interactive. The center is a CSS preview, NOT an actual UE rendering.
Gesamtbiomasse is labeled Portfolio-NAV and ONLY comes from a separate verified
financial source; without it we show a dash, never inferred value from six tile
visual copies or 60 USDT virtual paper funding requests. The fictional sample
NAV 108.90, +8.90% and its rankings are marked MOCK/DEMO, not empirical trades.

Future local Unreal agent must create actual UMG Blueprint assets and compile
C++ starter on pinned UE version. No orders/risk controls/real-paper bookings
are provided in the HUD starter.

## Selected plant HUD drawer — user-chosen close-up meadow direction

Design and implementation: ../../docs/plans/2026-11-02-cloudgarden-hud-plant-focus.md

We now use ONE physical 3D garden, not separate floating island worlds.
When a plant is clicked, the main camera eases toward it a little and a
translucent lower HUD drawer opens with a visually enlarged copy of that
plant (a mere mesh/capture, NEVER another canonical organism or bot),
individual equity development, high water mark, PnL, drawdown, win rate,
age/generation and the trading/research gene summary.

The reference code computes validated per-plant synthetic histories in
scripts/build_plant_inspector.py; the standalone HTML preview accepts clicks
on the garden's plants. In Unreal the new C++ UMG starter
UCloudgardenPlantFocusWidget should be tied to a cosmetic preview
SceneCaptureComponent2D/RenderTarget and a smoothed main-camera director.

All financial data in the screenshot/mock are fictional; without per-plant
source data, the chart/high-water mark stay unavailable. Multiple tile copies
of one plant always open ONE canonical-source inspector, never duplicate NAV.
