# Agent F: Tile-scoped seed evolution and independent capitalization modes

Status: architecture plus runnable synthetic Python reference, not Unreal Chaos or real trading.
User correction (2026-11-02): each eligible representation of the same canonical plant MAY emit separate seed experiments in EACH parameter region. Earlier "only canonical home tile may emit" restriction is SUPERSEDED.

## Two independent switches per parameter tile

(1) evolutionEnabled true or false: if true, each eligible tile representation with an explicit emitter profile may conduct a distinct research seed-drop experiment. This does not require financial capital.

(2) capitalMode off, paper or live: off disables trading research funding, but does not disable ecology; paper requests a virtual budget and paperFeedMode historical or realtime, not actual venue orders; live is a future requested state but currently rejected unless independently authorized by a trusted backend/risk/venue execution owner. The UE5 interface cannot grant live authorization.

Tile experiments must be distinct from render copies and canonical organisms. A plantId is ONE strategy/identity/lineage, tileId::plantId is a view copy, experiment::tileId::plantId::genomeRevision is a tile-specific research experiment. Release ID additionally includes world seed, source day, tile and batch. These do not duplicate a broker order or capital allocation.

## Current synthetic example

| Tile | Evolution | Capitalization | Requested virtual paper budget |
|---|---|---|---:|
| Lookback x Stop-Loss | enabled | paper (historical) | 25.00 USDT |
| Lookback x Sensitivity | enabled | paper (historical) | 25.00 USDT |
| Sensitivity x TTP | enabled | off | 0.00 USDT |
| Lookback x TTP | enabled | paper (historical) | 10.00 USDT |

Total virtual paper budgets requested = 60.00 out of 100.00 synthetic paper ceiling. This is NOT funded, reserved, filled, deployed or booked. An authoritative shared paper ledger needs separate integration.

The fixture has 4 source plants, 6 eligible tile representations (plant-a x4 and plant-c x2), and 6 independent planned seed experiments. Plant-a has an explicitly provided profile emitting 3 seeds per tile; plant-c has one emitting 2 seeds per tile: 16 reference ballistic outcomes. A capital-off tile still conducts seed research.

## Prepared files

- contracts/parameter_tiles_v1.json: arbitrary pairwise registered axes and tile transforms, tile-scoped candidate research, numeric step precision.
- contracts/tile_research_activation_v1.json: independent evolution and capital modes per tile; virtual budget ceilings; live fail-closed.
- fixtures/tile_seed_profiles_v1.json: explicit synthetic launch genes by canonical plant; missing profile means no release.
- scripts/project_parameter_tiles.py: identity-preserving multi-tile projections, each eligible with unique tile experiment identity.
- scripts/plan_tile_experiments.py: valid, separate per-tile seed launches, deterministic reference ballistics, inverse gene-axis patches from landings on that specific tile, paper-budget arithmetic.
- scripts/test_parameter_tiles.py and scripts/test_tile_experiments.py: source and per-tile policy invariants, missing genes/profiles, toggles, live rejection and deterministic event IDs.

Commands from repo root:

    python3 apps/cloudgarden-ue5/scripts/project_parameter_tiles.py > /tmp/tile_views.json
    python3 apps/cloudgarden-ue5/scripts/plan_tile_experiments.py > /tmp/tile_experiments.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_parameter_tiles.py' -v
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_tile_experiments.py' -v

## Ecology ownership and physics

Each visual representation may submit an independently keyed tile-specific physics candidate after an approved simulation release. The runtime Unreal Chaos hit/settle must be captured with scene/physics/geometry versions, tile ID, plant ID, experiment ID, seed and release ID; multiple contact callbacks cannot create multiple children. The analytical Python model is NOT actual Unreal physics and is a deterministic reference fixture only.

A landing mapped into one tile proposes exactly that tile's two gene axis values for a hypothetical future child. All remaining inherited genes must come from an independently approved inheritance/crossover rule; no automatic birth/trade. An out-of-range landing is unassigned and cannot be clamped. Accepted physics observations are replayed from stored receipts rather than re-tossed on resume.

## Capital and safety boundary

- Each tile may switch paper experimentation on/off independent of evolution.
- Paper allocations are requests to a future authoritative research owner; source feeds may be historical or real-time, both remaining NON-LIVE capital.
- A live request must fail closed until a separate explicit user approval and trusted venue/OMS/risk service validates account, aggregate cross-tile exposure, order semantics, risk limits and a kill switch. Unreal cannot hold exchange API keys or route orders.
- The same plant on four tiles does not cause four actual trading positions. Distinct funded strategies must have explicit separate trial IDs and allocation authority, with risk aggregation at account level.
- No historical trading performance, statistical significance, or empirical fitness may be attributed to synthetic seed-distribution outcomes.

## First local Unreal editor acceptance

Render the same source plant in four tiles. Demonstrate four separate physical seed releases, one per eligible tile (and two releases for plant-c), each tagged with its tile experiment identity. Show evolution and capital switches independently. With paper off, seeds may continue. With evolution paused, paper allocation request may persist. Reject a live request. Route numeric Chaos terminal landing observations back only to the specific tile experiment; no duplicate canonical plant/fills. Preserve Atlas/Explore seamless identity across all region copies.

No Unreal project/editor/Chaos/MCP was run when this document was authored.
