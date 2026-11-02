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
