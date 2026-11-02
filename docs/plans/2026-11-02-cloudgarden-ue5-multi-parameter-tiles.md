# Agent F: Parameter-Tiles and shared plant identities (2026-11-02)

Status: implemented as stdlib Python projection plus synthetic fixtures and tests. This is **NOT a running UE5 world**, a live strategy schema or a new ledger.

## Prior work

- ECO-11/#61 contains a synthetic world with two-gene parameter meadows (IMBA Sensitivity × TTP, separate RSI area).
- Cloudgarden/#62 contains SourceGenome.channelLookback and SourceGenome.stopLossBps, but does NOT contain approved sensitivity/TTP fields in that trading schema.
- The prior Agent F Unreal fixture contained only one Lookback × Stop-Loss tile.

This change registers **arbitrary two-gene tile combinations**. One canonical plant can have zero, one or many *visual instances*. Tiles are charts embedded in a 3D landscape, not a separate organism/strategy or balance.

## Four prepared SOL tile regions

| Tile ID | X-axis | Y-axis | Patch origin, m |
|---|---|---|---|
| sol-imbase | trade.channel_lookback (20–500 candles) | trade.stop_loss_bps (50–500 bps) | [0,0,0] |
| sol-lookback-sensitivity | trade.channel_lookback (20–500) | research.sensitivity (12–25 index) | [14,0,0] |
| sol-sensitivity-ttp | research.sensitivity (12–25) | research.ttp_pct (0.25–1.65%) | [0,14,0] |
| sol-lookback-ttp | trade.channel_lookback (20–500) | research.ttp_pct (0.25–1.65%) | [14,14,0] |

Every tile occupies a 10 m × 10 m patch for the demonstration. Every tile has **exactly two numeric coordinate axes**. A three-parameter question is represented by several pairwise tiles, not by stuffing three coordinates into a single two-axis tile. More pairs are registered in JSON, without adding another branch to every plant class.

**Source separation:** the two trade axes are strictly checked against the Cloudgarden scene plant's existing numeric source fields. The sensitivity/TTP values are synthetic demo sidecar values inspired by the distinct fleet geneLibrary. They are NOT real Cloudgarden SourceGenome fields; Agent B must approve any later mapping. Note that bps and % are different units.

## Yes/no membership per tile

- The plant must be active and born on or before the selected world day.
- The plant's asset region must match the tile.
- Both exact axis genes must exist; **absence is not zero and never filled with defaults**.
- Both values must have the correct type and be within the registered numeric domain. Out-of-range values are NOT clamped.
- Each tested pair emits a machine-readable decision, with reason: has_both_valid_genes, archived, not_born, asset_mismatch, missing_axes, invalid_axis_type, or axis_out_of_range.
- A successful decision emits one instance ID (tileId::plantId), with normalized two-axis coordinates, world position, the shared canonical plant ID and genome revision.

The synthetic fixture contains four canonical plants:
- plant-a owns all four axes, hence **4 instances** at different world positions.
- plant-c has Lookback, Stop-Loss and Sensitivity but lacks TTP, hence **2 instances**.
- plant-b is archived and remains available for genealogy but has **0 active instances**.
- plant-d lacks complete axis pairs and gets **0 active instances**.

There are therefore **6 view instances and still only 4 source plant records**. Clicking any plant-a instance selects the SAME underlying plant-a, which highlights all four representations; the genome, lineage and research inspector are shared. Neither scene projection nor changing Atlas/Explore creates money, trading events or an additional organism.

## One physical event source

Different tile representations of a plant are analytical projections, not four separately living individuals. Hence only the canonical home tile (source plant tileId) can ever be a source for **authorized** ecological seed-release candidates. All other tile copies have zero game/economic side effects; clicks and camera collision are visual interaction only. Even home-tile releases require external explicit release authority, deduplication and future-day ecology acceptance. Never let a visual clone emit a canonical birth or exchange order.

If we later want distinct physics-based mutation experiments on each tile, issue **separately named research experiments** (plantId, tileId, experimentId, sourceRevision), each producing untrusted candidates that require explicit adoption. The canonical organism must not reproduce once per tile by accident.

A seed landing on a registered tile can eventually be inverse-mapped to that tile's pair of candidate child gene values. The existing reference seed inverse example still targets sol-imbase; generalizing its source-owned mutation policy to every tile requires approval of the source schema and ecology ownership. Tile placement itself is already generalized.

## Files and use

- apps/cloudgarden-ue5/contracts/parameter_tiles_v1.json — registered axes, units/provenance, tile layouts, IDs, membership and physics policy.
- apps/cloudgarden-ue5/fixtures/tile_genomes_v1.json — sparse, optional synthetic numeric genes joined by immutable plant ID.
- apps/cloudgarden-ue5/scripts/project_parameter_tiles.py — deterministic source-to-many-view instance projection, no Unreal installation or SOL data required.
- apps/cloudgarden-ue5/scripts/test_parameter_tiles.py — regression tests covering multi-tile positions, source consistency, missing/out-of-range genes, archived state, shared identity, repeated camera-independent runs, and physical origin restrictions.

Run at repository root:

    python3 apps/cloudgarden-ue5/scripts/project_parameter_tiles.py > /tmp/cloudgarden_tile_instances.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_parameter_tiles.py' -v

The emitted JSON has sceneRef, tileSummaries, decisions, instances, plantToInstances, canonicalPlantCount, and renderInstanceCount. The latter is a graphics count, never a trading/strategy/equity count.

## UE5 handoff

Build one TileCatalogDataAsset and one CloudgardenTileProjectionSubsystem. Store source plants in a canonical immutable plantId index; store renderer instance handles in a **separate tileId::plantId lookup**. Epic's Instanced Static Mesh Components allow efficient repeated visual meshes and per-instance custom data. UE5 ISM array indices may change with rebuilds, so maintain the mapping on rebuild: https://dev.epicgames.com/documentation/en-us/unreal-engine/instanced-static-mesh-component-in-unreal-engine

- A tile actor receives X and Y registered gene IDs, ranges/units, world origin and dimensions.
- Add/remove a visual instance only when membership changes; do not add/remove a canonical plant.
- Render coordinates from numeric genes as normalized U,V within that tile. Convert meters to Unreal centimeters once at import; current draft only supports yaw=0. Reject unsupported rotations rather than silently moving organisms.
- Any plant ID chosen in Atlas or immersive Explore highlights each instance and opens one inspector. One snapshot update changes all projections at once.
- Clone instance materials/meshes may differ by LOD but must share exact plant identity and evidence status.
- Collision bodies can support selection/obstacle detection; only home-tile **approved seed emitters** can produce ecology candidate receipts.

### Acceptance

Positive: 4 tiles, 6 visible copies, one source genome/account per plant. plant-a projects four distinct coordinates with identical source ID, plant-c two, missing TTP gives no third/fourth, archived plant-b none.

Negative: no defaults, no clamping, no bool pretending to be a number, no unauthorized gene overriding source values, no duplicate axis tiles/overlapping physical regions, no duplicate seed event authority, no money writes.

No local market data or actual Unreal project/editor run has been used or claimed for this test.
