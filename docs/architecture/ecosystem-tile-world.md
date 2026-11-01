# ECO-14: Multi-Asset tiled genetic landscape (paper-only)

**Purpose.** The world is made from many adjacent *parameter tiles* grouped by asset regions, not one PCA/UMAP behavior-map. Every tile has two real, numeric, mutatable strategy inputs with bounds and steps. Each strategy has **one canonical home tile** for birth, root neighborhood, predation and capital identity. It may additionally be projected into other compatible tiles as a **read-only second view**; projection does not clone strategy_id, capital, equity, trades or seeds.

## Data and navigation
- `tileWorld.ts`: TypeScript pure types + runtime checks, per-tile local-normalized gene coordinates, world-grid positions, home population and visible population, region selection, research-only mycelium proposals. Version `1.0` is confined to synthetic Fleet demo; no API/venue calls.
- `tileWorldDemo.ts`: Six **SYNTHETIC** parameter tiles across crypto (SOL IMBA, SOL RSI, BTC trend), FX (EUR/USD breakout), commodities (WTI RSI), indices (Nasdaq trend). All values, strategies, feeds and module names are illustrations, not historical backtest claims. One SOL sample strategy projects onto a second tile, with no duplicate home identity.
- `TileWorldOverview.tsx`: World overview/region filter/tile focus; a tile focus shows plants in actual parameter coordinates and identifies projection-only plants. Mycelium is shown as aggregated research edges, **not thousands of rendered fungi**. Integration into Fleet App is read-only.
- `tests/tileWorld.smoke.mjs`: Node 22 TypeScript-strip smoke tests for dedup, incompatible genomes, tile coordinate choices, Mycelium safety, and stable read-only world state.

## World/Tile/UI separation
World grid coordinates are **presentation only** and must never determine the genotype. The `axisX` / `axisY` are actual trading parameters; a seed's local (x,y) only changes if the corresponding genome values change. The ecology engine must remain independent from scrolling, zooming and rendering.

Home population and world totals **only** count canonical strategy IDs. Read-only projections may appear on several tiles without doubling NAV. An unassigned strategy is a visible research record rather than being silently dropped. Market assets have data source / contract metadata and `dataStatus`; never convert synthetic market history into claims of verified execution.

## Mycelium, not money
A `MyceliumKnowledgeEdge` is a study / hypothesis from one tile to another. It is accepted only if module IDs and the named gene coincide. The output always says `requiresOutOfSampleValidation: true` and `fundingTransfer: false`, including if it is labelled OOS-verified in its source. Cross-asset trading requires dedicated separate market data, fees, funding, quote currencies and independent validation.

## Unreal Engine 5 rendering option
The engine of record is a deterministic, external paper simulator (ECO-01/03); **Unreal must not own** capital calculations, evolution RNG, strategy state or order execution. A future UE5 client may subscribe to versioned, immutable `SimulationSnapshot` or tile-world render snapshots containing: `schema_version`, `simulation_id`, `as_of`, `tiles`, `plants`, `root_network`, `water_depth`, `camera_focus`, `synthetic`; changes passed via event IDs. The current React/Tauri demo stays a lightweight fallback until render CPU/GPU/RAM benchmarks justify Unreal.

## Limitations
The current tile world is presentation/knowledge-suggestion only, not a live multi-asset portfolio. The game has no verified FX/oil/Nasdaq trading feeds or exchange transfers. The current TypeScript fixtures are independent from forthcoming Python ECO-01 wire definitions; an explicit adapter and runtime validation must accompany a future backend integration. Signal parity, order execution and account currency reconciliation are **not implemented**.
