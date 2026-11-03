# Agent F — Tile research event journal and mode-change previews (2026-11-02)

**Status: synthetic executable reference, NOT a persistent event service, a broker endpoint, funded paper ledger, a UE5 project or a Chaos receipt.** Extends the independent per-tile seed experiments in the preceding Agent F contracts.

## The replay boundary

A plant is a single canonical strategy identity. A tile is an independent two-gene coordinate projection and seed research experiment. Each eligible (plantId, tileId, genomeRevision) can generate a deterministic release. Four canonical source plants yield six eligible tile instances, six synthetic release events and sixteen analytical seed landings in the current fixture. Render copies do not create extra canonical plants, orders or account balances.

The new script apps/cloudgarden-ue5/scripts/tile_research_journal.py serializes a **42-event reference journal**, currently:
- 4 tile.mode_declared records (virtual/off mode snapshots; NEVER funded money state)
- 6 seed.release_reference records, each scoped to tile, plant, research experiment and RNG seed
- 16 seed.terminal_landing_reference records, each attached to an earlier release ID
- 16 offspring.axis_patch_proposed or offspring.outside_tile_unassigned records, **not before the next day**

Each record includes a strictly increasing sequence, unique eventId, day, tileId, plantId, payload, previousHash and eventHash. The headHash and planDigest bind the series to a separately known trusted source plan, using sha256 of canonical JSON. Exact re-runs of a source plan produce the same journal; duplicates, changed content, wrong tile, switched ordering and unauthorized current-day births are rejected by reference replay. **A hash chain is not a cryptographic signature and does not authorize a malicious client's event.** Re-validating a checksum or a chain without an independently trusted source plan is insufficient.

This is an **analytical flight fixture**, NOT an actual Chaos collision sample. To integrate real Chaos, a trusted research owner must validate the raw observation's engine version, step configuration, physical geometry hash, scene revision, release/seed IDs, final settled contact, tile membership and source time, and explicitly accept the receipt. That accepted receipt should then be persisted and replayed without re-simulating Chaos (cross-hardware physics need not be bit-identical). Do not promote this synthetic journal or untrusted samples to canonical ecology automatically.

## Capital/evolution mode changes are requests, not actions

scripts/preview_tile_mode.py accepts:
- Exact known source configuration hash (compare-and-swap; stale views rejected)
- A valid tileId
- Requested mode off or paper; exact 2-decimal USDT virtual budget and paperFeedMode
- Optional evolutionEnabled boolean

It validates the proposed entire tile configuration, preserving the **shared paper pool ceiling**, and returns a machine-readable preview with before/proposed settings, old/new config hashes, future effective day and deterministic requestId. It does not alter the in-memory source, persist the request, fund paper equity, book fills or submit orders. The JSON is explicitly marked request_only_requires_backend_acceptance.

A request for live mode is rejected regardless of checkbox state, because no trusted OMS/risk/venue gate exists in this frontend preparation. Eventually the backend must be the sole writer, persist the request ID idempotently, verify source revision and aggregate per-account risk, acknowledge acceptance/denial, and provide a fresh signed/authenticated state snapshot. An Unreal editor/MCP caller has no credential or broker access.

**Evolution mode is independent of funding.** A capital-off tile may continue to throw research seeds. Pausing ecological exploration does not delete a virtual paper budget request or close a position. Any future action involving real orders or active positions must be implemented by the independent execution owner with explicit semantics, not inferred from the UE button.

## Current runnable preflight

From repo root:

    python3 apps/cloudgarden-ue5/scripts/tile_research_journal.py > /tmp/cloudgarden_tile_journal.json
    python3 apps/cloudgarden-ue5/scripts/preview_tile_mode.py > /tmp/cloudgarden_mode_request_preview.json
    python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_tile_journal.py' -v

The sample preview asks to switch the Lookback × Stop-Loss tile from requested paper 25.00 USDT to off while keeping its evolution enabled. Therefore the requested virtual pool falls from 60.00 to **35.00 USDT**; no accounting changes take place. The output is deliberately NOT an approved new activation config.

The regression tests check: deterministic journal and replay; six releases/six distinct experiment keys, sixteen landing/candidate records; no canonical births/orders; tampered payloads even after recomputing a hash chain; replay/dedup/causality; separate switches; stale snapshot rejection; invalid or oversubscribed budgets; and live-mode hard block.

## Local Unreal-agent implementation checkpoint

The local UE agent should import the source tile configuration and canonical plant/experiment IDs, then implement a read-only research event timeline. Display seed launches, terminal hit receipts and ghost offspring patches in the correct tile without duplicating canonical plants. When the operator changes a tile mode, show a preview of source and proposed values, budget request and next effective day; a local-only preview cannot change a live account. The engine should remain operable in both Atlas and Explore without changing the journal.

MCP tool plans in apps/cloudgarden-ue5/contracts/local_mcp_tools_v1.json may include a read-only journal validation and read-only mode-change preview operation. The existence of this tool manifest does NOT mean an MCP bridge or UE assets already exist. A local agent should create and test those separately.

## Integration restrictions and next ownership

- Agent B: authoritative money and source ABI; this branch does not invent an overlapping ledger.
- Agent C: market/TradingView parity; the research journaling has no market history attached.
- Agent D/E: evolution acceptance/fitness and controlled trading policy; tile seed patches are hypotheses only.
- User: capital allocation and live enablement would require a separate explicit authorization path, not a preview.
- Local UE5 agent: real Editor/PCG/Chaos/UMG runtime, screenshot smoke tests and benchmarking. None of these editor tasks has been run in CI.

Nothing here deploys to Railway, merges the main branch or opens/orders on Kraken/KuCoin.
