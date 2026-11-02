# Cloudgarden UE5 — preparation only

This folder is NOT a compiled Unreal Engine 5 project. It contains only a
**synthetic visual scene fixture** for the planned read-only game client.
Nothing here connects to the trading engine, private SOL history or any venue.

Architecture / implementation decision record:
../../docs/plans/2026-11-02-cloudgarden-unreal5-agent-f.md

Fixture:
- fixtures/scene_demo_v0.json — fictional plants, two real-named *example*
  SourceGenome visual axes, simulated appearance and genealogy.
- scripts/validate_fixture.py — stdlib validation, zero market data dependencies.

Local fixture verification from repository root:

    python3 apps/cloudgarden-ue5/scripts/validate_fixture.py

No competing ECO-01/Cloudgarden capital ledger or final exchange ABI is defined.
Agent B owns the canonical source contract. UE5 will consume a read-only
rendering projection after that contract is settled, not write to it.

Initial recommendation: a top-down/oblique 3D strategy atlas with zoomable
parameter meadows, interactive individual plants, and separate underworld view.
Do not port trading/PnL, strategy, risk or bot execution code into Unreal.
