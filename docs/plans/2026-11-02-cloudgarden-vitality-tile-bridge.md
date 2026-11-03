# Cloudgarden G2 vitality to Unreal parameter tiles (2026-11-02)

## Implemented in draft branches

- Cloudgarden G2 (#71) owns one `vigor`, one `genes.patience`, and one `inactiveDays` per canonical plant. G2 applies a tiny inactivity penalty **once per economic day**, starting with consecutive no-fill day 15, *not* per rendered tile or frame.
- G2b (#74) exposes `exportVitalitySnapshot(world, revisionByPlantId)`. The caller **must supply genuine source-genome revisions**, one per existing plant. Nothing is inferred from the UE scene. Export includes archived plants for lineage but no capital/PnL.
- This branch (#72) validates and projects one snapshot to zero/many eligible tiles; four visual copies share the same `plantId` physiology and display values. It never calls `advanceSimulationDay`.
- Agent F (#68) already builds four pairwise tiles with required two-gene membership and one physical world; TTP and Sensitivity remain synthetic research-sidecar example axes.

## Formula / behavior

From G2 `simulation.ts`:

```
if consecutive_days_without_executed_fill <= 14:
    factor = 1
else:
    factor = 1 - 0.0003 * (1.2 - patience)    # patience in [0.05, 0.95]

vigor_next = clamp(round4(vigor * own_day_pnl_growth_factor * factor), 0, 100)
```

A genuine confirmed fill resets the single canonical plant streak. Unreal only receives the computed values and `inactivityDecayEligible` flag. UE seed physics cannot directly change vigor, orders, profit, or canonical trade genes.

## Validation

```sh
python3 -m unittest discover -s apps/cloudgarden-ue5/scripts -p 'test_vitality_tiles.py' -v
python3 apps/cloudgarden-ue5/scripts/project_vitality_tiles.py
npm test --prefix apps/cloudgarden
npm run build --prefix apps/cloudgarden
```

Synthetic fixture: `apps/cloudgarden-ue5/fixtures/vitality_snapshot_v1.json`.
The canonical producer's live `world.seed` and snapshot `sceneRef` will not match the synthetic fixture's IDs. A real bridge must supply synchronized, independently validated scene/gene records and revisions.

## Explicitly pending

Actual Unreal Engine editor/Chaos execution, authenticated local data transport, approved canonical genome revision sourcing, real-world synchronized SceneRef, and merged integration tests across both draft PR stacks. The literal `authority` field in JSON is **not** authentication. Until these are implemented, no real G2 historical run is rendered inside UE5 by this code. No main merge, Railway deployment, exchange action, or financial write is authorized by this preparation.
