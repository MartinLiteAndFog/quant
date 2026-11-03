# Working together in Unreal Editor

## Ownership

The user owns visual assets, levels, lighting, cameras, materials and plant appearance. Engineering owns historical execution, ecological state, selection, data transport and the HUD. Do not alter designer-owned assets or replace an authored level without an explicit request. Unreal asset files are binary: only one person or agent edits a particular level or Blueprint at a time.

The existing project is `apps/cloudgarden-unreal/Cloudgarden.uproject`. Create designer content under `Content/Cloudgarden/`: `Maps/LV_Cloudgarden_Art`, `Plants/BP_PlantVisual`, `Materials/` and `Meshes/`. The plant visual can initially be an ordinary Actor Blueprint with a Scene root at the soil surface, and movable mesh components below that root. It does not need trading logic.

## Integration contract to implement

These are planned interfaces, not current Editor controls:

- A level-owned garden director selects the visual Blueprint, coordinate scale and garden origin. It preserves authored ground, light and cameras. Procedural environment creation becomes an explicit fallback rather than running over the authored level.
- Each simulated plant retains an immutable identity and exact parameter anchor. Crowding offsets affect its crown only, with a visible connection to the true anchor; a visual asset cannot silently change its mathematical location.
- A typed, read-only visual state carries identity, age, biomass, vitality, phenotype, root/leaf/water/sun genes, measured growth contributions and active/archived status. The Blueprint receives an update event when that state changes, not on every render frame.
- The user maps these measurements to height, leaf density, colour, flowers and animation. Visual controls never change capital, trade signals, performance measurements or evolutionary selection.
- Growth display ranges, visual height limits and material responses are exposed as editable presentation parameters. Simulation parameters remain separately labelled and governed.
- Selection collision resolves to the owning simulated plant even when a leaf or another Blueprint child is clicked. Pooled visuals are completely reset before reuse.

## Current limitation

The current game mode still spawns its own camera/light and procedural ground and plants. Saving an art level is safe, but using it as the finished game requires the director and visual-state integration above. Do not claim the Blueprint integration is already available. The current work remains the native Slate HUD migration; integrate designer assets as a separate verified change afterward.

## Acceptance before handing over

An authored test level must run with its original ground, lights and camera intact. A user-supplied plant Blueprint must receive actual snapshot values, stay selectable after multiple updates, preserve its exact anchor, and remain bounded to the concurrent plant ceiling. Restarting and packaging must preserve the selected level and include all referenced assets. Validate these behaviours before advertising the interface as ready.
