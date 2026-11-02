# Cloudgarden — selected-plant focus interaction (2026-11-02)

User-selected art direction: first close-up meadow concept, **ONE contiguous physical garden** instead of four floating islands. Logical Lookback × Stop-Loss / Sensitivity / TTP parameter regions remain in the research model, but are subtle AR/coordinate overlays within the garden, NOT four separate physical worlds. The HUD is transparent/hairline and scientific, with restrained foliage and information-design aesthetics.

## Click on a plant

1. Hit test a source plant in the world OR any of its tile-specific visual representations. Resolve tileId::plantId -> **the single canonical plantId**. A leaderboard click must also use the same source ID.
2. Keep the main garden and same source day. Ease the main world camera toward the selected plant (approx 0.65s, 12% dolly, max offset 2.5 m, avoid terrain obstruction), with small side reframing to keep the focused plant visible above the HUD. Never teleport/reload a level or alter prices/evolution.
3. Open an **unobtrusive translucent HUD drawer along the lower left/center**. On its far left, show a separately rendered *visual representation* of the very same source plant, at a larger scale, rotatable later. The local Unreal implementation uses an isolated preview mesh/appearance instance rendered through SceneCaptureComponent2D to a UTextureRenderTarget2D used by WBP_PlantFocusDrawer. No canonical plant Actor/strategy/ledger is duplicated; set physics, collision, seed emitters, ticking simulation and market authority OFF in the preview.
4. Immediately to the right, show canonical plant ID, status, age and generation, its individual equity/PnL, observed high-water mark, max drawdown, win rate and closed-trade sample; a **separate individual equity time series** follows, then a compact gene readout. A line can only be drawn when per-plant history exists. If individual financial evidence is missing show an explicit dash/blank chart, not portfolio NAV copied into the plant.
5. Keep source selection across Atlas ↔ Explore mode; all compatible tile representations are highlighted at once. Switching the leaderboard winner/loser tab also preserves focus, while clicking a different plant atomically replaces the preview, numbers and main camera target. Escape/close hides drawer and restores earlier camera framing without deleting or changing selected source ID.

## Important meaning distinction

- Total biomass in the top HUD is the **portfolio NAV** from one ledger.
- Selected plant's "Biomasseentwicklung" is its own invested strategy-equity history, not the global NAV and NOT a photorealistic estimate of leaf mass. The purely physiological physical biomass, if later computed from the Unreal environment, must carry different physical units.
- Its "Höchststand" is the historical high water mark of the selected plant's own equity series. Max drawdown is computed over that same time series. A profitable plant on four tiles does not get four times the NAV.
- Source must report if numbers are synthetic, verified historical paper or authenticated live; current implementation is synthetic only, and missing data remains null. No cross-plant aggregation or fake high water mark from a scene mesh.

## Implemented reference and handoff

- apps/cloudgarden-ue5/scripts/build_plant_inspector.py computes one read-only inspector object per source plant, a separate 0–N tile visual instance list, optional individual financial time series and safe preview camera commands. It contains the strictly visual focus contract.
- apps/cloudgarden-ue5/fixtures/hud_mock_finance_v1.json now includes **fictional** plant-level equity series, with plant-specific birth/archive day boundaries, exact reconciliation to the user's individual demo PnL and high-water marks.
- apps/cloudgarden-ue5/scripts/build_hud_snapshot.py includes plantInspectors and plantFocusInteraction, shared by both camera modes.
- apps/cloudgarden-ue5/hud/hud_preview_template.html has an interactive one-garden conceptual browser scene. Clicking a visible plant opens a lower translucent drawer with a *2D stand-in only* for the future UMG SceneCapture texture, individual line chart, peak and metrics; CSS scales the scene slightly. Browser animation is not UE5 camera operation or a rendered mesh clone.
- apps/cloudgarden-ue5/ue5-starter/Source/CloudgardenUE5/Public/CloudgardenPlantFocusWidget.h and Private/CloudgardenPlantFocusWidget.cpp define the UMG focus widget, read-only typed inspector values, selected source ID, isolated preview RenderTarget and Blueprint camera-focus events. **These files are not UE-compiled** in GitHub CI.
- apps/cloudgarden-ue5/scripts/test_plant_focus.py checks no new canonical objects/capital/physics, histories, high water mark, one plant on many tile views, archive/birth bounds, and no synthetic PnL fabrication.

## Unreal implementation tasks

- Scene: one UE level/landscape with one physical meadow. Parameter tiles are read-only analytic overlays and optional separately authorized research seed emitters; rendering a tile copy for visualization does not turn it into a second independently funded strategy.
- Selection: hit proxy for each tile visual instance; resolve through source index; store canonical plantId and current scene revision. Do not serialize a copied 3D mesh as a new plant.
- Main camera: use Timeline/interpolation 0.65s, short 12% zoom, aim/offset, spring smoothing and collision checks, with cancellation/replacement if clicked rapidly.
- Offscreen preview: separate SceneCaptureComponent2D/SceneCapture2D Actor, controlled light/background, source phenotypic mesh and material; set PreviewOnly visibility, no collision/physics/seed emission/tick, set ShowOnly to cosmetic preview meshes only. Render target RT_CloudgardenPlantFocus (e.g. 512×640); limit updates to selection or appearance change; do not render each Tick without need.
- Widget: WBP_PlantFocusDrawer derived from UCloudgardenPlantFocusWidget. Bind UImage to render target and metrics to own typed data. Display high water mark and equity chart by series; empty state if no data. Connect RequestSoftWorldCameraFocus and RequestRestoreWorldCamera to the camera director, not any execution/risk service.
- Performance: at least 100/1000 plants testing focus changes, RenderTarget memory, capture/render frame budget, responsive UMG, controller navigation. No blind full-world capture for every single plant.
- Acceptance: selecting plant-a in any of four compatible parameter views opens **one** identical inspector; click another plant rebinds preview immediately; camera zoom eases smoothly; switching Atlas/Explore preserves selection and same scene/day; close returns framing; no additional PnL/seed release or account allocations. Verify actual UE Editor compilation separately.

## Mock values (not market observations)

Plant-a: individual starting equity 20.00 USDT, ending 31.50, peak 31.50, net +11.50, win rate 8/13 and drawdown from the fictional points. Plant-b (archived day 5) ends 17.50; plant-c is born on day 6, so its individual history starts on day 6. Plants without individually sourced equity have no line or high-water mark.

No Unreal scene, browser engine screenshot, market feed or venue order is claimed to have been run by this preflight.
