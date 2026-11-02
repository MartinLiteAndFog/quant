# Cloudgarden Unreal — Art Direction v1 (2026-11-02)

**Scope:** native `apps/cloudgarden-unreal` UE 5.8 application on PR #62. This is the first controlled technical-art pass, not a final screenshot/mesh-material quality claim. No live trading, account, ecology-tick, NAV, seed funding or paper execution code is changed.

## North-star image

One natural **bright, calm isometric botanical meadow**, not a tiled spreadsheet disguised as vegetation or four floating islands. Above the vegetation, a quiet financial-research HUD uses **dark, translucent sage** panels, thin mint boundaries, off-white numbers and muted secondary labels. Daylight should emphasize botanical volume rather than neon green. The user-supplied "Cloudgarden Pflanzen-HUD mit EVO AN" image is the visual reference but is not committed to the repository, so exact screenshot parity is **not yet verified**.

The garden occupies the visually dominant center. At the top: identity, date, play/pause/step. Left: canonical NAV, net PnL, deposits/fees/fills and active parameter plane. Right: portfolio rankings. Bottom: the selected plant and its *own* measured history, effective risk rules and parent/genome; never a fabricated individual curve. Always distinguish `PAPER` and `EVO PAUSIERT` from live trading.

## 3 plant silhouette families

| Form | Shape language | Legible at 50 plants |
| --- | --- | --- |
| **Grass** | narrow, rising alternating lanceolate blades, light sun-touched green | slender fan, open negative space |
| **Bush** | broader tapered leaves on a branching stem, slightly deeper evergreen | broad silhouette, non-spherical leaf construction |
| **Cactus** | thick waxy green central stem and four sturdy branches | unmistakably upright and compact; no fake flowers or berries |

All three forms are created by the native pooled actors, rather than requiring imported meshes in this pass. Full botanical high-poly assets, ribbed cactus surfaces, subsurface-scattering/wax material and texture-atlas foliage belong to a later Unreal Editor authoring stage.

## One organism — one appearance

The same canonical `plantId` may be projected in several two-parameter views. Its **geometry, palette and phyllotaxis phase must be reproducible from its own immutable identity and current real genes**, never from render index, sorted order, tile, random draw or capital rank.

- `phenotype` → Grass/Bush/Cactus silhouette.
- `biomass` → existing height `90 + 3.6 × clamp(biomass, 0, 100)` and 12–24 foliage elements (except cactus branches).
- `leaf` gene → small continuous width variation *within* that silhouette.
- `water` gene → slightly cooler leaf/stem color (water uptake represented visually only).
- `sun` gene → slightly warmer color; not a PnL indicator.
- `plantId` → deterministic phyllotaxis angle, stable under replay and parameter-plane changes.
- `root` and current plane axes → exact documented anchor and roots; crowding may displace the **crown** but never the anchor.
- `vigor` and `patience` → explanatory inspector/HUD data only for this pass. Do not color-code trading profit as biological vitality.
- Missing or out-of-range visual genes → bounded neutral defaults. No fictional performance, flowers, fruits or seed earnings.

Implemented in `Source/Cloudgarden/CloudgardenArtStyle.h`: all functions are pure and use only the supplied values. `Cloudgarden.cpp` now imports them without changing the ecological engine. All colors are still engine-basic-shape dynamic material parameters, awaiting a native look-dev pass.

## Botanical palette

| Surface | Character | Engine color approximation (linear RGB) |
| --- | --- | --- |
| meadow | sage/moss | 0.30 / 0.43 / 0.235 |
| soil | warm muted earth | 0.24 / 0.205 / 0.145 |
| groundwater | faded blue/teal | 0.20 / 0.48 / 0.51 |
| selected root marker | mint | 0.56 / 0.86 / 0.67 |
| grass/bush leaves | warm/cool gene-conditioned | in `Leaf(...)` |
| cactus/stem | desaturated jade | in `Stem(...)` |
| overlay | dark translucent sage | 0.045 / 0.12 / 0.095 / alpha 0.86 |
| HUD ink/accent | cream-white / mint | in `HudInk`, `HudMint` |

**Material warning:** these are code values, not an sRGB texture palette or proof of perceptually accurate rendering. Material-instance lighting, color-management, exposure, translucency order and color blindness must be checked in the actual UE 5.8 game.

## Practical Unreal production stages

### Stage A — completed in this branch
- Centralized deterministic linear-color scheme for ground, soil, groundwater, plant branches/leaves and HUD.
- Small heritable leaf-width variation and water/sun tonal influence without any trading/economy effect.
- Stable plant-ID-based leaf rotations: the same organism looks the same when sorting/reprojecting.
- Native engine's existing shape + material instances + rendering pool are reused; no extra spawned actors or daily allocations.

### Stage B — to do in the Unreal Editor
- Replace engine BasicShapes material with authored `M_PlantOpaque`, `M_CactusWaxy`, `M_LeafTwoSided`, `M_Soil`, `M_Water` and parameterized instances. Material scalar/vector parameter naming must be agreed with the C++ renderer before swapping; no silent fallback that makes every plant gray.
- Better broadleaf mesh / grass blade with normal variation, soft underside and thickness, three distinct species silhouettes at 50-count camera framing.
- Ground is a continuous meadow with subtle soil depth; groundwater stays below soil and maps to one canonical ledger only.
- World-level soft directional daylight, subdued shadows, controlled exposure and no saturating post effects.
- An actual mini plant viewport in the bottom inspector (SceneCapture2D) with **non-simulating read-only clone**: no physics, harvesting, signals, capital or extra growth tick.
- Native screenshots and click/hover/keyboard input verification; no use of TradingView reference numbers as live results.

### Stage C — scaling
- Count native scene actors/components, game thread time, draw calls, and RAM for 1/20/50 plants and repeated 1s snapshots. Existing geometry pool bounds *allocation count*, not draw calls.
- Replace per-leaf actor overhead with InstancedStaticMesh/HISM (one authoritative canonical plant controller) if the 50-plant profile requires it. Keep leaves pickable by resolving instance hit → owning plant.
- Integrate older UE5 Agent F tile/HUD contracts (#68/#72) selectively into *this* native runtime; they are **not a separate second simulator or a second financial ledger**.

## Acceptance checks on macOS Unreal 5.8

1. Confirm native editor **compiles** this header and `Cloudgarden.cpp` after rebasing on newest PR #62 (not possible in this remote run).
2. Capture native at 1440×900 and 1920×1080 with Grass/Bush/Cactus, one close-up, and 50-plants wide angle. Does each silhouette read at game zoom?
3. Reselect/reorder a plant and switch parameter planes: its color, leaf width and phyllotaxis do not shift; exact root anchor remains.
4. Verify financial HUD figures and status are snapshot-driven (unchanged from parent branch), effective stop not a fabricated value.
5. Check click selection including leaf owners; verify panels do not cover center garden or lose legibility on narrow screens.
6. Run Game/Draw/GPU memory metrics and a 1000+ day snapshot playback; report actual numbers instead of inferring performance from tests.
7. Validate the user-provided HUD reference side by side and resolve any palette/layout differences before calling art final.

**No Unreal Editor verification is claimed in this PR.** This work intentionally stays in presentation code and markdown.
