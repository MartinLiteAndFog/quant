# Cloudgarden design contract

Cloudgarden is a light, readable strategy game built around a working historical simulation. The garden is its main play surface. Cream paper, sage foliage, warm soil and muted water retain the existing garden identity. The HUD and inspector use clear text, tabular figures and explicit units; decorative financial numbers are not permitted.

## Composition

`GameSurface.tsx` owns the HUD, controls, selected plant inspector and research disclosures. On desktop the HUD stays available above the garden and inspector. At narrower widths the inspector follows the garden in document order. The four primary figures are shared NAV, net trading result after costs, external contributions and cumulative fills. Contributions must be subtracted from trading result.

`IsometricGarden.tsx` owns projected game geometry, plant selection, the selected coordinate plane, soil and groundwater. It renders living plants only. Archived specimens can remain available through ancestry and research records. A previous 20-entitlement save must be described as such; a dynamic world can support up to 200 living specimens.

## Exact coordinates, separate crowns

The terrain is a continuous 420 × 420 coordinate surface, not 25 planting slots. Its visible five-by-five tiles provide a measuring grid. Ground vectors use orthographic isometry:

```text
screenX = (groundX − groundY) × √3 / 2
screenY = (groundX + groundY) / 2 − elevation
```

Both ground axes sit at 30°; plant elevation remains vertical. The plane selector changes the actual traits mapped to the two ground axes:

| Plane | Ground X | Ground Y | Bounds |
| --- | --- | --- | --- |
| Wurzel / Blatt | root gene | leaf gene | 0–1 on each axis |
| Wasser / Sonne | water gene | sun gene | 0–1 on each axis |
| Ökologische Handelsgene | ecological lookback | ecological entry threshold | 1–120 bars; 1–3,000 bps |

The last plane represents ecological trading traits. It must not be presented as the active Kraken channel/lookback configuration. The inspector exposes the effective execution profile separately from supplementary source modules.

`gardenPlaneCoordinates` applies an affine mapping from those fixed bounds. A dot and selection ellipse mark the exact root coordinate. The selected plant's numerical trait values appear beneath the map. Sorting, crowding or archiving may never jitter this mathematical anchor.

At high density, a deterministic outward phyllotaxis search separates crown centers while preserving anchors. Thin connectors explain the displacement. Crown scale decreases with population; the selected label remains explicit. All 200 specimens remain in the rendered scene, and a scrollable roster supplies large selection controls. These are two ways to select the same specimens, not population aggregation.

```text
                         crown C       crown A
                            ╲           ╱
                             ╲         ╱
exact trait coordinate ───────●──────────── root anchor
                              ╲
                               crown B
```

The diagram illustrates multiple equal coordinates. Branch displacement communicates overlap; it does not imply different gene values. The renderer exposes raw trait and anchor values as `data-plane-x/y` and `data-anchor-x/y` for verification.

## Soil, water and biological encoding

Grass, Bush and Cactus are distinct projected meshes derived from the plant phenotype. Biomass affects height; the small bar represents ecological vigor. Neither height nor vigor is a claim of financial performance.

The soil cross-section extends below zero elevation. The shared water plane stays below the terrain. It maps NAV relative to starting capital plus net external contributions, so a deposit is not represented as investment return. The depth is bounded to 12–104 scene units: profits raise the level and losses lower it, with saturation at the ends. The caption identifies this as a bounded schematic representation while the displayed capital figures remain exact. Root length is determined by the root gene; the selected caption reports whether that modeled root depth reaches the modeled water depth. This is a visual ecological mapping, not a second capital account or permission to exceed an allocation.

Conception paths use actual event subjects, available parent identifiers and the central gene library. Local paths connect parents to their child; Hive paths connect the library; Hybrid can show both. Random conception does not fabricate parental lineage. Markers correspond to recent conception events rather than permanent arbitrary bee activity.

## Motion and interaction

The motion direction follows the user-invoked design-motion-principles skill: Jakub's restrained production polish first, Jhey's spatial game expression second, and Emil's restraint for repeated and keyboard interactions. There are no pulsing status indicators or attention loops.

Growth uses a short transform transition. Conception can show one finite flight. Pausing, switching off Bewegung, or requesting `prefers-reduced-motion: reduce` removes flight motion and growth transitions; static paths, markers, anchors, selection and text remain usable. Selection responds immediately and has no keyboard-triggered entrance animation.

Each map plant is keyboard focusable and supports Enter/Space. Its accessible name supplies identity, phenotype, vigor and allocation. The mobile roster supplies 44px controls. Focus styling must remain visible. Inspector tabs should use one active tab stop with arrow/Home/End navigation, correctly related tab and panel identifiers, and explicit active state.

## Evidence and verification limits

The committed `isometricCoordinates.test.ts` verifies affine coordinates, axis endpoints, archived exclusion and 200 coincident specimens at four corners and the center. It checks preserved anchors, separated crown centers, viewport bounds and selected motion state. This is a geometry/rendering contract, not a screenshot or browser interaction test. It uses an isolated Node CSS import hook for server rendering; it does not replace visual review.

Safari review is owned by the integrating agent. This independent documentation/review pass inspected current source, not the integrating agent's Safari screenshots. Source inspection confirms the compact desktop HUD rules, responsive composition, visible focus rules and reduced-motion alternatives exist. Actual Safari text size, desktop/mobile overflow, touch selection, keyboard focus travel, finite flight rendering and perceived plant overlap require the integrating agent's bounded browser review. A screenshot alone cannot prove interaction behavior.

Historical archive repetition is training on repeated observations. Modeled execution costs, unverified venue identity, the Renko proxy and absence of funding must stay inspectable. No visual style, cumulative archive day count or ecological fitness score may imply independent strategy profitability.
