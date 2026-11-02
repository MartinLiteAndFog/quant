# Botanical rendering contract

The native Unreal renderer uses only the replay snapshot for ecological and financial numbers. Plant height is `90 + 3.6 × clamp(biomass, 0, 100)` world units. A non-cactus plant has 12–24 alternating leaves, with the count determined by biomass. Each lanceolate leaf consists of two tapered geometric halves. Grass has narrower blades than Bush; Cactus has a thick central stem and four branches. There are no invented flowers, fruit, age or performance values.

Root anchors follow the exact selected trait coordinates. Crown displacement handles coincident coordinates; the branch from root anchor to crown communicates that visual displacement. Picking either the primary stem or a leaf selects the same plant. Leaf actors have the plant as their owner; they never become separate simulated specimens.

Geometry is reused by render slot. A snapshot update hides old slots and assigns the next snapshot to pooled actors. The pool follows the high-water mark of simultaneous plants and geometry, rather than creating new actors for every historical day. Dynamic material instances are reused. The one-second snapshot poll coalesces faster replay writes. This design still requires runtime memory and dense-population verification; it is not a benchmark result.

The HUD shows effective protection: the lower of a positive source stop and the mandatory paper stop. A source hard-stop setting of zero therefore never appears as the effective stop. Parameter labels include units and identify Renko-bar lookbacks. Growth displays use the supplied decomposition (`rootAccess`, `leafCapture`, `waterResponse`, `sunResponse`, `rootLeafCost`, `netGrowth`, `vigorDelta`), without treating ecological biomass as trading profit.
