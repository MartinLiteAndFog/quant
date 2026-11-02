# Cloudgarden native target — evidence status

This is a requirement audit, not a completion claim. The Unreal target is distinct from the earlier React/WKWebView reference app. Source presence and a successful TypeScript build cannot prove native game behavior.

| Requirement | Current source evidence | Evidence still needed |
|---|---|---|
| Native two-garden isometric game | `Cloudgarden.cpp` defines native actors, orthographic camera and Canvas HUD; snapshot bridge supplies two gardens | Native build has succeeded and runtime logs loaded CloudgardenMode; rendered SOL/BTC gardens still require visual verification |
| Actual archive execution | `unreal_bridge.mjs` reads local exported archives through the causal engine; integration notes record actual first-day fills | Current full-archive repeat runs with retained capital/population and bounded state |
| Play/pause/step | Native command-file protocol and bridge handlers | End-to-end click → command → executed candles → changed snapshot → rendered HUD |
| Plant inspection | Native selection/controller and selected-data HUD | Actual mouse/keyboard selection and full effective settings, genes, ancestry, measured analysis |
| Exact positions and overlap growth | Reference app has tested exact anchors and separated crowns; native root/leaf position mapping exists | Native dense-population overlap behavior and selected coordinate-plane controls |
| Groundwater, roots, bees, predation | Ecological engine and reference scene exist | Native visual causality and understandable capital access |
| Atomic genes and causal availability | Reference gene catalog separates atomic modules, recipes and availability metadata | Native complete library interactions; arbitrary graph execution is not established |
| Evolution uses executed parameters | Kraken/EW engine has frozen birth profiles and prior-record selection | Current native inspector must expose the same effective parameters; independent future evaluation remains absent |
| Monthly contributions and 200 living cap | Historical engine separates external contributions, has dynamic births and safety ceiling | Current native HUD and long-run evidence for both gardens |
| Entirely local Desktop app | Native launcher and bridge development paths exist | Self-contained offline packaged native game, icon, launch verification without development dependencies |
| Independent profitable strategy | No such evidence | Frozen out-of-sample/forward evaluation, calibrated costs and provenance; repeated data are training |

The working tree is actively changing. Before release, replace each pending item with direct runtime or artifact evidence. Do not infer fulfillment from this matrix, source comments, a reference screenshot or the absence of a reported failure.

Native editor build has now succeeded and the runtime log identifies CloudgardenMode. Native packaging is in progress. These are narrower facts than a visually verified, fully interactive offline game.

## GitHub audit snapshot

PR #62 (`codex/cloudgarden-app`) currently contains the older 40-file reference prototype. A read-only contents comparison found six modified tracked-on-branch paths and 43 additional local text paths before this status file was added. The exact reviewed sync candidates are recorded locally in `/tmp/cloudgarden-sync-delta.json`; it is not a publish action. Generated outputs, market assets, native build/cache/save directories and the raster HUD reference were excluded.

The local reference README and historical specification have been rewritten for the newer two-archive persistent runtime. The remote versions remain stale until explicitly synchronized. Documentation describes contracts and limitations; it does not replace runtime verification.
