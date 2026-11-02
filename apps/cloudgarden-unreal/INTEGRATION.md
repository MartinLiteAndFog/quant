# Native Cloudgarden game integration

The Unreal project is the requested game target. The earlier React/WKWebView application is a reference prototype, not the final game.

Native Unreal actors, orthographic camera and HUD read `Saved/CloudgardenSnapshot.json`. The causal historical engine runs locally through `apps/cloudgarden/scripts/unreal_bridge.mjs`, using bundled local SOL and BTC archives. No remote API supplies prices or decisions.

The bridge accepts its Saved directory as an argument. Launch from the Cloudgarden prototype directory with `node --import tsx scripts/unreal_bridge.mjs ../cloudgarden-unreal/Saved`. Node and tsx are current development dependencies; a self-contained final game distribution is not yet established.

Commands are atomically replaced JSON files `CloudgardenCommands.json`: `{id: <increasing integer>, action: "play"|"pause"|"step"}`. Snapshots separate deposits from trading PnL; each garden has its own persisted, validated session. Unknown or invalid saved state fails rather than silently resetting capital.

Verified bridge step on actual source: SOL day1 30 fills / NAV99.8953; BTC day1 650 fills / NAV96.7779. Both started with100USDT and no external contribution yet. These are execution checks, not profitability evidence.

Open integration requirements: native build, actual rendered game/selection/control verification, full inspector and atomic gene library, ecological visualizations, packaged offline game and Desktop launcher. Do not describe a compiler pass or reference-app build as proof of these requirements.
