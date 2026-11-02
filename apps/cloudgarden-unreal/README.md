# Cloudgarden — native Unreal prototype

This is the actual Unreal Engine 5.8 game target, separate from the earlier browser prototype. Native actors form two isometric gardens; Unreal Canvas renders measured capital figures and plant inspection. No web view is embedded.

The existing causally tested local replay engine writes `Saved/CloudgardenSnapshot.json`. The native game consumes that immutable snapshot and writes `Saved/CloudgardenCommands.json` for Play/Pause/Step commands. Prices and performance are never synthesized by the renderer. A missing snapshot displays an explicit no-data state.

Build with the installed engine's `Engine/Build/BatchFiles/Mac/Build.sh CloudgardenEditor Mac Development -Project=<absolute Cloudgarden.uproject>`. The build process needs `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` and Apple's installed Metal Toolchain.

This target is under active implementation. Full visual, interaction, archive-loop, native packaging and causality verification remain required before completion. It is a local research simulation without account or exchange access.
