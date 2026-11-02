# Local Unreal 5.8 setup

Verified 2026-11-02 on this Mac:

- Unreal installation: `/Users/Shared/Epic Games/UE_5.8`.
- Full Xcode: `/Applications/Xcode.app` (first-launch check passes).
- Default xcode-select was `/Library/Developer/CommandLineTools`.
- Downloaded Apple Metal Toolchain 17F109 with `xcodebuild -downloadComponent MetalToolchain` under the full Xcode developer directory.
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun metal --version` succeeds (Apple metal 32023.883).
- Started editor with `scripts/start_unreal.command -log -abslog=/tmp/cloudgarden-unreal-ready.log`.
- Log confirms Metal GPU initialization, `Engine is initialized. Leaving FEngineLoop::Init()`, project browser discovery and 134.69 seconds initialization time.

The Cloudgarden Unreal project now exists at `apps/cloudgarden-unreal/Cloudgarden.uproject`. Its native editor build has succeeded, and a subsequent runtime log loaded CloudgardenMode. Native packaging is in progress. The UI connector previously attached to an older Xcode-error instance; a fully rendered and interactive game still requires direct visual verification. Earlier engine-start AutomationTest condition failures must not be confused with the later project build result.

For normal launcher startup, the owner can select full Xcode once:

```sh
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```

This requires an administrator password. The provided starter instead sets DEVELOPER_DIR only for its own process and does not require changing the global path. No license acceptance was performed by the assistant.
