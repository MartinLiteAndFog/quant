# Native macOS development package

Run `scripts/package_unreal.command` from any directory. It selects the full installed Xcode for this process, requires a working Metal compiler, then builds the Game and Editor targets, cooks `/Engine/Maps/Entry`, stages local engine content and archives a macOS Development game under `/tmp/CloudgardenUnrealPackage` by default. Pass another archive directory as its first argument.

The bundle identifier is `com.cloudgarden.game`, distinct from Unreal Editor. Local ad-hoc signing is enabled; this is not a notarized public distribution. The game must be launched and visually checked after packaging. A successful cook alone does not prove the game controls or simulation.

The current simulation bridge remains a separate local Node/tsx process. It requires the causal engine source, installed Node dependencies and the actual SOL/BTC day assets. These are not automatically included by Unreal's content cooker. Both bridge and game must point to the same writable snapshot/command directory. A package containing only Unreal's `.app` is therefore not a self-contained Cloudgarden release. The Desktop package must supply the bridge/runtime/assets and start/stop them explicitly before it can be described as offline-ready.

No Desktop installation is replaced by this build script. Development packages and logs remain available for review before copying a verified release.
