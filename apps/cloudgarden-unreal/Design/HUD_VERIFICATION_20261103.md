# HUD implementation and verification — 2026-11-03

## Current source

Native Slate replaces Canvas text and drawn hit regions. Runtime Roboto Light/Regular, actual measured capital charts, native buttons and scroll boxes are implemented. The latest three-reference clarification is represented as an open default HUD, a projected plant callout and a bottom drawer with overview/analysis/parameters/gene-library tabs. No central inspector window remains. The overview uses one persistent selected-plant portrait widget with an on-change scene capture and inverse-opacity Slate drawing. No plant illustration is substituted for the selected plant.

The rejected strong vertical shading was replaced by faint elliptical shading. Scene, plant art and designer assets remain unchanged. The artist collaboration contract is in `EDITOR_COLLABORATION.md`; the level/director/Blueprint integration described there is not implemented yet.

## Proven and unproven

- Earlier Slate glass package: full BuildCookRun passed, required dylib closure completed, local fonts/catalog included, inner/outer signatures verified, game rendered with actual SOL data. Native I/G views and scrolling were exercised. Local gene collection wrote `imba_channel`; a keyboard single-step advanced day47 to48 and remained paused. These checks precede the latest bottom-drawer/portrait interaction change.
- Latest drawer/portrait source: `CloudgardenSlateHUD.cpp` and `PlantPortrait.cpp` compiled and the game executable linked. Unreal's ApplePostBuildSync then failed because Xcode requires renewed license acceptance after its SDK changed to27.2. This is **not** a complete successful app build. The user must review and accept the license in Xcode; the agent must not accept it or work around that finalization gate.
- Latest drawer, portrait transparency/capture bounds, projected callout, subtle radial shading and two-size interaction checks remain pending in the running packaged game. Do not advertise exact reference fidelity or a finished desktop replacement.
- The old HUD test revealed an unbound `MouseWheelAxis` query writing thousands of warnings. The query was removed; the subsequent control log contained zero such warnings.
- The design detector returned no findings for the native files; its silence is not proof of C++ layout, contrast or interaction correctness.

## Renderer failure under investigation

The earlier packaged game asserted at `RendererPrivateUtils.h:317`, `UploadDataProxy == nullptr`, inside `FSceneCullingBuilder::ProcessPostSceneUpdate`. The crash is observed, not merely inferred from a missing computer-use window. A missing window alone is not evidence that a process exited.

A controlled `r.SceneCulling.Async.Update=0` run initially remained live and then failed with the same assertion. Its log confirms the setting was applied; the new crash stack invokes the update directly. This falsifies serialization as a fix. The proposed project/launcher setting was rolled back. Do not label this a confirmed upstream Mac bug or claim an exact Epic issue match.

Local engine source shows that `Reserve()` is forbidden after uploader data has moved into an RDG-owned proxy. The stack proves an uploader order/lifetime violation; it does not establish its initiating cause. It is not evidence that trade histories caused the crash.

Current logs unexpectedly load `/Engine/Maps/Templates/OpenWorld` although packaging explicitly cooks `/Engine/Maps/Entry` and the project has no GameDefaultMap. Next isolated control: same previously packaged executable and paused snapshot, explicit `/Engine/Maps/Entry`, unchanged rendering settings. A separate classic-mesh/non-Nanite renderer-profile experiment is available if needed; do not combine the two tests or call a feature switch a proven engine repair.

## Scope safeguards

GitNexus impact attempts for the native HUD/controller/portrait/scrim report unindexed symbols. The global change scan includes other shared trading/fleet work and does not cover these untracked native files. Review and publish only explicitly selected Cloudgarden files. Preserve source freeze while compiling: editing a file while the compiler reads it previously produced a partial-file diagnostic. Compile success alone does not establish runtime success.

The profitable plant disappearance remains a separately preserved, deferred report. No trading or ecological behavior was changed during this HUD implementation. No live orders, independent profitability, finished environment art or complete game is claimed.

## Follow-up after Xcode update

The user updated Xcode and the complete native Game build subsequently passed, including ApplePostBuildSync, in46.18s. The earlier license block is resolved. Latest drawer/portrait runtime checks remain pending. The explicit Entry-map control also terminated with the same uploader assertion (exit139), so the unintended OpenWorld template is not established as the cause and changing maps is not a demonstrated fix.
