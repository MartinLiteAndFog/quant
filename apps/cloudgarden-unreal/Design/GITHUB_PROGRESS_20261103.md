# Cloudgarden GitHub reconciliation — 2026-11-03

The following are shared project work, not disposable alternate implementations. GitHub metadata, current PR heads, source reads and head-specific Actions were checked. All listed PRs are open drafts and unmerged at this audit. A green workflow proves only the scope it runs.

| PR | Implemented on its branch | Verification / integration limit |
| --- | --- | --- |
| #62 `465c574a` | Current native Unreal project, causal SOL execution bridge, single-plant focus/50 cap, native Slate HUD and latest bottom-drawer/portrait source | Earlier native package rendered; latest C++ compile/link passed. Final packaged reference fidelity and renderer stability remain open. Xcode27.2 finalization retry passed: complete native Game build succeeded in46.18s. Latest runtime reference comparison remains pending. |
| #78 `8c55a31a` | Start/pause menu, SOL/BTC/date selection, funding controls; default monthly payments off and shortfall top-up to100 on; gentle ecology defaults | Current-head Actions41652391977 passed. Shared engine changes also affect Unreal's bridge despite the React UI scope. Not integrated into #62. |
| #77 `71423e96` | Native art palette, stable plant-ID phyllotaxis and bounded gene-driven leaf width/pigment | Current-head source-smoke Actions41532259879 passed. No Unreal compile/render/performance proof. Old main-file Canvas HUD and green palette conflict with the current native HUD. |
| #69 → #70 → #71 → #73 | Separate gameplay stack: research does not gate Play; own settled PnL drives growth; descendants reproduce; patience/actual-fill inactivity; exhausted original seed entitlements can be followed by funded capital-recycled births | G1/G2/G3 head-specific Actions passed. This is a parallel, older engine/save/API lineage, not already included in #78 or current #62. |
| #75 `f0623012` | Nine immutable TradingView seed references, original EW source port, MTF RSI and Renko SL/TTP reference audits | All three current-head workflows passed. Still not complete independent entry/HTF/flip/executable-market-fill parity; imported reference PnL must not fund NAV or evolutionary fitness. |
| #63 `7c61f693` | Source provenance audit, synthetic CI and private replay workflow/gate | Public synthetic Actions passed. Private source replay and venue/license assumptions are not independently established by that CI. |
| #65 `fb4ac6e4` | Read-only ECO/Cloudgarden accounting wire contract and cross-language fixtures | Both current-head workflows passed; reconcile with the actual native bridge rather than replacing it wholesale. |
| #68 `c4d9699` | Earlier HUD/selection/UMG starter and browser/mock contract work | Preflight Actions passed explicitly without Unreal Editor. Its mock equity is not evidence for current native strategy performance. |
| #72 / #74 | Canonical vitality export and multi-view projection adapter | Head-specific workflows passed; remain source/adapter work with distinct stacked dependencies. |

## Concrete contradictions to resolve serially

1. **Funding:** current native focus uses monthly100; #78 defaults to topping up only the shortfall after a finished day. These are different financial rules. Preserve separate external-deposit accounting and explicit save migration when adopting the newer policy.
2. **Research start:** #78 React defaults to three parents, evolution enabled, most recent30 observation days and selectable BTC. Current native focus is one SOL plant, evolution paused, full archive starting at its beginning. Keep these as explicit profiles rather than silently replacing the native focus.
3. **Ecology:** #78 disables predation and scales negative vitality change to zero, but it does not implement a separate switch for G2 inactivity: that branch does not yet contain the G2 patience/inactiveDays model. Native old saves do not receive the React-only gentle-restore transformation automatically.
4. **Engine API/save versions:** G3's `advanceSimulationDay(world, tradingDayOutcomes?)`, schema4 and source-engine path differ from current observed-climate/protected-position tick, Kraken replay, full-archive loop and funding path. Copying entire old files would lose current work. Port G1/G2/G3 mechanics individually and version the resulting save/bridge contract.
5. **Graphics:** reuse #77's canonical-ID geometry rules selectively. Do not restore its older Canvas HUD or opaque green palette. User-authored levels, meshes and materials remain artist-owned.

## Next integration order

Finish and verify the current reference-matched native HUD. Then semantically port G1 growth, G2 lineage/inactivity and G3 conserved recycling into the current engine, reconciling the50 strategy cap. Adopt #78 funding/menu rules with explicit profile/save migration. Bring across compatible bridge contracts and art helpers. Promote serially after the relevant checks; do not blindly merge all stacked branches or treat them as absent.

Local working directory is still on the shared fleet branch, while both Cloudgarden directories are untracked locally and published separately through the Cloudgarden PR. Do not stage the whole workspace or overwrite other agents' fleet/trading work.
