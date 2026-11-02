# UE5 UMG C++ HUD starter (uncompiled in CI)

Files under Source/CloudgardenUE5/ form a reusable module *starter* and are NOT
an installed Unreal Engine project. The local agent should copy them into an
actual UE5 C++ project with module name CloudgardenUE5 and a pinned UE version,
then add a Blueprint child named WBP_CloudgardenHUD.

## Wire format
Feed the read-only JSON from:

    python3 apps/cloudgarden-ue5/scripts/build_hud_snapshot.py > /tmp/cloudgarden-hud.json
    python3 apps/cloudgarden-ue5/scripts/build_hud_snapshot.py --mock-finance > /tmp/cloudgarden-hud-demo.json

Default shows NO portfolio NAV/Top-5 values because the synthetic scene has NO
authoritative financial ledger. --mock-finance populates deliberately fictional
UI demo values only. Never confuse the requested paper 60 USDT with the NAV.

UMG blueprint events OnHUDDataChanged and OnRankingTabChanged are invoked after
a verified JSON snapshot or UI-only mode change. SetRankingTab switches the same
two Top-5 arrays (winner vs loser), ranked by canonical plant net PnL, NOT tile
representation. SetExploreCompact collapses information density for first-person
mode without losing underlying KPI values. SetSelectedTile is local-only, not a
trading capital-mode operation.

The starter refuses missing/unknown financial provenance, a LIVE source status,
a source banner without DEMO, funded paper balance, and >5 ranking rows. This
is deliberate: before real data, build an independently authenticated backend
HUD adapter owned by the source/ledger agent. Never delete the source checks to
make a live feed appear.

## Compile and review
- Unreal Engine version is NOT available in the GitHub CI runner; API signatures
  and visual widget binding must be verified in the local engine.
- Create WBP_CloudgardenHUD under /Game/Cloudgarden/UI and bind visible fields.
- Do NOT expose BlueprintCallable source-mutation or broker action functions.
- Use stable UMG bindings only on snapshot/mode/selection changes; avoid tick-time
  JSON parsing. Show DE/EN-localized placeholders for unavailable metrics.
- UMG implementation spec lives in docs/plans/2026-11-02-cloudgarden-ue5-hud.md.
