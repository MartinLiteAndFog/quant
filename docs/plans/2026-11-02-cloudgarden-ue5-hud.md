# Cloudgarden Unreal 5 — HUD / game interface specification (Agent F, 2026-11-02)

**Status:** runnable Python/JSON HUD data model, offline interactive HTML design preview, and UMG C++ starter. No Unreal Editor build, Blueprint assets, actual market connection or Chaos event provider has been run. Scope stays synthetic and research-only.

## Visual hierarchy

Cloudgarden uses a tactical strategy-game HUD, not a trading-terminal clone: dark forest-glass panels, unobstructed 3D garden in the center, restrained green for gains, red for losses, amber for unknown risk. Readable tabular digits and strong provenance.

| Region | Permanent information | Interaction |
|---|---|---|
| Top rail | Identity, DEMO/PAPER/LIVE evidence state, asset area, current research day | Atlas ↔ Explore (camera only) |
| Left upper hero | **Gesamtbiomasse · Portfolio-NAV** (USDT), NAV line, net PnL, portfolio return | Never calculated from the number or physical size of repeated plants |
| Left lower | Portfolio max drawdown, closed-trade win rate, exposure, loss-limit and kill-switch state | Risk unknown until source is connected; no green safety signal from missing data |
| World center | All parameter-wiese tiles, repeated stable plant identities, active tile focus | Click tile/plant, same underlying source plant from any tile |
| Right upper | Segmented **Top 5 Gewinner / Top 5 Verlierer** | Exact five-or-fewer canonical plants, ranked by signed net PnL USDT, with ROI%, closed-trade count and archived flag |
| Right lower | Tile table: parameter axes, evolution switch, off/paper/future live request, virtual paper limit | Only a **request preview**, never a direct account or order action |
| Bottom continuous strip | Unique active plants, multiple visual instances, evolving tiles, seed releases, landings, unborn candidates, requested paper budget | Never conflates biomass/portfolio, paper funding requests and ecology |
| Event ticker | Recent research events and day | Only those with event day up to current day; unapproved future births cannot look historical |

Atlas is information-dense; Explore keeps the key NAV, leaderboard and focus accessible and folds secondary details. Camera switching does not reset selected plant, selection across tile instances or source chronology. Responsive safe-frame (1440×900 baseline, DPI scaling, gamepad/keyboard focus) and readable minimum text sizes must be verified in the local UE Editor.

## What the financial metrics mean

- **Gesamtbiomasse = Portfolio-NAV.** Financial net asset value from ONE authoritative capital ledger for one account/capital scope. The physical foliage/plant biomass, if later measured, is a separate physical measure with different units; it is not automatically USDT. A repeated plant across four parameter tiles contributes zero extra money or trades.
- **Portfolio-Wachstum (%)** uses start/end NAV for the same scope and period, with flow adjustments if real deposits/withdrawals occur. Current demo has no flows; no production performance convention is assumed.
- **Netto-PnL (USDT)** must be owned by the paper/live accounting ledger, net fees/funding. Real open-position valuation requires the source's actual mark-to-market policy; HUD never invents it.
- **Max. Drawdown (%)** is the largest observed peak-to-trough decrease of a validated chronological NAV series. Always label period/frequency when a real data provider exists.
- **Win rate (%)** = winning closed trades / total closed trades for the selected source and period; unavailable without proper trade evidence.
- **Top 5 winners/losers**: unique canonical plant ID, never rendered tile instance ID. Positive net PnL descending for winners, negative net PnL ascending for losers. Display at most five, tie-break by stable plant ID. Show realized PnL and return and trade sample. Historical archived plants may rank if they made or lost money during the selected window. Cross-tile experimental rankings need an explicit separate (tileId, experimentId) scope before they can be shown.
- **Risk/exposure/kill switch** must be read from a verified risk backend, not from biology or untrusted client state. Missing values are —/unbekannt. Do not hard-code assumed prop firm loss limits as authoritative.
- **Paper budget requested** is not portfolio NAV, not equity reserved, not margin and not actual funded account. Evolution ON/OFF is independent of capital OFF/PAPER/LIVE; a LIVE request remains hard blocked until an independent risk/venue/OMS approval.

## The executable HUD data contract

apps/cloudgarden-ue5/scripts/build_hud_snapshot.py assembles a schemaVersion 1 JSON view model from the already source-verified synthetic scene bootstrap.

The default output contains **NULL financial metrics** and empty Top 5 rankings because that scene has no authoritative financial ledger; falsely deriving NAV from plant graphics or 60.00 USDT requested paper budget is prohibited. It still displays valid synthetic ecology counts.

The optional, explicitly **fictional** fixture hud_mock_finance_v1.json exercises UI rendering: starting NAV 100.00, ending NAV 108.90, return +8.90%, net PnL +8.90, maximum observed DD 3.67%, 22 closed trades / 11 wins = 50% win rate. It has 1 profitable and 2 losing unique plants. This is **not** actual strategy performance or TradingView parity. The losing archived plant is preserved in ranking evidence. All finance fields are labeled availability=mock_only, source=synthetic_only.

JSON fields: schemaVersion/kind/authority/sceneRef, sourceBanner, financialSource, financialMetrics (value decimal string or null, unit, source, availability), portfolioNavSeries, leaderboard (winnerRows/loserRows with plantId/pnl/return/trades), ecologyMetrics, paperBudget, tileCards, focus, recentResearchEvents, risk, hudControls and alerts.

Generate:

    python3 apps/cloudgarden-ue5/scripts/build_hud_snapshot.py > /tmp/hud-no-ledger.json
    python3 apps/cloudgarden-ue5/scripts/build_hud_snapshot.py --mock-finance > /tmp/hud-fictional-only.json

## Clickable browser layout proof

Generate a self-contained offline HTML, with **no external network** and only generated/escaped JSON:

    python3 apps/cloudgarden-ue5/scripts/export_hud_preview.py > /tmp/cloudgarden-hud-preview.html

Open the file on the local machine. It has real interactive winner/loser tabs, Atlas/Explore HUD density changes, tile selection, and fictional-finance vs missing-data toggle. Capital tile actions explicitly display PREVIEW, not a mutation. The 3D middle is a CSS **placeholder**, not an Unreal render or screenshot.

## Unreal UMG implementation starter

Files are in apps/cloudgarden-ue5/ue5-starter/Source/CloudgardenUE5:
- Public/CloudgardenHUDWidget.h: FCloudgardenHUDRankRow, FCloudgardenHUDTileRow, ECloudgardenRankingTab, UCloudgardenHUDWidget read-only exposed values/Blueprint events
- Private/CloudgardenHUDWidget.cpp: restricted source snapshot JSON parser, toggle, atlas/explore compact flag, tile selection; does NOT book money or call venue endpoints
- CloudgardenUE5.Build.cs: Core/Engine/UMG/Json module setup

The local UE agent should create one Blueprint WBP_CloudgardenHUD child of the widget. Suggested children: WBP_HUDTopRail, WBP_PortfolioNAV, WBP_RiskStatus, WBP_Leaderboard, WBP_TileList, WBP_EcologyFooter, WBP_Events, and shared WBP_PlantInspector. Use event-triggered text binding OnHUDDataChanged/OnRankingTabChanged, not per-frame JSON parsing. The code has **not yet been compiled against a specific UE5 version**, so the local agent must resolve API-version and Blueprint layout details.

The C++ starter intentionally rejects unknown authority, unrecognized financial provenance, funded-paper claims and live status; an authenticated ledger adaptor (owned by the capital/source backend, not Unreal) is a future separate implementation. Never remove provenance validation merely to show attractive numbers.

## Quality gates

1. No ledger → NAV —, Top-5 empty, risk unknown, 60 USDT paper requests never appear as 60 USDT NAV.
2. Mock view → NAV 108.90 USDT, +8.90%, max DD 3.67%, 1 profitable plant, 2 losing plants. All clearly fictional.
3. The same profitable plant seen on 4 tiles ranks **once**.
4. Archived loser may rank in historical results, but never appears as active 3D tile instance.
5. Atlas↔Explore changes visual density only. No finance/seed/money side effects.
6. Browser buttons all have a real action. Capital preview does not commit/fund/close/submit.
7. Unknown/invalid source, stale scene, duplicated plant PnL or non-reconciling NAV: reject.
8. Actual UE Editor build, UMG wiring, gamepad accessibility and packaged FPS tests remain necessary and are not claimed by this preflight.
