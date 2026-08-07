# Fleet Corrected Return — Design

**Date:** 2026-08-07  
**Status:** Approved for implementation planning  
**App:** Fleet Cockpit (`apps/fleet-desktop`) + Fleet API (`src/quant/execution/fleet_api.py`)

## Goal

Add a dedicated **Corrected Return** chart mode that removes confirmed cash inflows/outflows from performance %, so deposits and withdrawals no longer distort analysis.

Existing **Equity $** and **Equity %** stay raw (true account levels / unadjusted rebase). Live KuCoin/Kraken transfer data will be wired later; the API shape must already always include cashflows so that connection is a producer swap, not a new consumer path.

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| UI surface | New mode **Corrected Return** (option C) — Equity $ / Equity % unchanged |
| Curve scope | Per bot always, with Jump-TWR fallback when ledger unavailable |
| Inactive / disabled bots | No curve in this mode (`unavailable`) |
| Active bots without ledger | Jump-TWR fallback (e.g. Quant with `cashflow_return_excluded`) |
| Implementation approach | Backend emits corrected curves (Approach 1) |
| Cashflows on read path | Always loaded with `/api/fleet/performance`, regardless of active chart mode |

## Out of scope

- Changing Equity $ or Equity % formulas
- Initiating transfers or any live trading side effects
- Spot-wallet / on-chain deposit origin beyond futures-boundary transfers
- Replacing the existing portfolio `cashflow_return` scalar (keep for compatibility)
- Implementing live exchange fetch in this first slice (shape ready; producer later)

## Current state

- Absolute equity and Equity % come from `equity_snapshots` (+ optional live stitch). Equity % is `_normalize_account_curve` (rebase to first point) — capital moves look like PnL.
- `cashflow_sync.py` already syncs KuCoin Funding↔Futures `TransferIn`/`TransferOut` and Kraken Spot↔Futures transfers into Postgres (`cashflow_events`).
- `_cashflow_corrected_return` already computes a **scalar** corrected portfolio return; desktop UI does not chart it.
- `_twr_account_curve` (jump threshold) exists but is not on the live display path.

## Architecture

```
cashflow_sync (daemon) ──► Postgres cashflow_events
                                │
venue live fetch (later) ───────┤  same normalize + merge
                                ▼
build_fleet_performance ──► always _load_cashflow_data per bot
                         ──► corrected_curve (ledger | jump_twr | empty)
                         ──► cashflows[] on each series row
                         ──► portfolio.corrected_curve
                                ▼
Fleet Desktop ──► ChartMode "corrected" reads corrected_curve
```

### Backend (`fleet_api.py`)

1. **`_cashflow_corrected_curve(points, cashflows)`**  
   Time-series version of `_cashflow_corrected_return`: at each equity sample after `t0`, compound interval growth after subtracting confirmed `reporting_amount` flows (same `equity_after` segmentation rules as today). Output: `[{t, equity_pct}, …]` starting at 0%.

2. **Per-bot build inside `build_fleet_performance`**  
   For every registry bot:
   - Always call `_load_cashflow_data` for the bot’s venue/account and window.
   - Attach `cashflows` (normalized rows in range) on the series object.
   - Decide method:
     - **unavailable** if bot is inactive/disabled (see Visibility) → `corrected_curve: []`
     - **ledger** if sync coverage is sufficient and reporting amounts usable → corrected curve
     - **jump_twr** otherwise (including active `cashflow_return_excluded` bots such as Quant) → `_twr_account_curve`
   - Attach `corrected_meta`: `method`, `available`, `reason`, `flow_count`, `net_cashflow`, optional `source` (`db` now; later `live` / `mixed`).

3. **Portfolio**  
   - `portfolio.corrected_curve`: equal-weight mean of bot corrected curves that have points (same pattern as Equity % portfolio aggregation), aligned on the shared clock.
   - Keep `portfolio.cashflow_return` scalar as today.

4. **Always-query rule**  
   Cashflow load runs on every performance response, not gated on UI mode or `cashflow_return_excluded`. Excluded/ledger-missing bots still get `cashflows` (may be empty) and Jump-TWR curves when active.

5. **Later live connection**  
   Replace or augment `_load_cashflow_data` with a merge of DB + live venue fetch using the same normalizers in `cashflow_sync.py`. Response fields stay stable. No desktop change required for the producer swap.

### Visibility rules

| Bot state | Corrected Return |
|-----------|------------------|
| Inactive or disabled | `corrected_curve: []`, `method: "unavailable"`, `reason: "inactive"` or `"disabled"` |
| Active + ledger OK | Ledger-corrected curve |
| Active + no ledger / excluded / sync gap | Jump-TWR curve (`method: "jump_twr"`) |

**Inactive/disabled definition (implementation):** treat as unavailable when health/status is `down`, or registry/config marks the bot disabled. Dry-run / up / live remain eligible for a curve. Refine only if registry gains an explicit `disabled` flag.

`performance_start` (Quant) continues to clip the display window for all curves including corrected.

### API shape additions

`GET /api/fleet/performance` — each `series[]` item gains:

```json
{
  "corrected_curve": [{"t": 0, "equity_pct": 0.0}],
  "corrected_meta": {
    "method": "ledger",
    "available": true,
    "reason": null,
    "flow_count": 2,
    "net_cashflow": -500.0,
    "source": "db"
  },
  "cashflows": [
    {
      "t": 123,
      "direction": "out",
      "reporting_amount": -500.0,
      "currency": "USDT",
      "flow_type": "TransferOut"
    }
  ]
}
```

`portfolio` gains `corrected_curve` (same point shape). Existing fields unchanged.

### Frontend (`apps/fleet-desktop`)

- Extend `ChartMode` with `"corrected"`.
- Mode toggle label: **Corrected Return**.
- `HeroChart` / legend / export: when mode is `corrected`, use `corrected_curve` (bots and portfolio). Skip bots with empty curves; legend shows `—`.
- Optional subtle method hint (`ledger` / `jump_twr`) in legend or tooltip — no new drawer.
- Types updated for `corrected_curve`, `corrected_meta`, `cashflows`.
- Equity $ / Equity % / Trade % paths untouched.

## Error handling

| Failure | Behavior |
|---------|----------|
| Postgres / load error | Jump-TWR; empty `cashflows`; meta reason reflects sync failure |
| Incomplete ledger coverage | Jump-TWR for that bot |
| Non-reporting currency flows | Omit from ledger adjustment; do not fail whole bot if other flows OK; surface in meta if needed |
| Future live fetch timeout | Keep DB cashflows; `source: "db"` |

Never fail the whole `/api/fleet/performance` response because cashflow correction failed.

## Testing

- Unit: deposit/withdrawal removed from corrected curve growth; raw Equity % still jumps.
- Unit: no flows → Jump-TWR matches `_twr_account_curve`.
- Unit: inactive/disabled → empty corrected curve.
- API: response always includes `cashflows` + `corrected_curve` / `corrected_meta` without mode query param.
- Desktop: mode switch renders corrected series; empty bots omitted from chart.

## Rollout

1. Spec → implementation plan → backend curves + always-load cashflows + tests.
2. Desktop mode wiring.
3. Later PR: live KuCoin/Kraken fetch merged into the same load path (`source: live|mixed`).

## Success criteria

- Analyst can switch to Corrected Return and see performance without deposit/withdrawal distortion.
- Equity $ / Equity % still show true capital levels.
- Every performance poll carries cashflow payload ready for live producers.
- Active bots without ledger still show a corrected-style curve via Jump-TWR.
- Inactive/disabled bots do not clutter the Corrected Return chart.
