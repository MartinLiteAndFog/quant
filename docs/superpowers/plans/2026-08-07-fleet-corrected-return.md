# Fleet Corrected Return Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Fleet chart mode **Corrected Return** that strips confirmed cashflows from performance %, with always-on cashflow payloads in `/api/fleet/performance`, while leaving Equity $ / Equity % unchanged.

**Architecture:** Extend `fleet_api.py` to always load ledger cashflows per bot, build a `corrected_curve` (ledger-segmented TWR when sync coverage is good, else Jump-TWR), mark inactive/disabled bots unavailable, and expose the same fields on the portfolio. Fleet Desktop adds a fourth chart mode that reads `corrected_curve`. Live venue fetch is deferred; response shape is ready.

**Tech Stack:** Python (`fleet_api.py`, unittest), React + TypeScript (`apps/fleet-desktop`), existing Postgres `cashflow_events` / `cashflow_sync_state`.

**Spec:** `docs/superpowers/specs/2026-08-07-fleet-corrected-return-design.md`

## Global Constraints

- Do not change Equity $ (`account_curve_abs`) or Equity % (`account_curve`) formulas.
- Do not place orders, initiate transfers, or touch live trading flags.
- Cashflows are always loaded on the performance read path (not gated on UI mode).
- Active bots without ledger use Jump-TWR; inactive/disabled get empty `corrected_curve`.
- `cashflow_return_excluded` does **not** suppress Jump-TWR for active bots (Quant stays chartable).
- Keep `portfolio.cashflow_return` scalar for compatibility.
- Run `gitnexus_impact` before editing symbols; `gitnexus_detect_changes` before commits when GitNexus MCP/CLI is available.

## File Structure

| File | Responsibility |
|------|----------------|
| `src/quant/execution/fleet_api.py` | `_cashflow_corrected_curve`, `_bot_corrected_payload`, wire into `build_fleet_performance`, align + portfolio |
| `tests/test_fleet_api.py` | Unit + wiring tests for corrected curves / meta / inactive |
| `apps/fleet-desktop/src/types.ts` | `ChartMode`, `BotSeries`, `PortfolioSeries` types |
| `apps/fleet-desktop/src/components/HeroChart.tsx` | Plot `corrected` mode for bots + portfolio |
| `apps/fleet-desktop/src/App.tsx` | Mode toggle, legend, portfolio visibility, CSV export |

---

### Task 1: Cashflow-corrected curve (time series)

**Files:**
- Modify: `src/quant/execution/fleet_api.py` (add `_cashflow_corrected_curve` near `_cashflow_corrected_return` ~1241)
- Test: `tests/test_fleet_api.py` (extend `CashflowCorrectedReturnTests`)

**Interfaces:**
- Consumes: same point/flow dict shapes as `_cashflow_corrected_return`
- Produces: `_cashflow_corrected_curve(points: List[Dict[str, Any]], cashflows: List[Dict[str, Any]]) -> List[Dict[str, Any]]` with `{t, equity_pct}` points; empty list when insufficient data. Final point's `equity_pct` must match `_cashflow_corrected_return` when the scalar is not `None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_fleet_api.py` inside `CashflowCorrectedReturnTests`:

```python
def test_corrected_curve_matches_scalar_end_return(self) -> None:
    from quant.execution.fleet_api import (
        _cashflow_corrected_curve,
        _cashflow_corrected_return,
    )

    points = [
        {"t": 100, "equity": 100.0},
        {"t": 200, "equity": 160.0},
        {"t": 300, "equity": 176.0},
    ]
    flows = [{"t": 150, "reporting_amount": 50.0, "equity_after": None}]
    curve = _cashflow_corrected_curve(points, flows)
    self.assertEqual(curve[0], {"t": 100, "equity_pct": 0.0})
    scalar = _cashflow_corrected_return(points, flows)
    self.assertIsNotNone(scalar)
    self.assertAlmostEqual(curve[-1]["equity_pct"], float(scalar), places=5)
    # After deposit removed: 100 → 110 (+10%), then 110 → 176 is wrong;
    # unresolved deposit of +50 at t=150 inside [100,200]: adjusted_end=110 → +10%;
    # then [200,300]: 160 → 176 = +10%; compound ≈ 21%.
    self.assertAlmostEqual(curve[1]["equity_pct"], 10.0, places=5)
    self.assertAlmostEqual(curve[2]["equity_pct"], 21.0, places=5)

def test_corrected_curve_empty_when_insufficient_points(self) -> None:
    from quant.execution.fleet_api import _cashflow_corrected_curve

    self.assertEqual(
        _cashflow_corrected_curve([{"t": 1, "equity": 100.0}], []),
        [],
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace && python -m pytest tests/test_fleet_api.py::CashflowCorrectedReturnTests::test_corrected_curve_matches_scalar_end_return tests/test_fleet_api.py::CashflowCorrectedReturnTests::test_corrected_curve_empty_when_insufficient_points -v`

Expected: FAIL with `ImportError` / `AttributeError` for `_cashflow_corrected_curve`.

- [ ] **Step 3: Implement `_cashflow_corrected_curve`**

Before editing, run GitNexus impact on `_cashflow_corrected_return` (upstream) if available; warn on HIGH/CRITICAL.

Add immediately after `_cashflow_corrected_return` in `fleet_api.py`. Reuse the same interval/flow logic; emit a point after each equity sample. Pseudocode structure:

```python
def _cashflow_corrected_curve(
    points: List[Dict[str, Any]],
    cashflows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Time series of cashflow-corrected equity growth (0% at first point)."""
    clean = [
        {"t": int(point["t"]), "equity": float(point["equity"])}
        for point in points
        if point.get("equity") is not None
        and _is_finite(point.get("equity"))
        and float(point["equity"]) > 0
    ]
    clean.sort(key=lambda point: point["t"])
    if len(clean) < 2:
        return []
    flows = sorted(
        [
            flow
            for flow in cashflows
            if flow.get("reporting_amount") is not None
            and _is_finite(flow.get("reporting_amount"))
        ],
        key=lambda flow: int(flow["t"]),
    )
    growth = 1.0
    flow_index = 0
    last_value = clean[0]["equity"]
    out: List[Dict[str, Any]] = [{"t": clean[0]["t"], "equity_pct": 0.0}]
    for point in clean[1:]:
        interval_flows: List[Dict[str, Any]] = []
        while flow_index < len(flows) and int(flows[flow_index]["t"]) <= point["t"]:
            if int(flows[flow_index]["t"]) > clean[0]["t"]:
                interval_flows.append(flows[flow_index])
            flow_index += 1
        unresolved = 0.0
        for flow in interval_flows:
            amount = float(flow["reporting_amount"])
            equity_after = flow.get("equity_after")
            if equity_after is None or not _is_finite(equity_after) or float(equity_after) <= 0:
                unresolved += amount
                continue
            before = float(equity_after) - amount
            if before <= 0 or last_value <= 0:
                return []
            growth *= before / last_value
            last_value = float(equity_after)
        adjusted_end = float(point["equity"]) - unresolved
        if adjusted_end <= 0 or last_value <= 0:
            return []
        growth *= adjusted_end / last_value
        last_value = float(point["equity"])
        out.append(
            {"t": int(point["t"]), "equity_pct": round((growth - 1.0) * 100.0, 6)}
        )
    return out
```

Optionally refactor `_cashflow_corrected_return` to return `curve[-1]["equity_pct"] if curve else None` to keep one implementation — preferred if tests still pass.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /workspace && python -m pytest tests/test_fleet_api.py::CashflowCorrectedReturnTests -v`

Expected: PASS (existing scalar tests + new curve tests).

- [ ] **Step 5: Commit**

```bash
git add src/quant/execution/fleet_api.py tests/test_fleet_api.py
git commit -m "feat(fleet): add cashflow-corrected equity curve series"
```

---

### Task 2: Per-bot corrected payload helper

**Files:**
- Modify: `src/quant/execution/fleet_api.py`
- Test: `tests/test_fleet_api.py`

**Interfaces:**
- Consumes: `_load_cashflow_data`, `_cashflow_corrected_curve`, `_twr_account_curve`, `_downsample_points`
- Produces:

```python
def _bot_corrected_payload(
    *,
    abs_points: List[Dict[str, Any]],
    venue: str,
    account: str,
    since: Optional[pd.Timestamp],
    until_ts: int,
    bot_status: Optional[str],
    bot_disabled: bool,
) -> Dict[str, Any]:
    """Always loads cashflows. Returns corrected_curve, corrected_meta, cashflows."""
```

Return shape:

```python
{
  "corrected_curve": [{"t": int, "equity_pct": float}, ...],
  "corrected_meta": {
    "method": "ledger" | "jump_twr" | "unavailable",
    "available": bool,
    "reason": Optional[str],  # inactive|disabled|insufficient_equity|ledger_sync_unavailable|...
    "flow_count": int,
    "net_cashflow": Optional[float],
    "source": "db",
  },
  "cashflows": [  # API-safe subset
    {
      "t": int,
      "direction": str,
      "reporting_amount": Optional[float],
      "currency": Optional[str],
      "flow_type": Optional[str],
    },
    ...
  ],
}
```

**Method selection rules (exact):**
1. If `bot_disabled` → `method=unavailable`, `reason=disabled`, empty curve, still attempt cashflow load for payload.
2. Else if `bot_status == "down"` → `method=unavailable`, `reason=inactive`, empty curve, still load cashflows.
3. Else load flows+state via `_load_cashflow_data` for `[since or epoch, until_ts]`.
4. If `abs_points` has fewer than 2 positive equity points → `unavailable` / `insufficient_equity`.
5. Else if state has `last_success_at` and `coverage_end`, and `coverage_start` is not after window start, and at least one flow has usable `reporting_amount` **or** coverage proves no flows needed: try `_cashflow_corrected_curve`. If non-empty → `method=ledger`.
6. Else → `_twr_account_curve(abs_points)` then downsample like other % curves → `method=jump_twr`, `reason` from sync gap if applicable (`ledger_sync_unavailable`) else `None`.

For step 5 when sync coverage is good but there are zero flows: still use `_cashflow_corrected_curve` (identical to pure compound without adjustments) **or** Jump-TWR — pick **ledger with empty flows** so `method=ledger` when coverage is authoritative. If curve build fails → fall through to Jump-TWR.

Ledger coverage check (mirror `_build_cashflow_return` spirit, but do not fail the bot when sync is missing — fall back to Jump-TWR instead of marking portfolio unavailable):

```python
requested_start = int(since.timestamp()) if since is not None else int(abs_points[0]["t"])
coverage_ok = bool(
    state
    and state.get("last_success_at")
    and state.get("coverage_end")
    and state.get("coverage_start") is not None
    and int(pd.Timestamp(state["coverage_start"]).timestamp()) <= requested_start
)
```

- [ ] **Step 1: Write failing tests**

```python
class BotCorrectedPayloadTests(unittest.TestCase):
    def test_inactive_bot_has_empty_curve_but_still_loads_cashflows(self) -> None:
        from quant.execution.fleet_api import _bot_corrected_payload

        flows = [{"t": 150, "reporting_amount": -20.0, "direction": "out",
                  "currency": "USDT", "flow_type": "TransferOut",
                  "amount": -20.0, "status": "completed", "equity_after": None,
                  "source_ref": "x"}]
        state = {
            "coverage_start": pd.Timestamp(90, unit="s", tz="UTC"),
            "coverage_end": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_success_at": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_error": None,
            "source": "test",
        }
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=(flows, state),
        ) as load:
            out = _bot_corrected_payload(
                abs_points=[
                    {"t": 100, "equity": 100.0},
                    {"t": 200, "equity": 80.0},
                ],
                venue="kucoin",
                account="a",
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                until_ts=200,
                bot_status="down",
                bot_disabled=False,
            )
        load.assert_called_once()
        self.assertEqual(out["corrected_curve"], [])
        self.assertEqual(out["corrected_meta"]["method"], "unavailable")
        self.assertEqual(out["corrected_meta"]["reason"], "inactive")
        self.assertEqual(len(out["cashflows"]), 1)

    def test_active_without_sync_uses_jump_twr(self) -> None:
        from quant.execution.fleet_api import _bot_corrected_payload

        pts = [
            {"t": 100, "equity": 100.0},
            {"t": 200, "equity": 102.0},
            {"t": 300, "equity": 300.0},
            {"t": 400, "equity": 306.0},
        ]
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=([], None),
        ):
            out = _bot_corrected_payload(
                abs_points=pts,
                venue="kucoin",
                account="a",
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                until_ts=400,
                bot_status="live",
                bot_disabled=False,
            )
        self.assertEqual(out["corrected_meta"]["method"], "jump_twr")
        self.assertAlmostEqual(out["corrected_curve"][-1]["equity_pct"], 4.04, places=5)

    def test_active_with_ledger_uses_cashflow_curve(self) -> None:
        from quant.execution.fleet_api import _bot_corrected_payload

        pts = [
            {"t": 100, "equity": 100.0},
            {"t": 200, "equity": 160.0},
        ]
        flows = [{
            "t": 150, "reporting_amount": 50.0, "direction": "in",
            "currency": "USDT", "flow_type": "TransferIn",
            "amount": 50.0, "status": "completed", "equity_after": None,
            "source_ref": "y",
        }]
        state = {
            "coverage_start": pd.Timestamp(90, unit="s", tz="UTC"),
            "coverage_end": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_success_at": pd.Timestamp(200, unit="s", tz="UTC"),
            "last_error": None,
            "source": "test",
        }
        with patch(
            "quant.execution.fleet_api._load_cashflow_data",
            return_value=(flows, state),
        ):
            out = _bot_corrected_payload(
                abs_points=pts,
                venue="kucoin",
                account="a",
                since=pd.Timestamp(100, unit="s", tz="UTC"),
                until_ts=200,
                bot_status="live",
                bot_disabled=False,
            )
        self.assertEqual(out["corrected_meta"]["method"], "ledger")
        self.assertAlmostEqual(out["corrected_curve"][-1]["equity_pct"], 10.0, places=5)
        self.assertEqual(out["corrected_meta"]["flow_count"], 1)
        self.assertEqual(out["corrected_meta"]["net_cashflow"], 50.0)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd /workspace && python -m pytest tests/test_fleet_api.py::BotCorrectedPayloadTests -v`

Expected: FAIL — `_bot_corrected_payload` missing.

- [ ] **Step 3: Implement `_bot_corrected_payload`**

Impact-analyze `_load_cashflow_data` / `_twr_account_curve` before editing call sites if GitNexus available.

Implement per rules above. Serialize `cashflows` to the API-safe subset only. Downsample ledger/Jump-TWR curves with `_downsample_points(..., max_points=180, value_key="equity_pct", min_interval_sec=900)` so they match other % series density before clock align.

- [ ] **Step 4: Run tests — expect PASS**

Run: `cd /workspace && python -m pytest tests/test_fleet_api.py::BotCorrectedPayloadTests tests/test_fleet_api.py::CashflowCorrectedReturnTests tests/test_fleet_api.py::TwrAccountCurveTests -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/quant/execution/fleet_api.py tests/test_fleet_api.py
git commit -m "feat(fleet): build per-bot corrected return payload with ledger/TWR"
```

---

### Task 3: Wire into `build_fleet_performance` + portfolio + align

**Files:**
- Modify: `src/quant/execution/fleet_api.py` — `build_fleet_performance`, `_align_series_to_shared_clock`, `_build_portfolio_curve` (or small helper `_portfolio_corrected_curve`)
- Test: `tests/test_fleet_api.py`

**Interfaces:**
- Consumes: `_bot_corrected_payload`
- Produces: each series row includes `corrected_curve`, `corrected_meta`, `cashflows`; portfolio includes `corrected_curve` (equal-weight mean of bot corrected curves that are non-empty), same forward-fill clock as Equity %.

**Wiring details:**
1. In the per-bot loop of `build_fleet_performance`, after `account_curve` is built, resolve:
   - `account = str(b.get("equity_account") or b.get("strategy_instance") or "")`
   - `bot_status = (health_by_id.get(bot_id) or {}).get("status")`
   - `bot_disabled = bool(b.get("disabled"))`
   - Call `_bot_corrected_payload` with `abs_points=acct_pts` (pre-absolute-curve raw points or `account_curve_abs` before align — use the same equity samples that feed abs curve, i.e. `acct_pts` after live stitch), `since=bot_since`, `until_ts=now_ts`.
   - Merge returned keys into the series dict.
2. In `_align_series_to_shared_clock`, if `corrected_curve` non-empty, forward-fill it like `account_curve` onto `[t0, min(t1, last_t)]`.
3. After align, build portfolio corrected % as equal-weight mean of non-empty bot `corrected_curve`s. Prefer extending `_build_portfolio_curve` to also emit `corrected_curve` from `s.get("corrected_curve")`, **or** add `_equal_weight_pct_mean(curves) -> List[Dict]` and set `portfolio["corrected_curve"] = ...` after `_build_portfolio_curve`. Do not alter abs/raw % logic.
4. Existing `portfolio["cashflow_return"] = _build_cashflow_return(...)` stays.

- [ ] **Step 1: Write failing integration-style unit test**

```python
class BuildFleetCorrectedCurveTests(unittest.TestCase):
    def test_performance_series_includes_corrected_fields(self) -> None:
        from quant.execution.fleet_api import build_fleet_performance

        registry = [{
            "id": "a",
            "display_name": "A",
            "strategy_instance": "a",
            "venue": "kucoin",
            "symbol": "SOL-USDT",
            "color": "#fff",
        }]
        acct = [
            {"t": 100, "equity": 100.0, "currency": "USDT"},
            {"t": 200, "equity": 110.0, "currency": "USDT"},
        ]
        corrected = {
            "corrected_curve": [
                {"t": 100, "equity_pct": 0.0},
                {"t": 200, "equity_pct": 10.0},
            ],
            "corrected_meta": {
                "method": "jump_twr",
                "available": True,
                "reason": None,
                "flow_count": 0,
                "net_cashflow": 0.0,
                "source": "db",
            },
            "cashflows": [],
        }
        with patch("quant.execution.fleet_api.fleet_bot_registry", return_value=registry), \
             patch("quant.execution.fleet_api.list_fleet_bots", return_value={"bots": [{"id": "a", "status": "live", "equity": 110.0}]}), \
             patch("quant.execution.fleet_api._load_display_trades_for_bot", return_value=pd.DataFrame()), \
             patch("quant.execution.fleet_api._load_account_points_for_bot", return_value=acct), \
             patch("quant.execution.fleet_api._bot_corrected_payload", return_value=corrected), \
             patch("quant.execution.fleet_api._build_cashflow_return", return_value={"available": False}):
            out = build_fleet_performance(hours=1.0)
        row = out["series"][0]
        self.assertIn("corrected_curve", row)
        self.assertIn("corrected_meta", row)
        self.assertIn("cashflows", row)
        self.assertEqual(row["corrected_meta"]["method"], "jump_twr")
        self.assertIn("corrected_curve", out["portfolio"])
```

- [ ] **Step 2: Run test — expect FAIL** (missing fields / helper not called).

Run: `cd /workspace && python -m pytest tests/test_fleet_api.py::BuildFleetCorrectedCurveTests -v`

- [ ] **Step 3: Implement wiring**

Impact-analyze `build_fleet_performance`, `_align_series_to_shared_clock`, `_build_portfolio_curve` before edits.

Key snippets to add in the series append dict:

```python
corrected = _bot_corrected_payload(
    abs_points=acct_pts,
    venue=venue,
    account=str(b.get("equity_account") or instance),
    since=bot_since,
    until_ts=now_ts,
    bot_status=(live_row.get("status") if live_row else None),
    bot_disabled=bool(b.get("disabled")),
)
# ... include corrected["corrected_curve"], corrected["corrected_meta"], corrected["cashflows"]
```

Align:

```python
corr = s.get("corrected_curve") or []
if corr:
    row["corrected_curve"] = _forward_fill_on_grid(
        corr,
        value_key="equity_pct",
        t0=t0,
        t1=min(t1, int(corr[-1]["t"])),
        interval_sec=interval,
    )
else:
    row["corrected_curve"] = []
```

Portfolio equal-weight mean: copy the pct aggregation loop from `_build_portfolio_curve` but over `corrected_curve` lists; attach as `portfolio["corrected_curve"]`.

- [ ] **Step 4: Run tests**

Run: `cd /workspace && python -m pytest tests/test_fleet_api.py -v --tb=short`

Expected: PASS (full fleet_api suite).

- [ ] **Step 5: Commit + push**

```bash
git add src/quant/execution/fleet_api.py tests/test_fleet_api.py
git commit -m "feat(fleet): expose corrected curves on performance API"
git push -u origin cursor/fleet-corrected-return-1738
```

---

### Task 4: Fleet Desktop Corrected Return mode

**Files:**
- Modify: `apps/fleet-desktop/src/types.ts`
- Modify: `apps/fleet-desktop/src/components/HeroChart.tsx`
- Modify: `apps/fleet-desktop/src/App.tsx`

**Interfaces:**
- Consumes: API `corrected_curve` / `corrected_meta` on series + portfolio
- Produces: `ChartMode` includes `"corrected"`; UI toggle **Corrected Return**

- [ ] **Step 1: Update types**

In `types.ts`:

```typescript
export type ChartMode = "trade" | "account" | "account_abs" | "corrected";

export interface CorrectedMeta {
  method: "ledger" | "jump_twr" | "unavailable" | string;
  available: boolean;
  reason?: string | null;
  flow_count?: number;
  net_cashflow?: number | null;
  source?: string;
}

export interface CashflowPoint {
  t: number;
  direction?: string;
  reporting_amount?: number | null;
  currency?: string | null;
  flow_type?: string | null;
}

// On BotSeries:
corrected_curve?: CurvePoint[];
corrected_meta?: CorrectedMeta;
cashflows?: CashflowPoint[];

// On PortfolioSeries:
corrected_curve?: CurvePoint[];
```

- [ ] **Step 2: HeroChart — curve selection + portfolio**

Update `curveForBot` / `curveForPortfolio` / `sharedDomain` / portfolio plot gate:

```typescript
function curveForBot(bot: BotSeries, mode: ChartMode) {
  if (mode === "account_abs") {
    return (bot.account_curve_abs || []).map((p) => ({ t: p.t, value: p.equity }));
  }
  if (mode === "corrected") {
    return (bot.corrected_curve || []).map((p) => ({ t: p.t, value: p.equity_pct }));
  }
  const curve = mode === "trade" ? bot.trade_curve : bot.account_curve;
  return (curve || []).map((p) => ({ t: p.t, value: p.equity_pct }));
}

function curveForPortfolio(portfolio: PortfolioSeries, mode: ChartMode) {
  if (mode === "account_abs") {
    return (portfolio.account_curve_abs || []).map((p) => ({ t: p.t, value: p.equity }));
  }
  if (mode === "corrected") {
    return (portfolio.corrected_curve || []).map((p) => ({ t: p.t, value: p.equity_pct }));
  }
  return (portfolio.account_curve || []).map((p) => ({ t: p.t, value: p.equity_pct }));
}
```

In `sharedDomain`, treat `corrected` like `account` for portfolio domain.

Portfolio white line condition: include `mode === "corrected"` alongside account modes (not trade).

- [ ] **Step 3: App.tsx — mode toggle, legend, portfolio, export**

```typescript
const CHART_MODES: Array<{ id: ChartMode; label: string }> = [
  { id: "account_abs", label: "Equity $" },
  { id: "account", label: "Equity %" },
  { id: "corrected", label: "Corrected Return" },
  { id: "trade", label: "Trade %" },
];
```

`legendValue`: for `corrected`, use last `corrected_curve` point like Equity %; if empty show `—`. Optionally append thin method hint only in title/tooltip if already trivial — skip new UI chrome.

`portfolioLegendValue`: same for `corrected`.

```typescript
const portfolioOn =
  showPortfolio &&
  !isolatedId &&
  (chartMode === "account_abs" || chartMode === "account" || chartMode === "corrected");
```

Export: also dump `corrected_curve` rows with `mode: "corrected"`.

- [ ] **Step 4: Typecheck / build if available**

Run: `cd /workspace/apps/fleet-desktop && npm run build` (or `npx tsc --noEmit` if script exists)

Expected: success / no type errors on changed files.

- [ ] **Step 5: Commit + push + update PR**

```bash
git add apps/fleet-desktop/src/types.ts apps/fleet-desktop/src/components/HeroChart.tsx apps/fleet-desktop/src/App.tsx
git commit -m "feat(fleet-desktop): add Corrected Return chart mode"
git push -u origin cursor/fleet-corrected-return-1738
```

---

### Task 5: Verification + GitNexus + PR update

**Files:** none new (verification only)

- [ ] **Step 1: Full backend test suite for fleet**

Run: `cd /workspace && python -m pytest tests/test_fleet_api.py tests/test_cashflow_sync.py -v --tb=short`

Expected: PASS.

- [ ] **Step 2: Confirm Equity paths untouched**

Run existing normalize/TWR tests still green; spot-check `build_fleet_performance` still sets `account_curve` via `_normalize_account_curve` (grep/assert in a quick test or code review).

- [ ] **Step 3: GitNexus detect_changes (if available)**

Run: `npx gitnexus detect_changes` or MCP equivalent on staged/working tree. Confirm scope limited to fleet performance + desktop chart mode.

- [ ] **Step 4: Update PR body with implementation summary**

Note: live KuCoin/Kraken fetch remains a follow-up; API always returns `cashflows` from DB now.

- [ ] **Step 5: Final push if any fixups**

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| New Corrected Return mode; Equity $/ % unchanged | 3, 4 |
| Per-bot curve + Jump-TWR fallback | 1, 2 |
| Inactive/disabled → empty curve | 2 |
| Active without ledger → Jump-TWR | 2 |
| Always load cashflows on performance | 2, 3 |
| `corrected_curve` / `corrected_meta` / `cashflows` API | 2, 3 |
| Portfolio corrected curve | 3 |
| Keep `cashflow_return` scalar | 3 |
| Desktop mode + legend | 4 |
| Tests for deposit/withdrawal / fallback / inactive | 1, 2, 3 |
| Live fetch deferred | Out of scope (shape ready) |

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-07-fleet-corrected-return.md`.

User signaled **Go** — proceed with **inline execution** of tasks 1→5 on branch `cursor/fleet-corrected-return-1738` unless redirected.
