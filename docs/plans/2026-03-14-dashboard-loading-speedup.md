# Dashboard Loading Speedup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `/api/dashboard/chart` response time from ~3-5s to <500ms for cached polls and <1.5s on cold first-load.

**Architecture:** The current chart endpoint runs ~15 sequential I/O operations with massive redundancy (renko parquet read 3x, closed_trades Postgres query 6x, regime SQLite query 2x, `build_trading_diary` called 2x with an internal duplicate query). We fix this in three phases: (1) add response-level TTL cache (instant win for 4s polling), (2) deduplicate I/O so the first/cold load is 3-4x faster, (3) let the frontend render immediately with skeleton UI instead of blocking on the slowest query.

**Tech Stack:** Python/FastAPI (backend), React/TanStack Query (frontend), Pandas, PostgreSQL, SQLite, Redis

---

## Problem Analysis

A single `GET /api/dashboard/chart` request currently makes these I/O calls **sequentially**:

| # | Function | I/O | What it reads |
|---|----------|-----|---------------|
| 1 | `load_renko_bars()` | Disk | `renko_latest.parquet` **(read #1)** |
| 2 | `load_trade_markers()` | Postgres | `closed_trades` **(query #1)** |
| 3 | `load_live_fill_markers()` | Disk + KuCoin API | `fills_cache.parquet` + API refresh |
| 4 | `load_active_levels()` | Disk | `execution_state.json` |
| 5 | `load_trade_segments()` | Postgres | `closed_trades` **(query #2, duplicate!)** |
| 6 | `build_fibo_levels()` | Disk | `renko_latest.parquet` **(read #2, duplicate!)** |
| 7 | `load_renko_health()` | Disk | `renko_latest.parquet` **(read #3, duplicate!)** |
| 8 | `build_regime_overlay()` | SQLite | `regime_state_ts` **(query #1)** |
| 9 | `build_regime_scores()` | SQLite | `regime_state_ts` **(query #2, duplicate!)** |
| 10 | `build_equity_curve()` | Postgres | calls `build_trading_diary()` → `closed_trades` **(queries #3+#4)** |
| 11 | `build_trading_diary()` | Postgres | `closed_trades` **(queries #5+#6, all duplicates!)** |
| 12 | `load_real_equity_history()` | Postgres + API | `equity_snapshots` **(query #7)** |
| 13 | `load_kraken_equity_history()` | Postgres + Redis | `equity_snapshots` **(query #8)** |
| 14 | `load_kraken_metrics()` | Redis + Disk | Redis key or JSON file |
| 15 | `build_combined_equity()` | None | Pure computation |

**Redundancies:**
- `renko_latest.parquet` read **3 times** (load_renko_bars, build_fibo_levels, load_renko_health)
- `closed_trades` queried **6 times** (load_trade_markers, load_trade_segments, build_trading_diary×2 inside build_equity_curve, build_trading_diary×2 direct)
- `regime_state_ts` queried **2 times** with identical params (build_regime_overlay, build_regime_scores)
- `build_trading_diary` called **twice** (once by build_equity_curve, once directly) — and internally it calls `load_closed_trades_from_postgres` **twice** (bug at lines 1109-1122: duplicate call)

**Frontend:**
- Dashboard blocks rendering until ALL four queries (chart, status, position, fills) return
- The chart query at 4s polling interval hammers the uncached endpoint every 4 seconds

---

### Task 1: Add TTL cache to the chart endpoint

**Why first:** This is the highest-impact, lowest-risk change. The existing `_cache_get`/`_cache_put` pattern is already used for `/api/status` and `/api/position`. Adding it to `/api/dashboard/chart` means the 4-second polling interval will hit cache 100% of the time after the first cold load (default TTL is 8 seconds > 4 second poll interval).

**Files:**
- Modify: `src/quant/execution/webhook_server.py:50-51` (add cache dict)
- Modify: `src/quant/execution/webhook_server.py:621-918` (wrap endpoint)
- Test: `tests/test_webhook_dashboard_api.py`

**Step 1: Write the failing test**

In `tests/test_webhook_dashboard_api.py`, add a test method to the existing `WebhookDashboardApiTests` class:

```python
def test_chart_response_is_cached(self):
    """Second call within TTL should return cached response without re-reading data."""
    import time
    os.environ["DASHBOARD_API_CACHE_SEC"] = "10"

    r1 = api_dashboard_chart(symbol="SOL-USDT", hours=168, max_points=100)
    ts1 = r1.get("ts")
    self.assertTrue(r1["ok"])

    time.sleep(0.05)
    r2 = api_dashboard_chart(symbol="SOL-USDT", hours=168, max_points=100)
    ts2 = r2.get("ts")

    # Cached response should have the exact same timestamp
    self.assertEqual(ts1, ts2, "Second call should return cached response")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_webhook_dashboard_api.py::WebhookDashboardApiTests::test_chart_response_is_cached -v`
Expected: FAIL — `ts1 != ts2` because there is no caching yet.

**Step 3: Implement the cache**

In `src/quant/execution/webhook_server.py`, add the cache dict near lines 50-51:

```python
_STATUS_CACHE: Dict[str, Dict[str, Any]] = {}
_POSITION_CACHE: Dict[str, Dict[str, Any]] = {}
_CHART_CACHE: Dict[str, Dict[str, Any]] = {}
```

Then at the top of `api_dashboard_chart`, add cache check (right after the function signature, before `try:`):

```python
@app.get("/api/dashboard/chart")
def api_dashboard_chart(
    symbol: str = DEFAULT_SYMBOL,
    hours: int = 24 * 7,
    max_points: int = 3000,
) -> Dict[str, Any]:
    cache_key = f"{_normalize_symbol(symbol)}:{hours}:{max_points}"
    cached = _cache_get(_CHART_CACHE, cache_key)
    if cached is not None:
        return cached
    try:
        # ... existing code unchanged ...
```

Then at the end, just before the `return` statement (~line 844), cache it:

```python
        result = {
            "ok": True,
            "symbol": symbol,
            "bars": bars,
            # ... all existing fields ...
        }
        _cache_put(_CHART_CACHE, cache_key, result)
        return result
```

Also cache the error branch (line 884-918):

```python
    except Exception as e:
        err_result = {
            "ok": False,
            # ... existing error fields ...
        }
        _cache_put(_CHART_CACHE, cache_key, err_result)
        return err_result
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_webhook_dashboard_api.py::WebhookDashboardApiTests::test_chart_response_is_cached -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `python -m pytest tests/test_webhook_dashboard_api.py tests/test_dashboard_state.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add src/quant/execution/webhook_server.py tests/test_webhook_dashboard_api.py
git commit -m "perf: add TTL cache to /api/dashboard/chart endpoint"
```

---

### Task 2: Fix duplicate `load_closed_trades_from_postgres` call inside `build_trading_diary`

**Why now:** This is a pure bug — lines 1109-1122 in `dashboard_state.py` call `load_closed_trades_from_postgres` twice and `_read_trades_df()` twice. The second call on line 1119 overwrites the first result and discards the `df_source` tracking. Fixing this removes 2 unnecessary Postgres queries per chart request.

**Files:**
- Modify: `src/quant/execution/dashboard_state.py:1109-1126`
- Test: `tests/test_dashboard_state.py`

**Step 1: Write the failing test**

In `tests/test_dashboard_state.py`, add:

```python
def test_build_trading_diary_queries_postgres_once(self):
    """build_trading_diary should only call load_closed_trades_from_postgres once."""
    call_count = 0
    original_fn = ds.load_closed_trades_from_postgres

    def counting_wrapper(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_fn(*args, **kwargs)

    with patch.object(ds, "load_closed_trades_from_postgres", side_effect=counting_wrapper):
        ds.build_trading_diary(max_points=100)

    self.assertLessEqual(call_count, 1, f"Expected at most 1 Postgres call, got {call_count}")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_state.py::DashboardStateTests::test_build_trading_diary_queries_postgres_once -v`
Expected: FAIL — `call_count` will be 2.

**Step 3: Delete the duplicate block**

In `src/quant/execution/dashboard_state.py`, remove lines 1119-1125 (the second call to `load_closed_trades_from_postgres` and its fallback). The code should go from:

```python
def build_trading_diary(max_points: int = 500) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []

    df = load_closed_trades_from_postgres(
        venue="kucoin",
        symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
        max_points=max_points,
    )
    df_source = "postgres:closed_trades"
    if df.empty:
        df = _read_trades_df()
        df_source = "trades_parquet"

    df = load_closed_trades_from_postgres(
        venue="kucoin",
        symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
        max_points=max_points,
    )
    if df.empty:
        df = _read_trades_df()

    if not df.empty:
```

to:

```python
def build_trading_diary(max_points: int = 500) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []

    df = load_closed_trades_from_postgres(
        venue="kucoin",
        symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
        max_points=max_points,
    )
    df_source = "postgres:closed_trades"
    if df.empty:
        df = _read_trades_df()
        df_source = "trades_parquet"

    if not df.empty:
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dashboard_state.py::DashboardStateTests::test_build_trading_diary_queries_postgres_once -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/execution/dashboard_state.py tests/test_dashboard_state.py
git commit -m "fix: remove duplicate load_closed_trades_from_postgres in build_trading_diary"
```

---

### Task 3: Deduplicate renko parquet reads with a request-scoped shared DataFrame

**Why:** `_read_renko_df()` reads and parses the same parquet file 3 times per chart request (for bars, fibo, health). We add `_df` parameter overloads to avoid re-reading.

**Files:**
- Modify: `src/quant/execution/dashboard_state.py:210-270` (load_renko_bars, load_renko_health, build_fibo_levels)
- Modify: `src/quant/execution/webhook_server.py:628-739` (pass shared df)
- Test: `tests/test_dashboard_state.py`

**Step 1: Write the failing test**

In `tests/test_dashboard_state.py`, add:

```python
def test_renko_functions_accept_preloaded_df(self):
    """Renko functions should accept a pre-loaded DataFrame to avoid redundant reads."""
    df = pd.DataFrame({
        "ts": pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC"),
        "open": range(100, 110),
        "high": range(101, 111),
        "low": range(99, 109),
        "close": range(100, 110),
    })
    renko_path = self.tmp_path / "renko_latest.parquet"
    df.to_parquet(renko_path, index=False)
    os.environ["DASHBOARD_RENKO_PARQUET"] = str(renko_path)

    bars = ds.load_renko_bars(max_points=100, _df=df)
    self.assertEqual(len(bars), 10)

    health = ds.load_renko_health(_df=df)
    self.assertTrue(health["ok"])
    self.assertEqual(health["bars"], 10)

    fibo = ds.build_fibo_levels(max_points=100, _df=df)
    self.assertIn("long", fibo)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_state.py::DashboardStateTests::test_renko_functions_accept_preloaded_df -v`
Expected: FAIL — `TypeError: unexpected keyword argument '_df'`

**Step 3: Add `_df` parameter to the three functions**

In `src/quant/execution/dashboard_state.py`, modify each function signature to accept an optional pre-loaded DataFrame:

**`load_renko_bars` (line 210):**
```python
def load_renko_bars(max_points: int = 5000, _df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    df = _df if _df is not None else _read_renko_df()
    if df.empty:
        return []
    # ... rest unchanged
```

**`load_renko_health` (line 236):**
```python
def load_renko_health(_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    df = _df if _df is not None else _read_renko_df()
    if df.empty:
        # ... rest unchanged
```

**`build_fibo_levels` (line 260):**
```python
def build_fibo_levels(max_points: int = 5000, lookback: Optional[int] = None, _df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    lb = int(lookback or int(os.getenv("LIVE_IMBA_LOOKBACK", "250")))
    lb = max(2, lb)
    df = _df if _df is not None else _read_renko_df()
    if df.empty:
        # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dashboard_state.py::DashboardStateTests::test_renko_functions_accept_preloaded_df -v`
Expected: PASS

**Step 5: Wire up shared DataFrame in the chart endpoint**

In `src/quant/execution/webhook_server.py`, in `api_dashboard_chart`, read the renko df once and pass it:

```python
    try:
        from quant.execution.dashboard_state import _read_renko_df
        renko_df = _read_renko_df()
        bars = load_renko_bars(max_points=int(max(100, max_points)), _df=renko_df)
        # ... (lines 629-636 unchanged) ...
        # ... later ...
        fibo = build_fibo_levels(max_points=int(max(100, max_points)), _df=renko_df)
        renko_health = load_renko_health(_df=renko_df)
```

**Step 6: Run full test suite**

Run: `python -m pytest tests/test_webhook_dashboard_api.py tests/test_dashboard_state.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/quant/execution/dashboard_state.py src/quant/execution/webhook_server.py tests/test_dashboard_state.py
git commit -m "perf: read renko parquet once per chart request via shared DataFrame"
```

---

### Task 4: Deduplicate closed_trades Postgres queries with a shared DataFrame

**Why:** `load_closed_trades_from_postgres` is called 4 times per chart request (after Task 2 fix) with identical parameters: once by `load_trade_markers`, once by `load_trade_segments`, once by `build_trading_diary` (via `build_equity_curve`), and once by the direct `build_trading_diary` call. We add `_trades_df` parameter overloads.

**Files:**
- Modify: `src/quant/execution/dashboard_state.py:374-500,1102-1180,1364-1386` (load_trade_markers, load_trade_segments, build_trading_diary, build_equity_curve)
- Modify: `src/quant/execution/webhook_server.py:628-761` (pass shared trades df)
- Test: `tests/test_dashboard_state.py`

**Step 1: Write the failing test**

In `tests/test_dashboard_state.py`, add:

```python
def test_trade_functions_accept_preloaded_df(self):
    """Trade functions should accept a pre-loaded trades DataFrame."""
    df = pd.DataFrame({
        "trade_id": ["t1"],
        "venue": ["kucoin"],
        "symbol": ["SOL-USDT"],
        "entry_ts": [pd.Timestamp("2025-01-01", tz="UTC")],
        "exit_ts": [pd.Timestamp("2025-01-02", tz="UTC")],
        "side": ["long"],
        "qty": [1.0],
        "entry_price": [100.0],
        "exit_price": [105.0],
        "pnl_pct": [5.0],
        "exit_event": ["tp1"],
    })

    markers = ds.load_trade_markers(max_points=100, _trades_df=df)
    self.assertGreater(len(markers), 0)

    segments = ds.load_trade_segments(max_points=100, _trades_df=df)
    self.assertGreater(len(segments), 0)

    diary = ds.build_trading_diary(max_points=100, _trades_df=df)
    self.assertGreater(len(diary.get("entries", [])), 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_state.py::DashboardStateTests::test_trade_functions_accept_preloaded_df -v`
Expected: FAIL — `TypeError: unexpected keyword argument '_trades_df'`

**Step 3: Add `_trades_df` parameter to the four functions**

**`load_trade_markers` (line 374):**
```python
def load_trade_markers(max_points: int = 5000, _trades_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    if _trades_df is not None:
        df = _trades_df
    else:
        df = load_closed_trades_from_postgres(
            venue="kucoin",
            symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
            max_points=max_points,
        )
        if df.empty:
            df = _read_trades_df()
    if df.empty:
        return []
    # ... rest unchanged from line 384 onwards
```

**`load_trade_segments` (line 429):**
```python
def load_trade_segments(max_points: int = 2000, _trades_df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
    if _trades_df is not None:
        df = _trades_df
    else:
        df = load_closed_trades_from_postgres(
            venue="kucoin",
            symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
            max_points=max_points,
        )
        if df.empty:
            df = _read_trades_df()
    if df.empty:
        return []
    # ... rest unchanged from line 444 onwards
```

**`build_trading_diary` (line 1102):**
```python
def build_trading_diary(max_points: int = 500, _trades_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    out: List[Dict[str, Any]] = []

    if _trades_df is not None:
        df = _trades_df
        df_source = "preloaded"
    else:
        df = load_closed_trades_from_postgres(
            venue="kucoin",
            symbol=os.getenv("DASHBOARD_SYMBOL", "SOL-USDT"),
            max_points=max_points,
        )
        df_source = "postgres:closed_trades"
        if df.empty:
            df = _read_trades_df()
            df_source = "trades_parquet"

    if not df.empty:
    # ... rest unchanged
```

**`build_equity_curve` (line 1364):**
```python
def build_equity_curve(max_points: int = 500, _trades_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Cumulative equity curve from normalized diary entries."""
    diary = build_trading_diary(max_points=max_points, _trades_df=_trades_df)
    # ... rest unchanged
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dashboard_state.py::DashboardStateTests::test_trade_functions_accept_preloaded_df -v`
Expected: PASS

**Step 5: Wire up shared trades DataFrame in the chart endpoint**

In `src/quant/execution/webhook_server.py`, in `api_dashboard_chart`:

```python
    try:
        from quant.execution.dashboard_state import _read_renko_df, load_closed_trades_from_postgres
        renko_df = _read_renko_df()
        trades_df = load_closed_trades_from_postgres(
            venue="kucoin",
            symbol=symbol,
            max_points=int(max(100, max_points)),
        )
        bars = load_renko_bars(max_points=int(max(100, max_points)), _df=renko_df)
        markers = load_trade_markers(max_points=int(max(1000, max_points * 50)), _trades_df=trades_df)
        # ... load_live_fill_markers, load_active_levels unchanged ...
        segments = load_trade_segments(max_points=int(max(100, max_points)), _trades_df=trades_df)
        fibo = build_fibo_levels(max_points=int(max(100, max_points)), _df=renko_df)
        renko_health = load_renko_health(_df=renko_df)
        # ... regime calls unchanged ...
        diary = build_trading_diary(max_points=int(max(100, max_points)), _trades_df=trades_df)
        equity = build_equity_curve(max_points=int(max(100, max_points)), _trades_df=trades_df)
```

Then remove the separate `diary = build_trading_diary(...)` call on line 761 since we now call it once and reuse the result for both `diary` and `build_equity_curve`.

**Step 6: Run full test suite**

Run: `python -m pytest tests/test_webhook_dashboard_api.py tests/test_dashboard_state.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/quant/execution/dashboard_state.py src/quant/execution/webhook_server.py tests/test_dashboard_state.py
git commit -m "perf: query closed_trades once per chart request via shared DataFrame"
```

---

### Task 5: Deduplicate regime SQLite queries

**Why:** `build_regime_overlay` and `build_regime_scores` both call `RegimeStore().get_history()` with the same `symbol`, `start_ts`, `end_ts`, and `limit`. We query once and extract both results from the same data.

**Files:**
- Modify: `src/quant/execution/dashboard_state.py:1304-1405` (build_regime_overlay, build_regime_scores)
- Modify: `src/quant/execution/webhook_server.py:740-752`
- Test: `tests/test_dashboard_state.py`

**Step 1: Write the failing test**

In `tests/test_dashboard_state.py`, add:

```python
def test_regime_functions_accept_preloaded_rows(self):
    """Regime functions should accept pre-loaded rows to avoid duplicate queries."""
    rows = [
        {"ts": "2025-01-01T00:00:00+00:00", "gate_on": 1, "confidence": 0.8, "regime_state": "trend", "regime_score": 0.6},
        {"ts": "2025-01-01T01:00:00+00:00", "gate_on": 0, "confidence": 0.3, "regime_state": "range", "regime_score": -0.2},
    ]
    overlay = ds.build_regime_overlay(symbol="SOL-USDT", hours=168, _rows=rows)
    self.assertGreater(len(overlay["spans"]), 0)

    scores = ds.build_regime_scores(symbol="SOL-USDT", hours=168, _rows=rows)
    self.assertGreater(len(scores["scores"]), 0)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dashboard_state.py::DashboardStateTests::test_regime_functions_accept_preloaded_rows -v`
Expected: FAIL — `TypeError: unexpected keyword argument '_rows'`

**Step 3: Add `_rows` parameter to both functions**

**`build_regime_overlay` (line 1304):**
```python
def build_regime_overlay(symbol: str, hours: int = 24 * 14, _rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if _rows is not None:
        rows = _rows
    else:
        store = RegimeStore()
        end_ts = pd.Timestamp.now("UTC")
        start_ts = end_ts - pd.Timedelta(hours=int(max(1, hours)))
        rows = store.get_history(symbol=symbol, start_ts=start_ts.isoformat(), end_ts=end_ts.isoformat(), limit=20000)
    if not rows:
        return {"spans": [], "points": [], "latest": None}
    # ... rest unchanged from line 1312
```

**`build_regime_scores` (line 1388):**
```python
def build_regime_scores(symbol: str, hours: int = 24 * 14, _rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List]:
    if _rows is not None:
        rows = _rows
    else:
        store = RegimeStore()
        end_ts = pd.Timestamp.now("UTC")
        start_ts = end_ts - pd.Timedelta(hours=int(max(1, hours)))
        rows = store.get_history(symbol=symbol, start_ts=start_ts.isoformat(), end_ts=end_ts.isoformat(), limit=20000)
    if not rows:
        return {"scores": [], "forecast": []}
    # ... rest unchanged from line 1397
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_dashboard_state.py::DashboardStateTests::test_regime_functions_accept_preloaded_rows -v`
Expected: PASS

**Step 5: Wire up shared regime rows in the chart endpoint**

In `src/quant/execution/webhook_server.py`, in `api_dashboard_chart`:

```python
        from quant.regime import RegimeStore
        regime_store = RegimeStore()
        regime_end_ts = pd.Timestamp.now("UTC")
        regime_start_ts = regime_end_ts - pd.Timedelta(hours=int(max(1, hours)))
        regime_rows = regime_store.get_history(
            symbol=symbol,
            start_ts=regime_start_ts.isoformat(),
            end_ts=regime_end_ts.isoformat(),
            limit=20000,
        )
        regime = build_regime_overlay(symbol=symbol, hours=int(max(1, hours)), _rows=regime_rows)
        # ... existing regime processing unchanged ...
        regime_score_data = build_regime_scores(symbol=symbol, hours=int(max(1, hours)), _rows=regime_rows)
```

**Step 6: Run full test suite**

Run: `python -m pytest tests/test_webhook_dashboard_api.py tests/test_dashboard_state.py -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add src/quant/execution/dashboard_state.py src/quant/execution/webhook_server.py tests/test_dashboard_state.py
git commit -m "perf: query regime history once per chart request via shared rows"
```

---

### Task 6: Frontend — show skeleton immediately, don't block on chart query

**Why:** Currently `Dashboard.tsx` shows a `Loading…` indicator and renders empty content until ALL four queries resolve. The chart query is the slowest (~3-5s cold). Status, position, and fills are individually cached and return in <100ms. We should render the sidebar immediately with available data and show a skeleton chart placeholder while the chart data loads.

**Files:**
- Modify: `frontend/src/components/layout/Dashboard.tsx:15-73`
- No backend changes needed.

**Step 1: Modify Dashboard to render progressively**

Change the `isLoading` logic so only the chart area shows a loading state, not the entire dashboard:

```tsx
export default function Dashboard() {
  const chartQuery = useChartData("SOL-USDT", 168);
  const statusQuery = useStatus();
  const positionQuery = usePosition();
  const fillsQuery = useFills();

  const chartData = chartQuery.data;
  const status = statusQuery.data ?? null;
  const position = positionQuery.data ?? null;
  const fills = fillsQuery.data ?? null;

  const chartLevels = chartData?.levels ?? undefined;
  const regimeState = chartData?.regime_state ?? null;
  const gateOn = chartData?.gate_on ?? null;

  return (
    <div className="relative min-h-screen bg-zinc-950 text-zinc-100">
      <div className="grid grid-cols-1 gap-3 p-3 lg:grid-cols-[1fr_20rem]">
        <div className="flex flex-col gap-3">
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-1">
            {chartQuery.isLoading ? (
              <ChartSkeleton />
            ) : (
              <PriceChart
                bars={chartData?.bars ?? []}
                markers={chartData?.markers}
                segments={chartData?.segments}
                levels={chartData?.levels}
                ttpTrailPct={chartData?.ttp_trail_pct}
                fibo={chartData?.fibo}
                livePrice={status?.ticker?.last ?? status?.ticker?.mid}
              />
            )}
          </div>
          {chartQuery.isLoading ? (
            <EquitySkeleton />
          ) : (
            <EquityCurve
              components={chartData?.equity_components}
              totalEquity={chartData?.equity_total}
            />
          )}
        </div>

        <div className="order-first lg:order-last">
          <Sidebar
            status={status}
            position={position}
            fills={fills}
            chartLevels={chartLevels}
            regimeState={regimeState}
            gateOn={gateOn}
            krakenMetrics={chartData?.kraken_metrics}
          />
        </div>
      </div>
    </div>
  );
}
```

Add the skeleton components in the same file:

```tsx
function ChartSkeleton() {
  return (
    <div className="flex h-[500px] items-center justify-center">
      <div className="flex items-center gap-2 text-sm text-zinc-500">
        <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
        Loading chart…
      </div>
    </div>
  );
}

function EquitySkeleton() {
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <div className="flex h-[200px] items-center justify-center">
        <div className="flex items-center gap-2 text-sm text-zinc-500">
          <span className="h-2 w-2 animate-pulse rounded-full bg-amber-400" />
          Loading equity…
        </div>
      </div>
    </div>
  );
}
```

Remove the old `LoadingIndicator` component and the `isLoading` gate that blocked everything.

**Step 2: Verify the frontend builds**

Run: `cd frontend && npm run build`
Expected: Build succeeds with no errors.

**Step 3: Commit**

```bash
git add frontend/src/components/layout/Dashboard.tsx
git commit -m "perf: render sidebar immediately, show skeleton while chart loads"
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Cold first load (chart endpoint) | ~3-5s | ~1-1.5s (3x fewer I/O ops) |
| Subsequent polls (every 4s) | ~3-5s each | <10ms (cached, TTL=8s) |
| Renko parquet reads per request | 3 | 1 |
| `closed_trades` Postgres queries per request | 6 | 1 |
| Regime SQLite queries per request | 2 | 1 |
| `build_trading_diary` calls per request | 2 (+ 2 internal dupes) | 1 |
| Time to first sidebar render | ~3-5s (blocked) | <500ms (independent) |

## Task Summary

| Task | What | Risk | Impact |
|------|------|------|--------|
| 1 | TTL cache on chart endpoint | Very low (proven pattern) | Eliminates 100% of repeat polls |
| 2 | Fix duplicate query in build_trading_diary | Very low (pure bug fix) | -2 Postgres queries |
| 3 | Shared renko DataFrame | Low | -2 parquet reads |
| 4 | Shared trades DataFrame | Low | -3 Postgres queries |
| 5 | Shared regime rows | Low | -1 SQLite query |
| 6 | Frontend progressive rendering | Low (UI only) | Instant sidebar, skeleton chart |
