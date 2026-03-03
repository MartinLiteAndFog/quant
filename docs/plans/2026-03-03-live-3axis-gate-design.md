# Live 3-Axis PC Gate for Kraken Bot Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the static backtesting gate CSV with a live gate that computes gate_on/gate_off from the real-time 3-axis state space (X=Drift, Y=Elasticity, Z=Instability) already being refreshed every 5 minutes by the dashboard.

**Architecture:** Add a new function `get_live_gate_from_statespace()` to `gate_provider.py` that reads the live state space parquet (already maintained by `dashboard_statespace.py` → `refresh_state_space_cache()`), applies the 3-axis gate thresholds (same logic as `make_pc_3axis_gate_v2.py`), and returns gate_on/gate_off. The existing `get_live_gate_state()` gains a new priority: state_space_parquet → predictions_parquet → gate_csv → default_off. No changes to the Kraken bot — it already consumes `/api/gate/solusd`.

**Tech Stack:** Python 3.9+, numpy, pandas (all existing). No new dependencies.

---

## Current Data Flow (what exists)

```
KuCoin 1m candles
  → renko_cache_updater (every 5min, background thread in webhook_server)
  → renko_latest.parquet
  → refresh_state_space_cache() (every 5min, background thread in webhook_server)
  → state_space_latest.parquet (has: ts, X_raw, Y_res, Z_res, conf_x, conf_y, conf_z)
```

The state space parquet contains the same 3 axes the gate needs:
- **X_raw** = Drift (same concept as `drift_eff` in `make_pc_3axis_gate_v2.py`)
- **Y_res** = Elasticity (same concept as `elas`)
- **Z_res** = Instability (same concept as `instab`)

Values are already normalized to [-1, +1] range.

## Target Data Flow (what we build)

```
state_space_latest.parquet (refreshed every 5min)
  → get_live_gate_from_statespace() (new function in gate_provider.py)
  → applies 3-axis thresholds to latest X_raw, Y_res, Z_res values
  → returns gate_on / gate_off
  → served at /api/gate/solusd
  → consumed by Kraken bot via KRAKEN_GATE_URL
```

## Key Design Decision: Threshold Mapping

The original gate in `make_pc_3axis_gate_v2.py` uses rank-percentile thresholds on the full historical dataset (quantiles computed on training data). The live state space values are already [-1, +1] normalized signals. We need fixed thresholds that work on this normalized range.

**Approach:** Use configurable thresholds via env vars with sensible defaults derived from the backtesting quantiles. The gate fires "on" when 2-of-3 conditions are met:
- `|X_raw| <= threshold_drift` (low absolute drift → not trending strongly)
- `|Y_res| >= threshold_elasticity` (high elasticity → mean-reverting)
- `Z_res <= threshold_instability` (low instability → stable regime)

Defaults derived from examining the state space distributions in backtesting.

---

### Task 1: Add `get_live_gate_from_statespace()` to gate_provider.py

**Files:**
- Modify: `src/quant/execution/gate_provider.py`
- Test: `tests/test_gate_provider_live.py` (create)

**Step 1: Write the failing test**

```python
# tests/test_gate_provider_live.py
import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quant.execution.gate_provider import get_live_gate_from_statespace


class TestLiveGateFromStatespace(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ss_path = Path(self.tmp.name) / "state_space_latest.parquet"
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = str(self.ss_path)

    def tearDown(self):
        self.tmp.cleanup()
        os.environ.pop("DASHBOARD_STATESPACE_PARQUET", None)
        for k in list(os.environ):
            if k.startswith("LIVE_GATE_"):
                del os.environ[k]

    def test_returns_gate_on_when_conditions_met(self):
        # Low drift, high elasticity, low instability → gate ON
        df = pd.DataFrame({
            "ts": pd.date_range("2026-03-03T12:00:00Z", periods=5, freq="5min", tz="UTC"),
            "X_raw": [0.05, 0.03, 0.02, 0.01, 0.0],      # low drift
            "Y_res": [0.6, 0.7, 0.8, 0.7, 0.75],           # high elasticity
            "Z_res": [-0.5, -0.4, -0.3, -0.35, -0.3],      # low instability
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_from_statespace()
        self.assertEqual(result["gate_on"], 1)
        self.assertEqual(result["source"], "statespace_live")

    def test_returns_gate_off_when_trending(self):
        # High drift → gate OFF
        df = pd.DataFrame({
            "ts": pd.date_range("2026-03-03T12:00:00Z", periods=5, freq="5min", tz="UTC"),
            "X_raw": [0.8, 0.85, 0.9, 0.88, 0.92],         # high drift
            "Y_res": [0.1, 0.05, 0.0, -0.1, -0.05],        # low elasticity
            "Z_res": [0.6, 0.7, 0.8, 0.75, 0.85],           # high instability
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_from_statespace()
        self.assertEqual(result["gate_on"], 0)

    def test_returns_none_when_file_missing(self):
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = "/nonexistent/path.parquet"
        result = get_live_gate_from_statespace()
        self.assertIsNone(result)

    def test_custom_thresholds_via_env(self):
        os.environ["LIVE_GATE_DRIFT_THRESH"] = "0.01"  # very tight
        df = pd.DataFrame({
            "ts": pd.date_range("2026-03-03T12:00:00Z", periods=3, freq="5min", tz="UTC"),
            "X_raw": [0.05, 0.04, 0.03],   # above tight threshold
            "Y_res": [0.8, 0.8, 0.8],
            "Z_res": [-0.5, -0.5, -0.5],
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_from_statespace()
        self.assertEqual(result["gate_on"], 0)  # drift too high for tight threshold
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gate_provider_live.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_live_gate_from_statespace'`

**Step 3: Write minimal implementation**

Add to `src/quant/execution/gate_provider.py`:

```python
def get_live_gate_from_statespace() -> Optional[Dict[str, Any]]:
    """
    Compute gate_on/gate_off from the live state space parquet.
    Returns None if state space data is unavailable or too old.
    
    3-axis gate logic (2-of-3):
      g1: |X_raw| <= drift_thresh     (not trending strongly)
      g2: Y_res >= elasticity_thresh  (mean-reverting)
      g3: Z_res <= instability_thresh (stable regime)
      gate_on = (g1 + g2 + g3) >= 2
    """
    ss_path = Path(os.getenv(
        "DASHBOARD_STATESPACE_PARQUET",
        _live_default("live/state_space_latest.parquet"),
    ))
    if not ss_path.exists():
        return None

    try:
        df = pd.read_parquet(ss_path)
    except Exception:
        return None

    need = {"ts", "X_raw", "Y_res", "Z_res"}
    if not need.issubset(set(df.columns)) or df.empty:
        return None

    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "X_raw", "Y_res", "Z_res"]).sort_values("ts")
    if df.empty:
        return None

    # Staleness check: reject data older than threshold
    max_age_sec = float(os.getenv("LIVE_GATE_MAX_AGE_SEC", "1800"))  # 30min
    last_ts = df["ts"].iloc[-1]
    age = (pd.Timestamp.now("UTC") - last_ts).total_seconds()
    if age > max_age_sec:
        return None

    last = df.iloc[-1]
    x = float(last["X_raw"])
    y = float(last["Y_res"])
    z = float(last["Z_res"])

    drift_thresh = float(os.getenv("LIVE_GATE_DRIFT_THRESH", "0.3"))
    elasticity_thresh = float(os.getenv("LIVE_GATE_ELASTICITY_THRESH", "0.0"))
    instability_thresh = float(os.getenv("LIVE_GATE_INSTABILITY_THRESH", "0.3"))

    g1 = int(abs(x) <= drift_thresh)
    g2 = int(y >= elasticity_thresh)
    g3 = int(z <= instability_thresh)
    gate_on = int((g1 + g2 + g3) >= 2)

    ts_str = pd.Timestamp(last_ts).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "ts": ts_str,
        "gate_on": gate_on,
        "gate_off": int(1 - gate_on),
        "source": "statespace_live",
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
        "g1_drift": g1,
        "g2_elasticity": g2,
        "g3_instability": g3,
        "age_sec": round(age, 1),
    }
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gate_provider_live.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/execution/gate_provider.py tests/test_gate_provider_live.py
git commit -m "feat(gate): add live 3-axis gate from state space parquet"
```

---

### Task 2: Wire statespace gate into `get_live_gate_state()` priority chain

**Files:**
- Modify: `src/quant/execution/gate_provider.py:70-94` (the `get_live_gate_state` function)
- Test: `tests/test_gate_provider_live.py` (extend)

**Step 1: Write the failing test**

```python
# Add to tests/test_gate_provider_live.py
from quant.execution.gate_provider import get_live_gate_state

class TestGateStatePriorityChain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ss_path = Path(self.tmp.name) / "state_space_latest.parquet"
        os.environ["DASHBOARD_STATESPACE_PARQUET"] = str(self.ss_path)
        os.environ["PC_PREDICTIONS_PARQUET"] = "/nonexistent/predictions.parquet"
        os.environ["PC_GATE_CSV"] = "/nonexistent/gate.csv"

    def tearDown(self):
        self.tmp.cleanup()
        for k in ["DASHBOARD_STATESPACE_PARQUET", "PC_PREDICTIONS_PARQUET", "PC_GATE_CSV"]:
            os.environ.pop(k, None)

    def test_prefers_statespace_over_csv_fallback(self):
        df = pd.DataFrame({
            "ts": pd.date_range("2026-03-03T12:00:00Z", periods=3, freq="5min", tz="UTC"),
            "X_raw": [0.0, 0.0, 0.0],
            "Y_res": [0.5, 0.5, 0.5],
            "Z_res": [-0.5, -0.5, -0.5],
        })
        df.to_parquet(self.ss_path, index=False)
        result = get_live_gate_state()
        self.assertEqual(result["source"], "statespace_live")

    def test_falls_back_to_default_when_no_data(self):
        result = get_live_gate_state()
        self.assertEqual(result["source"], "default_off")
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gate_provider_live.py::TestGateStatePriorityChain -v`
Expected: FAIL (source won't be "statespace_live" because get_live_gate_state doesn't call the new function yet)

**Step 3: Modify `get_live_gate_state()` to try statespace first**

In `src/quant/execution/gate_provider.py`, modify the top of `get_live_gate_state()`:

```python
def get_live_gate_state() -> Dict[str, Any]:
    # Priority 1: live state space (refreshed every 5 min by dashboard)
    ss_gate = get_live_gate_from_statespace()
    if ss_gate is not None:
        return ss_gate

    # Priority 2: predictions parquet (full PC engine output)
    pred_path = Path(os.getenv("PC_PREDICTIONS_PARQUET", ...))
    # ... rest of existing code unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gate_provider_live.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/execution/gate_provider.py tests/test_gate_provider_live.py
git commit -m "feat(gate): prefer live statespace gate over static CSV fallback"
```

---

### Task 3: Add axis values and gate details to `/api/gate/solusd` response

**Files:**
- Modify: `src/quant/execution/webhook_server.py:621-640` (the `api_gate_solusd` endpoint)
- Test: `tests/test_webhook_dashboard_api.py` (extend existing gate test)

**Step 1: Extend the endpoint to pass through axis values**

The new `get_live_gate_from_statespace()` already returns x/y/z and individual gate components. The webhook endpoint just needs to forward them.

```python
@app.get("/api/gate/solusd")
def api_gate_solusd() -> Dict[str, Any]:
    try:
        out = get_live_gate_state()
        return {
            "ok": True,
            "ts": out.get("ts"),
            "gate_on": int(out.get("gate_on", 0) or 0),
            "gate_off": int(out.get("gate_off", 1) or 1),
            "source": out.get("source"),
            "x": out.get("x"),
            "y": out.get("y"),
            "z": out.get("z"),
            "g1_drift": out.get("g1_drift"),
            "g2_elasticity": out.get("g2_elasticity"),
            "g3_instability": out.get("g3_instability"),
            "age_sec": out.get("age_sec"),
        }
    except Exception as e:
        # ... existing error handling ...
```

**Step 2: Run tests**

Run: `python -m pytest tests/test_webhook_dashboard_api.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/quant/execution/webhook_server.py tests/test_webhook_dashboard_api.py
git commit -m "feat(api): expose 3-axis values in gate endpoint response"
```

---

### Task 4: Deploy and configure env vars

**No code changes.** Set these env vars in Railway:

**Quant (webhook) service:**
- `LIVE_GATE_DRIFT_THRESH=0.3` (tune later based on live observations)
- `LIVE_GATE_ELASTICITY_THRESH=0.0`
- `LIVE_GATE_INSTABILITY_THRESH=0.3`
- `LIVE_GATE_MAX_AGE_SEC=1800`
- `ENABLE_DASHBOARD_RENKO_UPDATER=1` (ensure renko data flows)

**Kraken worker service:**
- `KRAKEN_GATE_URL=https://quant-production-5533.up.railway.app/api/gate/solusd`
- `KRAKEN_TARGET_SIZE=1` (1 SOL, safe starting size)
- `KRAKEN_LEVERAGE=5`

**Step 1: Merge to main and push**
**Step 2: Set env vars in Railway dashboard**
**Step 3: Verify gate endpoint returns `source: statespace_live`**

```bash
curl -s https://quant-production-5533.up.railway.app/api/gate/solusd | python3 -m json.tool
```

Expected: `"source": "statespace_live"` with real x/y/z values and a recent timestamp.

---

## Threshold Tuning Guide

The 3 thresholds control when the gate is ON (allowing countertrend trading):

| Threshold | Default | Meaning | Tighter = more selective |
|-----------|---------|---------|-------------------------|
| `LIVE_GATE_DRIFT_THRESH` | 0.3 | Max abs(drift) to be "not trending" | Lower value |
| `LIVE_GATE_ELASTICITY_THRESH` | 0.0 | Min elasticity for "mean-reverting" | Higher value |
| `LIVE_GATE_INSTABILITY_THRESH` | 0.3 | Max instability for "stable" | Lower value |

Start with defaults. Monitor the gate endpoint's x/y/z values over a few days to understand the distributions, then tighten thresholds to match backtesting performance.

## Exact Backtest Parity (default)

**GATE_PREFER_CSV=1** (default): Gate CSV is tried first. Uses `gate_base_2of3` from the same source as the backtest (`make_pc_3axis_gate_v2.py`). Kraken bot behaves exactly like the backtest.

**GATE_PREFER_CSV=0**: Use live state space for fresh data. Thresholds may differ slightly from backtest.

## Rollback

If the live gate produces bad signals: set `GATE_PREFER_CSV=1` (default) and ensure `PC_GATE_CSV` points to the gate CSV. The gate provider uses `gate_base_2of3` for exact backtest parity.
