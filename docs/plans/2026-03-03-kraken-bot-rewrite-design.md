# Kraken Bot Rewrite — Full Backtest Parity

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `kraken_bot.py` so the Kraken Futures bot trades with the exact same logic as the combined backtest: IMBA signals trigger entries, gate selects the exit engine (flip_engine for gate ON, TP2 for gate OFF), gate transitions force regime_exit → flat.

**Architecture:** The bot polls two quant-service endpoints every 10s: `/api/gate/solusd` (existing) and `/api/signals/latest/solusd` (new). It maintains a state machine with position tracking, entry price, best favorable price, and mode. Orders are executed on Kraken Futures via the existing `KrakenFuturesClient`. Renko bar data for swing SL is fetched from a new `/api/renko/latest/solusd` endpoint.

**Tech Stack:** Python 3.9+, existing `KrakenFuturesClient`, pandas, numpy. No new dependencies.

---

## Backtest Logic Reference

### Gate ON → Flip Engine (`flip_engine.py:274-416`)

```
FLAT + signal → enter in signal direction, mode=TTP
IN_POSITION + TTP trail hit → flip to opposite side, mode=WAIT
IN_POSITION + opposite IMBA → flip immediately, mode=TTP
WAIT + swing SL hit → flat
WAIT + same/any IMBA → re-arm TTP
gate turns OFF → regime_exit → flat
```

- TTP trail: `best_fav * (1 ∓ ttp_trail_pct)` depending on side
- Swing SL: `min(low, lookback)` for long, `max(high, lookback)` for short
- Defaults: ttp_trail_pct=0.012, min_sl=0.015, max_sl=0.030, swing_lookback=250

### Gate OFF → TP2 Engine (`renko_runner_tp2.py:252-440`)

```
FLAT + signal → enter in signal direction
TP1 hit (partial 50%) → scale out, arm BE
TP2 hit → full exit → flat
BE hit → flat (if TP1 was done)
Swing SL hit → flat
Opposite IMBA → close + optional flip
gate turns ON → regime_exit → flat
```

- TP1: entry * (1 ± tp1_pct), partial exit tp1_frac
- TP2: entry * (1 ± tp2_pct), full exit
- BE: entry price (armed after TP1)
- Swing SL: clamped between min_sl_pct and max_sl_pct
- Defaults: tp1_pct=0.015, tp2_pct=0.030, tp1_frac=0.5, min_sl=0.030, max_sl=0.080, swing_lookback=180

---

### Task 1: Add `/api/signals/latest/solusd` endpoint to webhook_server.py

**Files:**
- Modify: `src/quant/execution/webhook_server.py`
- Test: `tests/test_webhook_dashboard_api.py` (extend)

**Step 1: Write the failing test**

```python
# Add to tests/test_webhook_dashboard_api.py
def test_signals_latest_endpoint(self):
    """GET /api/signals/latest/solusd returns latest IMBA signal."""
    from fastapi.testclient import TestClient
    from quant.execution.webhook_server import app
    client = TestClient(app)
    resp = client.get("/api/signals/latest/solusd")
    self.assertEqual(resp.status_code, 200)
    body = resp.json()
    self.assertIn("ok", body)
    self.assertIn("signal", body)
    self.assertIn("ts", body)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_webhook_dashboard_api.py::WebhookDashboardApiTests::test_signals_latest_endpoint -v`
Expected: FAIL (404)

**Step 3: Implement endpoint**

Reuse `_latest_signal()` logic from `live_executor.py` (or import it). Add to `webhook_server.py`:

```python
@app.get("/api/signals/latest/solusd")
def api_signals_latest_solusd() -> Dict[str, Any]:
    try:
        root = _signals_root()
        sig = _latest_signal_from_jsonl(root, "SOL-USDT")
        if sig is None:
            return {"ok": True, "ts": _now_utc_iso(), "signal": 0, "source": "no_signal"}
        return {
            "ok": True,
            "ts": str(sig["ts"]),
            "signal": int(sig["signal"]),
            "source": "jsonl",
        }
    except Exception as e:
        return {"ok": False, "ts": _now_utc_iso(), "signal": 0, "error": str(e)}
```

The helper `_latest_signal_from_jsonl(root, symbol)` mirrors the logic from `live_executor.py:_latest_signal()`: reads newest JSONL file, scans lines in reverse, returns first non-zero signal with ts.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_webhook_dashboard_api.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/quant/execution/webhook_server.py tests/test_webhook_dashboard_api.py
git commit -m "feat(api): add /api/signals/latest/solusd endpoint"
```

---

### Task 2: Add `/api/renko/latest/solusd` endpoint for swing SL data

**Files:**
- Modify: `src/quant/execution/webhook_server.py`
- Test: `tests/test_webhook_dashboard_api.py` (extend)

**Step 1: Write the failing test**

```python
def test_renko_latest_endpoint(self):
    resp = client.get("/api/renko/latest/solusd?lookback=180")
    self.assertEqual(resp.status_code, 200)
    body = resp.json()
    self.assertIn("ok", body)
    self.assertIn("swing_low", body)
    self.assertIn("swing_high", body)
```

**Step 2: Implement endpoint**

Reads `renko_latest.parquet`, computes rolling min(low) and max(high) over lookback, returns latest values:

```python
@app.get("/api/renko/latest/solusd")
def api_renko_latest_solusd(lookback: int = 250) -> Dict[str, Any]:
    try:
        path = Path(os.getenv("DASHBOARD_RENKO_PARQUET", "data/live/renko_latest.parquet"))
        if not path.exists():
            return {"ok": False, "error": "no_renko_data"}
        df = pd.read_parquet(path)
        if df.empty:
            return {"ok": False, "error": "empty_renko_data"}
        lb = min(max(lookback, 1), 500)
        swing_low = float(df["low"].rolling(lb, min_periods=1).min().iloc[-1])
        swing_high = float(df["high"].rolling(lb, min_periods=1).max().iloc[-1])
        return {
            "ok": True,
            "ts": str(df["ts"].iloc[-1]),
            "swing_low": swing_low,
            "swing_high": swing_high,
            "last_close": float(df["close"].iloc[-1]),
            "n_bars": len(df),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}
```

**Step 3: Run tests, commit**

```bash
git add src/quant/execution/webhook_server.py tests/test_webhook_dashboard_api.py
git commit -m "feat(api): add /api/renko/latest/solusd for swing SL data"
```

---

### Task 3: Rewrite kraken_bot.py — state machine + both engines

**Files:**
- Modify: `src/quant/execution/kraken_bot.py` (full rewrite of run_once and supporting code)
- Create: `tests/test_kraken_bot_statemachine.py`

This is the core task. The bot maintains persistent state across poll cycles.

**Step 1: Define the state model**

```python
@dataclass
class BotState:
    pos_side: int = 0          # +1 long, -1 short, 0 flat
    entry_px: float = 0.0
    best_fav: float = 0.0
    size_full: float = 0.0     # full position size in SOL
    size_rem: float = 0.0      # remaining after TP1 partial
    mode: str = "FLAT"         # FLAT, FLIP_TTP, FLIP_WAIT, TP2_OPEN, TP2_BE
    engine: str = "none"       # "flip" or "tp2" or "none"
    gate_on: int = 0
    last_signal_ts: str = ""   # dedup: only act on new signals
    tp1_done: bool = False
```

**Step 2: Implement `run_once()` as state machine**

```python
def run_once(client, state, gate, signal, mark_price, swing_low, swing_high, params):
    """
    Pure function: takes current state + market data, returns (new_state, actions).
    Actions are a list of dicts: [{"action": "enter_long", "size": 1.0}, ...]
    Caller executes actions on Kraken.
    """
```

The state machine logic (pseudocode, matching backtest exactly):

```
# 1. Gate transition check
if gate changed since last cycle:
    if pos_side != 0:
        actions.append(close_all → "regime_exit")
        state = FLAT
    state.engine = "flip" if gate_on else "tp2"

# 2. If FLAT: check for new signal
if state.mode == "FLAT" and signal is new and signal != 0:
    actions.append(enter in signal direction)
    state = FLIP_TTP if engine=="flip" else TP2_OPEN

# 3. FLIP_TTP mode
if state.mode == "FLIP_TTP":
    update best_fav
    ttp_stop = best_fav * (1 - trail) if long else best_fav * (1 + trail)
    if mark hits ttp_stop:
        actions.append(close + enter opposite)
        state = FLIP_WAIT, side flipped
    if new opposite signal:
        actions.append(close + enter opposite)
        state = FLIP_TTP, side flipped

# 4. FLIP_WAIT mode
if state.mode == "FLIP_WAIT":
    swing_sl = compute from swing_low/swing_high clamped by min/max sl
    if mark hits swing_sl:
        actions.append(close_all)
        state = FLAT
    if new signal (any direction):
        state.mode = FLIP_TTP  (re-arm TTP)

# 5. TP2_OPEN mode
if state.mode == "TP2_OPEN":
    tp1_px = entry * (1 ± tp1_pct)
    tp2_px = entry * (1 ± tp2_pct)
    swing_sl = compute from swing_low/swing_high clamped by min/max sl
    if mark hits tp2:
        actions.append(close_all)
        state = FLAT
    elif mark hits tp1 and not tp1_done:
        actions.append(close partial tp1_frac)
        state = TP2_BE, tp1_done=True
    elif mark hits swing_sl:
        actions.append(close_all)
        state = FLAT
    if new opposite signal:
        actions.append(close_all)
        if flip_on_opposite:
            actions.append(enter opposite)
            state = TP2_OPEN (new trade)
        else:
            state = FLAT

# 6. TP2_BE mode
if state.mode == "TP2_BE":
    be_px = entry_px
    tp2_px = entry * (1 ± tp2_pct)
    if mark hits be:
        actions.append(close_all)
        state = FLAT
    elif mark hits tp2:
        actions.append(close remainder)
        state = FLAT
```

**Step 3: Implement action executor**

Separate function that takes the action list and calls `client.place_market()` / `client.close_position()`:

```python
def execute_actions(client, actions, dry_run):
    for a in actions:
        if dry_run:
            log.info("DRY_RUN: would %s", a)
            continue
        if a["action"] == "close_all":
            client.close_position()
        elif a["action"] == "close_partial":
            client.place_market(a["close_side"], size=a["size"], reduce_only=True)
        elif a["action"].startswith("enter_"):
            client.place_market(a["side"], size=a["size"])
```

**Step 4: Implement polling loop**

```python
def main():
    state = load_state() or BotState()  # persist across restarts
    while True:
        gate = fetch_gate(gate_url)
        signal = fetch_signal(signal_url)
        mark = client.get_mark_price()
        renko = fetch_renko(renko_url, lookback)
        
        new_state, actions = run_once(client_info, state, gate, signal, mark,
                                       renko.swing_low, renko.swing_high, params)
        execute_actions(client, actions, dry_run)
        save_state(new_state)
        publish_metrics(new_state, actions)
        state = new_state
        sleep(poll_sec)
```

**Step 5: Write tests**

Test the state machine as a **pure function** (no Kraken API calls):

```python
class TestBotStateMachine(unittest.TestCase):
    def test_flat_signal_enters_flip_ttp(self):
        state = BotState(mode="FLAT", engine="flip", gate_on=1)
        new_state, actions = run_once_logic(state, gate_on=1, signal=1, mark=100.0,
                                             swing_low=95.0, swing_high=105.0, params=default_params())
        self.assertEqual(new_state.mode, "FLIP_TTP")
        self.assertEqual(new_state.pos_side, 1)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]["action"], "enter_long")

    def test_flip_ttp_trail_hit_flips(self):
        state = BotState(mode="FLIP_TTP", engine="flip", pos_side=1, entry_px=100.0, best_fav=105.0)
        # TTP trail hit: 105 * (1 - 0.012) = 103.74. Mark at 103.0 → triggers.
        new_state, actions = run_once_logic(state, gate_on=1, signal=0, mark=103.0, ...)
        self.assertEqual(new_state.mode, "FLIP_WAIT")
        self.assertEqual(new_state.pos_side, -1)  # flipped

    def test_gate_transition_regime_exit(self):
        state = BotState(mode="FLIP_TTP", engine="flip", pos_side=1, gate_on=1)
        new_state, actions = run_once_logic(state, gate_on=0, signal=0, mark=100.0, ...)
        self.assertEqual(new_state.mode, "FLAT")
        self.assertAny(a["action"] == "close_all" for a in actions)

    def test_tp2_tp1_partial_exit(self):
        state = BotState(mode="TP2_OPEN", engine="tp2", pos_side=1, entry_px=100.0, size_full=1.0, size_rem=1.0)
        # TP1 at 101.5 (1.5%). Mark at 102.0 → triggers.
        new_state, actions = run_once_logic(state, gate_on=0, signal=0, mark=102.0, ...)
        self.assertEqual(new_state.mode, "TP2_BE")
        self.assertTrue(new_state.tp1_done)
        self.assertAlmostEqual(new_state.size_rem, 0.5)

    # ... more tests for SL, BE, opposite signal flip, etc.
```

**Step 6: Run tests, commit**

```bash
git add src/quant/execution/kraken_bot.py tests/test_kraken_bot_statemachine.py
git commit -m "feat(kraken): rewrite bot with full backtest-parity state machine"
```

---

### Task 4: State persistence + metrics

**Files:**
- Modify: `src/quant/execution/kraken_bot.py`

The bot state must survive restarts (Railway redeploys). Save to JSON on the volume mount:

```python
def save_state(state: BotState, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state)), encoding="utf-8")

def load_state(path: Path) -> Optional[BotState]:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return BotState(**d)
```

Path: `/data/live/kraken/bot_state.json` (on Railway volume).

Metrics publishing stays similar to current: write to `metrics.json` + append to `equity.csv`.

**Commit:**

```bash
git commit -m "feat(kraken): add state persistence and metrics for bot state machine"
```

---

### Task 5: Env vars + deploy

**Kraken service env vars:**

```
KRAKEN_SIGNAL_URL=https://quant-production-5533.up.railway.app/api/signals/latest/solusd
KRAKEN_RENKO_URL=https://quant-production-5533.up.railway.app/api/renko/latest/solusd
KRAKEN_GATE_URL=https://quant-production-5533.up.railway.app/api/gate/solusd
KRAKEN_TARGET_SIZE=1
KRAKEN_DRY_RUN=1
KRAKEN_TRADING_ENABLED=1

# Flip engine (gate ON)
KRAKEN_TTP_TRAIL_PCT=0.012
KRAKEN_FLIP_MIN_SL_PCT=0.015
KRAKEN_FLIP_MAX_SL_PCT=0.030
KRAKEN_FLIP_SWING_LOOKBACK=250

# TP2 engine (gate OFF)
KRAKEN_TP1_PCT=0.015
KRAKEN_TP2_PCT=0.030
KRAKEN_TP1_FRAC=0.5
KRAKEN_TP2_MIN_SL_PCT=0.030
KRAKEN_TP2_MAX_SL_PCT=0.080
KRAKEN_TP2_SWING_LOOKBACK=180
KRAKEN_FLIP_ON_OPPOSITE=1
```

**Steps:**
1. Push to main
2. Set env vars on Kraken Railway service
3. Keep `KRAKEN_DRY_RUN=1` initially → watch logs
4. Verify state machine transitions in logs
5. Set `KRAKEN_DRY_RUN=0` when satisfied

---

## Differences from Backtest (unavoidable)

| Aspect | Backtest | Live Bot |
|--------|----------|----------|
| Price check | Bar OHLC (high/low) | Mark price every 10s |
| SL/TP precision | Exact bar high/low | May miss intra-bar spikes |
| Swing lookback | Previous bar's rolling min/max | Latest renko snapshot (refreshed every 5 min) |
| Signal latency | Same bar | Webhook → JSONL → poll (seconds) |
| Fills | Instant at exact price | Market order, may slip |

These are inherent to live trading and cannot be eliminated. The 10s polling on renko bars is close enough.
