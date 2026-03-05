# Live Trading Walkthrough (Beginner-Friendly)

This document explains what the system currently does, from IMBA signal generation to trade execution in both services.

It also explains why your observed issue can happen:

- no new IMBA signal when you expected one
- manual long position, then opposite signal, but only close (no reverse)

---

## 1) The Two Services (Important)

You currently run two trade executors:

1. `quant` service (KuCoin executor path)
2. `kraken` service (Kraken executor path)

Both read signal/gate/renko data from the `quant` API endpoints.

So one bad signal state or one state mismatch can affect both.

---

## 2) End-to-End Data Flow (Linear)

### Step 1: Build live candles and Renko bars

File:

- `src/quant/execution/live_signal_worker.py`

What happens:

- Pulls recent 1m candles from KuCoin
- Converts candles into Renko bars

Code pointers:

- `_fetch_recent_1m_ohlcv(...)`
- `renko_from_close(...)`

Quick inspect:

```bash
grep -n "_fetch_recent_1m_ohlcv\\|renko_from_close\\|run_once(" src/quant/execution/live_signal_worker.py
sed -n '291,360p' src/quant/execution/live_signal_worker.py
```

---

### Step 2: Compute IMBA signals

File:

- `src/quant/execution/live_signal_worker.py`
- strategy code in `src/quant/strategies/imba.py`

What happens:

- IMBA signals are computed from Renko OHLC
- Signals are "sticky" (they do not constantly alternate every loop)

Code pointers:

- `compute_imba_signals(...)`
- `trend_signals_from_imba(...)`

Quick inspect:

```bash
grep -n "compute_imba_signals\\|trend_signals_from_imba\\|active_mode" src/quant/execution/live_signal_worker.py
sed -n '330,390p' src/quant/execution/live_signal_worker.py
```

---

### Step 3: Write signals to JSONL files (storage)

File:

- `src/quant/execution/live_signal_worker.py`

What happens:

- Writes mode-specific streams:
  - `.../countertrend/YYYYMMDD.jsonl`
  - `.../trendfollower/YYYYMMDD.jsonl`
- Writes active stream:
  - `.../SOL-USDT/YYYYMMDD.jsonl` (this is what executors consume)
- Uses dedupe logic by timestamp, so same signal timestamp is not emitted repeatedly

Code pointers:

- `_append_signal_jsonl_dedupe(...)`
- `state.last_signal_ts`
- `emitted_now`

Quick inspect:

```bash
grep -n "active_new\\|_append_signal_jsonl_dedupe\\|last_signal_ts\\|emitted_now" src/quant/execution/live_signal_worker.py
sed -n '420,490p' src/quant/execution/live_signal_worker.py
```

Why you can see old timestamp for long time:

- If no new opposite IMBA event is generated, latest signal stays the same (example: still `+1` from 15:39)

---

### Step 4: Expose latest signal via API

File:

- `src/quant/execution/webhook_server.py`

Endpoint:

- `/api/signals/latest/solusd`

What happens:

- Reads newest non-zero signal from JSONL
- Returns `{ ts, signal }`

Code pointers:

- `_latest_signal_from_jsonl(...)`
- `api_signals_latest_solusd(...)`

Quick inspect:

```bash
grep -n "_latest_signal_from_jsonl\\|api_signals_latest_solusd" src/quant/execution/webhook_server.py
sed -n '146,187p' src/quant/execution/webhook_server.py
sed -n '897,913p' src/quant/execution/webhook_server.py
```

---

### Step 5: Gate and renko endpoints (context for exits)

File:

- `src/quant/execution/webhook_server.py`
- gate logic in `src/quant/execution/gate_provider.py`

Endpoints:

- `/api/gate/solusd`
- `/api/renko/latest/solusd`

What happens:

- Gate decides which exit engine is active (`flip` or `tp2`)
- Renko endpoint provides swing high/low used for SL logic

Quick inspect:

```bash
grep -n "api_gate_solusd\\|api_renko_latest_solusd" src/quant/execution/webhook_server.py
sed -n '868,940p' src/quant/execution/webhook_server.py
```

---

### Step 6: Kraken bot consumes API and decides actions

File:

- `src/quant/execution/kraken_bot.py`

What happens:

1. Fetch gate, signal, mark, equity
2. Build action list in state machine
3. Execute actions on Kraken
4. Save bot state to disk

Code pointers:

- `run_once(...)`
- `run_once_logic(...)`
- `execute_actions(...)`
- `save_state(...)` / `load_state(...)`

Quick inspect:

```bash
grep -n "def run_once_logic\\|def execute_actions\\|def run_once\\|load_state\\|save_state" src/quant/execution/kraken_bot.py
sed -n '179,366p' src/quant/execution/kraken_bot.py
sed -n '421,520p' src/quant/execution/kraken_bot.py
```

---

### Step 7: KuCoin live executor also consumes signals

File:

- `src/quant/execution/live_executor.py`

What happens:

- Reads same signal files / event stream
- Tries to execute on KuCoin
- Keeps own runtime state file

Quick inspect:

```bash
grep -n "def run_once\\|_latest_backtest_event\\|flip_to_" src/quant/execution/live_executor.py
sed -n '542,690p' src/quant/execution/live_executor.py
```

---

## 3) Where Data Is Stored

Signal and state files are critical:

- Signals:
  - `SIGNALS_DIR/SOL-USDT/YYYYMMDD.jsonl`
  - `SIGNALS_DIR/SOL-USDT/countertrend/YYYYMMDD.jsonl`
  - `SIGNALS_DIR/SOL-USDT/trendfollower/YYYYMMDD.jsonl`
- Signal worker state:
  - `LIVE_SIGNAL_STATE` (default `data/live/live_signal_state.json`)
- Kraken bot state:
  - `KRAKEN_STATE_JSON` (default `data/live/bot_state.json`)
- Renko cache:
  - `DASHBOARD_RENKO_PARQUET`

If state file and exchange position diverge, behavior can look "wrong" even if code path is deterministic.

---

## 4) Why Your Specific Failure Can Happen

### A) "Signal did not generate as supposed"

What likely happened:

- The worker kept reporting latest calculated signal from old timestamp (`latest_calc_ts=15:39`) because no new opposite IMBA event was produced.
- This is expected in sticky trend mode when market did not produce a new opposite signal condition.

Important:

- "No new signal line emitted" does not always mean worker is broken.
- It can mean "same active regime/signal still valid."

---

### B) "Manual long 40, then opposite came, but only close happened"

There are two concrete reasons currently possible:

1. **State mismatch after manual intervention**
   - Kraken bot logic is state-file driven.
   - If you manually open position on exchange, bot state may still think another mode/size/side.
   - Then bot may send action that flattens but does not end with the expected reverse size.

2. **TP2_BE branch behavior mismatch**
   - In `kraken_bot.py`, in `TP2_BE` mode on opposite signal, code does:
     - `close_all`
     - return flat
   - It does **not** apply `flip_on_opposite` there.
   - Backtest (`renko_runner_tp2.py`) applies opposite signal close + optional re-open opposite across active trade flow.

Quick inspect (important):

```bash
grep -n "TP2_BE\\|flip_on_opposite\\|signal_exit" src/quant/execution/kraken_bot.py
sed -n '369,412p' src/quant/execution/kraken_bot.py
```

Backtest reference:

```bash
grep -n "signal_exit\\|flip_on_opposite" src/quant/backtest/renko_runner_tp2.py
sed -n '427,433p' src/quant/backtest/renko_runner_tp2.py
```

---

## 5) All Involved Files (Core)

- `src/quant/execution/live_signal_worker.py`
- `src/quant/execution/webhook_server.py`
- `src/quant/execution/gate_provider.py`
- `src/quant/execution/renko_cache_updater.py`
- `src/quant/execution/kraken_bot.py`
- `src/quant/execution/kraken_futures.py`
- `src/quant/execution/live_executor.py`
- `src/quant/strategies/imba.py`
- `src/quant/backtest/renko_runner_tp2.py`
- `src/quant/strategies/flip_engine.py`

---

## 6) Precise Fix Plan for the Original Reliability Issue

Goal:

- opposite signal must reliably flip as in backtest
- manual interventions must not desync bot behavior

### Plan Step 1: Add position/state reconciliation in Kraken bot

Before `run_once_logic(...)`:

- Read live Kraken position (`get_position`)
- Compare with saved `BotState`
- If mismatch, reconcile state:
  - set `pos_side`, `entry_px`, `size_full`, `size_rem`, `mode` consistently
- Log explicit reconciliation event

Why:

- manual order changes are currently invisible to state machine logic

---

### Plan Step 2: Make TP2_BE opposite-signal path match backtest intent

In `TP2_BE` branch (`kraken_bot.py`):

- on opposite signal:
  - close all
  - if `flip_on_opposite` true: open opposite position immediately (same as TP2_OPEN behavior)

Why:

- prevents "close only" when strategy expects reverse

---

### Plan Step 3: Add deterministic tests for your failure case

Add tests in `tests/test_kraken_bot_statemachine.py`:

1. manual-like desync reconcile test
2. TP2_BE opposite signal with `flip_on_opposite=True` must close+enter opposite
3. TP2_BE opposite signal with `flip_on_opposite=False` must close only

Why:

- regression-proof for this exact issue

---

### Plan Step 4: Add runtime debug lines for every action decision

Log for each tick:

- `mode`, `engine`, `pos_side`, `signal`, `signal_ts`, `new_signal`, `reason`
- action list before execution
- exchange position before and after

Why:

- you can verify "why close" vs "why flip" from logs without guessing

---

### Plan Step 5: Operational guardrail for manual trading

When manual trade is done:

- either pause bot, then resume
- or run explicit "re-sync state" endpoint/command before next loop

Why:

- avoids hidden divergence between exchange reality and bot state file

---

## 7) Fast Live Checklist (for future runs)

1. Check signal freshness:
   - `/api/signals/latest/solusd`
2. Check gate:
   - `/api/gate/solusd`
3. Check renko swing values:
   - `/api/renko/latest/solusd?lookback=50`
4. Check bot mode/action logs:
   - look for `reason=...`, `action=...`, `mode=...`
5. If manual order happened:
   - re-sync state before trusting next flip

