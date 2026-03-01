# Railway Runbook (Current Live Setup)

This is the canonical operations doc for the current Railway deployment.
Use this file for maintenance, troubleshooting, and handover.

## 1) Services and responsibilities

- `quant` (web service, has public domain)
  - serves `/dashboard`
  - serves `/api/*`
  - reads dashboard files (`renko_latest.parquet`, `execution_state.json`, `regime.db`)
- `Signal` worker service (combined worker process)
  - runs `live_signal_worker`
  - runs `live_executor`
  - writes signals and execution state files

## 2) Working domain

- Root: `https://quant-production-5533.up.railway.app/`
- Dashboard: `https://quant-production-5533.up.railway.app/dashboard`
- Chart API: `https://quant-production-5533.up.railway.app/api/dashboard/chart?symbol=SOLUSDT&hours=336&max_points=4000`

## 3) Start commands

### `quant` service

Use standard web start (Dockerfile/uvicorn path).

### `Signal` worker service (combined)

```bash
bash -lc "python -u -m quant.execution.live_signal_worker --symbol SOLUSDT --signals-dir /data/live/signals & python -u -m quant.execution.live_executor --symbol SOLUSDT --signals-dir /data/live/signals; wait"
```

## 4) Required environment variables

Set these in both services where applicable.

### Shared/API

- `KUCOIN_FUTURES_API_KEY`
- `KUCOIN_FUTURES_API_SECRET`
- `KUCOIN_FUTURES_PASSPHRASE`
- `PYTHONUNBUFFERED=1`

### Paths (IMPORTANT: absolute, volume-backed)

- `SIGNALS_DIR=/data/live/signals`
- `REGIME_DB_PATH=/data/live/regime.db`
- `DASHBOARD_RENKO_PARQUET=/data/live/renko_latest.parquet`
- `DASHBOARD_LEVELS_JSON=/data/live/execution_state.json`
- `LIVE_SIGNAL_STATE=/data/live/live_signal_state.json`
- `LIVE_EXECUTOR_STATE=/data/live/live_executor_state.json`
- `LIVE_TRAILING_STATE=/data/live/live_trailing_state.json`

### Strategy/gate

- `LIVE_SYMBOL=SOLUSDT`
- `LIVE_DEFAULT_GATE_ON=1`  # Gate ON -> countertrend (IMBA)
- `LIVE_IMBA_LOOKBACK=250`
- `LIVE_RENKO_BOX=0.1`
- `LIVE_CANDLES_LIMIT=1500`
- `LIVE_IMBA_SL_ABS=1.5`

### Executor safety

- `LIVE_TRADING_ENABLED=0`  # set to 1 for live
- `LIVE_EXECUTOR_DRY_RUN=1` # set to 0 for live order placement
- `LIVE_EXECUTOR_MAX_EUR=40`
- `LIVE_EXECUTOR_LEVERAGE=4`
- `LIVE_EXECUTOR_SYMBOL_ALLOWLIST=SOLUSDT,SOL-USDT`
- `LIVE_EXECUTOR_POLL_SEC=5`

### Trailing

- `LIVE_TTP_TRAIL_PCT=0.012`
- `LIVE_WAIT_SL_PCT=0.02`

### Dashboard refresh controls

- `DASHBOARD_RENKO_AUTO_REFRESH_ON_READ=1`
- `DASHBOARD_RENKO_STALE_MIN=1`
- `DASHBOARD_RENKO_REFRESH_COOLDOWN_SEC=15`
- `DASHBOARD_RENKO_POLL_SEC=60`
- `DASHBOARD_UI_REFRESH_MS=4000`
- `DASHBOARD_STATESPACE_REFRESH_MS=15000`
- `DASHBOARD_API_CACHE_SEC=8`
- `DASHBOARD_FILLS_REFRESH_COOLDOWN_SEC=20`
- `DASHBOARD_FILLS_FETCH_LIMIT=200`
- `DASHBOARD_FILL_MARKER_LIMIT=1200`
- `DASHBOARD_LOG_THROTTLE_SEC=60`
- `DASHBOARD_RENKO_DAYS_BACK=14`
- `DASHBOARD_RENKO_STEP_HOURS=6`  # code clamps effective fetch window to API-safe chunks

Optional isolation if the same account runs multiple systems:

- `DASHBOARD_FILLS_CLIENT_OID_PREFIXES=entry-,tp_flip-,TP_,SL_,manual-`
- `DASHBOARD_FILLS_INCLUDE_EMPTY_CLIENT_OID=0`

### Log-rate controls (Railway log quota)

- `KUCOIN_LOG_THROTTLE_SEC=30`
- `LIVE_EXECUTOR_LOG_THROTTLE_SEC=60`
- `LIVE_EXECUTOR_NO_SIGNAL_LOG_SEC=60`
- `LIVE_SIGNAL_LOG_THROTTLE_SEC=60`
- `LIVE_SIGNAL_STATUS_LOG_SEC=60`
- `LIVE_SIGNAL_ERROR_LOG_SEC=30`

## 5) Current strategy routing

- `gate_on=1` -> `countertrend` (IMBA stream)
- `gate_on=0` -> `trendfollower` (inverse IMBA stream)

Signal files:

- active routed stream: `/data/live/signals/SOLUSDT/YYYYMMDD.jsonl`
- strategy history:
  - `/data/live/signals/SOLUSDT/countertrend/YYYYMMDD.jsonl`
  - `/data/live/signals/SOLUSDT/trendfollower/YYYYMMDD.jsonl`

## 6) Expected logs

- Signal worker:
  - `live-signal emitted ...` when a new flip is emitted
  - `live-signal status ... emitted_now=0` when no new flip occurred
- Executor:
  - `simulated action=...` in dry-run
  - `enter result=...` / `flip result=...` in live mode

## 7) Dashboard features currently implemented

- Renko chart + live price line (ticker mid-based)
- Gate shading overlay (strong color mode)
- IMBA fib lines:
  - long fib (green), mid fib (gray), short fib (red)
- Trade entry/exit markers
- Entry->exit connector segments with PnL coloring:
  - green for profitable trades, red for losing trades
- Level overlays from execution state: `SL`, `TTP`, `TP1`, `TP2`
- `renko_health` in API payload to diagnose stale data

## 8) Health and freshness checks

### API check (from local machine)

```bash
python3 - <<'PY'
import json, urllib.request
url = "https://quant-production-5533.up.railway.app/api/dashboard/chart?symbol=SOLUSDT&hours=336&max_points=4000"
with urllib.request.urlopen(url, timeout=20) as r:
    d = json.loads(r.read().decode("utf-8"))
print("renko_health:", d.get("renko_health"))
print("bars_count:", len(d.get("bars", [])))
print("last_3_bars:", d.get("bars", [])[-3:])
PY
```

### Worker signal file check (in Railway SSH)

```bash
tail -n 20 /data/live/signals/SOLUSDT/$(date -u +%Y%m%d).jsonl
```

## 9) Go-live procedure

1. Verify dry-run stability:
   - signal status logs continue
   - executor simulated actions make sense
2. Enable live trading:
   - `LIVE_TRADING_ENABLED=1`
3. Enable real execution:
   - `LIVE_EXECUTOR_DRY_RUN=0`
4. Keep small risk initially (`MAX_EUR`, leverage conservative).

Rollback (instant):

- `LIVE_EXECUTOR_DRY_RUN=1` and/or `LIVE_TRADING_ENABLED=0`

Manual flatten command (short -> flat), with expected-trade logging so dashboard fill reason is visible:

```bash
PYTHONPATH=src python3 -m quant.execution.manual_orders --symbol SOL-USDT --action cancel_short
```

## 10) Known caveats

- Railway minimal shells may not have `curl`, `rg`, `ps`.
  Use Python snippets and `/proc` checks instead.
- If chart looks stale, always inspect `renko_health.age_sec` and `last_ts`.
- If no new signals appear but worker is healthy, `emitted_now=0` can be a legitimate no-flip state.

## 11) Current status snapshot (2026-02-24)

### Live execution

- Live order submission is working on KuCoin Futures (`SOLUSDTM`).
- Confirmed fills seen via `list_fills(symbol="SOLUSDT")` with:
  - `liquidity: maker`
  - `marginMode: CROSS`
  - successful buy fills for size `2` contracts.

### Dashboard

- Implemented:
  - gate shading, fib lines, live fill markers, explicit `SL active` / `TTP active`
  - position label clarified as contracts
  - notional calculation now uses contract multiplier.
- Required env for correct notional:
  - `CONTRACT_MULTIPLIER_SOLUSDT=0.1`

### Open consistency issue (important)

- `Signal` service and `quant` service currently do not reliably share the same live `execution_state.json` content.
- Symptom:
  - Worker writes rich state (`countertrend/lookback/...`)
  - Web service reads a stale/minimal state object.
- Impact:
  - Dashboard can show outdated/missing SL/TTP/TP levels.

## 12) Next tasks for tomorrow

1. **Unify runtime storage (highest priority)**
   - Run web + workers in one service with one shared volume, or
   - move state/signals to shared DB/Redis instead of local files.

2. **Verify level propagation end-to-end**
   - Ensure `execution_state.json` seen by dashboard is exactly the worker-written file.
   - Re-check `SL active` / `TTP active` and level lines under live position.

3. **Final ops hardening**
   - Reduce noisy non-actionable `cancel_all` warnings.
   - Add structured `LIVE ORDER FILLED` log line (order_id, qty, price, side).

4. **Post-trade visualization completion**
   - Confirm entry/exit connectors and PnL colors for closed trades in live mode.
   - Validate marker timestamps against brick mapping.

5. **Go-live guardrail cleanup**
   - Review and freeze required env vars in Railway.
   - Keep a one-command rollback checklist (`DRY_RUN=1` and/or `LIVE_TRADING_ENABLED=0`).
