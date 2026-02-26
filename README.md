# quant

## Operations docs

- Railway runbook: `docs/RAILWAY_RUNBOOK.md`
- Deployment notes: `docs/RAILWAY.md`
- Live deployment notes: `docs/LIVE_DEPLOY.md`

## Session handoff (2026-02-24)

- Live Futures execution path is functional (confirmed KuCoin fills on `SOLUSDTM`).
- Dashboard and worker feature set was expanded (gate routing, fib overlays, trade markers, SL/TTP status).
- Remaining high-priority issue for next session: unify shared state storage between `quant` and `Signal` services so dashboard always reads the same live execution state written by worker.
- See `docs/RAILWAY_RUNBOOK.md` sections:
  - "Current status snapshot (2026-02-24)"
  - "Next tasks for tomorrow"

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # (we'll add later)
```

## State Space v0.1
Run the 3-axis market state space pipeline (`X_raw`, `Y_res`, `Z_res`) on OHLCV parquet/csv.

### Input contract
- Required columns: `ts`, `open`, `high`, `low`, `close`
- Optional: `volume` (used for rolling VWAP-based equilibrium)

### CLI usage
```bash
PYTHONPATH=src python3 scripts/run_state_space_v01.py \
  --input /absolute/path/to/SOL-USDT_1m.parquet \
  --output data/state_space/solusdt_state_space_v01.parquet
```

### Outputs
- Main output parquet: axis raw/residualized values, axis diagnostics, and debug sensor columns.
- Sidecar basin file: `<output_stem>_basins.csv` with top densest voxel centers from `(X_raw, Y_res, Z_res)`.

### Programmatic usage
```python
from quant.state_space.config import StateSpaceConfig
from quant.state_space.pipeline import compute_state_space

out = compute_state_space(df, StateSpaceConfig())
# out.attrs["basins"] -> list of top voxel basins
```
