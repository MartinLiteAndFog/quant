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
