from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.predictive_coding.config import PCConfig
from quant.predictive_coding.model import TemporalPCModel
from quant.predictive_coding.targets import build_obs_features
from quant.predictive_coding.trade_logic import TradeDecisionLayer


def _read_renko(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Required cols
    for c in ["ts", "close"]:
        if c not in df.columns:
            raise ValueError(f"renko parquet missing '{c}' column")

    # Keep duplicates! Renko can have multiple bricks per same minute.
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).reset_index(drop=True)

    # Optional: src_ts for nicer plots
    if "src_ts" in df.columns:
        df["src_ts"] = pd.to_datetime(df["src_ts"], utc=True, errors="coerce")

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    # Stable order: preserve parquet order; add brick_id as canonical index
    df["brick_id"] = np.arange(len(df), dtype=np.int64)
    return df


def run_backtest_renko(df: pd.DataFrame, cfg: PCConfig) -> dict:
    close = df["close"].values.astype(np.float64)
    ts = df["ts"].values
    src_ts = df["src_ts"].values if "src_ts" in df.columns else None
    brick_id = df["brick_id"].values.astype(np.int64)

    # Features on brick-close (these are now in "brick units", not minutes)
    obs_all = build_obs_features(close)

    model = TemporalPCModel(cfg)
    trade_logic = TradeDecisionLayer(cfg)

    pred_rows = []
    event_rows = []
    trade_pairs = []
    equity_rows = []

    equity = 1.0
    peak_equity = 1.0
    open_trade = None

    start = 60
    end = len(close)

    for i in range(start, end):
        px = float(close[i])
        obs = obs_all[i]

        if np.any(~np.isfinite(obs)):
            continue

        is_warmup = (i - start) < cfg.warmup_bars

        pred = model.step(price_now=px, obs=obs.astype(np.float64))

        if is_warmup:
            continue

        probs = {}
        for h in cfg.horizons:
            probs[h] = {
                "mu": float(pred["mu"][h]),
                "sigma": float(pred["sigma"][h]),
                "p_up": float(pred["p_up"][h]),
                "price_level": float(pred["price_level"][h]),
                "price_upper": float(pred["price_upper"][h]),
                "price_lower": float(pred["price_lower"][h]),
            }

        # TradeDecisionLayer compatibility
        try:
            signal, events = trade_logic.update(probs, px, bar_idx=int(brick_id[i]))
        except TypeError:
            signal, events = trade_logic.update(probs, px, int(brick_id[i]))

        # ---- prediction row (log tension proxies) ----
        row = {
            "ts": ts[i],
            "brick_id": int(brick_id[i]),
            "close": px,
            "signal": int(signal),
        }
        if src_ts is not None:
            row["src_ts"] = src_ts[i]

        v_temp = float(getattr(model, "v_temporal", np.nan))
        row["v_temporal"] = v_temp
        row["sigma_temporal"] = float(np.sqrt(v_temp + cfg.eps)) if np.isfinite(v_temp) else np.nan

        v_obs = getattr(model, "v_obs", None)
        if v_obs is not None:
            v_obs = np.asarray(v_obs, dtype=float)
            row["v_obs_mean"] = float(np.nanmean(v_obs))
            row["sigma_obs_mean"] = float(np.nanmean(np.sqrt(v_obs + cfg.eps)))
        else:
            row["v_obs_mean"] = np.nan
            row["sigma_obs_mean"] = np.nan

        for h in cfg.horizons:
            p = probs[h]
            row[f"mu_{h}"] = p["mu"]
            row[f"sigma_{h}"] = p["sigma"]
            row[f"p_up_{h}"] = p["p_up"]
            row[f"price_level_{h}"] = p["price_level"]
            v_h = getattr(model, "v_h", None)
            if isinstance(v_h, dict) and h in v_h:
                row[f"v_h_{h}"] = float(v_h[h])

        pred_rows.append(row)

        # ---- events + equity ----
        for ev in events:
            ev_row = {
                "ts": ts[i],
                "brick_id": int(brick_id[i]),
                "event": ev.get("event", ""),
                "side": int(ev.get("side", 0)),
                "price": float(ev.get("price", px)),
                "pnl_pct": float(ev.get("pnl_pct", 0.0)),
                "seq": int(ev.get("seq", 0)),
                "horizon": int(ev.get("horizon", 0)),
                "edge": float(ev.get("edge", 0.0)),
            }
            if src_ts is not None:
                ev_row["src_ts"] = src_ts[i]
            event_rows.append(ev_row)

            if ev_row["event"] == "entry":
                open_trade = {
                    "entry_ts": ts[i],
                    "entry_brick_id": int(brick_id[i]),
                    "entry_px": ev_row["price"],
                    "side": ev_row["side"],
                    "horizon": ev_row["horizon"],
                    "edge": ev_row["edge"],
                    "p_at_entry": float(probs.get(ev_row["horizon"], {}).get("p_up", 0.5)),
                }
                if src_ts is not None:
                    open_trade["entry_src_ts"] = src_ts[i]

            elif ev_row["event"] in ("sl_exit", "tp_exit", "timeout_exit", "flip_exit"):
                if open_trade is not None:
                    pnl = ev_row["pnl_pct"] - float(cfg.total_cost)
                    equity *= (1.0 + pnl)
                    trade_pairs.append({
                        **open_trade,
                        "exit_ts": ts[i],
                        "exit_brick_id": int(brick_id[i]),
                        "exit_px": ev_row["price"],
                        "pnl_pct": float(pnl),
                        "exit_event": ev_row["event"],
                        **({"exit_src_ts": src_ts[i]} if src_ts is not None else {}),
                    })
                    open_trade = None

        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        equity_rows.append({
            "ts": ts[i],
            "brick_id": int(brick_id[i]),
            "equity": float(equity),
            "drawdown": float(dd),
            **({"src_ts": src_ts[i]} if src_ts is not None else {}),
        })

    predictions_df = pd.DataFrame(pred_rows)
    events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame()
    trades_df = pd.DataFrame(trade_pairs) if trade_pairs else pd.DataFrame()
    equity_df = pd.DataFrame(equity_rows)

    n_trades = len(trades_df)
    stats = {
        "total_return_pct": float((equity - 1.0) * 100.0),
        "max_drawdown_pct": float(equity_df["drawdown"].max() * 100.0) if len(equity_df) else 0.0,
        "trade_count": int(n_trades),
        "hit_rate": float((trades_df["pnl_pct"] > 0).mean() * 100.0) if n_trades else 0.0,
        "avg_trade_pnl_bps": float(trades_df["pnl_pct"].mean() * 10_000.0) if n_trades else 0.0,
        "fee_bps": float(cfg.fee_bps),
        "slippage_bps": float(cfg.slippage_bps),
        "total_fee_drag_bps": float(n_trades * (cfg.fee_bps + cfg.slippage_bps)),
        "bricks_processed": int(len(predictions_df)),
        "warmup_bars": int(cfg.warmup_bars),
        "note": "Horizons are in BRICKS (not minutes). obs windows are also in bricks.",
    }

    return {
        "predictions": predictions_df,
        "events": events_df,
        "trades": trades_df,
        "equity": equity_df,
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser(description="Predictive-Coding Runner on Renko Bricks (Causal v0.2)")
    ap.add_argument("--input", required=True, help="Path to RENKO parquet (ts/open/high/low/close/dir/src_ts)")
    ap.add_argument("--run-id", default=None)

    # model
    ap.add_argument("--d-latent", type=int, default=32)
    ap.add_argument("--n-inference-steps", type=int, default=5)
    ap.add_argument("--lr-x", type=float, default=0.05)
    ap.add_argument("--lr-A", type=float, default=1e-4)
    ap.add_argument("--lr-W", type=float, default=1e-4)
    ap.add_argument("--lr-C", type=float, default=1e-4)
    ap.add_argument("--warmup-bars", type=int, default=5000)  # renko needs more warmup often
    ap.add_argument("--beta-obs", type=float, default=0.2)

    # trade params
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--z-min", type=float, default=0.35)
    ap.add_argument("--min-edge-bps", type=float, default=15.0)
    ap.add_argument("--sl-pct", type=float, default=0.02)
    ap.add_argument("--tp-pct", type=float, default=0.04)
    ap.add_argument("--cooldown-bars", type=int, default=10)

    args = ap.parse_args()

    cfg = PCConfig(
        d_latent=args.d_latent,
        n_inference_steps=args.n_inference_steps,
        lr_x=args.lr_x,
        lr_A=args.lr_A,
        lr_W=args.lr_W,
        lr_C=args.lr_C,
        warmup_bars=args.warmup_bars,
        beta_obs=args.beta_obs,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        margin=args.margin,
        z_min=args.z_min,
        min_edge_bps=args.min_edge_bps,
        sl_pct=args.sl_pct,
        tp_pct=args.tp_pct,
        cooldown_bars=args.cooldown_bars,
    )

    print(f"[PC] Loading {args.input}")
    df = _read_renko(args.input)
    print(f"[PC] Loaded {len(df)} bricks, ts {df['ts'].iloc[0]} -> {df['ts'].iloc[-1]}")

    results = run_backtest_renko(df, cfg)

    stats = results["stats"]
    print("\n=== PC Renko Backtest Results (Causal v0.2) ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/runs") / run_id / "pc_v02_renko"
    out_dir.mkdir(parents=True, exist_ok=True)

    results["predictions"].to_parquet(out_dir / "predictions.parquet", index=False)
    if len(results["events"]):
        results["events"].to_parquet(out_dir / "events.parquet", index=False)
    if len(results["trades"]):
        results["trades"].to_parquet(out_dir / "trades.parquet", index=False)
    results["equity"].to_parquet(out_dir / "equity.parquet", index=False)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\n[PC] Wrote results to {out_dir}")


if __name__ == "__main__":
    main()