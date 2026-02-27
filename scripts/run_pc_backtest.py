from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant.predictive_coding.config import PCConfig
from quant.predictive_coding.model import TemporalPCModel
from quant.predictive_coding.probability import compute_probabilities
from quant.predictive_coding.targets import build_obs_features, build_targets, get_valid_range
from quant.predictive_coding.trade_logic import TradeDecisionLayer


def _read_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "ts" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "ts"})
        else:
            raise ValueError("parquet missing 'ts' column and not datetime-indexed")
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def run_backtest(df: pd.DataFrame, cfg: PCConfig) -> dict:
    close = df["close"].values.astype(np.float64)
    ts = df["ts"].values
    n = len(close)

    obs_all = build_obs_features(close)
    targets_all = build_targets(close, cfg.horizons)
    start, end = get_valid_range(close, cfg.horizons, obs_lookback=60)

    model = TemporalPCModel(cfg)
    trade_logic = TradeDecisionLayer(cfg)

    pred_rows = []
    event_rows = []
    trade_pairs = []
    equity_rows = []

    equity = 1.0
    peak_equity = 1.0
    open_trade = None

    for t in range(start, end):
        obs = obs_all[t]
        tgt = {h: targets_all[t, i] for i, h in enumerate(cfg.horizons)}

        if np.any(np.isnan(obs)) or any(np.isnan(v) for v in tgt.values()):
            continue

        is_warmup = t < start + cfg.warmup_bars
        mu, sigma = model.step(obs, tgt, is_warmup=is_warmup)
        probs = compute_probabilities(mu, sigma, float(close[t]))

        if is_warmup:
            continue

        signal, events = trade_logic.update(probs, float(close[t]), t)

        row = {"ts": ts[t], "close": float(close[t])}
        for h in cfg.horizons:
            p = probs[h]
            row[f"mu_{h}"] = p["mu"]
            row[f"sigma_{h}"] = p["sigma"]
            row[f"p_up_{h}"] = p["p_up"]
            row[f"price_level_{h}"] = p["price_level"]
        row["signal"] = signal
        pred_rows.append(row)

        for ev in events:
            ev_row = {
                "ts": ts[t],
                "event": ev["event"],
                "side": ev["side"],
                "price": ev["price"],
                "pnl_pct": ev["pnl_pct"],
                "seq": ev["seq"],
                "horizon": ev.get("horizon", 0),
            }
            event_rows.append(ev_row)

            if ev["event"] == "entry":
                open_trade = {
                    "entry_ts": ts[t],
                    "entry_px": ev["price"],
                    "side": ev["side"],
                    "horizon": ev.get("horizon", 0),
                    "edge": ev.get("edge", 0.0),
                    "p_at_entry": probs.get(ev.get("horizon", 1), {}).get("p_up", 0.5),
                }
            elif ev["event"] in ("sl_exit", "tp_exit", "timeout_exit", "flip_exit"):
                if open_trade is not None:
                    pnl = ev["pnl_pct"] - cfg.total_cost
                    equity *= (1 + pnl)
                    trade_pairs.append({
                        **open_trade,
                        "exit_ts": ts[t],
                        "exit_px": ev["price"],
                        "pnl_pct": pnl,
                        "exit_event": ev["event"],
                    })
                    open_trade = None

        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        equity_rows.append({"ts": ts[t], "equity": equity, "drawdown": dd})

    predictions_df = pd.DataFrame(pred_rows)
    events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame()
    trades_df = pd.DataFrame(trade_pairs) if trade_pairs else pd.DataFrame()
    equity_df = pd.DataFrame(equity_rows)

    n_trades = len(trades_df)
    stats = {
        "total_return_pct": (equity - 1) * 100,
        "max_drawdown_pct": float(equity_df["drawdown"].max() * 100) if len(equity_df) else 0.0,
        "trade_count": n_trades,
        "hit_rate": float((trades_df["pnl_pct"] > 0).mean() * 100) if n_trades else 0.0,
        "avg_trade_pnl_bps": float(trades_df["pnl_pct"].mean() * 10_000) if n_trades else 0.0,
        "avg_winner_bps": float(trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].mean() * 10_000) if n_trades and (trades_df["pnl_pct"] > 0).any() else 0.0,
        "avg_loser_bps": float(trades_df.loc[trades_df["pnl_pct"] <= 0, "pnl_pct"].mean() * 10_000) if n_trades and (trades_df["pnl_pct"] <= 0).any() else 0.0,
        "fee_bps": cfg.fee_bps,
        "slippage_bps": cfg.slippage_bps,
        "total_fee_drag_bps": n_trades * (cfg.fee_bps + cfg.slippage_bps),
        "bars_processed": end - start - cfg.warmup_bars,
        "warmup_bars": cfg.warmup_bars,
    }

    return {
        "predictions": predictions_df,
        "events": events_df,
        "trades": trades_df,
        "equity": equity_df,
        "stats": stats,
    }


def main():
    ap = argparse.ArgumentParser(description="Predictive-Coding Backtest Runner")
    ap.add_argument("--input", required=True, help="Path to OHLCV parquet")
    ap.add_argument("--run-id", default=None)

    ap.add_argument("--d-latent", type=int, default=32)
    ap.add_argument("--n-inference-steps", type=int, default=5)
    ap.add_argument("--lr-x", type=float, default=0.05)
    ap.add_argument("--lr-A", type=float, default=1e-4)
    ap.add_argument("--lr-W", type=float, default=1e-4)
    ap.add_argument("--lr-C", type=float, default=1e-4)
    ap.add_argument("--warmup-bars", type=int, default=200)
    ap.add_argument("--beta-obs", type=float, default=0.2)

    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--z-min", type=float, default=0.15)
    ap.add_argument("--min-edge-bps", type=float, default=5.0)
    ap.add_argument("--sl-pct", type=float, default=0.015)
    ap.add_argument("--tp-pct", type=float, default=0.03)
    ap.add_argument("--cooldown-bars", type=int, default=3)

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
    df = _read_ohlcv(args.input)
    print(f"[PC] Loaded {len(df)} bars, {df['ts'].iloc[0]} to {df['ts'].iloc[-1]}")

    results = run_backtest(df, cfg)

    stats = results["stats"]
    print("\n=== Predictive-Coding Backtest Results ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/runs") / run_id / "pc"
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
