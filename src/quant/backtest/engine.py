# src/quant/backtest/engine.py

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    stats: dict


def run_backtest(df: pd.DataFrame, pos: pd.Series, fee_bps: float = 4.0) -> BacktestResult:
    """
    Extremely simple backtest:
    - enter/hold position from pos (+1 long, -1 short)
    - PnL uses close-to-close returns
    - fee charged on position changes (turnover) as fee_bps (basis points) on notional
    Assumes 1 unit notional for simplicity (we add sizing later).
    """
    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["pos"] = pos.reindex(df.index).fillna(0).astype(int)

    # strategy return: position from previous bar applied to current return
    df["strat_ret_gross"] = df["pos"].shift(1).fillna(0) * df["ret"]

    # turnover: absolute change in position (0->1, 1->-1, etc.)
    df["turnover"] = (df["pos"] - df["pos"].shift(1).fillna(0)).abs()

    fee_rate = fee_bps / 10_000.0
    df["fees"] = df["turnover"] * fee_rate

    df["strat_ret_net"] = df["strat_ret_gross"] - df["fees"]
    df["equity"] = (1.0 + df["strat_ret_net"]).cumprod()

    stats = {
        "rows": int(len(df)),
        "start": str(df["ts"].iloc[0]),
        "end": str(df["ts"].iloc[-1]),
        "total_return_pct": float((df["equity"].iloc[-1] - 1.0) * 100.0),
        "max_drawdown_pct": float(((df["equity"] / df["equity"].cummax()) - 1.0).min() * 100.0),
        "turnover_sum": float(df["turnover"].sum()),
    }

    equity_curve = df[["ts", "close", "pos", "equity", "strat_ret_net", "fees"]].copy()
    return BacktestResult(equity_curve=equity_curve, stats=stats)
