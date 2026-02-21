import pandas as pd
from dataclasses import dataclass
from typing import List

@dataclass
class RenkoBrick:
    ts: pd.Timestamp
    direction: int  # +1 up, -1 down
    open: float
    close: float

def renko_from_close(df: pd.DataFrame, box: float) -> pd.DataFrame:
    """
    Fixed-box Renko using CLOSE prices.
    Emits one or multiple bricks per bar if price moved enough.
    Returns bricks dataframe with: ts, dir, open, close.
    """
    if box <= 0:
        raise ValueError("box must be > 0")

    df = df.sort_values("ts").reset_index(drop=True)
    closes = df["close"].astype(float).tolist()
    tss = pd.to_datetime(df["ts"], utc=True).tolist()

    if not closes:
        return pd.DataFrame(columns=["ts", "dir", "open", "close"])

    bricks: List[RenkoBrick] = []

    last_close = closes[0]
    # anchor to nearest box grid so bricks are stable
    anchor = (last_close // box) * box
    last_brick_close = anchor

    for ts, price in zip(tss, closes):
        diff = price - last_brick_close
        n = int(diff // box) if diff >= 0 else int((-diff) // box)

        if n == 0:
            continue

        direction = 1 if diff > 0 else -1
        for _ in range(n):
            o = last_brick_close
            c = o + direction * box
            bricks.append(RenkoBrick(ts=ts, direction=direction, open=o, close=c))
            last_brick_close = c

    out = pd.DataFrame([{"ts": b.ts, "dir": b.direction, "open": b.open, "close": b.close} for b in bricks])
    return out
