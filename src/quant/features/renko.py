import math
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
    Fixed-box Traditional Renko using source CLOSE prices.

    TradingView-compatible Traditional Renko geometry:
      - continuation requires a 1-box move;
      - reversal requires a 2-box move from the previous brick close;
      - the first reversal brick opens one box away from the previous close;
      - if one source bar emits multiple bricks, timestamps are spaced by 1 ms
        so downstream de-duplication does not discard intermediate bricks.

    Returns a dataframe with: ts, dir, open, close.

    Important: parity still depends on using the same venue/feed and source
    timeframe as TradingView. This function deliberately remains close-driven.
    """
    if box <= 0:
        raise ValueError("box must be > 0")

    df = df.sort_values("ts").reset_index(drop=True)
    closes = pd.to_numeric(df["close"], errors="coerce")
    tss = pd.to_datetime(df["ts"], utc=True, errors="coerce")

    valid = closes.notna() & tss.notna()
    closes = closes[valid].astype(float).tolist()
    tss = tss[valid].tolist()

    if not closes:
        return pd.DataFrame(columns=["ts", "dir", "open", "close"])

    bricks: List[RenkoBrick] = []

    # Keep the historical grid convention used by this project.
    eps = max(abs(float(closes[0])), abs(float(box)), 1.0) * 1e-12
    anchor = math.floor((float(closes[0]) + eps) / float(box)) * float(box)
    last_brick_close = round(float(anchor), 12)

    # 0 = not established yet, +1 = uptrend, -1 = downtrend.
    trend = 0
    tol = max(abs(float(box)) * 1e-9, 1e-12)

    for ts, price in zip(tss, closes):
        price = float(price)
        emitted = 0

        def emit(direction: int, open_px: float, close_px: float) -> None:
            nonlocal emitted
            out_ts = pd.Timestamp(ts) + pd.Timedelta(milliseconds=emitted)
            bricks.append(
                RenkoBrick(
                    ts=out_ts,
                    direction=int(direction),
                    open=round(float(open_px), 12),
                    close=round(float(close_px), 12),
                )
            )
            emitted += 1

        if trend >= 0:
            # Initial/continuing up move: one box per brick.
            if price >= last_brick_close + box - tol:
                while price >= last_brick_close + box - tol:
                    o = last_brick_close
                    c = o + box
                    emit(+1, o, c)
                    last_brick_close = round(c, 12)
                    trend = +1

            # Reversal from an established uptrend needs two boxes.
            elif trend == +1 and price <= last_brick_close - 2.0 * box + tol:
                o = last_brick_close - box
                c = last_brick_close - 2.0 * box
                emit(-1, o, c)
                last_brick_close = round(c, 12)
                trend = -1

                while price <= last_brick_close - box + tol:
                    o = last_brick_close
                    c = o - box
                    emit(-1, o, c)
                    last_brick_close = round(c, 12)

            # Before a trend exists, a one-box down move establishes it.
            elif trend == 0 and price <= last_brick_close - box + tol:
                while price <= last_brick_close - box + tol:
                    o = last_brick_close
                    c = o - box
                    emit(-1, o, c)
                    last_brick_close = round(c, 12)
                    trend = -1

        else:
            # Continuing down move: one box per brick.
            if price <= last_brick_close - box + tol:
                while price <= last_brick_close - box + tol:
                    o = last_brick_close
                    c = o - box
                    emit(-1, o, c)
                    last_brick_close = round(c, 12)
                    trend = -1

            # Reversal from an established downtrend needs two boxes.
            elif price >= last_brick_close + 2.0 * box - tol:
                o = last_brick_close + box
                c = last_brick_close + 2.0 * box
                emit(+1, o, c)
                last_brick_close = round(c, 12)
                trend = +1

                while price >= last_brick_close + box - tol:
                    o = last_brick_close
                    c = o + box
                    emit(+1, o, c)
                    last_brick_close = round(c, 12)

    return pd.DataFrame(
        [{"ts": b.ts, "dir": b.direction, "open": b.open, "close": b.close} for b in bricks]
    )
