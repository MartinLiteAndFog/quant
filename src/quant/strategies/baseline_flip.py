import pandas as pd

def generate_signals(df: pd.DataFrame, flip_every: int = 25) -> pd.Series:
    """
    Dummy strategy: alternate +1/-1 every N bars.
    Returns a position series aligned to df rows (position held during bar).
    """
    pos = []
    cur = 1
    for i in range(len(df)):
        if i % flip_every == 0 and i != 0:
            cur *= -1
        pos.append(cur)
    return pd.Series(pos, index=df.index, name="pos")
