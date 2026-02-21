import pandas as pd

def generate_signals_on_bricks(bricks: pd.DataFrame, flip_every: int = 10) -> pd.Series:
    """
    Dummy strategy on Renko bricks: alternate position every N bricks.
    """
    pos = []
    cur = 1
    for i in range(len(bricks)):
        if i % flip_every == 0 and i != 0:
            cur *= -1
        pos.append(cur)
    return pd.Series(pos, index=bricks.index, name="pos")
