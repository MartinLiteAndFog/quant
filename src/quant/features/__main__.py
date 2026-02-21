import glob
import pandas as pd
from ..config import CFG
from ..utils.log import get_logger
from .renko import renko_from_close

log = get_logger("quant.features")

def latest_parquet() -> str:
    safe_symbol = CFG.symbol.replace("/", "-").replace(":", "-")
    pattern = f"{CFG.data_dir}/exchange={CFG.exchange}/symbol={safe_symbol}/timeframe={CFG.timeframe}/*.parquet"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files found at {pattern}")
    return files[-1]

def main():
    path = latest_parquet()
    df = pd.read_parquet(path)
    # read RENKO_BOX from env via CFG? (keep simple here)
    import os
    box = float(os.getenv("RENKO_BOX", "0.1"))

    bricks = renko_from_close(df, box=box)
    log.info(f"[green]loaded[/green] {path}")
    log.info(f"[cyan]renko[/cyan] box={box} bricks={len(bricks)}")
    if len(bricks) > 0:
        log.info(f"last brick: ts={bricks['ts'].iloc[-1]} dir={bricks['dir'].iloc[-1]} close={bricks['close'].iloc[-1]}")

if __name__ == "__main__":
    main()
