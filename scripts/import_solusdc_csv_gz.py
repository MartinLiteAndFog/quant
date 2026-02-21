import gzip
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

IN_PATH = Path("/Users/martinpeter/Downloads/SOLUSDC.csv.gz")

# Ziel: kompatibel zu deinem Store-Layout
OUT_BASE = Path("data/raw")
EXCHANGE = "local"
SYMBOL = "SOL-USDC"
TIMEFRAME = "1m"

COLS = [
    "ts", "open", "high", "low", "close",
    "volume_base", "volume_quote",
    "taker_buy_base_vol", "taker_buy_quote_vol",
    "trade_count",
]

def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"missing: {IN_PATH}")

    # pandas kann gzip selbst, aber wir machen es explizit robust
    df = pd.read_csv(
        IN_PATH,
        sep="|",
        header=None,
        names=COLS,
        compression="gzip",
        dtype={
            "ts": "int64",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume_base": "float64",
            "volume_quote": "float64",
            "taker_buy_base_vol": "float64",
            "taker_buy_quote_vol": "float64",
            "trade_count": "int64",
        },
    )

    # ts: Unix seconds -> UTC Timestamp
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Backtest-Standardspalten
    out_df = df[["ts", "open", "high", "low", "close"]].copy()
    out_df["volume"] = df["volume_base"]

    # Sanity checks
    out_df = out_df.dropna()
    out_df = out_df.sort_values("ts").drop_duplicates("ts", keep="last").reset_index(drop=True)

    # Output-Pfad
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_BASE / f"exchange={EXCHANGE}" / f"symbol={SYMBOL}" / f"timeframe={TIMEFRAME}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{SYMBOL}_{TIMEFRAME}_{now}.parquet"

    out_df.to_parquet(out_path, index=False)

    print(f"read rows:  {len(df):,}")
    print(f"write rows: {len(out_df):,}")
    print(f"range: {out_df['ts'].iloc[0]} -> {out_df['ts'].iloc[-1]}")
    print(f"saved: {out_path}")

if __name__ == "__main__":
    main()
