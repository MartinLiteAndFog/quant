import pandas as pd
from pathlib import Path

INPUT_DIR = Path("data/orderflow/binance_trades")
OUTPUT_FILE = Path("data/orderflow/binance_cvd_1s.parquet")

cvd = 0
all_rows = []

files = sorted(INPUT_DIR.glob("*.csv"))

print("Found files:", len(files))

for f in files:

    print("Processing:", f.name)

    df = pd.read_csv(
        f,
        usecols=["price", "qty", "time", "is_buyer_maker"]
    )

    df["time"] = pd.to_datetime(df["time"], unit="ms")

    # 1 second aggregation
    df["second"] = df["time"].dt.floor("1s")

    df["buy_volume"] = df.apply(
        lambda r: r["qty"] if r["is_buyer_maker"] == False else 0,
        axis=1
    )

    df["sell_volume"] = df.apply(
        lambda r: r["qty"] if r["is_buyer_maker"] == True else 0,
        axis=1
    )

    agg = df.groupby("second").agg(
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum")
    )

    agg["delta"] = agg["buy_volume"] - agg["sell_volume"]

    for idx, row in agg.iterrows():

        cvd += row["delta"]

        all_rows.append(
            {
                "ts": idx,
                "buy_volume": row["buy_volume"],
                "sell_volume": row["sell_volume"],
                "delta": row["delta"],
                "cvd": cvd
            }
        )

out = pd.DataFrame(all_rows)

out.to_parquet(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Rows:", len(out))