import pandas as pd
from pathlib import Path
import zipfile

INPUT_DIR = Path("data/orderflow/binance_trades")
OUTPUT_FILE = Path("data/orderflow/binance_cvd_1s_full.parquet")

cvd = 0


def process_file(csv_file, writer):
    global cvd

    chunk_iter = pd.read_csv(
        csv_file,
        usecols=["price", "qty", "time", "is_buyer_maker"],
        chunksize=2_000_000,
    )

    for chunk in chunk_iter:
        chunk["time"] = pd.to_datetime(chunk["time"], unit="ms")
        chunk["second"] = chunk["time"].dt.floor("1s")

        chunk["buy_volume"] = chunk.apply(
            lambda r: r["qty"] if r["is_buyer_maker"] is False or r["is_buyer_maker"] == False else 0,
            axis=1,
        )

        chunk["sell_volume"] = chunk.apply(
            lambda r: r["qty"] if r["is_buyer_maker"] is True or r["is_buyer_maker"] == True else 0,
            axis=1,
        )

        agg = chunk.groupby("second").agg(
            buy_volume=("buy_volume", "sum"),
            sell_volume=("sell_volume", "sum"),
        )

        agg["delta"] = agg["buy_volume"] - agg["sell_volume"]

        rows = []
        for idx, row in agg.iterrows():
            cvd += row["delta"]
            rows.append(
                {
                    "ts": idx,
                    "buy_volume": row["buy_volume"],
                    "sell_volume": row["sell_volume"],
                    "delta": row["delta"],
                    "cvd": cvd,
                }
            )

        df_out = pd.DataFrame(rows)
        writer.append(df_out)


class ParquetAppender:
    def __init__(self, path):
        self.path = path
        self.first = not path.exists()

    def append(self, df):
        if self.first:
            df.to_parquet(self.path, index=False)
            self.first = False
        else:
            existing = pd.read_parquet(self.path)
            df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(self.path, index=False)


def main():
    writer = ParquetAppender(OUTPUT_FILE)

    files = sorted(
        f for f in INPUT_DIR.glob("*.zip")
        if f.name >= "SOLUSDT-trades-2026-02-05.zip"
    )

    print("Files found:", len(files))

    for f in files:
        print("Processing:", f.name)

        with zipfile.ZipFile(f) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as csv_file:
                process_file(csv_file, writer)

    print("Finished building dataset")
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()