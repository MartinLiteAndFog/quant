import requests
from datetime import datetime, timedelta
from pathlib import Path

BASE_URL = "https://data.binance.vision/data/futures/um/daily/trades/SOLUSDT"
OUTDIR = Path("data/orderflow/binance_trades")

OUTDIR.mkdir(parents=True, exist_ok=True)

START = datetime(2023, 1, 1)
END = datetime.today()   # small test range first


def download_file(url, path):
    r = requests.get(url, stream=True)

    if r.status_code != 200:
        print("Missing:", url)
        return

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Downloaded:", path.name)


d = START

while d <= END:

    date_str = d.strftime("%Y-%m-%d")

    filename = f"SOLUSDT-trades-{date_str}.zip"

    url = f"{BASE_URL}/{filename}"

    path = OUTDIR / filename

    if path.exists():
        print("Exists:", filename)
    else:
        download_file(url, path)

    d += timedelta(days=1)

print("Finished")