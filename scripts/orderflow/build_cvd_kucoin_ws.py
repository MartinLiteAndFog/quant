import asyncio
import json
import pandas as pd
import websockets
import requests
import uuid
from datetime import datetime, timezone
from pathlib import Path

SYMBOL = "SOLUSDTM"

OUTDIR = Path("data/orderflow")
OUTDIR.mkdir(parents=True, exist_ok=True)

cvd = 0
current_minute = None
buy_volume = 0.0
sell_volume = 0.0
rows = []


def minute_floor(ts):
    return ts.replace(second=0, microsecond=0)


def get_ws_endpoint():
    r = requests.post("https://api-futures.kucoin.com/api/v1/bullet-public")
    data = r.json()["data"]

    token = data["token"]
    server = data["instanceServers"][0]

    endpoint = server["endpoint"]
    ping_interval = server["pingInterval"]

    url = f"{endpoint}?token={token}&connectId={uuid.uuid4()}"

    return url, ping_interval


async def ping_loop(ws, interval):
    while True:
        await asyncio.sleep(interval / 1000)
        try:
            await ws.send(json.dumps({"type": "ping"}))
        except:
            return


async def kucoin_ws():

    url, ping_interval = get_ws_endpoint()

    async with websockets.connect(url) as ws:

        asyncio.create_task(ping_loop(ws, ping_interval))

        sub = {
            "id": str(uuid.uuid4()),
            "type": "subscribe",
            "topic": f"/contractMarket/execution:{SYMBOL}",
            "privateChannel": False,
            "response": True
        }

        await ws.send(json.dumps(sub))

        print("Subscribed to", SYMBOL)

        while True:

            msg = await ws.recv()
            data = json.loads(msg)

            if "data" not in data:
                continue

            trade = data["data"]

            size = float(trade["size"])
            side = trade["side"]

            ts_raw = int(trade["ts"])

            # normalize timestamp (KuCoin can send ns/µs/ms)
            if ts_raw > 1e18:
                ts_raw = ts_raw / 1e9
            elif ts_raw > 1e15:
                ts_raw = ts_raw / 1e6
            elif ts_raw > 1e12:
                ts_raw = ts_raw / 1e3

            ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)

            minute = minute_floor(ts)

            await process_trade(minute, side, size)


async def process_trade(minute, side, size):
    global current_minute, buy_volume, sell_volume, cvd, rows

    if current_minute is None:
        current_minute = minute

    if minute != current_minute:

        delta = buy_volume - sell_volume
        cvd += delta

        rows.append({
            "ts": current_minute,
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "delta": delta,
            "cvd": cvd
        })

        if len(rows) >= 2:
            flush()

        buy_volume = 0
        sell_volume = 0
        current_minute = minute

    if side == "buy":
        buy_volume += size
    else:
        sell_volume += size


def flush():
    global rows

    df = pd.DataFrame(rows)

    path = OUTDIR / "sol_cvd.parquet"

    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, df])

    df.to_parquet(path, index=False)

    print("Saved rows:", len(df))

    rows = []


if __name__ == "__main__":
    asyncio.run(kucoin_ws())