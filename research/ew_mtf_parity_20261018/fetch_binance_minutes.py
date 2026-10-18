"""Public spot klines only; no credentials, account access or trading requests."""
from pathlib import Path
import hashlib
import json
import time
import urllib.request
import pandas as pd
import argparse

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--interval',choices=['1m','1h'],default='1m')
parser.add_argument('--start',default='2026-07-01T00:00:00Z')
parser.add_argument('--end',default='2026-10-18T08:00:00Z')
args=parser.parse_args()
start=int(pd.Timestamp(args.start).timestamp()*1000)
end=int(pd.Timestamp(args.end).timestamp()*1000)
step=60000 if args.interval=='1m' else 3600000
cache=root/f'inputs/binance_api_{args.interval}_chunks'
cache.mkdir(exist_ok=True)
all_rows=[]; cursor=start; chunks=[]
while cursor<end:
    path=cache/f'{cursor}.json'
    url=f'https://api.binance.com/api/v3/klines?symbol=SOLUSDT&interval={args.interval}&startTime={cursor}&endTime={end-1}&limit=1000'
    if path.exists():
        raw=path.read_bytes()
    else:
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=20) as response: raw=response.read()
                break
            except Exception:
                if attempt==2: raise
                time.sleep(1+attempt)
        rows=json.loads(raw)
        if not isinstance(rows,list) or not rows: raise RuntimeError(f'No klines at {cursor}')
        path.write_bytes(raw)
    rows=json.loads(raw)
    if int(rows[0][0])!=cursor: raise RuntimeError('Unexpected gap at chunk boundary')
    all_rows.extend(rows)
    chunks.append({'path':str(path.relative_to(root)),'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'rows':len(rows)})
    cursor=int(rows[-1][0])+step
    if len(chunks)%20==0: print(f'{len(all_rows)} {args.interval} bars fetched',flush=True)
frame=pd.DataFrame(all_rows,columns=['open_ms','open','high','low','close','volume','close_ms','quote_volume','trades','taker_volume','taker_quote_volume','ignore'])
gaps=frame.loc[frame.open_ms.diff()>step,'open_ms'].tolist()
assert frame.open_ms.is_monotonic_increasing
if args.interval=='1m': assert not gaps
assert frame.open_ms.is_unique and int(frame.open_ms.iloc[-1])+step==end
frame.insert(0,'time',frame.open_ms//1000)
output='binance_spot_minutes.csv.gz' if args.interval=='1m' else 'binance_spot_hours_bootstrap.csv.gz'
frame.to_csv(root/'inputs'/output,index=False)
(root/f'results/binance_{args.interval}_source_manifest.json').write_text(json.dumps({'venue':'BINANCE spot','symbol':'SOLUSDT','interval':args.interval,'start_ms':start,'end_exclusive_ms':end,'rows':len(frame),'gap_following_open_ms':gaps,'chunks':chunks},indent=2)+'\n')
print(f'Verified ordered {args.interval} bars: {len(frame)}; historical gaps: {len(gaps)}',flush=True)
