"""Extract existing minute market data without assuming its venue provenance."""
from pathlib import Path
import hashlib
import json
import pandas as pd

root=Path(__file__).resolve().parent
source=Path('/Users/martinpeter/Downloads/SOLUSDT (3).csv.gz')
start=int(pd.Timestamp('2026-07-01T00:00:00Z').timestamp())
parts=[]; total=0; first=None; last=None
for chunk in pd.read_csv(source,sep='|',header=None,
                         names=['time','open','high','low','close','volume','quote_volume','taker_volume','taker_quote_volume','trades'],
                         chunksize=250000):
    total+=len(chunk)
    if first is None: first=int(chunk.time.iloc[0])
    last=int(chunk.time.iloc[-1])
    parts.append(chunk.loc[chunk.time>=start])
market=pd.concat(parts,ignore_index=True)
assert market.time.is_monotonic_increasing and market.time.is_unique
market.to_csv(root/'inputs/market_minutes.csv.gz',index=False)
digest=hashlib.file_digest(source.open('rb'),'sha256').hexdigest()
report={'source':str(source),'sha256':digest,'source_rows':total,'source_start':first,'source_end':last,
        'retained_rows':len(market),'start':int(market.time.min()),'end':int(market.time.max()),
        'missing_minutes':int(((market.time.diff().dropna()/60)-1).clip(lower=0).sum()),
        'venue':'UNVERIFIED: filename alone does not establish exchange or spot/futures',
        'ohlc_valid':bool(((market.high>=market[['open','close','low']].max(axis=1)) & (market.low<=market[['open','close','high']].min(axis=1))).all())}
(root/'results/market_inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
