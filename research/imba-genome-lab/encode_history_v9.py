from pathlib import Path
import zipfile, io, numpy as np, json, base64, gzip, pandas as pd
P=Path('/mnt/data');W=P/'imba_genome_v9';W.mkdir(exist_ok=True)
old=np.load(W/'historical_7326_equity_2021Dec_2022_2023_2024.npz', allow_pickle=True)
z7=np.load(P/'imba_genome_v7/full_4158_genomes_2026_daily.npz',allow_pickle=True)
z8=np.load(P/'imba_genome_v8/extra_3168_mode_curves_2025_2026.npz',allow_pickle=True)
ids=np.r_[z7['ids'],z8['ids']].astype(str)
assert np.array_equal(ids,old['ids'].astype(str))
E={}
for y in [2021,2022,2023,2024]:
    key='equity_'+('2021Dec' if y==2021 else str(y))
    E[y]=old[key].astype(np.float64)
E[2025]=np.r_[z7['equity_2025'],z8['equity_2025']].astype(np.float64)
E[2026]=np.r_[z7['equity'],z8['equity']].astype(np.float64)
F={}
for y,e in E.items():
    f=np.empty_like(e)
    f[:,0]=e[:,0];f[:,1:]=e[:,1:]/np.maximum(e[:,:-1],1e-12)
    if not np.isfinite(f).all() or np.min(f)<=0:print('BAD',y,'min',float(np.min(f)));raise ValueError('Invalid daily factor')
    F[y]=f.astype('<f4')
payload={}
for y in range(2022,2027):
    tail=F[y-1][:,-90:]
    if tail.shape[1]<90:
        tail=np.concatenate([np.ones((len(ids),90-tail.shape[1]),dtype='<f4'),tail],axis=1)
    joined=np.ascontiguousarray(np.c_[tail,F[y]],dtype='<f4')
    raw=joined.tobytes()
    enc=base64.b64encode(gzip.compress(raw,compresslevel=6)).decode('ascii')
    payload[str(y)]=enc
    print(y,'days',F[y].shape[1],'min/max',float(np.min(joined)),float(np.max(joined)),
          'rawMB',round(len(raw)/1e6,1),'gzipMB',round(len(enc)*.75/1e6,1),flush=True)
with open(W/'history_payload.json','w') as f:json.dump(payload,f,separators=(',',':'))
print('Total JSON MB',round((W/'history_payload.json').stat().st_size/1e6,2))