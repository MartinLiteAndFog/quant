"""Offline reconstruction of EW v3.2 primary/secondary pivots, patterns, W3/WC signals.

Verification is on genuine exported TradingView Renko OHLC and TradingView's
exported HTF MTF state (not an independent HTF reconstruction). Never live trade.
Original source: user-supplied Elliott Wave Detector Pro v3.2 Pine (1405 lines).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ew_wave3_pine_probe import atr_wilder,pine_pivots,score

def ratio(a,b,c):
 return abs(c-b)/abs(b-a) if b!=a else 0.0

def subcount(sec,a,b):return sum(a < p[0] < b for p in sec)

def classify_corr(r,d,dprev):
 return 'sideways' if (r<=50.0 or (d>dprev*1.5 and r<55.0)) else 'sharp'

def impulse(arr,sec,allow=True):
 if len(arr)<6:return None
 p0,p1,p2,p3,p4,p5=arr[-6:]
 bull=p0[1]<p1[1]
 structure= ((not p0[2] and p1[2] and not p2[2] and p3[2] and not p4[2] and p5[2]) if bull
             else (p0[2] and not p1[2] and p2[2] and not p3[2] and p4[2] and not p5[2]))
 if not structure:return None
 w1=abs(p1[1]-p0[1]);w3=abs(p3[1]-p2[1]);w5=abs(p5[1]-p4[1]);
 if w1==0:return None
 w2r=ratio(p0[1],p1[1],p2[1])*100;w4r=ratio(p2[1],p3[1],p4[1])*100
 if not ((p2[1]>p0[1] if bull else p2[1]<p0[1]) and not(w3<w1 and w3<w5)
         and (p4[1]>p1[1] if bull else p4[1]<p1[1])):return None
 alt=classify_corr(w2r,p2[0]-p1[0],p1[0]-p0[0]) != classify_corr(w4r,p4[0]-p3[0],p3[0]-p2[0])
 sw=[subcount(sec,x[0],y[0]) for x,y in zip([p0,p1,p2,p3,p4],[p1,p2,p3,p4,p5])]
 sw_ok=sw[0]>=4 and sw[1]>=2 and sw[2]>=4 and sw[3]>=2 and sw[4]>=4
 conf=.40
 conf+=.10 if abs(w2r-61.8)<10 else (.07 if abs(w2r-50)<10 else 0)
 conf+=.12 if abs(w3/w1-1.618)<.3 else (.05 if w3/w1>1 else 0)
 conf+=.08 if abs(w4r-38.2)<10 else 0
 conf+=.10 if alt else 0
 conf+=.10 if sw_ok else 0
 conf+=.05 if w3>w1 and w3>w5 else 0
 conf+=.05 if abs(w5/w1-1.0)<.3 else 0
 return dict(t='impulse',conf=min(1,conf),bull=bull,end=p5[0])

def diagonal(arr):
 if len(arr)<6:return None
 p0,p1,p2,p3,p4,p5=arr[-6:];bull=p0[1]<p1[1]
 structure=((not p0[2] and p1[2] and not p2[2] and p3[2] and not p4[2] and p5[2]) if bull
            else (p0[2] and not p1[2] and p2[2] and not p3[2] and p4[2] and not p5[2]))
 if not structure:return None
 w1=abs(p1[1]-p0[1]);w3=abs(p3[1]-p2[1]);w5=abs(p5[1]-p4[1])
 overlap=p4[1]<p1[1] if bull else p4[1]>p1[1]
 r1=p2[1]>p0[1] if bull else p2[1]<p0[1]
 r2=not(w3<w1 and w3<w5)
 w3b=p3[1]>p1[1] if bull else p3[1]<p1[1]
 w4h=p4[1]>p2[1] if bull else p4[1]<p2[1]
 s13=(p3[1]-p1[1])/max(p3[0]-p1[0],1)
 s24=(p4[1]-p2[1])/max(p4[0]-p2[0],1)
 converge=s24>s13 if bull else s24<s13
 contracting=converge and w3<w1 and w5<w3
 expanding=not converge and w3>w1 and w5>w3
 if not(overlap and r1 and r2 and w3b and w4h and (contracting or expanding)):return None
 return dict(t='diagonal' if contracting else 'expanding diagonal',conf=.65 if contracting and w5<w1 else .6,bull=bull,end=p5[0])

def triangle(arr):
 if len(arr)<6:return None
 p0,p1,p2,p3,p4,p5=arr[-6:]
 valid=(p0[2] and not p1[2] and p2[2] and not p3[2] and p4[2] and not p5[2]
        and p2[1]<p0[1] and p4[1]<p2[1] and p3[1]>p1[1] and p5[1]>p3[1]) or (
         not p0[2] and p1[2] and not p2[2] and p3[2] and not p4[2] and p5[2]
         and p2[1]>p0[1] and p4[1]>p2[1] and p3[1]<p1[1] and p5[1]<p3[1])
 if not valid:return None
 lens=[abs(b[1]-a[1]) for a,b in zip([p0,p1,p2,p3,p4],[p1,p2,p3,p4,p5])]
 contract=sum(b<a for a,b in zip(lens,lens[1:]))
 if contract<3:return None
 conf=.55 if contract==4 else .45
 conf+=.1 if abs(ratio(p3[1],p4[1],p5[1])-.618)<.15 else 0
 r=ratio(p0[1],p1[1],p2[1]);conf+=.08 if .55<=r<=.85 else 0
 return dict(t='triangle',conf=min(conf,1),bull=not p0[2],end=p5[0])

def zigzag(arr):
 if len(arr)<4:return None
 p0,p1,p2,p3=arr[-4:];bear=p0[1]>p1[1]
 valid=(p0[2] and not p1[2] and p2[2] and not p3[2] and p2[1]<p0[1] and p3[1]<=p1[1]) if bear else (
        not p0[2] and p1[2] and not p2[2] and p3[2] and p2[1]>p0[1] and p3[1]>=p1[1])
 if not valid:return None
 a=abs(p1[1]-p0[1]);r=ratio(p0[1],p1[1],p2[1]);cr=abs(p3[1]-p2[1])/a if a>0 else 0
 if not(.3<=r<.8):return None
 conf=.45
 conf+=.15 if abs(r-.618)<.1 else (.1 if abs(r-.5)<.1 else 0)
 conf+=.12 if abs(cr-1.0)<.2 else (.08 if abs(cr-1.618)<.25 else 0)
 return dict(t='zigzag',conf=min(conf,1),bull=not bear,end=p3[0])

def flat(arr):
 if len(arr)<4:return None
 p0,p1,p2,p3=arr[-4:];bear=p0[1]>p1[1]
 valid=(p0[2] and not p1[2] and p2[2] and not p3[2]) if bear else (
        not p0[2] and p1[2] and not p2[2] and p3[2])
 if not valid:return None
 a=abs(p1[1]-p0[1]);r=ratio(p0[1],p1[1],p2[1]);cr=abs(p3[1]-p2[1])/a if a>0 else 0
 if not (.8-1e-12<=r<=1.45 and a>0):return None
 conf=.40
 conf+=.15 if .9<=r<=1.05 else (.1 if 1.05<r<=1.38 else 0)
 conf+=.12 if abs(cr-1.0)<.2 else (.08 if 1.0<cr<1.8 else 0)
 cnear=(p3[1]<=p1[1]+.1*a) if bear else (p3[1]>=p1[1]-.1*a)
 beyond=(p3[1]<p1[1]) if bear else (p3[1]>p1[1])
 conf+=.08 if cnear else 0
 typ=('expanded flat' if beyond else 'running flat') if r>1.05 else 'regular flat'
 return dict(t=typ,conf=min(conf,1),bull=not bear,end=p3[0])

def findbest(arr,sec):
 if len(arr)<4:return None
 best=None
 for x in [impulse(arr,sec),diagonal(arr),triangle(arr),zigzag(arr),flat(arr)]:
  if x is not None and (best is None or x['conf']>best['conf']):best=x
 if best is None and len(arr)>=7:best=impulse(arr[:-1],sec)
 return best

def reconstruct(df,pri_len=13,sec_len=5,atr_len=14,pri_mult=3,sec_mult=1.5):
 high=df.high.to_numpy(float);low=df.low.to_numpy(float);close=df.close.to_numpy(float)
 atr=atr_wilder(high,low,close,atr_len);ph,pl=pine_pivots(high,low,pri_len);sh,sl=pine_pivots(high,low,sec_len)
 pri=[];sec=[];ctf=None;n=len(df)
 push=np.zeros(n,bool);changed=np.zeros(n,bool);w3=np.zeros(n,np.int8);wc=np.zeros(n,np.int8);signal=np.zeros(n,np.int8)
 ptype=np.empty(n,dtype=object); ptype[:]=None
 pri_at=[]
 def add(arr,price,hi,ix,val,mult):
  if not arr:arr.append([ix,price,hi]);return 2
  last=arr[-1]
  if last[2]!=hi:
   if np.isfinite(val) and abs(price-last[1])>=val*mult-1e-10:
    arr.append([ix,price,hi]);return 2
  elif (price>last[1] if hi else price<last[1]):
   arr[-1]=[ix,price,hi];return 1
  return 0
 for i in range(n):
  if not np.isfinite(atr[i]):continue
  ch=False;new=False
  for yes,hi in ((ph[i],True),(pl[i],False)):
   if yes:
    code=add(pri,high[i-pri_len] if hi else low[i-pri_len],hi,i-pri_len,atr[i],pri_mult)
    ch |= code>0;new |= code==2
  for yes,hi in ((sh[i],True),(sl[i],False)):
   if yes:add(sec,high[i-sec_len] if hi else low[i-sec_len],hi,i-sec_len,atr[i],sec_mult)
  if len(pri)>100:pri.pop(0)
  if len(sec)>300:sec.pop(0)
  changed[i]=ch;push[i]=new
  if ch:ctf=findbest(pri,sec)
  ptype[i]=ctf['t'] if ctf else 'none'
  if new and len(pri)>=3:
   a,b,c=pri[-3:];bull=a[1]<b[1]
   r=ratio(a[1],b[1],c[1]);holds=(c[1]>a[1]) if bull else (c[1]<a[1])
   sub=subcount(sec,a[0],b[0]);w3ok=.382<=r<=.786 and holds and sub>=4
   if w3ok:
    # Pine htfAllows: do not trade if HTF1 bias explicitly opposite.
    bias = df['Audit htf1 trend bias'].iat[i] if 'Audit htf1 trend bias' in df else 0
    if bias !=(-1 if bull else 1):w3[i]=1 if bull else -1
   pat_here=ctf is not None and ctf['end']==pri[-1][0]
   # Use the TV-exported MTF confidence and alignment as *inputs*, not reconstructed MTF.
   confidence=float(df['Audit CTF MTF confidence'].iat[i]) if 'Audit CTF MTF confidence' in df else (ctf['conf'] if ctf else 0)
   conflicting= df['Audit alignment label'].iat[i]==-2 if 'Audit alignment label' in df else False
   if ctf and pat_here and confidence>=.35 and not conflicting and (ctf['t']=='zigzag' or 'flat' in ctf['t']):
    wc[i]= -1 if ctf['bull'] else 1
   signal[i]=(1 if w3[i]==1 or wc[i]==1 else -1 if w3[i]==-1 or wc[i]==-1 else 0)
  if ch:pri_at.append({'index':i,'primary_count':len(pri),'type':ctf['t'] if ctf else 'none','confidence':ctf['conf'] if ctf else None,'pushed':new})
 return {'push':push,'changed':changed,'w3':w3,'wc':wc,'signal':signal,'pattern_type':ptype,'pivot_log':pri_at}

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--csv',default='/mnt/data/BINANCE_SOLUSDT, 5 (5)(1).csv');p.add_argument('--out',default='/mnt/data/ew_pine_signal_match.csv');args=p.parse_args()
 d=pd.read_csv(args.csv);res=reconstruct(d)
 start=100;target=d['Long Entry'].to_numpy().astype(int)-d['Short Entry'].to_numpy().astype(int)
 print('renko rows',len(d));print('primary',score(d['Audit primary pushed'].to_numpy()!=0,res['push'],start))
 print('combined LONG',score(target==1,res['signal']==1,start));print('combined SHORT',score(target==-1,res['signal']==-1,start));
 print('Raw W3',{'long':int(sum(res['w3']==1)),'short':int(sum(res['w3']==-1))});print('Wave C',{'long':int(sum(res['wc']==1)),'short':int(sum(res['wc']==-1))});
 print('pattern crosstab when primary changed')
 v=d.iloc[100:].copy();v['our_pattern']=res['pattern_type'][100:];v=v[np.asarray(res['changed'][100:])];print(pd.crosstab(v['our_pattern'],v['Audit CTF pattern']).to_string())
 o=d[['time','open','high','low','close','Long Entry','Short Entry','Audit primary pushed','Audit CTF pattern','Audit CTF raw confidence','Audit CTF MTF confidence','Audit alignment label']].copy()
 o['python_primary_pushed']=res['push'];o['python_ew_pattern']=res['pattern_type'];o['python_w3']=res['w3'];o['python_wave_c']=res['wc'];o['python_combined_signal']=res['signal'];o['is_signal_match']=target==res['signal']
 o.to_csv(args.out,index=False)