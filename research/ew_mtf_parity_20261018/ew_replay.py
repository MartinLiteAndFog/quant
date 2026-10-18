# SPDX-License-Identifier: MPL-2.0
# Derived from EW_v3_2_MTF_AUDIT.pine
# Copyright: Elliott Wave Advanced Multi-Degree Analyzer — v3.2 MTF Edition
# Python numerical port and parity investigation, 2026-10-18.
"""Numerical port of EW_v3_2_MTF_AUDIT.pine, excluding drawing/forecast UI.

HTF raw input replay is diagnostic, NOT an independent market-data backtest.
Pine source locations are documented beside each calculation group.
"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from pine_numeric import PineFloat


@dataclass
class Pivot:
    idx: int
    price: float
    high: bool


@dataclass
class Pattern:
    kind: str
    confidence: float
    bull: bool
    end: int


@dataclass
class Context:
    phase: str = "unknown"
    kind: str = "none"
    bull: bool = True
    confidence: float = 0.
    count: int = 0
    bias: int = 0


@dataclass
class Config:
    primary: int = 13
    secondary: int = 5
    atr_length: int = 14
    primary_mult: float = 3.
    secondary_mult: float = 1.5
    htf_mult: float = 2.5
    weight: float = .20
    require_sub: bool = True
    mtf: bool = True


def add_pivot(arr, price, high, idx, atr, mult):
    """Pine addZigPivot, lines 342–362; ATR-mode only."""
    price, atr = PineFloat(price), PineFloat(atr)
    if not arr:
        arr.append(Pivot(idx, price, high))
        return 2
    last = arr[-1]
    if last.high != high:
        if math.isfinite(atr) and abs(price - last.price) >= atr * mult:
            arr.append(Pivot(idx, price, high))
            return 2
    elif (price > last.price if high else price < last.price):
        arr[-1] = Pivot(idx, price, high)
        return 1
    return 0


def ratio(a, b, c):
    return abs(c-b)/abs(b-a) if b != a else 0.


def subcount(sec, a, b):
    return sum(a < p.idx < b for p in sec)


def corrective(kind):
    return kind in ("zigzag", "triangle") or "flat" in kind


def impulse(pivots, sec):
    if len(pivots) < 6:
        return None
    p = pivots[-6:]
    a,b,c,d,e,f = [v.price for v in p]
    bull = a < b
    if [v.high for v in p] != [not bull,bull,not bull,bull,not bull,bull]:
        return None
    w1,w3,w5 = abs(b-a),abs(d-c),abs(f-e)
    if not ((c > a and e > b) if bull else (c < a and e < b)):
        return None
    if (w3 < w1 and w3 < w5) or w1 == 0:
        return None
    r2,r4 = ratio(a,b,c)*100,ratio(c,d,e)*100
    dur = [p[i+1].idx-p[i].idx for i in range(5)]
    sharp2 = r2 > 50 and not (dur[1] > dur[0]*1.5 and r2 < 55)
    sharp4 = r4 > 50 and not (dur[3] > dur[2]*1.5 and r4 < 55)
    sw = all(subcount(sec,p[i].idx,p[i+1].idx) >= (4 if i%2 == 0 else 2) for i in range(5))
    conf = .40
    conf += .10 if abs(r2-61.8)<10 else .07 if abs(r2-50)<10 else 0
    conf += .12 if abs(w3/w1-1.618)<.3 else .05 if w3/w1>1 else 0
    conf += .08 if abs(r4-38.2)<10 else 0
    conf += .10 if sharp2 != sharp4 else 0
    conf += .10 if sw else 0
    conf += .05 if w3>w1 and w3>w5 else 0
    conf += .05 if abs(w5/w1-1)<.3 else 0
    return Pattern("impulse",min(conf,1),bull,p[-1].idx)


def best_pattern(pivots, sec=()):
    """Pine lines 421–735. Preserve candidate order and strict > ties."""
    n = len(pivots)
    if n < 4:
        return None
    candidates = [impulse(pivots,sec)]
    if n >= 6:
        p=pivots[-6:]; a,b,c,d,e,f=[v.price for v in p]; bull=a<b
        structure=[v.high for v in p] == [not bull,bull,not bull,bull,not bull,bull]
        w1,w3,w5=abs(b-a),abs(d-c),abs(f-e)
        slope13=(d-b)/max(p[3].idx-p[1].idx,1)
        slope24=(e-c)/max(p[4].idx-p[2].idx,1)
        conv=slope24>slope13 if bull else slope24<slope13
        contract=conv and w3<w1 and w5<w3
        expand=not conv and w3>w1 and w5>w3
        rules=(e<b and c>a and d>b and e>c) if bull else (e>b and c<a and d<b and e<c)
        if structure and rules and not (w3<w1 and w3<w5) and (contract or expand):
            candidates.append(Pattern("diagonal" if contract else "expanding diagonal",.65 if contract and w5<w1 else .60,bull,p[-1].idx))
        tri=(c<a and e<c and d>b and f>d) if p[0].high else (c>a and e>c and d<b and f<d)
        alternating=all(p[i].high != p[i+1].high for i in range(5))
        lengths=[abs(p[i+1].price-p[i].price) for i in range(5)]
        contracts=sum(lengths[i+1]<lengths[i] for i in range(4))
        if alternating and tri and contracts>=3:
            conf=.55 if contracts==4 else .45
            conf+=.10 if abs(ratio(d,e,f)-.618)<.15 else 0
            conf+=.08 if .55<=ratio(a,b,c)<=.85 else 0
            candidates.append(Pattern("triangle",conf,not p[0].high,p[-1].idx))
    p=pivots[-4:]; a,b,c,d=[v.price for v in p]; bearish=a>b
    structure=[v.high for v in p] == [bearish,not bearish,bearish,not bearish]
    alen=abs(b-a); retr=ratio(a,b,c); cr=abs(d-c)/alen if alen else 0
    zz_structure=(c<a and d<=b) if bearish else (c>a and d>=b)
    if structure and zz_structure and .30<=retr<.80:
        conf=.45
        conf+=.15 if abs(retr-.618)<.1 else .10 if abs(retr-.5)<.1 else 0
        conf+=.12 if abs(cr-1)<.2 else .08 if abs(cr-1.618)<.25 else 0
        candidates.append(Pattern("zigzag",conf,not bearish,p[-1].idx))
    if structure and .80<=retr<=1.45 and alen>0:
        conf=.40
        conf+=.15 if .90<=retr<=1.05 else .10 if 1.05<retr<=1.38 else 0
        conf+=.12 if abs(cr-1)<.2 else .08 if 1<cr<1.8 else 0
        near=d<=b+.1*alen if bearish else d>=b-.1*alen
        beyond=d<b if bearish else d>b
        conf+=.08 if near else 0
        kind=("expanded flat" if beyond else "running flat") if retr>1.05 else "regular flat"
        candidates.append(Pattern(kind,conf,not bearish,p[-1].idx))
    valid=[c for c in candidates if c is not None]
    return max(valid,key=lambda c:PineFloat(c.confidence)) if valid else impulse(pivots[:-1],sec) if n>=7 else None


def analyze_htf(pivots):
    """Pine lines 766–845, including correction direction inversion."""
    ctx=Context(count=len(pivots))
    pat=best_pattern(pivots)
    if pat:
        ctx.phase="corrective" if corrective(pat.kind) else "motive"
        ctx.kind=pat.kind; ctx.bull=not pat.bull if corrective(pat.kind) else pat.bull
        ctx.confidence=pat.confidence; ctx.bias=1 if ctx.bull else -1
    elif len(pivots)>=4:
        p=pivots[-6:]; highs=[v.price for v in p if v.high]; lows=[v.price for v in p if not v.high]
        up=sum(v[i]>v[i-1] for v in (highs,lows) for i in range(1,len(v)))
        total=len(highs)+len(lows)-2; down=total-up
        if up!=down:
            ctx.bull=up>down; ctx.bias=1 if ctx.bull else -1; r=max(up,down)/total
            ctx.phase="motive" if r>=.6 else "corrective"
            ctx.confidence=.7+(r-.8)*1.5 if r>=.8 else .4+(r-.6)*1.5 if r>=.6 else .3
        else:
            ctx.phase="corrective"; ctx.confidence=.2
        if ctx.phase=="motive" and ((ctx.bull and lows[-1]<highs[-2]) or (not ctx.bull and highs[-1]>lows[-2])):
            ctx.phase="corrective"; ctx.confidence*=.7
        ctx.kind=("trend_up" if ctx.bull else "trend_down") if ctx.phase=="motive" else "ranging" if ctx.phase=="corrective" else "none"
        ctx.confidence=min(ctx.confidence,1.)
    elif len(pivots)>=2:
        ctx.bull=pivots[-1].price>pivots[-2].price; ctx.bias=1 if ctx.bull else -1; ctx.confidence=.1
    return ctx


def alignment(pat, contexts, active):
    has=[on and h.count>=3 and h.confidence>.1 for h,on in zip(contexts,active)]
    if pat is None or not any(has):
        return 0.,-1
    corr=corrective(pat.kind); trend=not pat.bull if corr else pat.bull
    scores=[]
    for h,ok in zip(contexts,has):
        s=0.
        if ok and h.bias:
            same=trend==h.bull; s=.5 if same else -.5
            if h.phase=="motive" and same: s+=.3 if corr else .5
            elif h.phase=="corrective" and not corr: s-=.2
            s*=h.confidence
        scores.append(s)
    score=scores[0]*.6+scores[1]*.4 if all(has) else sum(scores)
    score=PineFloat(max(-1.,min(1.,score)))
    return score,2 if score>=.4 else 1 if score>=.1 else 0 if score>=-.1 else -2


def pattern_code(kind):
    return 1 if kind=="zigzag" else 2 if "flat" in kind else 3 if kind=="impulse" else 4 if "diagonal" in kind else 5 if kind=="triangle" else 0


def atr_series(frame, length):
    h,l,c=(frame[k].to_numpy(float) for k in ('high','low','close'))
    prev=np.r_[np.nan,c[:-1]]
    tr=np.fmax(h-l,np.fmax(abs(h-prev),abs(l-prev)))
    out=np.full(len(tr),np.nan)
    if len(tr)>=length:
        out[length-1]=np.mean(tr[:length])
        for i in range(length,len(tr)): out[i]=out[i-1]*(1-1/length)+tr[i]/length
    return out


def pivot_series(values, length, high):
    """Rightmost tie wins: equal older extreme allowed, equal newer rejected."""
    out=np.full(len(values),np.nan)
    for i in range(2*length,len(values)):
        j=i-length; v=values[j]; left=values[j-length:j]; right=values[j+1:i+1]
        ok=(np.all(v>=left) and np.all(v>right)) if high else (np.all(v<=left) and np.all(v<right))
        if ok: out[i]=v
    return out


def replay_export(frame, config=Config()):
    """Replay CTF from OHLC and HTF from exported raw values; no target leakage.

    Audit new-period fields are supplied diagnostic inputs. Reference trend,
    confidence, patterns, signals and pivot counts are NEVER read here.
    """
    c=config; atr=atr_series(frame,c.atr_length)
    sources=[pivot_series(frame[k].to_numpy(float),n,hi) for n in (c.primary,c.secondary) for k,hi in [('high',True),('low',False)]]
    primary=[]; secondary=[]; hp=[[],[]]; hc=[Context(),Context()]; pat=None
    raw=0.; conf=0.; score=0.; label=-1; rows=[]
    cols={k:frame[k].to_numpy() for k in frame.columns if k.startswith('Audit htf')}
    for i in range(len(frame)):
        changed=False; pushed=False
        for s in range(4):
            price=sources[s][i]
            if not math.isfinite(price): continue
            pri=s<2; arr=primary if pri else secondary
            code=add_pivot(arr,price,s%2==0,i-(c.primary if pri else c.secondary),atr[i],c.primary_mult if pri else c.secondary_mult)
            if pri: changed=changed or code>0; pushed=pushed or code==2
        if len(primary)>100: primary.pop(0)
        if len(secondary)>300: secondary.pop(0)
        hchanged=[False,False]; active=[]
        for j in range(2):
            pre=f'Audit htf{j+1} '; active.append(c.mtf and bool(cols[pre+'active'][i]))
            if active[j] and cols[pre+'new period'][i]:
                for name,hi in [('pivot high raw',True),('pivot low raw',False)]:
                    price=cols[pre+name][i]
                    if math.isfinite(price):
                        code=add_pivot(hp[j],price,hi,int(cols[pre+'pivot idx'][i]),cols[pre+'ATR'][i],c.htf_mult)
                        hchanged[j]=hchanged[j] or code>0
                if len(hp[j])>50: hp[j].pop(0)
        if changed:
            pat=best_pattern(primary,secondary); raw=pat.confidence if pat else 0.; conf=raw
        for j in range(2):
            if hchanged[j]: hc[j]=analyze_htf(hp[j])
        if changed or any(hchanged):
            score,label=alignment(pat,hc,active)
            conf=max(.05,min(1.,raw+score*c.weight)) if raw>0 and c.mtf and c.weight!=0 else raw
        here=pat is not None and bool(primary) and pat.end==primary[-1].idx
        long=short=xb=xs=False
        if pushed and len(primary)>=3:
            p0,p1,p2=primary[-3:]; bull=p0.price<p1.price; r=ratio(p0.price,p1.price,p2.price)
            holds=p2.price>p0.price if bull else p2.price<p0.price
            sw=not c.require_sub or subcount(secondary,p0.idx,p1.idx)>=4
            allows=not active[0] or hc[0].bias!=(-1 if bull else 1)
            if .382<=r<=.786 and holds and sw and allows: long=bull; short=not bull
            if here and conf>=.35:
                if (pat.kind=='zigzag' or 'flat' in pat.kind) and not (c.mtf and label==-2):
                    long=long or not pat.bull; short=short or pat.bull
                if pat.kind=='impulse' or 'diagonal' in pat.kind: xb=pat.bull; xs=not pat.bull
        row={'Long Entry':int(long),'Short Entry':int(short),'Wave 5 Exit (bullish impulse)':int(xb),'Wave 5 Exit (bearish impulse)':int(xs),
             'Audit primary pushed':int(pushed),'Audit pattern here':int(here),'Audit CTF raw confidence':raw,'Audit CTF MTF confidence':conf,
             'Audit CTF pattern':pattern_code(pat.kind) if pat else 0,'Audit alignment score':score,'Audit alignment label':label}
        for j,h in enumerate(hc):
            pre=f'Audit htf{j+1} '
            row.update({pre+'changed':int(hchanged[j]),pre+'pivot count':h.count,pre+'trend bias':h.bias,pre+'phase':1 if h.phase=='motive' else -1 if h.phase=='corrective' else 0,pre+'pattern':pattern_code(h.kind),pre+'confidence':h.confidence})
        rows.append(row)
    return pd.DataFrame(rows,index=frame.index)
