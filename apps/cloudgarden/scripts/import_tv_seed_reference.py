"""Offline TradingView Strategy Builder export importer for Cloudgarden research.

Requires artifact_tool. Read-only source XLSX. Excludes the entire alerts section
(including potentially live webhook tokens). Produces immutable reference seeds,
per-trade fixtures and independently checkable reconciliation diagnostics.
Never sends an order and never claims to reproduce the third-party signal code.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from artifact_tool import Blob, SpreadsheetFile

HERE = Path(os.environ.get('TV_SEED_SOURCE_DIR', Path.cwd()))
XLSX = sorted(HERE.glob('Strategy_Builder*.xlsx'))
UTC = dt.timezone.utc


def iso_excel(value):
    if isinstance(value, (float, int)) and math.isfinite(value):
        return (dt.datetime(1899, 12, 30, tzinfo=UTC) + dt.timedelta(days=float(value))).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
    if value == 'Open':
        return None
    raise ValueError(f'Unsupported Excel trade date: {value!r}')


def as_number(s, title):
    try:
        v = float(s)
        if not math.isfinite(v): raise ValueError()
        return v
    except Exception as exc:
        raise ValueError(f'Invalid {title}: {s!r}') from exc


def attr(properties, key, after=None):
    matches = [v for i, (k, v) in enumerate(properties) if k == key and (after is None or i >= after)]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one {key!r}, got {len(matches)}')
    return matches[0]


def parse_one(path):
    book = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    rawprops = book.worksheets.get_item('Properties').get_range('A1:B215').values
    rawprops = [(str(k).strip(), v) for k, v in rawprops if k is not None]
    alert_start = next(i for i, (k, v) in enumerate(rawprops) if 'ALERTS' in k)
    alert_end = next(i for i, (k, v) in enumerate(rawprops) if k == 'Initial capital')
    # Alert payloads may contain real webhook auth tokens; they never enter the export.
    safe_props = rawprops[:alert_start] + rawprops[alert_end:]
    assert not any(k == 'Long Entry' and v not in ('Off', 'On') for k, v in safe_props)
    header = book.worksheets.get_item('Trades').get_range('A1:Q1').values[0]
    if header[:5] != ['Trade number', 'Type', 'Date and time', 'Signal', 'Price USDT']:
        raise ValueError('Unexpected TradingView Trades sheet schema')
    trade_rows = [r for r in book.worksheets.get_item('Trades').get_range('A2:Q1200').values if isinstance(r[0], (int, float))]
    grouped = defaultdict(list)
    for row in trade_rows: grouped[int(row[0])].append(row)
    report = book.worksheets.get_item('Trades analysis').get_range('A1:B20').values
    declared_count = int(next(v for k,v in report if k=='Total trades'))
    closed=[];open_trades=[]; mismatched=[]
    for n,rows in sorted(grouped.items()):
        if len(rows)!=2: raise ValueError(f'Trade #{n} must be entry+exit, got {len(rows)} rows')
        entr=next((r for r in rows if str(r[1]).startswith('Entry ')),None)
        ext=next((r for r in rows if str(r[1]).startswith('Exit ')),None)
        if entr is None or ext is None: raise ValueError(f'Trade #{n}: missing entry/exit')
        side=str(entr[1]).split(' ',1)[1]
        if side not in ('long','short') or ext[1]!='Exit '+side:raise ValueError(f'Trade #{n}: mismatching side')
        price=as_number(entr[4],'entry price')
        qty=as_number(entr[5],'entry quantity')
        if qty<=0 or price<=0:raise ValueError('Invalid source quantity/price')
        record={
            'number':n,'side':side,
            'entryUtc':iso_excel(entr[2]),
            'entrySignal':str(entr[3]),
            'entryPrice':price,'quantity':qty,
        }
        if ext[2]=='Open':
            record['status']='open_not_booked'
            open_trades.append(record)
            continue
        record['status']='closed_tradingview_reference'
        record['exitUtc']=iso_excel(ext[2])
        record['exitSignal']=str(ext[3])
        record['exitPrice']=as_number(ext[4],'exit price')
        record['referenceNetPnlUsdt']=as_number(ext[7],'net pnl')
        record['referenceCommissionUsdt']=as_number(ext[9],'commission')
        record['referenceReturnPct']=as_number(ext[8],'return pct')
        record['durationBricks']=int(ext[16])
        if record['entryUtc']>record['exitUtc']:raise ValueError('Time reversal in TV trade')
        if record['referenceCommissionUsdt']<0:raise ValueError('Negative commission')
        # Spreadsheet displays rounded fills/quantity. This is a diagnostic, not a
        # second PnL authority: do not silently recompute the TradingView PnL.
        sign=1 if side=='long' else -1
        gross=(record['exitPrice']-price)*qty*sign
        residual=round(gross-record['referenceCommissionUsdt']-record['referenceNetPnlUsdt'],4)
        record['displayPrecisionPnlResidualUsdt']=residual
        if abs(residual)>0.25:mismatched.append({'trade':n,'residualUsdt':residual})
        closed.append(record)
    if len(closed)!=declared_count:raise ValueError(f'Closed trades {len(closed)} != TV report {declared_count}')
    if len(open_trades)>1:raise ValueError('Multiple concurrent open trades for pyramiding=0')
    keymap={k:v for k,v in safe_props}
    sl=as_number(attr(safe_props,'SL %'),'stopLossPercent')
    dev=as_number(attr(safe_props,'Dev %'),'trailDeviationPercent')
    lev=as_number(next(v for k,v in safe_props if 'Leverage' in k),'leverage')
    order_pct=as_number(next(v for k,v in safe_props if 'Order Size' in k),'order percent')
    return {
      'id':'tvseed-'+hashlib.sha256(path.read_bytes()).hexdigest()[:12],
      'displayName':path.stem.replace('Strategy_Builder_@Tradingade_','TV '),
      'originalFilename':path.name,
      'originalXlsxSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
      'instrument':{'symbol':str(keymap['Symbol']),'chartType':str(keymap['Chart type']),
        'sourceTimeframe':str(keymap['Timeframe']), 'renkoBoxSizeQuote':None,
        'tickSize':as_number(keymap['Tick size'],'tick size')},
      'originalSettings':{'safePropertyRows':safe_props,
        'strategyEngine':'Tradingade Strategy Builder',
        'longEntrySource':str(keymap['Long¹']),
        'shortEntrySource':str(keymap['Short¹']),
        'longOscillatorSource':str(keymap['L¹']),
        'shortOscillatorSource':str(keymap['S¹']),
        'oscillatorLongEnabled':next(v for k,v in safe_props if k=='🟢¹ -')=='On',
        'oscillatorShortEnabled':next(v for k,v in safe_props if k=='🔴¹ -')=='On',
        'stopLossPercent':sl,'trailDeviationPercent':dev,
        'takeProfitMode':str(keymap['💰 Take Profit Type']),
        'leverage':lev,'orderSizePercent':order_pct,
        'initialCapitalUsdt':as_number(keymap['Initial capital'],'capital'),
        'commissionPercent':as_number(keymap['Commission'],'commission'),
        'slippage':str(keymap['Slippage']),
        'executionDelay':str(keymap['Order execution delay']),
        'intrabarModel':str(keymap['Bar detalization']),
        'execSchedule':str(keymap['Script execution'])},
      'candidateGenes':{
         'stopLossBps':round(sl*100,4),
         'trailingDeviationBps':round(dev*100,4),
         'leverage':lev,
         'orderSizePercent':order_pct,
         'entryLookbackBars':None,
         'ewPivotLength':None,
         'rsiFormulaVersion':None,
      },
      'eligibility':{'referenceTradeReplay':True,'independentSignalParity':False,
          'reason':'External Elliott Wave Detector Pro and MTF RSI implementations/inputs unverified',
          'liveOrderEligible':False},
      'tradingRange':keymap['Trading range'],
      'referenceTradeCount':len(closed),
      'openPositionCount':len(open_trades),
      'roundedPnlResidualAlerts':mismatched,
      'trades':{'closed':closed,'open':open_trades},
    }


def main():
    if not XLSX:raise ValueError('No XLSX files found')
    seeds=[parse_one(p) for p in XLSX]
    catalog={
       'schemaVersion':1,
       'claim':'tradingview_trade_reference_only_not_signal_parity',
       'source':'User-supplied TradingView Strategy Builder Excel exports 2026-10-17/18',
       'venueDataNote':'BINANCE SOL 1m parquet/source is separate; no matching BTC 1m source provided',
       'secretsExcluded':True,
       'seeds':seeds,
    }
    outfile=Path(os.environ.get('TV_SEED_OUTPUT_PATH', str(HERE/'quant_tradingview_first_plants_reference_2026-11-02.json')))
    outfile.write_text(json.dumps(catalog,ensure_ascii=False,indent=2,default=str)+'\n')
    totals={'variants':len(seeds),'closedTrades':sum(s['referenceTradeCount'] for s in seeds),
          'openPositions':sum(s['openPositionCount'] for s in seeds),
          'pnlDiagnosticAlerts':sum(len(s['roundedPnlResidualAlerts']) for s in seeds)}
    print('TOTALS',totals)
    for s in seeds: print(s['id'],s['originalSettings']['leverage'],s['referenceTradeCount'],s['openPositionCount'],s['originalFilename'])
    return outfile,totals

if __name__=='__main__': main()