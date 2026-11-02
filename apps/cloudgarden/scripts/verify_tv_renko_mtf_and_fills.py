#!/usr/bin/env python3
"""Offline TradingView Renko MTF RSI and reference-trade price audit.

Inputs are user-supplied private Binance 1m gzip (pipe-delimited), TradingView
Renko CSV and sanitized Cloudgarden v2 seed reference JSON. Do not commit the
inputs or outputs. This reproduces *historical plotted values*, then evaluates a
separate confirmed-only 30m gate and reprices fixed reference event times at the
first Binance 1m open after the originating 5m source candle closes.

No strategy signals are inferred from future reference trades; this is NOT a
validated forward/backtest/live execution engine.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

BOX = 0.1
RSI_LENGTH = 14
DEFAULT_H1_SPLIT = 1788739200  # 2026-09-07T00:00:00Z
UTC = dt.timezone.utc


def utc_seconds(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp())


def bars_from_rows(rows: Iterable[tuple[int, float, float]], seconds: int) -> dict[int, tuple[float, int]]:
    """Use completed, consecutively present 1m closes only. Rows: ts,open,close."""
    if seconds < 60 or seconds % 60:
        raise ValueError('The source timeframe must be a positive multiple of 60 seconds')
    raw: dict[int, tuple[float, int]] = {}
    last = -1
    for ts, _op, close in rows:
        if not (ts % 60 == 0 and ts > last and math.isfinite(close) and close > 0):
            raise ValueError('Raw source must have strictly increasing UTC minute timestamps and positive prices')
        last = ts
        bucket = ts // seconds * seconds
        count = raw.get(bucket, (0.0, 0))[1] + 1
        raw[bucket] = (close, count)
    n = seconds // 60
    return {ts: (close, count) for ts, (close, count) in raw.items() if count == n}


def load_raw(path: Path, start_seconds: int):
    rows: list[tuple[int, float, float]] = []
    minute_opens: dict[int, float] = {}
    last_ts = None
    total = 0
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            columns = line.rstrip().split('|')
            if len(columns) < 5:
                raise ValueError('Invalid Binance gzip minute candle layout')
            ts = int(columns[0]); total += 1
            if ts < start_seconds:
                continue
            op = float(columns[1]); close = float(columns[4])
            if last_ts is not None and ts <= last_ts:
                raise ValueError('Source timestamp duplicate/out of order')
            if ts % 60 or not (math.isfinite(op) and math.isfinite(close) and op > 0 and close > 0):
                raise ValueError('Bad minute timestamp/open/close')
            rows.append((ts, op, close))
            minute_opens[ts] = op
            last_ts = ts
    if not rows:
        raise ValueError('No minute candles at/after requested warmup')
    return rows, minute_opens, {'total_raw_minutes': total,
                                 'loaded_minutes': len(rows),
                                 'latest_loaded_utc': dt.datetime.fromtimestamp(rows[-1][0], UTC).isoformat()}


def construct_traditional_renko(src: Iterable[tuple[int, float]], box: float = BOX,
                                  source_period_seconds: int = 1800):
    """Traditional close-source Renko: one-box continuation, two-box reversal.

    Each element is (chart_timestamp_ms, brick_close, source_available_ms).
    The source timestamp is the *underlying period opening label*.
    """
    src = list(src)
    if not src:
        return []
    if box <= 0:
        raise ValueError('box must be positive')
    anchor = round(math.floor((src[0][1] + 1e-10) / box) * box, 12)
    level = anchor
    trend = 0
    out: list[tuple[int, float, int]] = []
    tol = box * 1e-9
    for ts, price in src:
        if not (math.isfinite(price) and price > 0):
            raise ValueError('Bad source close')
        ms = ts * 1000
        end_ms = (ts + source_period_seconds) * 1000
        emitted = 0
        def emit(d: int, close: float) -> None:
            nonlocal emitted, trend, level
            level = round(close, 12)
            out.append((ms + emitted, level, end_ms))
            emitted += 1
            trend = d
        if trend >= 0:
            if price >= level + box - tol:
                while price >= level + box - tol:
                    emit(1, level + box)
            elif trend == 1 and price <= level - 2 * box + tol:
                emit(-1, level - 2 * box)
                while price <= level - box + tol:
                    emit(-1, level - box)
            elif trend == 0 and price <= level - box + tol:
                while price <= level - box + tol:
                    emit(-1, level - box)
        else:
            if price <= level - box + tol:
                while price <= level - box + tol:
                    emit(-1, level - box)
            elif price >= level + 2 * box - tol:
                emit(1, level + 2 * box)
                while price >= level + box - tol:
                    emit(1, level + box)
    return out


def wilder_rsi(prices: list[float], length: int = RSI_LENGTH) -> list[float]:
    if length < 1:
        raise ValueError('RSI length must be positive')
    n = len(prices)
    result = [float('nan')] * n
    if n <= length:
        return result
    changes = [prices[i] - prices[i - 1] for i in range(1, n)]
    gain = sum(max(d, 0) for d in changes[:length]) / length
    loss = sum(max(-d, 0) for d in changes[:length]) / length
    def compute(g: float, l: float) -> float:
        if l == 0:
            return 100.0 if g > 0 else 50.0
        return 100 - 100 / (1 + g / l)
    result[length] = compute(gain, loss)
    for i in range(length + 1, n):
        change = changes[i - 1]
        gain = (gain * (length - 1) + max(change, 0)) / length
        loss = (loss * (length - 1) + max(-change, 0)) / length
        result[i] = compute(gain, loss)
    return result


def renko_rsi_series(complete_bars: dict[int, tuple[float, int]], minutes: int,
                     hourly_before: dict[int, tuple[float, int]] | None = None,
                     split_seconds: int = DEFAULT_H1_SPLIT):
    src = sorted((bucket, v[0]) for bucket, v in complete_bars.items()
                 if hourly_before is None or bucket >= split_seconds)
    if hourly_before is not None:
        src = sorted([(bucket, v[0]) for bucket, v in hourly_before.items() if bucket < split_seconds] + src)
    # Mixed 5m H1 history: source confirmation is 60m before the split, 5m after.
    if hourly_before is not None:
        # Single state across prefix/suffix is important: do NOT reset Renko on split.
        # Build using a unified function with period determined per source timestamp.
        out = construct_traditional_renko_mixed(src, split_seconds, BOX)
    else:
        out = construct_traditional_renko(src, BOX, source_period_seconds=minutes * 60)
    values = wilder_rsi([brick[1] for brick in out])
    return [(ts, rsi, end) for (ts, _, end), rsi in zip(out, values) if math.isfinite(rsi)]


def construct_traditional_renko_mixed(src: list[tuple[int, float]], split_seconds: int, box: float):
    # Reuses the exact same rules while attaching per-source period-ending availability.
    # build with a dummy constant window, then map each emitted chart timestamp back to
    # its original period label (milliseconds offset is less than 1000).
    rows = construct_traditional_renko(src, box=box, source_period_seconds=300)
    return [(ts, close, (ts // 1000 + (3600 if ts // 1000 < split_seconds else 300)) * 1000)
            for ts, close, _ in rows]


def series_lookup(rows: list[tuple[int, float, int]], known_only: bool):
    """For confirmed-only mode, release all HTF bricks at the source candle END.

    Multiple bricks per source period are ordered; the last observation becomes
    available at close. This is a strict, deterministic completed-source policy.
    """
    points = [(end if known_only else ts, i, value) for i, (ts, value, end) in enumerate(rows)]
    points.sort(key=lambda x: (x[0], x[1]))
    return [p[0] for p in points], [p[2] for p in points]


def lookup(times: list[int], values: list[float], timestamp_ms: int) -> float | None:
    i = bisect.bisect_right(times, timestamp_ms) - 1
    return values[i] if i >= 0 else None


def read_tv(path: Path):
    result = []
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            stamp = round(float(row['time']) * 1000)
            result.append((stamp, {minute: float(row[f'{minute}m']) for minute in (5, 15, 30)}))
    return result


def entry_time_ms(value: str):
    return round(dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() * 1000)


def asof_first_open(open_by_minute: dict[int, float], reference_event_ms: int, source_minutes: int = 5):
    """First real minute open at the END of the 5m originating source candle.

    This is an explicit research assumption. It is NOT the TV simulated fill.
    """
    bucket_ms = source_minutes * 60 * 1000
    available_ms = ((reference_event_ms // bucket_ms) + 1) * bucket_ms
    stamp = available_ms // 1000
    return stamp, open_by_minute.get(stamp)


def audit(raw_gz: Path, tv_csv: Path, seed_manifest: Path, out_dir: Path,
          warmup: int, split: int):
    rows, minute_opens, meta = load_raw(raw_gz, warmup)
    candles = {minutes: bars_from_rows(rows, 60 * minutes) for minutes in (5, 15, 30, 60)}
    rsi_streams = {
        5: renko_rsi_series(candles[5], 5, hourly_before=candles[60], split_seconds=split),
        15: renko_rsi_series(candles[15], 15),
        30: renko_rsi_series(candles[30], 30),
    }
    chart = {period: series_lookup(stream, False) for period, stream in rsi_streams.items()}
    confirmed = {period: series_lookup(stream, True) for period, stream in rsi_streams.items()}
    latest_full_30 = max(candles[30])
    # The TV series is a historical chart plot. Only compare when 30m raw source is complete.
    cutoff_ms = (latest_full_30 + 1800) * 1000 - 1
    summary = {'audit_version': 1,
               'claim': 'offline_historical_chart_reconstruction_and_reference_time_stress_test_not_live_backtest',
               'input_source': {'market': str(raw_gz.name), 'chart': str(tv_csv.name),
                                'seeds': str(seed_manifest.name)},
               'meta': meta, 'comparison_cutoff_utc': dt.datetime.fromtimestamp(cutoff_ms / 1000, UTC).isoformat(),
               'source_mtf_semantics': 'independent_5_15_30_minute_renko_feeds',
               'causal_gate_assumption': '30m_Renko_RSI_known_only_after_complete_source_30m_bar;5m_signal_known_after_complete_5m_bar',
               'reference_trade_reprice_assumption': 'TV_entry_exit_event_time_fixed;first_available_next_1m_open;no_ledger_or_position_resimulation',
               'periods': {}, 'seed_results': {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_rows = []
    compared = defaultdict(list)
    for stamp, original in read_tv(tv_csv):
        if stamp > cutoff_ms:
            continue
        record = {'chart_timestamp_utc': dt.datetime.fromtimestamp(stamp / 1000, UTC).isoformat(),
                  'ts_ms': stamp}
        for period in (5, 15, 30):
            val = lookup(*chart[period], stamp)
            error = abs(original[period] - val) if val is not None else None
            if error is not None:
                compared[period].append(error)
            record[f'tv_{period}m'] = original[period]
            record[f'ours_{period}m'] = val
            record[f'error_{period}m'] = error
        chart_rows.append(record)
    for period in (5, 15, 30):
        errors = compared[period]
        summary['periods'][f'{period}m'] = {'rows': len(errors),
                   'exact_tolerance_1e_6': sum(e < 1e-6 for e in errors),
                   'mae_rsi_points': statistics.mean(errors) if errors else None,
                   'max_abs_error': max(errors) if errors else None,
                   'source_bricks': len(rsi_streams[period])}
    write_csv(out_dir / 'tradingview_mtf_renko_rsi_rows.csv', chart_rows)
    v2 = json.loads(seed_manifest.read_text(encoding='utf-8'))
    if v2.get('schemaVersion') != 2 or v2.get('sourceTimeZone') != 'Europe/Berlin' or v2.get('secretsExcluded') is not True:
        raise ValueError('Only the sanitized, timezone-corrected v2 reference manifest is supported')
    gate_rows, repriced_rows = [], []
    for seed in v2['seeds']:
        if seed['instrument']['symbol'] != 'BINANCE:SOLUSDT':
            continue  # No matching KuCoin or BTC raw-minute feed in this audit.
        setting = seed['originalSettings']
        if any(type(setting.get(k)) is not bool for k in ('oscillatorLongEnabled', 'oscillatorShortEnabled')):
            raise ValueError('RSI gate flags must be parsed booleans, not truthy strings')
        if not (isinstance(setting.get('commissionPercent'), (int, float))
                and 0 <= setting['commissionPercent'] <= 5):
            raise ValueError('Invalid commission percent')
        entries = [*seed['trades']['closed'], *seed['trades']['open']]
        per_seed = {'complete_reference_trades': len(seed['trades']['closed']),
                    'reference_entry_count': len(entries), 'entry_rsi_checked': 0,
                    'chart_gate_pass': 0, 'confirmed_only_gate_pass': 0,
                    'exits_real_reprice_count': 0, 'exits_without_underlying': 0,
                    'exits_same_available_minute': 0,
                    'real_reprice_reference_pnl_usdt': 0.0,
                    'real_reprice_counterfactual_pnl_usdt': 0.0}
        for trade in entries:
            ts = entry_time_ms(trade['entryUtc'])
            if ts > cutoff_ms:
                continue
            on = bool(setting['oscillatorLongEnabled']) if trade['side'] == 'long' else bool(setting['oscillatorShortEnabled'])
            chart_rsi = lookup(*chart[30], ts)
            confirmed_rsi = lookup(*confirmed[30], ts + 300000)
            if chart_rsi is None or confirmed_rsi is None:
                continue
            passes = lambda value: (not on) or (value > 55 if trade['side'] == 'long' else value < 75)
            chart_gate, confirmed_gate = passes(chart_rsi), passes(confirmed_rsi)
            per_seed['entry_rsi_checked'] += 1
            per_seed['chart_gate_pass'] += int(chart_gate)
            per_seed['confirmed_only_gate_pass'] += int(confirmed_gate)
            gate_rows.append({'seed_id': seed['id'], 'trade_number': trade['number'],
                              'side': trade['side'], 'reference_entry_utc': trade['entryUtc'],
                              'rsi_chart_plot': chart_rsi, 'rsi_confirmed_only': confirmed_rsi,
                              'rsi_filter_enabled': on, 'chart_gate_pass': chart_gate,
                              'confirmed_only_gate_pass': confirmed_gate,
                              'status': 'reference_entry_time_only_not_generated_signal'})
        for trade in seed['trades']['closed']:
            start_ms = entry_time_ms(trade['entryUtc'])
            end_ms = entry_time_ms(trade['exitUtc'])
            te, entry_op = asof_first_open(minute_opens, start_ms)
            tx, exit_op = asof_first_open(minute_opens, end_ms)
            report = {'seed_id': seed['id'], 'trade_number': trade['number'], 'side': trade['side'],
                      'tv_entry_utc': trade['entryUtc'], 'tv_exit_utc': trade['exitUtc'],
                      'tv_exit_reason': trade['exitSignal'],
                      'tv_entry_price': trade['entryPrice'], 'tv_exit_price': trade['exitPrice'],
                      'tv_net_pnl_usdt': trade['referenceNetPnlUsdt'],
                      'first_market_entry_ts_utc': dt.datetime.fromtimestamp(te, UTC).isoformat(),
                      'first_market_exit_ts_utc': dt.datetime.fromtimestamp(tx, UTC).isoformat(),
                      'first_market_entry_open': entry_op,
                      'first_market_exit_open': exit_op}
            if entry_op is None or exit_op is None:
                per_seed['exits_without_underlying'] += 1
                report['status'] = 'no_underlying_1m_at_required_time'
            elif tx <= te:
                per_seed['exits_same_available_minute'] += 1
                report['status'] = 'same_5m_candle_cannot_sequentially_execute'
            else:
                side = 1 if trade['side'] == 'long' else -1
                qty = trade['quantity']
                fee_fraction = setting['commissionPercent'] / 100
                net = (exit_op - entry_op) * qty * side - (entry_op + exit_op) * qty * fee_fraction
                per_seed['exits_real_reprice_count'] += 1
                per_seed['real_reprice_reference_pnl_usdt'] += trade['referenceNetPnlUsdt']
                per_seed['real_reprice_counterfactual_pnl_usdt'] += net
                report.update({'status': 'fixed_reference_times_repriced_not_executed',
                              'price_entry_diff_bps': (entry_op / trade['entryPrice'] - 1) * 10000,
                              'price_exit_diff_bps': (exit_op / trade['exitPrice'] - 1) * 10000,
                              'counterfactual_net_pnl_usdt': net,
                              'difference_net_pnl_usdt': net - trade['referenceNetPnlUsdt']})
            repriced_rows.append(report)
        summary['seed_results'][seed['id']] = per_seed
    write_csv(out_dir / 'tradingview_mtf_rsi_entry_gate.csv', gate_rows)
    write_csv(out_dir / 'tradingview_market_1m_fixed_event_reprice.csv', repriced_rows)
    summary['entry_gates_total'] = {'checked': len(gate_rows),
                                    'chart_gate_pass': sum(x['chart_gate_pass'] for x in gate_rows),
                                    'confirmed_only_gate_pass': sum(x['confirmed_only_gate_pass'] for x in gate_rows),
                                    'rsi_enabled': sum(x['rsi_filter_enabled'] for x in gate_rows)}
    summary['reprice_status_counts'] = dict(Counter(x['status'] for x in repriced_rows))
    (out_dir / 'tradingview_renko_mtf_audit_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return summary


def write_csv(path: Path, records: list[dict]):
    if not records:
        return
    columns = sorted(set().union(*(record.keys() for record in records)))
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-binance-gz', type=Path, required=True)
    parser.add_argument('--tv-renko-csv', type=Path, required=True)
    parser.add_argument('--seed-manifest', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--warmup', default='2026-07-25T00:00:00Z', help='1m raw prehistory; UTC ISO')
    parser.add_argument('--h1-before', default='2026-09-07T00:00:00Z', help='Observed export source switch; UTC ISO')
    args = parser.parse_args()
    result = audit(args.raw_binance_gz, args.tv_renko_csv, args.seed_manifest,
                   args.out_dir, utc_seconds(args.warmup), utc_seconds(args.h1_before))
    print(json.dumps({'cutoff': result['comparison_cutoff_utc'],
                      'periods': result['periods'],
                      'entry_gates': result['entry_gates_total'],
                      'reprice_status': result['reprice_status_counts']}, indent=2))


if __name__ == '__main__':
    main()