from dataclasses import replace
from decimal import Decimal as D
import json
import os
import subprocess
import sys

import pytest
from quant.ecosystem.market_data import Costs, PaperOrder, Provenance, funding_cashflow, normalize, paper_fill, synthetic_week, timestamp


def rows():
    return [dict(timestamp=f'2024-01-01T00:0{i}:00Z', open='100', high='110', low='90', close='105', volume='100') for i in (0, 1, 3)]


def test_quality_missing_boundaries_and_hash():
    p = Provenance('fixture', 'CC0', 'SOL-DEMO', 'linear_futures', synthetic=True)
    ds = normalize(rows(), p, end='2024-01-01T00:05:00Z')
    assert ds.report['missing_bars'] == 2
    assert ds.fingerprint == normalize(reversed(rows()), p).fingerprint
    assert ds.report['synthetic']
    with pytest.raises(ValueError, match='duplicate'):
        normalize(rows()+rows()[:1], p)
    with pytest.raises(ValueError, match='timezone'):
        normalize([dict(rows()[0], timestamp='2024-01-01T00:00:00')], p)
    with pytest.raises(ValueError, match='OHLCV'):
        normalize([dict(rows()[0], high='99')], p)


def order():
    return PaperOrder('o1', 's1', 'buy', D('2'), timestamp('2024-01-01T00:00:00Z'), timestamp('2024-01-01T00:01:00Z'), timestamp('2024-01-01T00:01:00Z'))


def test_causal_next_open_long_short_costs():
    ds = synthetic_week()
    fill = paper_fill(order(), ds)
    assert fill.executed_at == timestamp('2024-01-01T00:02:00Z')
    assert fill.reference_price == D('100.02')
    assert fill.price == D('100.040004')
    assert fill.fee == fill.notional*D('.0006')
    assert fill.slippage_cost == D('0.040008')
    sell = paper_fill(replace(order(), side='sell'), ds)
    assert sell.price == D('99.999996')
    with pytest.raises(ValueError, match='timely'):
        paper_fill(replace(order(), received_at=timestamp('2024-01-01T00:10:00Z')), ds)
    with pytest.raises(ValueError, match='chronology'):
        paper_fill(replace(order(), available_at=timestamp('2023-12-31T23:59:00Z')), ds)
    with pytest.raises(ValueError, match='quantity'):
        paper_fill(replace(order(), quantity=D('0')), ds)


def test_gap_does_not_execute_at_missing_or_stale_price():
    ds = normalize(rows(), Provenance('fixture', 'CC0', 'SOL', 'spot'))
    with pytest.raises(ValueError, match='timely'):
        paper_fill(order(), ds, Costs(max_delay_seconds=60))
    assert paper_fill(order(), ds).executed_at == timestamp('2024-01-01T00:03:00Z')


@pytest.mark.parametrize('side,rate,expected', [('long','.001','-.2'), ('short','.001','.2'), ('long','-.001','.2')])
def test_funding_sign_and_boundaries(side, rate, expected):
    kwargs = dict(side=side, quantity='2', multiplier='.1', mark_price='1000', rate=rate, opened_at='2024-01-01T00:00:00Z', closed_at='2024-01-01T08:00:00Z')
    assert funding_cashflow(**kwargs, funding_at='2024-01-01T08:00:00Z') == D(expected)
    assert funding_cashflow(**kwargs, funding_at=kwargs['opened_at']) == 0
    assert funding_cashflow(**kwargs, funding_at='2024-01-01T16:00:00Z') == 0


def test_week_and_cli(tmp_path):
    first, second = synthetic_week(), synthetic_week()
    assert len(first.bars) == 10080
    assert first.report['missing_bars'] == 0
    assert first.fingerprint == second.fingerprint
    subprocess.run([sys.executable, '-m', 'quant.ecosystem.market_data', '--output', str(tmp_path)], check=True, capture_output=True)
    report = json.loads((tmp_path/'quality.json').read_text())
    assert report['fingerprint'] == first.fingerprint
    assert len((tmp_path/'bars.jsonl').read_text().splitlines()) == 10080
