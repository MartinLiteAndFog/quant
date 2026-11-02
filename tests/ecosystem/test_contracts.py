import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant.ecosystem.contracts import TICK_PHASES, dump_snapshot, load_snapshot

FIXTURES = Path(__file__).resolve().parents[1] / 'fixtures' / 'ecosystem'


def fixture():
    return json.loads((FIXTURES / 'grass.json').read_text())


@pytest.mark.parametrize('name', ['grass', 'bush', 'cactus'])
def test_roundtrip(name):
    snapshot = load_snapshot((FIXTURES / f'{name}.json').read_text())
    assert dump_snapshot(load_snapshot(dump_snapshot(snapshot))) == dump_snapshot(snapshot)
    assert tuple(e.phase for e in snapshot.events) == TICK_PHASES
    assert isinstance(json.loads(dump_snapshot(snapshot))['groundwater']['equity'], str)


def test_migration():
    legacy = json.loads((FIXTURES / 'legacy-0.1.json').read_text())
    original = copy.deepcopy(legacy)
    assert dump_snapshot(load_snapshot(legacy)) == dump_snapshot(load_snapshot(fixture()))
    assert legacy == original


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(schema_version='2.0'),
    lambda d: d['groundwater'].update(equity=100.0),
    lambda d: d['groundwater'].update(reserve='9'),
    lambda d: d['groundwater']['accounts'].append(d['groundwater']['accounts'][0]),
    lambda d: d['groundwater']['entitlements'][0].update(amount='-1'),
    lambda d: d['groundwater']['entitlements'][0].update(amount='91'),
    lambda d: d['genomes'][0]['genes'].pop('protective_stop'),
    lambda d: d['genomes'][0]['genes'].update(protective_stop=False),
    lambda d: d['genomes'][0]['genes'].update(max_leverage=99),
    lambda d: d['events'].reverse(),
    lambda d: d['events'][1].update(event_id=d['events'][0]['event_id']),
    lambda d: d['events'][0].update(known_at='2027-01-01T00:00:00Z'),
    lambda d: d.update(as_of='2026-01-01T00:00:00'),
    lambda d: d['plants'][0]['fitness'].update(end='2027-01-01T00:00:00Z'),
    lambda d: d['groundwater']['entitlements'][0].update(used_margin='95'),
])
def test_invalid_contract_rejected(mutation):
    data = fixture()
    mutation(data)
    with pytest.raises((ValueError, ValidationError)):
        load_snapshot(data)


def test_loss_preserves_locked_margin_and_freezes_new_allocations():
    data = fixture()
    data['groundwater'].update(risk_breach=True, allocations_frozen=True)
    data['groundwater']['entitlements'][0].update(used_margin='95')
    snapshot = load_snapshot(data)
    assert snapshot.groundwater.entitlements[0].used_margin == 95
    assert snapshot.groundwater.entitlements[0].amount == 90


def test_pending_orders_not_double_promised():
    data = fixture()
    data['groundwater']['accounts'][0]['free_margin'] = '20'
    claim = data['groundwater']['entitlements'][0]
    claim.update(used_margin='40', reserved_order_margin='30')
    load_snapshot(data)
    claim['reserved_order_margin'] = '29'
    with pytest.raises(ValueError):
        load_snapshot(data)


def test_negative_nav_preserved_fail_closed():
    data = fixture()
    water = data['groundwater']
    water.update(equity='-2', reserve='0', distributable='0', risk_breach=True, allocations_frozen=True)
    water['accounts'][0].update(equity='-2', free_margin='0')
    water['entitlements'][0].update(amount='0', realizable_margin='0', used_margin='10')
    assert load_snapshot(data).groundwater.equity == -2
    water['allocations_frozen'] = False
    with pytest.raises(ValueError):
        load_snapshot(data)
