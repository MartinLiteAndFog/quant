"""Regression coverage for complete Kraken flips, including fractional dust."""
from decimal import Decimal
from unittest.mock import patch

import pytest

from quant.execution import kraken_tv_executor as ktv
from test_kraken_tv_executor import DummyKrakenClient, _config, _signal


class ReduceOnlyClient(DummyKrakenClient):
    def __init__(self, position_signed, residuals=(), **kwargs):
        super().__init__(position_signed=position_signed, **kwargs)
        self.residuals = list(residuals)
        self.position_at_open = []

    def place_market(self, side, size, symbol=None, reduce_only=False, cli_ord_id=None):
        self.market_orders.append(dict(side=side, size=size, symbol=symbol, reduce_only=reduce_only))
        if reduce_only:
            assert (side == 'sell') == (self.position_signed > 0)
            assert Decimal(str(size)) >= abs(Decimal(str(self.position_signed)))
            self.position_signed = self.residuals.pop(0) if self.residuals else 0.0
        else:
            self.position_at_open.append(self.position_signed)
            assert self.position_signed == 0.0, 'Must close fully before opening'
            self.position_signed = size if side == 'buy' else -size
        return {'ok': True, 'data': {'sendStatus': {'status': 'filled'}}}


@pytest.mark.parametrize('position', [11.2, 11.21, 11.26, -11.21, 0.01, -0.01, -0.06])
@pytest.mark.parametrize('verify', [True, False])
def test_flip_covers_whole_position_before_open(position, verify):
    client = ReduceOnlyClient(position)
    side = 'sell' if position > 0 else 'buy'
    config = _config(dry_run=False, flat_first_flip=False, verify_after_order=verify)
    result = ktv.execute_kraken_tv_signal(_signal(side=side), config, client)
    assert result['close_signed_after'] == 0.0
    assert client.market_orders[0]['reduce_only'] is True
    assert client.position_at_open == [0.0]


def test_partial_close_retries_actual_remaining_size():
    client = ReduceOnlyClient(-11.26, residuals=[-0.06, -0.01, 0.0])
    with patch.object(ktv.time, 'sleep'):
        ktv.execute_kraken_tv_signal(_signal(side='buy'), _config(dry_run=False), client)
    assert [o['size'] for o in client.market_orders if o['reduce_only']] == [11.3, 0.1, 0.1]
    assert client.position_at_open == [0.0]


def test_persistent_residual_aborts_without_open_and_releases_dedupe():
    client = ReduceOnlyClient(-11.26, residuals=[-0.06] * 3)
    signal = _signal(side='buy', bar_index=567)
    config = _config(dry_run=False, dedup_ttl_sec=300)
    with patch.object(ktv.time, 'sleep'), pytest.raises(RuntimeError, match='did not flatten'):
        ktv.execute_kraken_tv_signal(signal, config, client)
    assert len(client.market_orders) == 3
    assert all(o['reduce_only'] for o in client.market_orders)
    assert client.position_at_open == []
    assert signal.fingerprint not in ktv._SEEN_FINGERPRINTS


def test_position_read_failure_aborts_before_open():
    class FailingRead(ReduceOnlyClient):
        def get_position(self, symbol=None):
            if self.market_orders:
                raise RuntimeError('position read failed')
            return super().get_position(symbol)
    client = FailingRead(11.21)
    with pytest.raises(RuntimeError, match='position read failed'):
        ktv.execute_kraken_tv_signal(_signal(), _config(dry_run=False), client)
    assert all(o['reduce_only'] for o in client.market_orders)


def test_wait_for_flat_does_not_accept_substep_dust():
    with patch.object(ktv.time, 'sleep'):
        _, signed = ktv._wait_for_flat(ReduceOnlyClient(-0.01), 'PF_SOLUSD', 0.1)
    assert signed == -0.01


def test_flip_still_closes_when_equity_cannot_fund_reopen():
    client = ReduceOnlyClient(0.01, equity_usd=0.0)
    result = ktv.execute_kraken_tv_signal(_signal(), _config(dry_run=False), client)
    assert result['close_signed_after'] == 0.0
    assert client.position_at_open == []
    assert len(client.market_orders) == 1


def test_dry_run_plans_close_all_without_orders():
    client = ReduceOnlyClient(11.21)
    result = ktv.execute_kraken_tv_signal(_signal(), _config(dry_run=True), client)
    assert [(o['reduce_only'], o['size']) for o in result['order_plan']] == [(True, 11.3), (False, 9.0)]
    assert client.market_orders == []
