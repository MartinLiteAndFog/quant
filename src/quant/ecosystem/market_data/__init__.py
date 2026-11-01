"""Read-only OHLCV normalization and deterministic next-open paper execution."""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json

UTC = timezone.utc
D = Decimal


def timestamp(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('timestamps must include timezone')
    return result.astimezone(UTC)


def number(value):
    result = D(str(value))
    if not result.is_finite():
        raise ValueError('nonfinite number')
    return result


@dataclass(frozen=True)
class Provenance:
    source: str
    license: str
    symbol: str
    market_type: str
    contract_multiplier: str = '1'
    interval_seconds: int = 60
    synthetic: bool = False
    timezone: str = 'UTC'
    timestamp_semantics: str = 'open'

    def __post_init__(self):
        if not self.source or not self.license or not self.symbol:
            raise ValueError('source, license and symbol required')
        if self.market_type not in ('spot', 'linear_futures'):
            raise ValueError('only spot and linear futures supported')
        if number(self.contract_multiplier) <= 0 or self.interval_seconds <= 0:
            raise ValueError('positive multiplier and interval required')
        if self.timezone != 'UTC' or self.timestamp_semantics != 'open':
            raise ValueError('normalized contract requires UTC open timestamps')


@dataclass(frozen=True)
class Candle:
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class Dataset:
    bars: tuple
    provenance: Provenance
    report: dict
    fingerprint: str


def normalize(rows, provenance, *, start=None, end=None):
    """Validate UTC open bars in [start,end); gaps remain gaps, never filled.

    Duplicates and malformed bars fail closed. Report leading/trailing gaps when
    explicit boundaries exist. Sorting is deterministic; no interpolation.
    """
    bars, seen = [], set()
    interval = provenance.interval_seconds
    start = timestamp(start) if start is not None else None
    end = timestamp(end) if end is not None else None
    if start is not None and end is not None and end <= start:
        raise ValueError('end must follow start')
    for boundary in (start, end):
        if boundary is not None and boundary.timestamp() % interval:
            raise ValueError('boundary not aligned')
    for row in rows:
        time = timestamp(row['timestamp'])
        if time.timestamp() % interval:
            raise ValueError('bar not aligned')
        if time in seen:
            raise ValueError('duplicate candle')
        seen.add(time)
        values = [number(row[key]) for key in ('open', 'high', 'low', 'close', 'volume')]
        o, h, l, c, v = values
        if min(o, h, l, c) <= 0 or v < 0 or h < max(o, l, c) or l > min(o, h, c):
            raise ValueError('invalid OHLCV')
        if (start is None or time >= start) and (end is None or time < end):
            bars.append(Candle(time, *values))
    bars.sort(key=lambda bar: bar.timestamp)
    if not bars:
        raise ValueError('no candles in requested range')
    begin = start or bars[0].timestamp
    finish = end or (bars[-1].timestamp + timedelta(seconds=interval))
    gaps = []
    previous = begin
    for bar in bars:
        if bar.timestamp > previous:
            gaps.append({'start': previous.isoformat(), 'end': bar.timestamp.isoformat(), 'missing_bars': int((bar.timestamp-previous).total_seconds())//interval})
        previous = bar.timestamp + timedelta(seconds=interval)
    if previous < finish:
        gaps.append({'start': previous.isoformat(), 'end': finish.isoformat(), 'missing_bars': int((finish-previous).total_seconds())//interval})
    canonical = {'provenance': asdict(provenance), 'bars': [asdict(bar) for bar in bars]}
    fingerprint = hashlib.sha256(json.dumps(canonical, default=str, sort_keys=True).encode()).hexdigest()
    report = {'schema_version': 1, 'bar_count': len(bars), 'start': begin.isoformat(), 'end_exclusive': finish.isoformat(), 'gaps': gaps, 'missing_bars': sum(g['missing_bars'] for g in gaps), 'synthetic': provenance.synthetic, 'provenance': asdict(provenance), 'survivorship': 'single supplied symbol; no universe survivorship claim', 'lookahead': 'closed bars available only at open timestamp plus interval', 'fingerprint': fingerprint}
    return Dataset(tuple(bars), provenance, report, fingerprint)


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    strategy_id: str
    side: str
    quantity: Decimal
    signal_at: datetime
    available_at: datetime
    received_at: datetime


@dataclass(frozen=True)
class Costs:
    fee_bps: Decimal = D('6')
    spread_bps: Decimal = D('2')
    slippage_bps: Decimal = D('1')
    max_delay_seconds: int = 120


@dataclass(frozen=True)
class PaperFill:
    order_id: str
    strategy_id: str
    side: str
    quantity: Decimal
    executed_at: datetime
    price: Decimal
    reference_price: Decimal
    fee: Decimal
    slippage_cost: Decimal
    notional: Decimal
    dataset_fingerprint: str


def paper_fill(order, dataset, costs=Costs()):
    """Market orders execute at first open strictly AFTER receipt/availability.

    Strict inequality avoids assuming zero-latency execution at an already
    observed open. No bar after bounded delay means rejection, not stale fills.
    """
    signal, available, received = [timestamp(t) for t in (order.signal_at, order.available_at, order.received_at)]
    if available < signal or received < signal:
        raise ValueError('invalid signal chronology')
    if order.side not in ('buy', 'sell') or not order.order_id or not order.strategy_id:
        raise ValueError('invalid order')
    quantity = number(order.quantity)
    fee, spread, slip = [number(x) for x in (costs.fee_bps, costs.spread_bps, costs.slippage_bps)]
    if quantity <= 0 or min(fee, spread, slip) < 0 or spread/2+slip >= 10000 or costs.max_delay_seconds <= 0:
        raise ValueError('invalid quantity or costs')
    ready = max(available, received)
    bar = next((b for b in dataset.bars if b.timestamp > ready), None)
    if bar is None or (bar.timestamp-available).total_seconds() > costs.max_delay_seconds:
        raise ValueError('no timely executable candle')
    if bar.volume <= 0:
        raise ValueError('zero volume candle')
    multiplier = number(dataset.provenance.contract_multiplier)
    direction = 1 if order.side == 'buy' else -1
    adjustment = bar.open * (spread/2+slip)/10000
    price = bar.open + direction*adjustment
    notional = quantity * multiplier * price
    return PaperFill(order.order_id, order.strategy_id, order.side, quantity, bar.timestamp, price, bar.open, notional*fee/10000, quantity*multiplier*adjustment, notional, dataset.fingerprint)


def funding_cashflow(*, side, quantity, multiplier, mark_price, rate, opened_at, closed_at, funding_at):
    """Linear futures funding, positive rate paid by longs. Holding (open,close]."""
    opened, closed, at = [timestamp(t) for t in (opened_at, closed_at, funding_at)]
    if side not in ('long', 'short') or closed < opened:
        raise ValueError('invalid position')
    q, m, p, r = [number(x) for x in (quantity, multiplier, mark_price, rate)]
    if min(q, m, p) <= 0:
        raise ValueError('invalid funding inputs')
    return (D('-1') if side == 'long' else D('1'))*q*m*p*r if opened < at <= closed else D('0')


def synthetic_week(start='2024-01-01T00:00:00Z'):
    """10080 deterministic minute bars; generated fixture, never historical SOL."""
    begin = timestamp(start)
    rows = []
    for i in range(7*24*60):
        price = D('100')+D(i%240)/100
        rows.append(dict(timestamp=(begin+timedelta(minutes=i)).isoformat(), open=str(price), high=str(price+1), low=str(price-1), close=str(price+D('.1')), volume='1000'))
    return normalize(rows, Provenance('generated synthetic_week v1', 'CC0 synthetic fixture', 'SOL-DEMO', 'linear_futures', synthetic=True))


def to_contract_fill(fill, order, dataset, *, currency='USDT'):
    """Export ECO-01 fill; funding is reported separately at settlement time."""
    from quant.ecosystem.contracts import TradeFill
    if fill.order_id != order.order_id or fill.dataset_fingerprint != dataset.fingerprint:
        raise ValueError('fill provenance mismatch')
    return TradeFill(fill_id=fill.order_id, strategy_id=fill.strategy_id,
                     symbol=dataset.provenance.symbol, signal_at=timestamp(order.signal_at),
                     available_at=max(timestamp(order.available_at), timestamp(order.received_at)),
                     filled_at=fill.executed_at, side=fill.side, quantity=fill.quantity,
                     price=fill.price, fee=fill.fee, funding=D('0'), currency=currency,
                     synthetic=dataset.provenance.synthetic)
