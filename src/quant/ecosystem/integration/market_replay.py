"""Causal OHLCV + external target signals -> ECO-03 paper NAV events.

No indicator signals are invented. The caller supplies *actual* signal timestamp,
availability, receipt, target position, and optional funding settlements.
This is a single-asset linear-futures/spot RESEARCH ledger, not a risk-cleared
execution or a five-year SOL backtest.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from hashlib import sha256
import json
from typing import Mapping, Sequence

from quant.ecosystem.market_data import (
    Costs, Dataset, PaperFill, PaperOrder, number,
    paper_fill, timestamp,
)
from quant.ecosystem.sim_clock.clock import IntradayEvent, SimulationState, DayResult
from quant.ecosystem.integration.evolution_cycle import (
    CycleConfig, step_evolution,
)
from quant.ecosystem.ecology.physiology import Weather

ZERO = Decimal("0")
UTC = timezone.utc


class ReplayError(ValueError):
    """Cannot establish a causal/complete paper market replay."""


@dataclass(frozen=True)
class TargetSignal:
    signal_id: str
    strategy_id: str
    target_quantity: Decimal
    signal_at: datetime
    available_at: datetime
    received_at: datetime

    def __post_init__(self):
        if not self.signal_id or not self.strategy_id:
            raise ReplayError("signal and strategy IDs required")
        q = _decimal(self.target_quantity, "target quantity", signed=True)
        a,b,c = (timestamp(t) for t in (self.signal_at,self.available_at,self.received_at))
        if b<a or c<a:
            raise ReplayError("signal cannot be known or received before it occurs")
        object.__setattr__(self,"target_quantity",q)
        object.__setattr__(self,"signal_at",a)
        object.__setattr__(self,"available_at",b)
        object.__setattr__(self,"received_at",c)


@dataclass(frozen=True)
class FundingTick:
    funding_id: str
    at: datetime
    known_at: datetime
    rate: Decimal
    mark_price: Decimal

    def __post_init__(self):
        if not self.funding_id:
            raise ReplayError("funding identifier required")
        at,known=timestamp(self.at),timestamp(self.known_at)
        if known<at or known.date()!=at.date():
            raise ReplayError("late/cross-day funding requires separate settlement pipeline")
        r=_decimal(self.rate,"funding rate",signed=True)
        p=_decimal(self.mark_price,"funding mark price")
        if p<=ZERO:
            raise ReplayError("funding mark price must be positive")
        object.__setattr__(self,"at",at)
        object.__setattr__(self,"known_at",known)
        object.__setattr__(self,"rate",r)
        object.__setattr__(self,"mark_price",p)


@dataclass(frozen=True)
class ReplayConfig:
    costs: Costs = Costs()
    max_abs_target_quantity: Decimal = Decimal("100")
    max_mark_staleness_seconds: int = 300
    fill_confirmation_latency_seconds: int = 1
    require_full_utc_days: bool = True

    def __post_init__(self):
        bound=_decimal(self.max_abs_target_quantity,"quantity cap")
        if bound<=ZERO:
            raise ReplayError("quantity cap must be positive")
        if (type(self.max_mark_staleness_seconds) is not int
            or self.max_mark_staleness_seconds<60
            or type(self.fill_confirmation_latency_seconds) is not int
            or not 0<=self.fill_confirmation_latency_seconds<=120
            or type(self.require_full_utc_days) is not bool):
            raise ReplayError("invalid data maturity or fill-confirmation policy")
        object.__setattr__(self,"max_abs_target_quantity",bound)


@dataclass(frozen=True)
class ReplayDay:
    day: date
    events: tuple[IntradayEvent,...]
    completed_trades: tuple[tuple[str,int],...]
    signals_seen: tuple[tuple[str,int],...]
    end_positions: tuple[tuple[str,Decimal],...]

    @property
    def net_pnl(self) -> Decimal:
        return sum((e.amount for e in self.events if e.kind in
                    ("pnl_delta","fee","funding")),ZERO)

    def trade_counts(self) -> dict[str,int]:
        return dict(self.completed_trades)


@dataclass(frozen=True)
class ReplayOutput:
    days: tuple[ReplayDay,...]
    fills: tuple[PaperFill,...]
    data_hash: str
    signal_hash: str
    provenance: dict
    synthetic: bool

    def events(self) -> tuple[IntradayEvent,...]:
        return tuple(x for day in self.days for x in day.events)


def _decimal(value, label: str, *, signed: bool=False) -> Decimal:
    if isinstance(value,float) or isinstance(value,bool):
        raise ReplayError(f"{label} requires Decimal or exact decimal string")
    try:
        d=number(value)
    except Exception as exc:
        raise ReplayError(f"invalid {label}") from exc
    if not signed and d<ZERO:
        raise ReplayError(f"negative {label}")
    return d


def _iso_hash(signals: Sequence[TargetSignal], funding: Sequence[FundingTick]) -> str:
    s={"signals":[{"id":i.signal_id,"strategy":i.strategy_id,
                   "q":str(i.target_quantity),
                   "signal_at":i.signal_at.isoformat(),
                   "available":i.available_at.isoformat(),
                   "received":i.received_at.isoformat()} for i in signals],
       "funding":[{"id":i.funding_id,"at":i.at.isoformat(),
                   "known_at":i.known_at.isoformat(),"rate":str(i.rate),
                   "mark":str(i.mark_price)} for i in funding]}
    return sha256(json.dumps(s,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def replay_market(
    dataset: Dataset,
    signals: Sequence[TargetSignal],
    *, funding: Sequence[FundingTick]=(),
    config: ReplayConfig=ReplayConfig(),
) -> ReplayOutput:
    """Emit NAV deltas from executed target positions with daily fair-price marks.

    Every fill is marked first at its same-bar *open* reference price; adverse
    spread/slippage lives in pnl_delta and is NOT deducted a second time.
    Fees and funding have separate event types; daily position returns and
    closed logical trades cannot be inferred by counting order fills.
    """
    if not dataset.bars or dataset.provenance.market_type not in ("spot","linear_futures"):
        raise ReplayError("unsupported or missing market dataset")
    bars=dataset.bars
    interval=dataset.provenance.interval_seconds
    if not isinstance(dataset.fingerprint,str) or not dataset.fingerprint:
        raise ReplayError("dataset fingerprint required")
    if len({s.signal_id for s in signals})!=len(signals):
        raise ReplayError("duplicate signal IDs")
    if len({f.funding_id for f in funding})!=len(funding):
        raise ReplayError("duplicate funding settlements")
    if dataset.provenance.market_type=="spot" and funding:
        raise ReplayError("spot data cannot settle futures funding")
    if any(x.target_quantity<ZERO for x in signals) and dataset.provenance.market_type=="spot":
        raise ReplayError("shorting requires explicit futures contract provenance")
    if any(abs(x.target_quantity)>config.max_abs_target_quantity for x in signals):
        raise ReplayError("target position exceeds external research quantity cap")
    if any(a.timestamp>=b.timestamp for a,b in zip(bars,bars[1:])):
        raise ReplayError("bars must be strictly time sorted")

    # Instructions are ordered by when they *can be acted upon*, not by a
    # historical signal bar that might only become known later.
    ordered=sorted(signals,key=lambda s:(max(s.available_at,s.received_at),s.signal_id))
    desired:dict[str,Decimal]={}
    last_fill:dict[str,datetime]={}
    actions:list[tuple[datetime,int,str,object]]=[]
    fills:list[PaperFill]=[]
    signal_counts:dict[date,dict[str,int]]={}
    for s in ordered:
        ready=max(s.available_at,s.received_at)
        if ready.date()<bars[0].timestamp.date():
            raise ReplayError("signal before dataset cannot be justified")
        if ready>bars[-1].timestamp:
            raise ReplayError("signal after final candle")
        signal_counts.setdefault(ready.date(),{}).setdefault(s.strategy_id,0)
        signal_counts[ready.date()][s.strategy_id]+=1
        actions.append((ready,0,s.signal_id,s))
        old=desired.get(s.strategy_id,ZERO)
        change=s.target_quantity-old
        if change==ZERO:
            continue
        side="buy" if change>ZERO else "sell"
        try:
            fill=paper_fill(
                PaperOrder("signal:"+s.signal_id,s.strategy_id,side,
                           abs(change),s.signal_at,s.available_at,s.received_at),
                dataset,config.costs,
            )
        except ValueError as exc:
            raise ReplayError(f"signal {s.signal_id}: {exc}") from exc
        prior=last_fill.get(s.strategy_id)
        if prior is not None and fill.executed_at<=prior:
            raise ReplayError("overlapping target changes would execute at same bar open")
        if (fill.executed_at+timedelta(seconds=config.fill_confirmation_latency_seconds)
                ).date()!=fill.executed_at.date():
            raise ReplayError("cross-day trade confirmation requires deferred-event adapter")
        desired[s.strategy_id]=s.target_quantity
        last_fill[s.strategy_id]=fill.executed_at
        fills.append(fill)
        actions.append((fill.executed_at,1,s.signal_id,(fill,change)))

    for tick in funding:
        if tick.at<bars[0].timestamp or tick.at>bars[-1].timestamp+timedelta(seconds=interval):
            raise ReplayError("funding timestamp outside data coverage")
        actions.append((tick.at,2,tick.funding_id,tick))

    # One conservative fair-price close per UTC day, using ONLY bars whose
    # close became observable on that same day (never the 23:59 open's
    # future-day close). First/last partial days need explicit opt-in.
    completed:dict[date,list]= {}
    coverage:dict[date,list]= {}
    for bar in bars:
        coverage.setdefault(bar.timestamp.date(),[]).append(bar.timestamp)
        known_at=bar.timestamp+timedelta(seconds=interval)
        if known_at.date()==bar.timestamp.date():
            completed.setdefault(bar.timestamp.date(),[]).append((known_at,bar))
    for day, times in coverage.items():
        if config.require_full_utc_days and (
            times[0].time()!=time.min or
            (datetime.combine(day+timedelta(days=1),time.min,UTC) -
             times[-1]).total_seconds()!=interval
        ):
            raise ReplayError("incomplete UTC calendar day: do not pretend full history")
    for day, items in completed.items():
        known_at,bar=items[-1]
        next_midnight=datetime.combine(day+timedelta(days=1),time.min,UTC)
        age=(next_midnight-known_at).total_seconds()
        if age>config.max_mark_staleness_seconds and config.require_full_utc_days:
            # Incomplete history must not count as a day-end NAV.
            actions.append((known_at,3,"stale-mark:"+day.isoformat(),(bar,True)))
        else:
            actions.append((known_at,3,"mark:"+day.isoformat(),(bar,False)))
    # A full day with only its last 23:59 candle has no causally known
    # same-day mark and cannot value any open position that night.

    actions.sort(key=lambda x:(x[0],x[1],x[2]))
    positions:dict[str,Decimal]={}
    last_ref:dict[str,Decimal]={}
    all_events:dict[date,list[IntradayEvent]]={}
    completed_trades:dict[date,dict[str,int]]={}
    active_intervals:dict[str,list[tuple[datetime,datetime|None]]]={}
    opened_at:dict[str,datetime]={}
    multiplier=_decimal(dataset.provenance.contract_multiplier,"contract multiplier")

    def add(event:IntradayEvent):
        all_events.setdefault(event.occurred_at.date(),[]).append(event)

    for at,_,key,payload in actions:
        if isinstance(payload,TargetSignal):
            add(IntradayEvent("signal:"+payload.signal_id,at,at,"signal",ZERO,
                              payload.strategy_id))
        elif isinstance(payload,tuple) and isinstance(payload[0],PaperFill):
            fill,delta=payload
            strategy=fill.strategy_id
            old_q=positions.get(strategy,ZERO)
            new_q=old_q+delta
            if abs(new_q)>config.max_abs_target_quantity:
                raise ReplayError("executed exposure exceeds target cap")
            fair=fill.reference_price
            prior_ref=last_ref.get(strategy)
            if old_q and prior_ref is None:
                raise ReplayError("open position without marked reference price")
            mark_movement=old_q*multiplier*(fair-prior_ref) if old_q else ZERO
            execution_drag=-delta*multiplier*(fill.price-fair)
            confirmation=at+timedelta(seconds=config.fill_confirmation_latency_seconds)
            add(IntradayEvent("fill-pnl:"+fill.order_id,at,confirmation,
                              "pnl_delta",mark_movement+execution_drag,strategy))
            add(IntradayEvent("fill-fee:"+fill.order_id,at,confirmation,
                              "fee",-fill.fee,strategy))
            # A completed logical round trip is not a raw fill count.
            if old_q and (new_q==ZERO or (old_q>ZERO)!=(new_q>ZERO)):
                completed_trades.setdefault(at.date(),{}).setdefault(strategy,0)
                completed_trades[at.date()][strategy]+=1
                start=opened_at.pop(strategy)
                active_intervals.setdefault(strategy,[]).append((start,at))
            if new_q and not old_q or (old_q and new_q and (old_q>ZERO)!=(new_q>ZERO)):
                opened_at[strategy]=at
            positions[strategy]=new_q
            last_ref[strategy]=fair
        elif isinstance(payload,FundingTick):
            if payload.known_at.date()!=at.date():
                raise ReplayError("cross-day funding not supported")
            if dataset.provenance.market_type!="linear_futures":
                raise ReplayError("funding requires linear futures")
            for strategy in sorted(k for k,q in positions.items() if q):
                q=positions[strategy]
                payment=-q*multiplier*payload.mark_price*payload.rate
                add(IntradayEvent("funding:"+payload.funding_id+":"+strategy,
                                  at,payload.known_at,"funding",payment,strategy))
        else:
            bar,stale=payload
            if stale and any(q for q in positions.values()):
                raise ReplayError("stale day-end mark on open position")
            for strategy,q in sorted(positions.items()):
                if not q:
                    continue
                old_ref=last_ref.get(strategy)
                if old_ref is None:
                    raise ReplayError("unmarked open position")
                amount=q*multiplier*(bar.close-old_ref)
                add(IntradayEvent("mark:"+at.date().isoformat()+":"+strategy,
                                  at,at,"pnl_delta",amount,strategy))
                last_ref[strategy]=bar.close
        # The last known position after each day's final action is snapshotted
        # outside this loop below, across the complete UTC date range.

    end_of_data=bars[-1].timestamp+timedelta(seconds=interval)
    for strategy,start in opened_at.items():
        active_intervals.setdefault(strategy,[]).append((start,end_of_data))

    # A missing OHLCV interval that overlaps an open position means its
    # intraday path, stop/liquidation exposure and fill assumptions are
    # unknowable. Reject instead of silently interpolating returns.
    for gap in dataset.report.get("gaps",()):
        start,end=timestamp(gap["start"]),timestamp(gap["end"])
        for ranges in active_intervals.values():
            if any(max(a,start)<min(b,end) for a,b in ranges):
                raise ReplayError("missing OHLCV while position open; unsafe replay")

    days=sorted(coverage)
    if any((b-a).days!=1 for a,b in zip(days,days[1:])):
        raise ReplayError("missing entire UTC market day")
    # A market dataset may finish before its day ends: optional partial-day
    # fixture mode allows a last observed mark, but rejects stale open risk.
    event_days=set(all_events)
    if event_days-set(days):
        raise ReplayError("event day not covered by market history")
    positions_now:dict[str,Decimal]={}
    results=[]
    for day in days:
        day_events=tuple(sorted(all_events.get(day,()),
                                key=lambda e:(e.available_at,e.event_id)))
        for event in day_events:
            if event.available_at.date()!=day:
                raise ReplayError("known-next-day event requires carryover adapter")
        for fill in fills:
            if fill.executed_at.date()==day:
                signed=fill.quantity if fill.side=="buy" else -fill.quantity
                positions_now[fill.strategy_id]=positions_now.get(fill.strategy_id,ZERO)+signed
        if any(q for q in positions_now.values()) and day not in completed:
            raise ReplayError("open position without causal daily closing mark")
        results.append(ReplayDay(day,day_events,
                                  tuple(sorted(completed_trades.get(day,{}).items())),
                                  tuple(sorted(signal_counts.get(day,{}).items())),
                                  tuple(sorted(positions_now.items()))))
    return ReplayOutput(tuple(results),tuple(fills),dataset.fingerprint,
                        _iso_hash(ordered,sorted(funding,key=lambda x:(x.at,x.funding_id))),
                        dataset.report["provenance"],dataset.provenance.synthetic)


def apply_replay_to_evolution(
    initial: SimulationState, replay: ReplayOutput,
    *, weather_by_day: Mapping[date,Weather],
    opportunities_by_day: Mapping[date,Mapping[str,int]],
    free_margin_by_day: Mapping[date,str],
    config: CycleConfig=CycleConfig(),
) -> SimulationState:
    """Bridge *already simulated* paper returns to ECO-17, never market orders.

    Independent ex-ante opportunity counts and free margin are MANDATORY.
    This adapter refuses to invent them from successful trades or bot genes.
    The market replay does not itself enforce ECO-17 allocation sizing and
    therefore must not be interpreted as a risk-cleared investable backtest.
    """
    state=initial
    for record in replay.days:
        if record.day not in weather_by_day or record.day not in opportunities_by_day or record.day not in free_margin_by_day:
            raise ReplayError("ECO-17 requires explicit causal weather, opportunities and free margin")
        state=step_evolution(
            state,record.day,record.events,weather_by_day[record.day],
            trades=record.trade_counts(),
            opportunities=opportunities_by_day[record.day],
            free_margin=free_margin_by_day[record.day],
            config=config,
        ).state
    return state
