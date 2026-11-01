"""ECO-19b: source-aware IMBA/Renko research candidates and TV alert diagnostics.

Causal boundary: 5-minute source bars become known only AFTER their fifth
one-minute candle closes. Historical Renko brick timestamps (with 1ms
offsets) are geometry identifiers, NEVER valid availability timestamps.

The existing project renko_from_close() is body/close driven and does not
fully reconstruct TradingView wick highs/lows. Consequently generated IMBA
signals are UNVERIFIED CANDIDATES, not evidence of trading profitability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from typing import Sequence

from quant.ecosystem.market_data import timestamp
from quant.ecosystem.market_data.streaming import (
    StreamImportError, get_manifest, iter_candles,
)

UTC = timezone.utc


class SignalAuditError(ValueError):
    """Incompatible or incomplete candle/gene/alert comparison."""


@dataclass(frozen=True)
class FiveMinuteBar:
    open_at: datetime
    known_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


@dataclass(frozen=True)
class ImbaCandidate:
    candidate_id: str
    strategy_id: str
    source_5m_open_at: datetime
    available_at: datetime
    direction: int
    renko_brick_at: datetime
    box: Decimal
    lookback: int


@dataclass(frozen=True)
class ReferenceAlert:
    reference_id: str
    direction: int
    confirmed_5m_close_at: datetime

    def __post_init__(self):
        if not self.reference_id or type(self.direction) is not int or self.direction not in (-1,1):
            raise SignalAuditError("invalid source alert identity/direction")
        dt=timestamp(self.confirmed_5m_close_at)
        if dt.second or dt.microsecond or int(dt.timestamp())%300:
            raise SignalAuditError("reference must have confirmed 5m-close timestamp")
        object.__setattr__(self,"confirmed_5m_close_at",dt)


@dataclass(frozen=True)
class ImbaResearchBatch:
    dataset_fingerprint: str
    strategy_id: str
    source_candles_5m: int
    brick_count: int
    candidates: tuple[ImbaCandidate,...]
    canonical_source_interval: int
    renko_source_interval: int = 300
    renko_wicks_reproduced: bool = False
    tv_feed_identity_verified: bool = False
    independent_oos_verified: bool = False
    capital_risk_cleared: bool = False


@dataclass(frozen=True)
class ReferenceComparison:
    candidate_count: int
    reference_count: int
    exact_matches: int
    direction_mismatches: int
    timestamp_mismatches: int
    unpaired_candidates: int
    unpaired_references: int
    matches_supplied_reference: bool
    enough_reference_signals_for_screening: bool
    full_TradingView_parity_established: bool = False
    safe_to_claim_oos_alpha: bool = False


def _number(value: str, label: str) -> Decimal:
    try:
        result=Decimal(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise SignalAuditError(f"bad {label}") from exc
    if not result.is_finite():
        raise SignalAuditError(f"nonfinite {label}")
    return result


def source_5min_bars(
    database: str,
    start: str | datetime,
    end: str | datetime,
    *, max_source_1min: int=10080,
) -> tuple[FiveMinuteBar,...]:
    """Resample only COMPLETE UTC-aligned 5×1minute source groups.

    Fail on gaps within or between buckets. This function is deliberately
    window-bounded, not a materialization of five years of minute candles.
    """
    manifest=get_manifest(database)
    if manifest["provenance"]["interval_seconds"] != 60:
        raise SignalAuditError("this research adapter requires exact 1m source bars")
    lo,hi=timestamp(start),timestamp(end)
    if int(lo.timestamp())%300 or int(hi.timestamp())%300 or lo>=hi:
        raise SignalAuditError("both source boundaries must align to full UTC 5m buckets")
    if type(max_source_1min) is not int or max_source_1min<5:
        raise SignalAuditError("invalid source row budget")
    expected=int((hi-lo).total_seconds()/60)
    if expected>max_source_1min:
        raise SignalAuditError("requested research window exceeds 1m bar budget")
    result:list[FiveMinuteBar]=[]
    group:list[dict]=[]
    checked=0
    for record in iter_candles(database,lo,hi):
        at=timestamp(record["timestamp"])
        exact=lo+timedelta(minutes=checked)
        if at!=exact:
            raise SignalAuditError("missing/misaligned 1m candle in requested 5m window")
        group.append(record)
        checked+=1
        if len(group)==5:
            started=at-timedelta(minutes=4)
            if int(started.timestamp())%300:
                raise SignalAuditError("5m bucket not UTC aligned")
            prices=[{k:_number(c[k],k) for k in ("open","high","low","close","volume")}
                    for c in group]
            result.append(FiveMinuteBar(
                started,started+timedelta(minutes=5),
                prices[0]["open"],max(c["high"] for c in prices),
                min(c["low"] for c in prices),prices[-1]["close"],
                sum((c["volume"] for c in prices),Decimal(0)),
            ))
            group.clear()
    if checked!=expected or group:
        raise SignalAuditError("incomplete source 5m bucket or missing final 1m bar")
    return tuple(result)


def candidate_imba_signals(
    database: str, *, start: str | datetime, end: str | datetime,
    trading_start: str | datetime, strategy_id: str,
    box: str | Decimal="0.1", lookback: int=50,
    max_source_1min: int=10080,
) -> ImbaResearchBatch:
    """Calculate *unverified* legacy IMBA flip candidates, never orders.

    Uses the already-existing repo implementations:
      quant.features.renko.renko_from_close()
      quant.strategies.imba.compute_imba_signals()
    Wicks are NOT TV-wick-parity modeled. Renko state starts from the
    *start* date's close anchor; supply a sufficiently long warmup.
    """
    if not strategy_id:
        raise SignalAuditError("strategy_id required")
    if type(lookback) is not int or lookback<2:
        raise SignalAuditError("IMBA lookback must be >=2")
    b=_number(str(box),"Renko box size")
    if b<=0:
        raise SignalAuditError("Renko box must be positive")
    lo,hi,trade=(timestamp(z) for z in (start,end,trading_start))
    if not lo<trade<hi:
        raise SignalAuditError("trading_start must follow warmup and precede end")
    bars=source_5min_bars(database,lo,hi,max_source_1min=max_source_1min)
    if len(bars)<lookback+1:
        raise SignalAuditError("insufficient warmup source bars")
    # Legacy geometry logic requires pandas/numpy. Installed as existing repo
    # dependencies, not as a new candidate/TV-parity implementation.
    import pandas as pd
    from quant.features.renko import renko_from_close
    from quant.strategies.imba import ImbaParams, compute_imba_signals
    frame=pd.DataFrame([
        {"ts":pd.Timestamp(x.open_at),"close":float(x.close)}
        for x in bars
    ])
    renko=renko_from_close(frame,box=float(b))
    if len(renko)<lookback+1:
        raise SignalAuditError("insufficient Renko bricks for sticky IMBA warmup")
    # Existing Renko builder only returns open/close body, no exchange-specific
    # historical TradingView wick. Never label these signals parity-verified.
    body=renko[["ts","open","close"]].copy()
    body["high"]=body[["open","close"]].max(axis=1)
    body["low"]=body[["open","close"]].min(axis=1)
    detected=compute_imba_signals(body,ImbaParams(lookback=lookback))
    manifest=get_manifest(database)
    selected:dict[datetime,ImbaCandidate]={}
    for _,row in detected.iterrows():
        brick_at=timestamp(row["ts"])
        # Multi-brick 1ms offsets are *display/source sequencing* only.
        # Bucket close-time, not brick event timestamp, is when it is known.
        bucket_unix=int(brick_at.timestamp())//300*300
        source_open=datetime.fromtimestamp(bucket_unix,UTC)
        available=source_open+timedelta(minutes=5)
        if available<trade or available>hi:
            continue
        if source_open<lo or source_open>=hi:
            raise SignalAuditError("Renko brick escaped source window")
        direction=int(row["signal"])
        if direction not in (-1,1):
            raise SignalAuditError("unknown IMBA direction")
        identity=sha256(
            (manifest["dataset_fingerprint"]+"|"+strategy_id+"|"+
             available.isoformat()+"|"+str(direction)+"|"+str(b)+"|"+
             str(lookback)).encode()
        ).hexdigest()[:24]
        # Several 1ms-offset Renko bricks can flip in a single source bucket.
        # They are not separately tradable inside that unclosed 5m candle:
        # retain only the final confirmed direction for that bucket.
        selected[available]=ImbaCandidate(
            "research-imba:"+identity,strategy_id,source_open,available,
            direction,brick_at,b,lookback,
        )
    return ImbaResearchBatch(
        manifest["dataset_fingerprint"],strategy_id,len(bars),len(renko),
        tuple(selected[key] for key in sorted(selected)),60,
    )


def compare_confirmed_alerts(
    batch: ImbaResearchBatch,
    reference: Sequence[ReferenceAlert],
    *, minimum_alerts: int=20,
) -> ReferenceComparison:
    """Diagnostic comparison only. Does not certify venue feed/wicks/OOS.

    Reference rows must already be converted to *confirmed 5m source-close*
    timestamps, NOT webhook arrival times. Both streams should refer to
    identical ticker, source venue, fixed-box size, lookback and period.
    """
    if type(minimum_alerts) is not int or minimum_alerts<1:
        raise SignalAuditError("invalid reference minimum")
    ordered=sorted(reference,key=lambda x:(x.confirmed_5m_close_at,x.reference_id))
    if len({x.reference_id for x in ordered})!=len(ordered):
        raise SignalAuditError("duplicate TV alert reference ID")
    if len({x.confirmed_5m_close_at for x in ordered})!=len(ordered):
        raise SignalAuditError("same confirmed source bucket has multiple reference alerts")
    ref_by_time={r.confirmed_5m_close_at:r.direction for r in ordered}
    cand_by_time={c.available_at:c.direction for c in batch.candidates}
    if len(cand_by_time)!=len(batch.candidates):
        raise SignalAuditError("duplicate candidate source bucket")
    shared=set(ref_by_time)&set(cand_by_time)
    direction_bad=sum(ref_by_time[at]!=cand_by_time[at] for at in shared)
    timedif=len(ref_by_time.keys() ^ cand_by_time.keys())
    matched=len(shared)-direction_bad
    complete=(len(ref_by_time)>=minimum_alerts
              and direction_bad==0 and set(ref_by_time)==set(cand_by_time))
    return ReferenceComparison(
        candidate_count=len(cand_by_time),
        reference_count=len(ref_by_time),exact_matches=matched,
        direction_mismatches=direction_bad,
        timestamp_mismatches=timedif,
        unpaired_candidates=len(set(cand_by_time)-set(ref_by_time)),
        unpaired_references=len(set(ref_by_time)-set(cand_by_time)),
        matches_supplied_reference=(direction_bad==0 and
                                   set(ref_by_time)==set(cand_by_time)
                                   and bool(ref_by_time)),
        enough_reference_signals_for_screening=len(ref_by_time)>=minimum_alerts,
        # Even a byte-perfect match cannot establish full TV parity because
        # this source-close body-only builder lacks TV historical wick parity.
        full_TradingView_parity_established=False,
        safe_to_claim_oos_alpha=False,
    )
