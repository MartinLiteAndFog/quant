"""Stable Blaupunkt pilot roster: three roots plus three daughters per root.

The EXISTING TradingView alarm stays on the legacy Blaupunkt root "blaupunkt".
Only the other 11 profiles require new webhook tokens and alerts.
No exchange or registration side effects are performed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PilotStrategy:
    strategy_id: str
    display_name: str
    parent_strategy_id: Optional[str] = None
    pool: str = "evolution"
    existing_legacy: bool = False


BLAUPUNKT_PILOT_ROSTER: tuple[PilotStrategy, ...] = (
    PilotStrategy("blaupunkt", "Elternstrategie 1 – Blaupunkt", existing_legacy=True),
    PilotStrategy("blaupunkt-d01", "Blaupunkt Tochter 1.1", "blaupunkt"),
    PilotStrategy("blaupunkt-d02", "Blaupunkt Tochter 1.2", "blaupunkt"),
    PilotStrategy("blaupunkt-d03", "Blaupunkt Tochter 1.3", "blaupunkt"),
    PilotStrategy("blaupunkt-p02", "Elternstrategie 2"),
    PilotStrategy("blaupunkt-p02-d01", "Blaupunkt Tochter 2.1", "blaupunkt-p02"),
    PilotStrategy("blaupunkt-p02-d02", "Blaupunkt Tochter 2.2", "blaupunkt-p02"),
    PilotStrategy("blaupunkt-p02-d03", "Blaupunkt Tochter 2.3", "blaupunkt-p02"),
    PilotStrategy("blaupunkt-p03", "Elternstrategie 3"),
    PilotStrategy("blaupunkt-p03-d01", "Blaupunkt Tochter 3.1", "blaupunkt-p03"),
    PilotStrategy("blaupunkt-p03-d02", "Blaupunkt Tochter 3.2", "blaupunkt-p03"),
    PilotStrategy("blaupunkt-p03-d03", "Blaupunkt Tochter 3.3", "blaupunkt-p03"),
)


def validate_roster() -> None:
    ids = [x.strategy_id for x in BLAUPUNKT_PILOT_ROSTER]
    if len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError("pilot must contain twelve unique IDs")
    roots = [x for x in BLAUPUNKT_PILOT_ROSTER if x.parent_strategy_id is None]
    if len(roots) != 3:
        raise ValueError("pilot must contain exactly three parents")
    index = {x.strategy_id: x for x in BLAUPUNKT_PILOT_ROSTER}
    for item in BLAUPUNKT_PILOT_ROSTER:
        if item.parent_strategy_id and (
            item.parent_strategy_id not in index or
            index[item.parent_strategy_id].parent_strategy_id is not None
        ):
            raise ValueError("every daughter must belong directly to a root parent")
    for root in roots:
        if sum(x.parent_strategy_id == root.strategy_id for x in BLAUPUNKT_PILOT_ROSTER) != 3:
            raise ValueError("each parent must have exactly three daughters")
    if [x.strategy_id for x in BLAUPUNKT_PILOT_ROSTER if x.existing_legacy] != ["blaupunkt"]:
        raise ValueError("legacy Blaupunkt must be the only existing parent")


validate_roster()
