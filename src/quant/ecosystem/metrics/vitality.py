"""ECO-07: statistics for strategy fitness, vitality and credibility.

No broker, backtest, simulator RNG or risk-limit mutation. All returns must
already be causal cashflow-adjusted *net* daily portfolio/strategy returns
from a separate execution/valuation engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from math import ceil, exp, isfinite, log, log1p, sqrt, tanh
from statistics import mean, stdev
from typing import Sequence


class MetricError(ValueError):
    """Inconsistent or unusable strategy evaluation observations."""


@dataclass(frozen=True)
class NetDay:
    day: date
    net_return: float
    trades: int
    eligible_opportunities: int
    expected_signals: int
    known_at: datetime

    def __post_init__(self) -> None:
        if type(self.day) is not date:
            raise MetricError("day must be a UTC calendar date")
        if type(self.net_return) not in (float, int) or not isfinite(self.net_return):
            raise MetricError("nonfinite daily net return")
        if not -1 < self.net_return <= 10:
            raise MetricError("daily net TWR must be in (-1, 10]")
        for name in ("trades", "eligible_opportunities", "expected_signals"):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise MetricError(f"invalid {name}")
        if self.expected_signals > self.eligible_opportunities:
            raise MetricError("expected signals cannot exceed eligible opportunities")
        if type(self.known_at) is not datetime or self.known_at.tzinfo is None or self.known_at.utcoffset() is None:
            raise MetricError("net outcomes must have causal, timezone-aware timestamps")
        day_end = datetime.combine(self.day, time.max, timezone.utc)
        if self.known_at.astimezone(timezone.utc) > day_end:
            raise MetricError("future-known outcome data in daily metric")


@dataclass(frozen=True)
class MetricConfig:
    annual_sessions: int = 365  # SOL continuous-market default; other assets need own calendar
    recent_half_life_opportunities: float = 20.0
    credibility_prior: float = 30.0
    min_effective_samples_for_interval: float = 3.0
    data_quality: float = 1.0
    oos_discount: float = 0.5
    tail_fraction: float = 0.05
    fitness_profit_weight: float = 0.55
    fitness_sharpe_weight: float = 0.25
    fitness_drawdown_weight: float = 0.20

    def __post_init__(self) -> None:
        if type(self.annual_sessions) is not int or self.annual_sessions <= 0:
            raise MetricError("annual_sessions must be positive")
        for name in ("recent_half_life_opportunities", "credibility_prior",
                     "min_effective_samples_for_interval"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise MetricError(f"{name} must be positive finite")
        for name in ("data_quality", "oos_discount", "tail_fraction",
                     "fitness_profit_weight", "fitness_sharpe_weight",
                     "fitness_drawdown_weight"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
                raise MetricError(f"{name} must be in [0,1]")
        weights = (self.fitness_profit_weight + self.fitness_sharpe_weight +
                   self.fitness_drawdown_weight)
        if abs(weights - 1) > 1e-9:
            raise MetricError("fitness component weights must sum to 1")
        if self.tail_fraction == 0:
            raise MetricError("tail_fraction must be positive")


@dataclass(frozen=True)
class MetricResult:
    sample_days: int
    opportunity_active_days: int
    eligible_opportunities: int
    expected_signals: int
    executed_trades: int
    effective_independent_samples: float
    opportunity_capture: float | None
    days_since_last_opportunity: int | None
    growth_multiple: float
    annualized_net_return: float
    lifetime_fitness: float
    current_vitality: float
    credibility: float
    bark_evidence: float
    sharpe: float | None
    max_drawdown: float
    current_drawdown: float
    underwater_days: int
    ulcer_index: float
    expected_shortfall_loss: float
    recent_opportunity_return: float | None
    historical_opportunity_return: float | None
    performance_drift: float | None
    lower_confidence_bound: float | None
    portfolio_correlation: float | None
    missing_calendar_days: int
    oos_verified: bool
    formula_version: str = "0.1-experimental"


def _clip(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return min(b, max(a, x))


def _corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    n = len(x)
    if n < 3 or n != len(y):
        return None
    mx, my = mean(x), mean(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx < 1e-15 or vy < 1e-15:
        return None
    return _clip(sum((a - mx) * (b - my) for a, b in zip(x, y)) / sqrt(vx * vy), -1, 1)


def _effective_sample_count(returns: Sequence[float], trade_count: int) -> float:
    """Conservative Newey-West-style autocorrelation discount on active days.

    With only daily data, independent *trades* cannot exceed independent daily
    observations; intraday serial dependence is not inferable here.
    """
    n = len(returns)
    if not n or not trade_count:
        return 0.0
    if n < 5:
        return float(min(n, trade_count))
    mu = mean(returns)
    denom = sum((x - mu) ** 2 for x in returns)
    if denom <= 1e-18:
        return float(min(n, trade_count))
    penalty = 1.0
    for lag in range(1, min(5, n // 4) + 1):
        covariance = sum((returns[k] - mu) * (returns[k - lag] - mu)
                         for k in range(lag, n))
        rho = covariance / denom
        # Positive dependence reduces evidence; negative dependence doesn't
        # inflate it above the count of active sessions or executed trades.
        penalty += 2 * max(0.0, rho) * (1 - lag / n)
    return _clip(n / max(1, penalty), 1, float(min(n, trade_count)))


def _drawdowns(returns: Sequence[float]) -> tuple[float, float, int, float, float]:
    high = 1.0
    wealth = 1.0
    max_dd = 0.0
    last_duration = 0
    ulcer_sq = 0.0
    for r in returns:
        wealth *= (1 + r)
        high = max(high, wealth)
        dd = _clip(1 - wealth / high)
        max_dd = max(max_dd, dd)
        ulcer_sq += dd ** 2
        last_duration = last_duration + 1 if dd > 1e-12 else 0
    current_dd = _clip(1 - wealth / high)
    return wealth, max_dd, last_duration, current_dd, sqrt(ulcer_sq / len(returns))


def _recent_opportunity_mean(active: Sequence[NetDay], half_life: float) -> float:
    # Events, not months, advance the decay clock. An idle cactus with no
    # valid opportunities and no mark-to-market movements retains its last
    # assessed expectation rather than "dying" merely due to calendar time.
    clock = 0.0
    total, weighted = 0.0, 0.0
    for r in reversed(active):
        weight = exp(-log(2) * clock / half_life)
        weighted += r.net_return * weight
        total += weight
        clock += max(1, r.eligible_opportunities)
    return weighted / total if total else 0.0


def evaluate_strategy(
    observations: Sequence[NetDay],
    *, config: MetricConfig = MetricConfig(), oos_verified: bool = False,
    portfolio_daily_returns: Sequence[float] | None = None,
) -> MetricResult:
    """Return independent, versioned dimensions; never an order instruction."""
    if not observations:
        raise MetricError("at least one daily observation is required")
    if type(oos_verified) is not bool:
        raise MetricError("OOS evidence must be explicit boolean")
    dates = [x.day for x in observations]
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise MetricError("observation days must be strictly increasing/unique")
    daily = [float(o.net_return) for o in observations]
    if portfolio_daily_returns is not None:
        if len(portfolio_daily_returns) != len(daily) or any(
            type(x) not in (int, float) or not isfinite(x) for x in portfolio_daily_returns
        ):
            raise MetricError("portfolio series must match finite daily observations")
    total_trades = sum(o.trades for o in observations)
    opps = sum(o.eligible_opportunities for o in observations)
    expected = sum(o.expected_signals for o in observations)
    active = [o for o in observations if o.eligible_opportunities > 0
              or o.trades > 0 or abs(o.net_return) > 1e-15]
    active_returns = [o.net_return for o in active]
    n_eff = _effective_sample_count(active_returns, total_trades)
    credibility = (n_eff / (n_eff + config.credibility_prior)
                   * config.data_quality
                   * (1.0 if oos_verified else config.oos_discount))
    credibility = _clip(credibility)
    growth, max_dd, underwater, current_dd, ulcer = _drawdowns(daily)
    loggrowth = sum(log1p(x) for x in daily)
    # Cap unrealistic synthetic-input extremes; do not call this a forecast.
    annual_return = exp(_clip(loggrowth * config.annual_sessions / len(daily), -40, 40)) - 1
    if len(daily) >= 3:
        sd = stdev(daily)
        sharpe = mean(daily) / sd * sqrt(config.annual_sessions) if sd > 1e-12 else None
    else:
        sharpe = None
    worst_n = max(1, ceil(len(daily) * config.tail_fraction))
    shortfall = max(0.0, -mean(sorted(daily)[:worst_n]))
    profit_part = 0.5 + 0.5 * tanh(annual_return / .5)
    sharpe_part = 0.5 + 0.5 * tanh(sharpe / 2) if sharpe is not None else 0.5
    risk_part = 1 - max_dd
    raw_score = 100 * (
        config.fitness_profit_weight * profit_part
        + config.fitness_sharpe_weight * sharpe_part
        + config.fitness_drawdown_weight * risk_part
    )
    fitness = _clip(raw_score / 100 * (1 - min(shortfall, 1)) * credibility) * 100
    if active:
        historical = mean(active_returns)
        recent = _recent_opportunity_mean(active, config.recent_half_life_opportunities)
        difference = recent - historical
        # This is an experimental calibration, not a proven trading threshold.
        sample_shrinkage = len(active) / (len(active) + 3.0)
        freshness = 0.5 + 0.42 * tanh(20 * recent) * sample_shrinkage
        drift_penalty = 0.22 * _clip(12 * max(0.0, -difference)) * sample_shrinkage
        vitality = _clip(freshness - drift_penalty - 0.25 * current_dd)
        if n_eff >= config.min_effective_samples_for_interval:
            sd_active = stdev(active_returns) if len(active_returns) > 1 else 0
            lower = recent - 1.645 * sd_active / sqrt(n_eff)
        else:
            lower = None
        inactive_days = (observations[-1].day - active[-1].day).days
    else:
        historical = recent = difference = lower = None
        vitality = 0.5
        inactive_days = None
    bark_support = _clip(
        credibility * (1 - max_dd) * (1.0 if annual_return > 0 else 0.0)
    )
    captured = min(1.0, total_trades / expected) if expected else None
    calendar_span = (dates[-1] - dates[0]).days + 1
    return MetricResult(
        sample_days=len(daily), opportunity_active_days=len(active),
        eligible_opportunities=opps, expected_signals=expected,
        executed_trades=total_trades, effective_independent_samples=n_eff,
        opportunity_capture=captured, days_since_last_opportunity=inactive_days,
        growth_multiple=growth, annualized_net_return=annual_return,
        lifetime_fitness=fitness, current_vitality=vitality,
        credibility=credibility, bark_evidence=bark_support,
        sharpe=sharpe, max_drawdown=max_dd, current_drawdown=current_dd,
        underwater_days=underwater, ulcer_index=ulcer,
        expected_shortfall_loss=shortfall,
        recent_opportunity_return=recent,
        historical_opportunity_return=historical,
        performance_drift=difference,
        lower_confidence_bound=lower,
        portfolio_correlation=(
            _corr(daily, portfolio_daily_returns) if portfolio_daily_returns is not None
            and len(daily) >= 20 else None
        ),
        missing_calendar_days=calendar_span - len(dates),
        oos_verified=oos_verified,
    )
