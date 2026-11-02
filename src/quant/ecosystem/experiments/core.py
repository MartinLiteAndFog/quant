"""Small, deterministic experiment harness for ECO-12.

The built-in evaluator is a synthetic protocol fixture. It is intentionally not a
market-data backtester and cannot establish trading edge or TradingView parity.
External evaluators can implement :class:`Evaluator` when the upstream market,
metrics, breeding, and compute contracts are available.
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


PROTOCOL_VERSION = "eco12.v1.1"
ARMS = ("A_random", "B_local", "C_hive", "D_hybrid")


@dataclass(frozen=True)
class AblationConfig:
    """Experiment sliders; unsupported mechanisms stay explicit in reports."""

    gene_cost_weight: float = 0.0
    groundwater_inertia: float = 0.5
    groundwater_hysteresis: float = 0.05
    predator_vitality_ceiling: float = 0.70

    def __post_init__(self) -> None:
        if self.gene_cost_weight < 0:
            raise ValueError("gene_cost_weight must be non-negative")
        if not 0.0 <= self.groundwater_inertia <= 1.0:
            raise ValueError("groundwater_inertia must be in [0, 1]")
        if not 0.0 <= self.groundwater_hysteresis <= 1.0:
            raise ValueError("groundwater_hysteresis must be in [0, 1]")
        if not 0.0 <= self.predator_vitality_ceiling <= 1.0:
            raise ValueError("predator_vitality_ceiling must be in [0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    protocol_version: str = PROTOCOL_VERSION
    benchmark_arm: str = "A_random"
    seeds: Tuple[int, ...] = tuple(range(30))
    walk_forward_folds: int = 3
    purge_bars: int = 1
    final_holdout_fraction: float = 0.20
    population_size: int = 12
    generations: int = 6
    evaluations_per_arm_fold: int = 72
    starting_equity: float = 100.0
    budget_fraction: float = 0.90
    reserve_fraction: float = 0.10
    fee_bps: float = 4.0
    slippage_bps: float = 2.0
    annualization_bars: int = 365
    minimum_seeds_for_verdict: int = 30
    minimum_net_edge: float = 0.0
    ablations: AblationConfig = field(default_factory=AblationConfig)

    @classmethod
    def from_mapping(cls, value: Dict[str, object]) -> "ExperimentConfig":
        payload = dict(value)
        if "seeds" in payload:
            payload["seeds"] = tuple(payload["seeds"])
        if isinstance(payload.get("ablations"), dict):
            payload["ablations"] = AblationConfig(**payload["ablations"])
        return cls(**payload)

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported protocol_version")
        if self.benchmark_arm not in ARMS:
            raise ValueError("benchmark_arm must identify one of the four arms")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique tuple")
        if self.walk_forward_folds < 1 or self.purge_bars < 0:
            raise ValueError("fold count must be positive and purge non-negative")
        if not 0.0 < self.final_holdout_fraction < 0.5:
            raise ValueError("final_holdout_fraction must be between 0 and 0.5")
        if self.population_size < 2 or self.generations < 1:
            raise ValueError("population_size >= 2 and generations >= 1 are required")
        if self.evaluations_per_arm_fold != self.population_size * self.generations:
            raise ValueError("compute budget must equal population_size * generations for every arm/fold")
        if not math.isclose(self.budget_fraction + self.reserve_fraction, 1.0):
            raise ValueError("budget and reserve fractions must sum to exactly 1")
        if self.starting_equity <= 0 or min(self.fee_bps, self.slippage_bps) < 0:
            raise ValueError("equity must be positive and costs non-negative")


@dataclass(frozen=True)
class MarketData:
    """Chronological simple returns plus provenance supplied by the caller."""

    returns: Tuple[float, ...]
    data_id: str
    source_kind: str  # "synthetic", "empirical", or "unavailable"
    source_label: str
    timestamps: Tuple[str, ...] = ()
    signal_parity: str = "unverified"

    def __post_init__(self) -> None:
        if self.source_kind not in ("synthetic", "empirical", "unavailable"):
            raise ValueError("source_kind must be synthetic, empirical, or unavailable")
        if self.source_kind == "unavailable" and self.returns:
            raise ValueError("unavailable data must not contain fabricated returns")
        if self.timestamps and len(self.timestamps) != len(self.returns):
            raise ValueError("timestamps and returns must have equal lengths")
        if self.timestamps and (tuple(sorted(self.timestamps)) != self.timestamps or len(set(self.timestamps)) != len(self.timestamps)):
            raise ValueError("timestamps must be unique and strictly chronological")
        if any(not math.isfinite(value) or value <= -1.0 for value in self.returns):
            raise ValueError("returns must be finite simple returns greater than -100%")


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: int
    train_end: int  # exclusive
    validation_start: int
    validation_end: int  # exclusive
    purge_bars: int


@dataclass(frozen=True)
class SplitPlan:
    development_end: int
    folds: Tuple[Fold, ...]
    final_start: int
    final_end: int


@dataclass(frozen=True)
class Genome:
    momentum_window: int
    threshold_bps: float
    complexity: int


@dataclass(frozen=True)
class Metrics:
    net_twr: float
    max_drawdown: float
    sharpe: float
    tail_loss_5pct: float
    turnover: float
    activity: float
    diversification: float
    population_complexity: float
    capital_utilization: float


class Evaluator(Protocol):
    """Interface for an upstream genuine-data evaluator."""

    evaluator_id: str

    def score(self, genome: Genome, returns: Sequence[float], config: ExperimentConfig) -> float:
        """Return a training-only scalar fitness after declared costs."""

    def evaluate_population(
        self,
        population: Sequence[Genome],
        returns: Sequence[float],
        config: ExperimentConfig,
    ) -> Metrics:
        """Evaluate a frozen population on a held-out return slice."""


class SyntheticEvaluator:
    """Simple causal momentum proxy used for wiring and deterministic fixtures."""

    evaluator_id = "synthetic_momentum_proxy.v1"

    def _path(self, genome: Genome, values: Sequence[float], config: ExperimentConfig):
        prior: List[float] = []
        equity = 1.0
        peak = 1.0
        drawdown = 0.0
        positions: List[float] = []
        net_returns: List[float] = []
        costs = (config.fee_bps + config.slippage_bps) / 10000.0
        threshold = genome.threshold_bps / 10000.0
        for value in values:
            signal = sum(prior) / len(prior) if prior else 0.0
            position = 1.0 if signal > threshold else (-1.0 if signal < -threshold else 0.0)
            previous = positions[-1] if positions else 0.0
            turnover = abs(position - previous)
            net = position * value - turnover * costs
            equity *= max(1e-9, 1.0 + net)
            peak = max(peak, equity)
            drawdown = min(drawdown, equity / peak - 1.0)
            positions.append(position)
            net_returns.append(net)
            prior.append(value)
            if len(prior) > genome.momentum_window:
                prior.pop(0)
        return net_returns, positions, drawdown

    def score(self, genome: Genome, returns: Sequence[float], config: ExperimentConfig) -> float:
        path, _, _ = self._path(genome, returns, config)
        net_twr = math.prod(1.0 + value for value in path) - 1.0
        return net_twr - config.ablations.gene_cost_weight * genome.complexity

    def evaluate_population(
        self,
        population: Sequence[Genome],
        returns: Sequence[float],
        config: ExperimentConfig,
    ) -> Metrics:
        if not population or not returns:
            raise ValueError("population and evaluation returns must be non-empty")
        paths = [self._path(genome, returns, config) for genome in population]
        merged = [sum(path[0][i] for path in paths) / len(paths) for i in range(len(returns))]
        positions = [sum(path[1][i] for path in paths) / len(paths) for i in range(len(returns))]
        equity = 1.0
        peak = 1.0
        drawdown = 0.0
        for value in merged:
            equity *= max(1e-9, 1.0 + value)
            peak = max(peak, equity)
            drawdown = min(drawdown, equity / peak - 1.0)
        mean = statistics.fmean(merged)
        std = statistics.pstdev(merged)
        sorted_returns = sorted(merged)
        tail_count = max(1, math.ceil(len(sorted_returns) * 0.05))
        turnover = sum(abs(positions[i] - (positions[i - 1] if i else 0.0)) for i in range(len(positions)))
        activity = sum(1 for value in positions if value != 0.0) / len(positions)
        # Correlation is omitted for this proxy; the equal-weight population's
        # active-member fraction is a transparent diversity diagnostic only.
        active_genomes = len({(g.momentum_window, g.threshold_bps) for g in population})
        return Metrics(
            net_twr=equity - 1.0,
            max_drawdown=drawdown,
            sharpe=(mean / std * math.sqrt(config.annualization_bars)) if std else 0.0,
            tail_loss_5pct=statistics.fmean(sorted_returns[:tail_count]),
            turnover=turnover,
            activity=activity,
            diversification=active_genomes / len(population),
            population_complexity=statistics.fmean(g.complexity for g in population),
            capital_utilization=activity * config.budget_fraction,
        )


@dataclass
class ArmResult:
    arm: str
    seed: int
    walk_forward_metrics: List[Dict[str, object]]
    final_metrics: Optional[Dict[str, object]]
    evaluations: int
    frozen_genome: Optional[Dict[str, object]]


@dataclass
class ExperimentReport:
    protocol_version: str
    dataset: Dict[str, object]
    evaluator_id: str
    verdicts: Dict[str, str]
    comparisons: Dict[str, Dict[str, object]]
    population_stability: Dict[str, Dict[str, object]]
    config: Dict[str, object]
    split_plan: Optional[Dict[str, object]]
    arms: List[Dict[str, object]]
    limitations: List[str]
    observed_runtime_seconds: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, allow_nan=False)

    def to_markdown(self) -> str:
        rows = ["# ECO-12 experiment report", "", f"- Protocol: `{self.protocol_version}`", f"- Data: {self.dataset.get('source_label')} (`{self.dataset.get('source_kind')}`)", f"- Evaluator: `{self.evaluator_id}`", "", "**Evidence firewall:** all seeds reuse the same final market holdout. The displayed seeded spread is descriptive ONLY, never an independent-market confidence interval. No inferential 'beats'/'does_not_beat' verdict is permitted.", "", "## Verdicts (all not_comparable)", "", "| Comparison vs benchmark | Verdict | Paired mean final net TWR difference | Seed-spread band (descriptive only) |", "|---|---|---:|---:|"]
        for name, verdict in self.verdicts.items():
            comp = self.comparisons.get(name, {})
            interval = comp.get("seed_spread_95_descriptive")
            interval_text = "—" if interval is None else f"[{interval[0]:.4%}, {interval[1]:.4%}]"
            mean = comp.get("paired_mean_difference")
            mean_text = "—" if mean is None else f"{mean:.4%}"
            rows.append(f"| {name} | `{verdict}` | {mean_text} | {interval_text} |")
        rows.extend(["", "## Population stability", "", "| Arm | Frozen-genome similarity across seeds | Final net-TWR standard deviation | Seeds |", "|---|---:|---:|---:|"])
        for arm, values in self.population_stability.items():
            rows.append("| {} | {:.3f} | {:.4%} | {} |".format(arm, values["frozen_genome_similarity"], values["seed_net_twr_std"], values["seed_count"]))
        rows.extend(["", "## Interpretation", "", "All reported values inherit the dataset and evaluator limitations below. Synthetic-proxy outcomes are wiring checks, never trading evidence and not a profitability claim.", "", "## Limitations", ""])
        rows.extend(f"- {item}" for item in self.limitations)
        return "\n".join(rows) + "\n"


def build_split_plan(length: int, config: ExperimentConfig) -> SplitPlan:
    """Create expanding purged folds followed by a separate untouched tail."""
    final_size = max(1, int(math.ceil(length * config.final_holdout_fraction)))
    development_end = length - final_size
    validation_size = development_end // (config.walk_forward_folds + 2)
    if validation_size <= 0:
        raise ValueError("not enough observations for the configured split plan")
    folds = []
    for index in range(config.walk_forward_folds):
        validation_start = validation_size * (index + 1)
        validation_end = validation_start + validation_size
        train_end = validation_start - config.purge_bars
        if train_end <= 0 or validation_end > development_end:
            raise ValueError("not enough observations for train, purge, and validation windows")
        folds.append(Fold(index, 0, train_end, validation_start, validation_end, config.purge_bars))
    return SplitPlan(development_end, tuple(folds), development_end, length)


def make_synthetic_market(length: int = 300, seed: int = 7) -> MarketData:
    """Build an explicitly labelled deterministic synthetic return fixture."""
    if length < 40:
        raise ValueError("synthetic fixture needs at least 40 observations")
    rng = random.Random(seed)
    values: List[float] = []
    regime = 0.00025
    for i in range(length):
        if i and i % max(12, length // 5) == 0:
            regime = -regime if (i // max(12, length // 5)) % 2 else regime * 0.5
        values.append(max(-0.2, min(0.2, regime + rng.gauss(0.0, 0.012))))
    return MarketData(tuple(values), f"synthetic-fixture:{seed}:{length}", "synthetic", "deterministic synthetic fixture", signal_parity="none")


def _random_genome(rng: random.Random) -> Genome:
    return Genome(rng.randint(2, 20), rng.choice((0.0, 2.0, 5.0, 10.0, 20.0)), rng.randint(1, 4))


def _mutate(genome: Genome, rng: random.Random) -> Genome:
    window = max(2, min(20, genome.momentum_window + rng.choice((-3, -1, 1, 3))))
    threshold = max(0.0, min(20.0, genome.threshold_bps + rng.choice((-5.0, -2.0, 2.0, 5.0))))
    complexity = max(1, min(4, genome.complexity + rng.choice((-1, 0, 1))))
    return Genome(window, threshold, complexity)


def _evolve(
    arm: str,
    train_returns: Sequence[float],
    seed: int,
    fold: int,
    config: ExperimentConfig,
    evaluator: Evaluator,
) -> Tuple[List[Genome], int]:
    # Every arm starts with the exact same seed population and candidate count.
    rng = random.Random((seed + 1) * 1000003 + fold * 7919)
    population = [_random_genome(rng) for _ in range(config.population_size)]
    scores = [(evaluator.score(g, train_returns, config), g) for g in population]
    evaluations = len(population)
    archive = sorted(scores, key=lambda item: item[0], reverse=True)[: config.population_size]
    while evaluations + config.population_size <= config.evaluations_per_arm_fold:
        ranked = sorted(scores, key=lambda item: item[0], reverse=True)
        elite = [g for _, g in ranked[: max(2, config.population_size // 3)]]
        children: List[Genome] = []
        for i in range(config.population_size):
            if arm == "A_random":
                parent = rng.choice([g for _, g in scores])
            elif arm == "B_local":
                anchor = elite[i % len(elite)]
                neighborhood = [g for g in elite if abs(g.momentum_window - anchor.momentum_window) <= 5]
                parent = rng.choice(neighborhood or elite)
            elif arm == "C_hive":
                parent = rng.choice([g for _, g in archive])
            else:
                local = rng.choice(elite)
                hive = rng.choice([g for _, g in archive])
                parent = Genome(
                    int(round((local.momentum_window + hive.momentum_window) / 2)),
                    (local.threshold_bps + hive.threshold_bps) / 2.0,
                    max(local.complexity, hive.complexity),
                )
            children.append(_mutate(parent, rng))
        scores = [(evaluator.score(g, train_returns, config), g) for g in children]
        evaluations += len(children)
        archive = sorted(archive + scores, key=lambda item: item[0], reverse=True)[: config.population_size]
    return [g for _, g in sorted(scores, key=lambda item: item[0], reverse=True)], evaluations


def _metric_dict(metrics: Metrics) -> Dict[str, object]:
    return asdict(metrics)



def _comparison(candidate: Sequence[float], benchmark: Sequence[float]) -> Dict[str, object]:
    """Descriptive *same-market* seeded variation, never inferential p-values.

    Every candidate/benchmark seed sees the SAME final market time-series. Seed
    replicates are algorithmic variation, NOT independent market episodes.
    Neither confidence intervals nor Holm/Bonferroni p-values are justified
    from this nesting, even if a caller labels the source 'empirical'.
    """
    common = {
        "verdict": "not_comparable",
        "statistical_unit": "one_shared_market_holdout",
        "independent_market_holdouts": 1,
        "inferential_interval_valid": False,
        "p_value_two_sided": None,
        "p_value_holm": None,
        "confidence_interval_95": None,
        "familywise_interval_95": None,
        "reason": (
            "All seeds reuse one final market holdout. Independent market "
            "episodes and verified source/strategy/venue provenance are "
            "required before an inferential A/B/C/D verdict."
        ),
    }
    if len(candidate) != len(benchmark) or not candidate:
        return {**common, "paired_mean_difference": None,
                "seed_spread_95_descriptive": None,
                "seeds": len(candidate)}
    differences = [candidate[i] - benchmark[i] for i in range(len(candidate))]
    mean = statistics.fmean(differences)
    if len(differences) > 1:
        # Only a *descriptive* spread across random search seeds; 1.96 is not
        # an inferential market-confidence threshold and must not be called one.
        se = statistics.stdev(differences) / math.sqrt(len(differences))
        interval = (mean - 1.96 * se, mean + 1.96 * se)
    else:
        interval = None
    return {
        **common, "paired_mean_difference": mean,
        "seed_spread_95_descriptive": interval,
        "seeds": len(differences),
    }

def run_experiment(
    config: ExperimentConfig,
    market: MarketData,
    evaluator: Optional[Evaluator] = None,
) -> ExperimentReport:
    """Run common-seed A/B/C/D evolution, walk-forward, then one final holdout."""
    started = time.perf_counter()
    evaluator = evaluator or SyntheticEvaluator()
    limitations = [
        "No TradingView signal or fill parity is asserted; signal_parity is provenance metadata only.",
        "The built-in proxy does not model actual ecosystem breeding contracts, lifecycle, funding, venue fills, or capital ledger behavior.",
        "Groundwater inertia, hysteresis, and predator ceiling are recorded as ablation knobs but are not simulated by this evaluator.",
        "The synthetic strategy universe is a one-step momentum proxy and cannot support a profitability claim.",
        "Observed wall runtime is descriptive; deterministic compute fairness uses candidate evaluations, not elapsed time.",
        "CRITICAL: All random seeds share ONE final market holdout. Seed replicas are not independent market periods; inferential intervals, p-values and positive/negative significance verdicts are disabled.",
        "A user-supplied 'empirical' source label or external Evaluator ID is not independently verified source, venue, funding, signal parity or OOS evidence.",
    ]
    if not market.returns:
        return ExperimentReport(
            PROTOCOL_VERSION,
            {"data_id": market.data_id, "source_kind": "unavailable", "source_label": market.source_label, "signal_parity": market.signal_parity},
            getattr(evaluator, "evaluator_id", "external"),
            {arm: "not_comparable" for arm in ARMS if arm != config.benchmark_arm},
            {arm: {"verdict": "not_comparable", "reason": "no data supplied"} for arm in ARMS if arm != config.benchmark_arm},
            {}, asdict(config), None, [], limitations + ["No market observations were supplied; no results were fabricated."], time.perf_counter() - started,
        )
    plan = build_split_plan(len(market.returns), config)
    arm_results: List[ArmResult] = []
    for arm in ARMS:
        for seed in config.seeds:
            fold_metrics = []
            fold_champions: List[Tuple[Genome, float]] = []
            eval_count = 0
            for fold in plan.folds:
                train = market.returns[fold.train_start:fold.train_end]
                validation = market.returns[fold.validation_start:fold.validation_end]
                population, count = _evolve(arm, train, seed, fold.index, config, evaluator)
                eval_count += count
                metrics = evaluator.evaluate_population(population, validation, config)
                fold_metrics.append(_metric_dict(metrics))
                champion = population[0]
                fold_champions.append((champion, metrics.net_twr))
            # The holdout is inspected once after walk-forward comparison has
            # frozen one genome; it is never passed into search or selection.
            frozen, _ = max(fold_champions, key=lambda item: item[1])
            final_metrics = evaluator.evaluate_population(
                [frozen], market.returns[plan.final_start:plan.final_end], config
            )
            arm_results.append(ArmResult(arm, seed, fold_metrics, _metric_dict(final_metrics), eval_count, asdict(frozen)))
    by_arm = {arm: [r for r in arm_results if r.arm == arm] for arm in ARMS}
    population_stability: Dict[str, Dict[str, object]] = {}
    for arm, rows in by_arm.items():
        genomes = [r.frozen_genome for r in rows if r.frozen_genome]
        distances = []
        for i, left in enumerate(genomes):
            for right in genomes[i + 1:]:
                distances.append((
                    abs(int(left["momentum_window"]) - int(right["momentum_window"])) / 18.0
                    + abs(float(left["threshold_bps"]) - float(right["threshold_bps"])) / 20.0
                    + abs(int(left["complexity"]) - int(right["complexity"])) / 3.0
                ) / 3.0)
        net_values = [float(r.final_metrics["net_twr"]) for r in rows if r.final_metrics]
        population_stability[arm] = {
            "frozen_genome_mean_pairwise_distance": statistics.fmean(distances) if distances else 0.0,
            "frozen_genome_similarity": 1.0 - (statistics.fmean(distances) if distances else 0.0),
            "seed_net_twr_std": statistics.pstdev(net_values) if len(net_values) > 1 else 0.0,
            "seed_count": len(rows),
        }
    benchmark_returns = [r.final_metrics["net_twr"] for r in by_arm[config.benchmark_arm] if r.final_metrics]
    verdicts: Dict[str, str] = {}
    comparisons: Dict[str, Dict[str, object]] = {}
    # A single historical market tail is shared by every random seed. It is
    # pseudoreplication to use seed count as independent OOS market sample size.
    # This version records only descriptive paired differences; a future
    # verified multi-holdout market-aware evaluator must implement valid
    # cluster/block inference and predeclared multiple-testing correction.
    verdicts: Dict[str, str] = {}
    comparisons: Dict[str, Dict[str, object]] = {}
    for arm in ARMS:
        if arm == config.benchmark_arm:
            continue
        candidate_returns = [
            r.final_metrics["net_twr"] for r in by_arm[arm] if r.final_metrics
        ]
        comp = _comparison(candidate_returns, benchmark_returns)
        verdicts[arm] = "not_comparable"
        comparisons[arm] = comp
    return ExperimentReport(
        PROTOCOL_VERSION,
        {"data_id": market.data_id, "source_kind": market.source_kind, "source_label": market.source_label, "observations": len(market.returns), "signal_parity": market.signal_parity},
        evaluator.evaluator_id,
        verdicts,
        comparisons,
        population_stability,
        asdict(config),
        asdict(plan),
        [asdict(item) for item in arm_results],
        limitations,
        time.perf_counter() - started,
    )
