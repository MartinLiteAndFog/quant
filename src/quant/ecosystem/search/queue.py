"""Deterministic candidate deduplication and bounded screening execution.

The evaluator is injected: this package does not define trading signals, fills,
or profitability assumptions. It schedules and caches pure calculations over
explicitly named data splits.
"""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Candidate:
    """A proposed configuration with a caller-stable identifier."""

    candidate_id: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        _canonical_json(self.config)


@dataclass(frozen=True)
class DataSplit:
    """A frozen, causal interval with half-open sample bounds [start, stop)."""

    name: str
    start: int
    stop: int
    dataset_fingerprint: str
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name or not self.dataset_fingerprint or not self.version:
            raise ValueError("split name, dataset fingerprint, and version are required")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("split bounds must satisfy 0 <= start < stop")

    @property
    def sample_count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class DataSplitPlan:
    """Chronological train/validation/OOS windows over one frozen dataset."""

    train: DataSplit
    validation: DataSplit
    out_of_sample: DataSplit

    def __post_init__(self) -> None:
        splits = (self.train, self.validation, self.out_of_sample)
        if tuple(split.name for split in splits) != ("train", "validation", "out_of_sample"):
            raise ValueError("split names must be train, validation, and out_of_sample")
        if len({split.dataset_fingerprint for split in splits}) != 1:
            raise ValueError("all splits must refer to the same dataset fingerprint")
        if not (self.train.stop <= self.validation.start and self.validation.stop <= self.out_of_sample.start):
            raise ValueError("split windows must be chronological and non-overlapping")


@dataclass(frozen=True)
class ComputeBudget:
    """Evaluation/sample caps plus an optional deadline for starting work.

    A deadline prevents starting additional batches; an evaluator already
    running in a worker is allowed to finish because the queue cannot safely
    interrupt arbitrary user-supplied evaluator code.
    """

    max_evaluations: int
    max_samples: int
    deadline_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_evaluations < 0 or self.max_samples < 0:
            raise ValueError("compute budgets cannot be negative")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds must be positive when supplied")


@dataclass(frozen=True)
class EvaluationResult:
    """Stable outcome for one proposal, including explicit skip reasons."""

    candidate_id: str
    cache_key: str
    status: str
    metrics: Mapping[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    reason: Optional[str] = None


@dataclass(frozen=True)
class ScreeningReport:
    results: Tuple[EvaluationResult, ...]
    unique_evaluations: int
    batch_count: int
    cache_hits: int
    budget_skips: int
    sample_count: int
    elapsed_seconds: float


Evaluator = Callable[[Mapping[str, Any], DataSplit, int], Mapping[str, Any]]


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("config must be JSON-compatible and contain finite numbers") from exc


def _stable_seed(cache_key: str) -> int:
    return int.from_bytes(hashlib.sha256(cache_key.encode("ascii")).digest()[:8], "big")


def _cache_key(config: Mapping[str, Any], split: DataSplit, evaluator_version: str) -> str:
    payload = {
        "config": json.loads(_canonical_json(config)),
        "split": {
            "dataset_fingerprint": split.dataset_fingerprint,
            "name": split.name,
            "start": split.start,
            "stop": split.stop,
            "version": split.version,
        },
        "evaluator_version": evaluator_version,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _run_evaluation(
    evaluator: Evaluator,
    config: Mapping[str, Any],
    split: DataSplit,
    seed: int,
) -> Dict[str, Any]:
    """Top-level worker entry point so process workers can pickle it."""
    return dict(evaluator(config, split, seed))


def _run_batch(
    evaluator: Evaluator,
    items: Iterable[Tuple[str, Mapping[str, Any], int]],
    split: DataSplit,
) -> Dict[str, Mapping[str, Any]]:
    """Run a batch in one worker task to amortize process scheduling overhead."""
    return {
        key: _run_evaluation(evaluator, config, split, seed)
        for key, config, seed in items
    }


class ScreeningQueue:
    """Evaluate each unique config/split once, with stable budget admission.

    Admission is made in sorted cache-key order, independent of proposal order.
    Process workers may finish in any order; result rows are sorted by candidate
    ID. Duplicate proposals share a result. The cache survives repeated runs on
    this queue instance.
    """

    def __init__(
        self,
        evaluator: Evaluator,
        evaluator_version: str,
        workers: int = 1,
        batch_size: int = 32,
    ) -> None:
        if not evaluator_version:
            raise ValueError("evaluator_version must be non-empty")
        if workers < 1:
            raise ValueError("workers must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.evaluator = evaluator
        self.evaluator_version = evaluator_version
        self.workers = workers
        self.batch_size = batch_size
        self._cache: Dict[str, Mapping[str, Any]] = {}

    def run(
        self,
        candidates: Iterable[Candidate],
        split: DataSplit,
        budget: ComputeBudget,
    ) -> ScreeningReport:
        started = time.monotonic()
        deadline = started + budget.deadline_seconds if budget.deadline_seconds is not None else None
        candidate_rows = []
        unique: Dict[str, Tuple[Mapping[str, Any], int]] = {}
        for candidate in candidates:
            key = _cache_key(candidate.config, split, self.evaluator_version)
            candidate_rows.append((candidate.candidate_id, key))
            if key not in self._cache and key not in unique:
                unique[key] = (json.loads(_canonical_json(candidate.config)), _stable_seed(key))

        admitted = []
        rejected = set()
        used_samples = 0
        for key in sorted(unique):
            if len(admitted) >= budget.max_evaluations or used_samples + split.sample_count > budget.max_samples:
                rejected.add(key)
                continue
            if deadline is not None and time.monotonic() >= deadline:
                rejected.add(key)
                continue
            admitted.append(key)
            used_samples += split.sample_count

        batches = [
            admitted[index : index + self.batch_size]
            for index in range(0, len(admitted), self.batch_size)
        ]
        outcomes: Dict[str, Mapping[str, Any]] = {}
        started_batches = 0
        if self.workers == 1:
            for batch in batches:
                if deadline is not None and time.monotonic() >= deadline:
                    rejected.update(batch)
                    continue
                items = [(key, unique[key][0], unique[key][1]) for key in batch]
                started_batches += 1
                outcomes.update(_run_batch(self.evaluator, items, split))
        elif admitted:
            executor = ProcessPoolExecutor(max_workers=self.workers)
            futures = {}
            next_batch = 0
            try:
                while next_batch < len(batches) or futures:
                    while next_batch < len(batches) and len(futures) < self.workers:
                        if deadline is not None and time.monotonic() >= deadline:
                            for pending_batch in batches[next_batch:]:
                                rejected.update(pending_batch)
                            next_batch = len(batches)
                            break
                        batch = batches[next_batch]
                        future = executor.submit(
                            _run_batch,
                            self.evaluator,
                            [(key, unique[key][0], unique[key][1]) for key in batch],
                            split,
                        )
                        futures[future] = batch
                        started_batches += 1
                        next_batch += 1
                    if futures:
                        completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in completed:
                            outcomes.update(future.result())
                            del futures[future]
            finally:
                executor.shutdown(wait=True, cancel_futures=True)

        self._cache.update(outcomes)
        results = []
        cache_hits = 0
        for candidate_id, key in sorted(candidate_rows, key=lambda row: (row[0], row[1])):
            if key in self._cache:
                hit = key not in outcomes
                cache_hits += int(hit)
                results.append(EvaluationResult(candidate_id, key, "evaluated", deepcopy(self._cache[key]), hit))
            else:
                results.append(
                    EvaluationResult(
                        candidate_id,
                        key,
                        "skipped",
                        reason="compute_budget_exhausted" if key in rejected else "not_evaluated",
                    )
                )
        return ScreeningReport(
            results=tuple(results),
            unique_evaluations=len(outcomes),
            batch_count=started_batches,
            cache_hits=cache_hits,
            budget_skips=sum(result.status == "skipped" for result in results),
            sample_count=len(outcomes) * split.sample_count,
            elapsed_seconds=max(0.0, time.monotonic() - started),
        )
