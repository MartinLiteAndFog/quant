from __future__ import annotations

import random
from typing import Any, Mapping

import pytest

from src.quant.ecosystem.search import Candidate, ComputeBudget, DataSplit, DataSplitPlan, ScreeningQueue


def seeded_evaluator(config: Mapping[str, Any], split: DataSplit, seed: int) -> Mapping[str, Any]:
    rng = random.Random(seed)
    return {"score": round(float(config["x"]) + rng.random() + split.sample_count / 1000, 12)}


def candidates(order=(0, 1, 2, 3)):
    configs = [{"x": 0.25}, {"x": 0.5}, {"x": 0.25}, {"x": 0.75}]
    return [Candidate("c%d" % index, configs[index]) for index in order]


def test_deduplicates_and_is_order_independent_with_process_workers():
    split = DataSplit("train", 0, 100, "bars-sha")
    serial = ScreeningQueue(seeded_evaluator, "eval-v1", workers=1).run(
        candidates(), split, ComputeBudget(max_evaluations=4, max_samples=400)
    )
    parallel = ScreeningQueue(seeded_evaluator, "eval-v1", workers=2).run(
        candidates((3, 2, 1, 0)), split, ComputeBudget(max_evaluations=4, max_samples=400)
    )
    by_id_serial = {row.candidate_id: (row.cache_key, dict(row.metrics), row.status) for row in serial.results}
    by_id_parallel = {row.candidate_id: (row.cache_key, dict(row.metrics), row.status) for row in parallel.results}
    assert by_id_serial == by_id_parallel
    assert serial.unique_evaluations == parallel.unique_evaluations == 3
    assert serial.results[0].cache_key == serial.results[2].cache_key


def test_cache_key_includes_split_and_evaluator_version():
    candidate = [Candidate("one", {"x": 1})]
    first = ScreeningQueue(seeded_evaluator, "eval-v1")
    base_split = DataSplit("train", 0, 5, "bars-sha")
    initial = first.run(candidate, base_split, ComputeBudget(2, 10))
    hit = first.run(candidate, base_split, ComputeBudget(2, 10))
    changed_split = first.run(candidate, DataSplit("validation", 5, 10, "bars-sha"), ComputeBudget(2, 10))
    changed_eval = ScreeningQueue(seeded_evaluator, "eval-v2").run(candidate, base_split, ComputeBudget(2, 10))
    assert initial.results[0].cache_key == hit.results[0].cache_key
    assert hit.cache_hits == 1 and hit.unique_evaluations == 0
    assert initial.results[0].cache_key != changed_split.results[0].cache_key
    assert initial.results[0].cache_key != changed_eval.results[0].cache_key


def test_returned_metrics_cannot_mutate_the_queue_cache():
    queue = ScreeningQueue(seeded_evaluator, "eval-v1")
    candidate = [Candidate("one", {"x": 1})]
    split = DataSplit("train", 0, 5, "bars-sha")
    first = queue.run(candidate, split, ComputeBudget(1, 5))
    expected = dict(first.results[0].metrics)
    first.results[0].metrics["score"] = -999
    cached = queue.run(candidate, split, ComputeBudget(1, 5))
    assert dict(cached.results[0].metrics) == expected
    assert cached.results[0].cache_hit


def test_budget_admission_is_stable_and_reports_skips():
    split = DataSplit("screen", 0, 10, "fixture")
    queue = ScreeningQueue(seeded_evaluator, "v1")
    first = queue.run(candidates(), split, ComputeBudget(max_evaluations=2, max_samples=20))
    second = ScreeningQueue(seeded_evaluator, "v1").run(
        candidates((3, 2, 1, 0)), split, ComputeBudget(max_evaluations=2, max_samples=20)
    )
    selected_first = {row.cache_key for row in first.results if row.status == "evaluated"}
    selected_second = {row.cache_key for row in second.results if row.status == "evaluated"}
    assert selected_first == selected_second
    assert first.unique_evaluations == 2
    assert first.budget_skips == 2
    assert first.sample_count == 20


def test_config_validation_rejects_non_finite_values():
    with pytest.raises(ValueError, match="JSON-compatible"):
        Candidate("bad", {"threshold": float("nan")})


def test_deadline_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        ComputeBudget(1, 1, deadline_seconds=0)


def test_split_plan_requires_chronological_non_overlapping_windows():
    with pytest.raises(ValueError, match="chronological"):
        DataSplitPlan(
            DataSplit("train", 0, 10, "bars"),
            DataSplit("validation", 8, 15, "bars"),
            DataSplit("out_of_sample", 15, 20, "bars"),
        )
