"""Reproducible synthetic smoke benchmark for ECO-10 screening.

Examples:
  python benchmarks/ecosystem/screening.py --quick
  python benchmarks/ecosystem/screening.py --counts 100 1000 10000 --samples 256 --workers 2

The fixture is synthetic screening work, not a market backtest or a 5-year
SOL throughput estimate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quant.ecosystem.search import Candidate, ComputeBudget, DataSplit, ScreeningQueue


def synthetic_screen(config: Mapping[str, Any], split: DataSplit, seed: int) -> Mapping[str, Any]:
    """Small deterministic CPU workload that visits every synthetic sample."""
    rng = random.Random(seed)
    scale = float(config["scale"])
    threshold = float(config["threshold"])
    hits = 0
    total = 0.0
    for _ in range(split.sample_count):
        value = rng.random() * scale
        total += math.sin(value)
        hits += value >= threshold
    return {"synthetic_mean": total / split.sample_count, "synthetic_hit_rate": hits / split.sample_count}


def _hardware() -> Dict[str, Any]:
    model = platform.processor() or "unknown"
    memory_bytes = None
    chip = None
    if sys.platform == "darwin":
        try:
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
        try:
            memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
        if chip is None or memory_bytes is None:
            try:
                overview = subprocess.check_output(
                    ["system_profiler", "-detailLevel", "mini", "SPHardwareDataType"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                for line in overview.splitlines():
                    label, separator, value = line.strip().partition(":")
                    if not separator:
                        continue
                    if label == "Chip":
                        chip = value.strip()
                    elif label == "Memory" and value.strip().endswith(" GB"):
                        memory_bytes = int(float(value.strip()[:-3]) * 1024 ** 3)
                    elif label == "Model Identifier":
                        model = value.strip()
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": chip or model,
        "logical_cpus": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "python": sys.version.split()[0],
    }


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def run_benchmark(count: int, samples: int, workers: int) -> Dict[str, Any]:
    proposals = []
    for index in range(count):
        config_index = index
        proposals.append(
            Candidate(
                "candidate-%06d" % index,
                {"scale": 0.5 + (config_index % 97) / 100.0, "threshold": (config_index % 53) / 100.0},
            )
        )
    unique_count = len({json.dumps(candidate.config, sort_keys=True) for candidate in proposals})
    split = DataSplit("synthetic-smoke", 0, samples, "synthetic-bars-v1")
    queue = ScreeningQueue(synthetic_screen, evaluator_version="synthetic-screen-v1", workers=workers)
    cpu_before = os.times()
    rss_before = _peak_rss_bytes()
    started = time.perf_counter()
    report = queue.run(proposals, split, ComputeBudget(max_evaluations=count, max_samples=count * samples))
    elapsed = time.perf_counter() - started
    cpu_after = os.times()
    cpu_seconds = (cpu_after.user - cpu_before.user) + (cpu_after.system - cpu_before.system)
    samples_processed = report.unique_evaluations * samples
    return {
        "proposals": count,
        "unique_configs_requested": unique_count,
        "unique_evaluations": report.unique_evaluations,
        "worker_batches": report.batch_count,
        "cache_hits": report.cache_hits,
        "workers": workers,
        "samples_per_evaluation": samples,
        "synthetic_samples_processed": samples_processed,
        "elapsed_seconds": round(elapsed, 6),
        "proposals_per_second": round(count / elapsed, 2) if elapsed else None,
        "unique_evaluations_per_second": round(report.unique_evaluations / elapsed, 2) if elapsed else None,
        "synthetic_samples_per_second": round(samples_processed / elapsed, 2) if elapsed else None,
        "aggregate_process_cpu_seconds": round(cpu_seconds, 6),
        "aggregate_cpu_cores_used": round(cpu_seconds / elapsed, 3) if elapsed else None,
        "parent_process_peak_rss_bytes_before": rss_before,
        "parent_process_peak_rss_bytes_after": _peak_rss_bytes(),
        "worker_rss_included": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--samples", type=int, default=256, help="synthetic samples visited per unique proposal")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="use 32 samples and run 100/1000/10000 proposals")
    parser.add_argument("--json", type=Path, help="optional output path for the JSON report")
    args = parser.parse_args()
    if any(count < 0 for count in args.counts) or args.samples < 1:
        parser.error("counts must be non-negative and samples must be positive")
    samples = 32 if args.quick else args.samples
    report = {
        "benchmark": "eco10-synthetic-screening-v1",
        "warning": "Synthetic CPU fixture only; not a market backtest or 5-year SOL throughput estimate.",
        "hardware": _hardware(),
        "runs": [run_benchmark(count, samples, args.workers) for count in args.counts],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
