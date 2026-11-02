"""Run ECO-12's labelled synthetic wiring demo or an empty abstention report."""

import argparse
import json
from pathlib import Path

from .core import ExperimentConfig, MarketData, make_synthetic_market, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-demo", action="store_true", help="run the deterministic, non-evidentiary synthetic fixture")
    parser.add_argument("--config", help="JSON experiment configuration")
    parser.add_argument("--json", default="eco12-result.json", help="machine-readable result path")
    parser.add_argument("--report", default="eco12-report.md", help="human-readable report path")
    args = parser.parse_args()
    config = ExperimentConfig.from_mapping(json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else ExperimentConfig()
    market = make_synthetic_market() if args.synthetic_demo else MarketData((), "none", "unavailable", "no dataset supplied")
    result = run_experiment(config, market)
    Path(args.json).write_text(result.to_json() + "\n", encoding="utf-8")
    Path(args.report).write_text(result.to_markdown(), encoding="utf-8")
    print("Wrote {} and {}".format(args.json, args.report))


if __name__ == "__main__":
    main()
