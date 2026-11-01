# ECO-12 statistical experiment protocol

Protocol `eco12.v1` compares four search policies against A, the random-mutation
benchmark: A random mutation, B local pollination, C central hive, and D hybrid.
This package owns only the experiment contract; it does not import or modify the
upstream market-data, ecology, metrics, breeding, or compute packages. Its
`Evaluator` protocol is the adapter point for those packages once integrated.

## Fairness and causal splits

All arms share the same ordered seed list, starting genome population, market
history, initial equity, 90% budget / 10% reserve, fees, slippage, and candidate
evaluation count per fold. The default uses 30 common seeds, 3 expanding
walk-forward folds, a one-bar purge, 12 candidates over 6 generations (72
evaluations per arm/fold), 4 bps fee plus 2 bps slippage, and a final 20% tail.
The final tail is excluded from search and walk-forward selection, then opened
once to score a frozen genome. The development portion after the last validation
window remains unused; this conservative gap prevents quietly enlarging the
training set.

The built-in `SyntheticEvaluator` is a deterministic, causal momentum proxy.
It verifies replay, equal compute, split handling, costs, and report generation.
It does not execute the application's strategies, simulate the ecosystem's
ledger/breeding/lifecycle rules, use historical venue fills, or establish
TradingView parity. A synthetic run always returns `not_comparable`. Calling
the module without a data source writes an abstention report and no outcomes.

## Metrics and verdicts

The evaluator contract reports net TWR, maximum drawdown, annualized Sharpe,
5% tail mean, turnover, activity, a clearly limited population diversity proxy,
genome complexity, and capital utilization. Each seed records walk-forward and
final metrics, candidate evaluations, and the frozen genome. Runtime is observed
and reported, but deterministic fairness is based on evaluation count.

The predeclared primary benchmark is A. B/C/D are paired to A by seed and each
challenger receives one family-wise comparison. The report uses a normal
approximation for paired two-sided p-values, Holm step-down adjustment, and a
conservative Bonferroni family-wise interval.
`beats` requires genuine empirical provenance, a non-synthetic evaluator, the
minimum number of common seeds, a positive predeclared edge, and adjusted
significance. `does_not_beat` requires the adjusted interval's upper bound to
be at or below the edge threshold. Everything else is `not_comparable`.
Exploratory gene, ecology, or threshold sweeps create additional hypotheses and
must be preregistered or corrected as a larger family; they cannot reuse the
same untouched final split repeatedly and call each pass confirmatory.

## Ablations and current limits

Gene cost defaults to zero and is applied as a training-selection penalty.
Groundwater inertia, hysteresis, and predator vitality ceiling are versioned
configuration fields so the protocol can record future ablations. The synthetic
proxy does not implement those mechanisms, and its report says so. Until real
contracts and a validated evaluator arrive, no result from this package can
promote a strategy or authorize capital, alerts, orders, transfers, or deployment.
An empirical data manifest must also establish source licensing, permitted use,
and exact instrument/product identity before its results can be considered
comparable. A pair label or vendor FAQ alone is insufficient qualification.

## Run

From the repository root:

```bash
PYTHONPATH=src python -m quant.ecosystem.experiments
PYTHONPATH=src python -m quant.ecosystem.experiments --synthetic-demo
PYTHONPATH=src python -m quant.ecosystem.experiments --config src/quant/ecosystem/experiments/experiment.example.json --synthetic-demo
```

The first command creates machine-readable and Markdown abstention reports. The
second runs the labelled synthetic wiring demo. Both write local output files;
neither connects to an exchange or external service.
