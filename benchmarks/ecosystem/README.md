# ECO-10 screening benchmark

Run `python benchmarks/ecosystem/screening.py --quick` for deterministic 100,
1,000, and 10,000-proposal smoke runs. Use `--samples N --workers N --json
report.json` to record a longer run. The fixture visits synthetic samples and
reports proposal throughput, unique evaluation throughput, synthetic samples
per second, worker batch count, aggregate process CPU time, and parent-process peak RSS.
An optional `deadline_seconds` budget stops new batches from starting when it
expires; batches already running are allowed to finish.

These measurements describe only the synthetic screening workload on the
reported machine. The peak RSS field excludes worker processes, so it is not a
total-memory measurement for multi-process runs. It is not a trading backtest,
does not establish strategy quality, and must not be extrapolated to five years
of SOL minute data. No real dataset is embedded or implied.

## Recorded smoke run

On a MacBook Pro Mac15,6 with Apple M3 Pro (11 CPU cores), 18 GB RAM, macOS
26.6.2, and Python 3.14.6, `--samples 1024` produced:

| Proposals | Unique evaluations after dedup | 1 worker elapsed / unique per second | 2 workers elapsed / unique per second |
|---:|---:|---:|---:|
| 100 | 100 | 0.008055 s / 12,415 | 0.079290 s / 1,261 |
| 1,000 | 1,000 | 0.080695 s / 12,392 | 0.118621 s / 8,430 |
| 10,000 | 5,141 | 0.449703 s / 11,432 | 0.358842 s / 14,327 |

This short run shows process startup dominates smaller batches; two workers were
about 1.25x faster for the 10,000-proposal fixture after deduplicating 4,859
repeated configurations. Each unique evaluation scanned 1,024 generated
synthetic samples. The result is only a scheduling smoke measurement; it gives
no estimate for market backtests or five years of SOL data.
