# ECO-02 offline market data and paper execution

The prototype ships generated **synthetic** data only: 10,080 minute bars (one week),
CC0 fixture provenance. No historical SOL dataset or TradingView parity is claimed.
No exchange, network, credential, order, or transfer API is used.

Run `PYTHONPATH=src python -m quant.ecosystem.market_data --output /tmp/eco-week`.
For supplied licensed data, pass `--csv path --provenance metadata.json`, optionally
`--start` and `--end` (exclusive). CSV columns are timestamp,open,high,low,close,volume.
Timestamps must include timezone offsets; bars normalize to UTC open timestamps.
Metadata fields: source, license, symbol, market_type (`spot` or `linear_futures`),
contract_multiplier (decimal string), interval_seconds, synthetic. UTC/open are
required normalized conventions. Never invent missing license or provenance.

Normalizer rejects duplicate, nonfinite, invalid OHLCV and off-grid records; sorts
bars deterministically; reports missing interior/boundary bars without interpolation.
Exports bars.jsonl and quality.json with provenance and content SHA-256. A single
symbol cannot establish universe survivorship safety. Dataset revisions/source
quality remain external responsibilities. No five-year dataset has been imported;
full-history memory/throughput has not been benchmarked.

`paper_fill(PaperOrder, Dataset, Costs)` handles market orders only. Signal time,
feature availability and receipt are distinct. First bar open **strictly after**
receipt and availability is eligible (conservative one-bar latency). The maximum
delay is measured from availability so late signals cannot execute indefinitely.
Zero volume rejects. Costs are fixed configurable bps: half spread plus adverse
slippage in price, fees on filled notional, contract multiplier included. These are
assumptions, not calibrated venue estimates. Volume participation, order book,
limit/stop intrabar ordering and liquidation are not modeled; do not claim them.

`funding_cashflow` is linear-futures-only, signed positive cash received; positive
funding rate charges longs and credits shorts. Holding interval is (open, close],
so same-timestamp opening pays none and closing pays. Caller supplies verified
funding rate/mark and calls exactly once per settlement. Stateless fills/funding
must be deduplicated by the orchestrating ledger. `PaperFill.slippage_cost` includes
spread and slippage already embedded in price; do not subtract it twice.

All prices/fees use Decimal. Canonical adapter `to_contract_fill` lazily imports
ECO-01, requiring that package when exporting. Funding is a separate settlement
rather than guessed inside an individual fill. Benchmark: no-trade/cash; research
verdict **not_comparable** (fixture validates mechanics, not trading edge).
