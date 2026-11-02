# Agent A: Cloudgarden data provenance, CI and replay audit

**State at implementation:** the original SOL Parquet has not been supplied to this agent or committed to GitHub. Original-byte verification and historical 140-day reproduction remain BLOCKED, never passed by substituting generated bars.

## Frozen reference (developer-reported, not independently reproduced)

- Expected original: research/solusdt_robust_growth_v1/immutable/SOL-USDT_1m_20260711T092900Z.parquet
- SHA256: fa3e689ddbc42b35b89f66db4240521a330896e2d89fc2d2df8624bc749fda64
- 3,109,186 rows; 2020-08-11 06:00 UTC through 2026-07-11 09:29 UTC; 10 gaps / 1,424 missing minutes; no duplicates.
- Reported 140 days, 2026-01-01 through 2026-05-20: 28,078 fills, final NAV 31.3791 from 100, modeled fees 45.9937, 20/20 seedling entitlements used.
- A matching checksum identifies these original bytes, not their exchange venue, contractual identity or license. A historical paper fill is not an exact live strategy fill.

## Provenance still unverified

| Information | Status | Required evidence |
|---|---|---|
| Original vendor and acquisition date | unverified | original download log, data provider/export reference |
| Exact exchange, ticker and instrument | unverified | exchange export/metadata |
| Spot or futures, settlement and contract multiplier | unverified | official historical product terms |
| Timestamp open/close meaning and source timezone | unverified | vendor timestamp specification (UTC alignment alone does not prove candle-open labels) |
| Real trading fees and historical funding | unverified | venue-specific funding records and fee schedules |
| Research/derivative redistribution license | unverified | provider license and rights evidence |

No trade is permitted on these findings. Current simulated 1x linear-futures exposure is a hypothesis; venue calibration, signal parity and OOS performance remain not_comparable. Do not distribute derivative day files without permission.

## Public PR CI (safe with no historical archive)

.github/workflows/cloudgarden-ci.yml runs npm ci, npm test, a checked-in SIX-MINUTE explicitly synthetic OHLCV-to-fill fixture (CC0), and npm run build. It rejects accidentally present public/market/manifest.json. Synthetic test pass is not market evidence.

## Manual private replay (only authorized read-only original + self-hosted runner)

.github/workflows/cloudgarden-private-history.yml is workflow_dispatch ONLY and requires a restricted self-hosted Linux x64 runner additionally labeled cloudgarden-private-history. Restrict the GitHub environment cloudgarden-private-history with required approvers. Set environment variable SOL_PRIVATE_PARQUET to the absolute path of the read-only local original accessible to that runner. Do not copy the raw archive to GitHub or a public CI runner.

The workflow checks SHA256 over original bytes, scans all UTC timestamps and OHLCV rows, validates full count/gaps/time boundaries, then uses export_sol_history.py to write temporary local daily assets. It executes the 140-day historical loop against those files. The verifier requires the frozen source SHA/row count, fill count, NAV, fees and seedling entitlement reference to agree. The workflow emits only metadata-only JSON in the GitHub job summary, never uploads any market artifact, then removes temporary assets.

Authorized local commands, from repo root:

    python -m pip install 'pyarrow==22.0.0'
    python apps/cloudgarden/scripts/audit_sol_source.py --source /AUTHORIZED/READ_ONLY/SOL.parquet --report /PRIVATE/TEMP/source-audit.json
    python apps/cloudgarden/scripts/export_sol_history.py --source /AUTHORIZED/READ_ONLY/SOL.parquet --output /PRIVATE/TEMP/market
    cd apps/cloudgarden
    npm ci
    CLOUDGARDEN_MARKET_DIR=/PRIVATE/TEMP/market CLOUDGARDEN_RESULT_MANIFEST=/PRIVATE/TEMP/replay.json npm run verify:history -- 140

When original data is missing, fail visibly. Do NOT fabricate candles. A matched private replay is integration evidence only, not alpha proof.

## Acceptance status

- Public CI: PENDING first PR run on Agent A commit.
- SHA256, 3,109,186 rows, exact UTC gap audit: BLOCKED missing private file.
- Exact provider, venue, license, contract multiplier, funding: UNVERIFIED.
- Independent 140-day replay manifest: BLOCKED missing private file and authorized runner.
- Live orders, deployments, automatic promotion: PROHIBITED.
