# Historical execution contract

## Data and clocks

The runtime supports distinct `SOL-USDT` and `BTC-USDT` manifests and daily asset directories. Symbol identity, source SHA256, day byte checksums, observed row counts and UTC timestamp bounds are validated before execution. Neither rendering nor ecology may manufacture market candles. Partial days execute their observations; gaps reset indicators according to the engine's discontinuity policy while preserving account exposure and equity.

SOL: 3,109,186 real minute rows, 2020-08-11–2026-07-11. BTC: 3,084,497 real minute rows, 2020-08-12–2026-06-24. Source location does not prove venue, instrument or license provenance. The normalized linear-futures account is an explicit model; unavailable exchange funding is excluded.

A date's trading is completed before its observed volume/range/direction drives the ecological update. Newborn plants first trade on later observations. Sizing budgets and halt state are frozen from the pre-day information set for concurrent Kraken accounts; source-fixture accounts freeze each contiguous segment's information set. Prior results, not later same-day outcomes, determine selection and allocation weights.

## Effective strategy

Initial compiled profiles are Kraken Legacy, Countervariante, Kraken 17.2 close, Kraken 17.2 reverse and Pure ImbaTP. A new world chooses three parent profiles. Sensitivity 17.2 with the source multiplier means 172 signal-Renko bars, not 17.2 candles. The latest templates retain 0.5% trailing distance, 2.8% partial TP, independent 3.15% CT TP1 and WAIT-stop differences. Original leverage and sizing remain provenance; execution caps notional at 1× and enforces a protective stop.

The source-derived engine aggregates completed five-minute observations into a Renko signal proxy. Renko prices never become execution prices. Decisions enqueue orders for subsequent observed minute opens; fees and adverse slippage are modeled. Protective stop/TP ambiguity resolves conservatively. A gap through an already armed stop fills at the observed worse open. Zero-volume observations do not provide executable liquidity.

Positions, orders, ATR, confirmed EW pivots, Renko history and CT TTP/WAIT or TF state persist between dates. The historical legacy source-fixture branch remains daily/segment-flat and must not be confused with the persistent Kraken branch.

## Atomic genes and availability

The catalog separates 12 single-purpose genes from seven example recipes. Each atom records required observations, historical lookback, confirmation delay, availability and earliest execution. EW pivots require their right-hand confirmation candles; drawing a mark at an older pivot index never authorizes trading at that older time. Confirmed EW bias may filter an IMBA entry; confirmed impulses may trigger exits. Risk exits remain independent of entry filters.

Graph validation rejects unknown genes, cycles, missing entry nodes and any absent or conditionally gated compulsory stop-loss. Logical functions are typed documented composition designs, not arbitrary runtime code. Named compiled strategies are executable; custom snippets and unsupported compositions remain non-executable research text. Metadata documents a contract; actual prefix-invariance, confirmation and execution-time tests are required to substantiate it.

Children combine and mutate bounded actual execution fields and EW genes once at birth. The frozen effective profile is persisted with the plant. Parent ranking and evidence weighting use previously completed measured results after the observation gate. This evolutionary selection is not an independent performance evaluation.

## Capital, population and repeat

Each garden starts independently at 100 USDT, protects 10% of current NAV and keeps one conserved ledger. At subsequent simulated calendar-month transitions, 100 EUR enters at a fixed disclosed 1:1 simulation conversion; restored states cannot book the same contribution twice. External deposits, realized price PnL, unrealized marks and costs are separate fields. The invariant is NAV = initial capital + deposits + trading result, subject to explicit ledger precision.

Dynamic births spend free capital; there is no fixed number of historical planting places. At most 200 plants are alive simultaneously. Predation cannot silently erase an open account. Recent histories, trades and retired-plant detail are bounded; cumulative counters preserve lifetime accounting. Legacy synthetic demos keep their original 20-slot rules.

A declared exit-only window starts in the final 60 observed archive minutes. The boundary verifies no pending exposure, then repeats the dataset with accounts, population and genomes intact. It does not reset NAV to 100 or replace losses with artificial equity. A capital halt continues observations and may be released by the next real simulated contribution. Missing exit liquidity fails explicitly rather than fabricating settlement.

## Native presentation and verification

The Unreal target consumes locally generated immutable snapshots and writes local play/pause/step commands. Native actors/HUD must expose the same measured ledger and effective genes as the reference engine. A TypeScript build or native compilation alone does not prove selection, controls, readability or packaged offline behavior; inspect the native status matrix and actual runtime evidence.

Required tests include source checksums, symbol isolation, whole-versus-split state equality, future-prefix invariance, cross-plant ordering invariance, pivot confirmation delay, bounded mutation, mandatory stops, contribution idempotence, population safety and long-run retention. Current daily-end drawdown metrics are not intraday drawdown and must be labeled accordingly.

Benchmark: cash/no trade. Verdict: `not_comparable` for independent market performance until provenance, cost calibration, frozen holdout/forward evaluation and promotion gates are established. Replaying identical history repeatedly is training; it does not generate independent evidence or prove a profitable adaptive trading system.
