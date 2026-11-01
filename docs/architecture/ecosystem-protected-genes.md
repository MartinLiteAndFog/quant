# ECO-04: protected demo genomes

Schema version 2 adds the ECO-01 canonical `protective_stop`, `max_leverage`,
`max_margin_fraction`, and `reproduction_budget` policies. Library migration
preserves v1 custom genes and adds missing defaults. Explicit library validation
rejects removed or altered mandatory definitions. Existing sensitivity/TTP axes
stay required for backward compatibility with the synthetic meadow.

The defaults (1–3 leverage, 1–90% margin, 8–15% reproduction) are bounded simulation
parameters, not live authorization. Global account limits take precedence.
`stop_loss_pct` remains optional tuning; deleting it leaves the protective stop
active and resolves to 1.8%, limited by the caller's positive global cap.

All five crossover algorithms retain required genes. Immutable policies skip
mutation entirely. Optional dropout remains 5%, or 10% for the pruner. Invalid
values, unknown genes, unsupported bee types and invalid RNG output reject.
Missing required values in legacy demo genomes are restored from safe defaults.
Complexity penalty defaults exactly to zero.

The module catalogue covers IMBA, TTP, protective stop, RSI and ADX. Entries are
versioned descriptors with closed-bar data access and a global-cap risk contract.
They explicitly carry `descriptor_only` and `parity: unverified`; this package
provides neither arbitrary-code execution nor TradingView parity.

Validation: original smoke test, 500 seeded crossover/migration invariants,
module-resolution/stop-cap tests, and TypeScript/Vite production build.
