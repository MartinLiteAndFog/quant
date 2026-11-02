# ECO-09 — Predation, dormancy and confirmed-flat archive (paper research)

This module is a pure deterministic lifecycle reducer. It never opens/closes an exchange position, cancels orders, calls Railway, makes account transfers or changes NAV.

Predators: weakest, weighted-random and age-biased. Selection uses a stable simulation/day/hunter RNG seed and selects only plants with vitality below the adjustable 0.70 threshold. Plants in the 70–80% protection band and above 80% are not hunted. Seedlings have a 30-day grace, small effective samples a limited 90-day grace. Dormant cactus/bushes with no genuine eligible opportunity or mark-to-market loss can remain sheltered up to 365 days. Evidence must be causal and independently measured, not invented by the strategy.

Bark shielding is bounded by min(max_bark_shield, bark_strength * independent_bark_evidence * credibility). Genetically strong bark without evidence does not block predation. Bark never overrides account-level protective stops or hard loss limits.

Predation is two-phase: (1) an event changes status to closing and releases ZERO budget (even if the plant is inactive), without any flatten operation; (2) archive only after explicit positions_flat, orders_cancelled, fills_reconciled, free_margin_confirmed and a fresh snapshot of zero locked/reserved margin. Then ArchiveRecord retains genome version, history reference and account/tile identity; the event describes prior entitlement available for the separate ECO-05 allocator to redistribute. No cash is minted and no individual mushroom is rendered.

Adapters to ECO-07 metric results and ECO-03 daily clock should be versioned after their PRs are merged. CLI checks: PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_predation.py -v.
