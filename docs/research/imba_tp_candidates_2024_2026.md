# IMBA TP +5% research candidates (2024 discovery, 2025–2026 OOS)

These are **research candidates only**, not enabled live bots. Selection used the 2024 top-10 entry configurations from a 600-point classic IMBA sweep; the fixed 5% full-exit rule was selected using 2024 outcomes. Do not treat 2024 as out-of-sample. Subsequent 2025/2026 were checked on the same Binance 5-minute signal / actual OHLC execution model (0.06% fee per side). Re-entry after a TP waits for the next opposite IMBA signal. This is **not** a verified parity match to the original production bot engine or a multi-bot capital-management backtest.

| Sensitivity | Renko box (USD) | 2024 % (train) | 2025 % (OOS) | 2026 % (OOS to Nov 4) |
|---:|---:|---:|---:|---:|
| 7 | 0.72 | 188.81 | -26.70 | 7.63 |
| 17 | 0.36 | 177.67 | 32.38 | -25.31 |
| 26 | 0.36 | 165.11 | -46.74 | 36.55 |
| 6 | 0.72 | 156.15 | -9.05 | -19.85 |
| 6 | 0.80 | 154.06 | 19.43 | -15.40 |
| 18 | 0.36 | 150.11 | 43.31 | -22.80 |
| 25 | 0.36 | 139.11 | -47.50 | 16.90 |
| 9 | 0.56 | 123.01 | 80.38 | -31.07 |
| 6 | 0.76 | 118.91 | 50.61 | -5.73 |
| 16 | 0.52 | 108.67 | -33.75 | 11.21 |

Candidate definition: classic IMBA; lookback = sensitivity ×10 Renko bricks; 5-minute confirmed-bar signal and next-bar-open entry; 5% fixed take profit on intrabar touches; flat until opposite signal after TP; zero additional stop-loss or trailing logic; 1× notional; 0.06% fee each side.

Provisional finding: In a reconstructed 2026 equal-weight comparison, adding these 10 candidates to the prior 2025-picked 60 increased average return from -33.48% to -29.38% for the profit-only family, and from -24.26% to -21.48% for quality-diverse. Replacing the lowest-ranked 10 (using **2025**, not 2026, ranks) gave -31.47% and -23.13%, respectively. **All tested configurations still lost money in 2026.** Differences from historic stored 2026 returns indicate parity must be independently validated.

Research next: confirm engine parity; evaluate daily return correlation and complementary regimes, verify selection timing, recover 2024-to-2025 full legacy selection (currently incomplete), then test capped 60-bot portfolios with dormant activation, capital reserve/withdrawals and leverage. No live portfolio configuration changed.
