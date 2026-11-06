# Genome v13 – Correlation, matched 462×462 exits, and Kraken TTP research

This is **research-only** and does not modify Kraken, KuCoin, Railway, or live-trading configuration.

## Historical data already in v13

The base dashboard contains computed daily factors for its historical exit programs.
The new selection controls compare only exact existing 1× strategy IDs:

- **Original IMBA-only** group: starts from the 462-parameter original grid.
- **Legacy TTP 3% activation / 1.5% trailing**: matched one-to-one with IMBA by identical USD-Renko box and sensitivity, using only existing `META` entries.
- **Legacy TTP 2% activation / 1% trailing**: same matching.
- **Mixed historical IMBA + TTP**: union of the matched groups. This reaches 924 only if all 462 pairs exist in the embedded dataset. The browser reports the actual matched count, never silently fills missing entries.

These historical TTP curves **do not** represent the newly requested Kraken mode. All comparisons use the same Top-N, ranking, stop scope and stop thresholds as the selected dashboard settings, and retain the existing capital model. The annual comparison for 2022–2026 resets to €1,000 each January. It is NOT a single chained account.

Existing 28-day Pearson correlation filtering remains part of actual candidate selection; the quick presets allow `rho <= 0.85` strict, `rho <= 0.90` with fallback, or off. These can be compared with historical portfolios.

## New research defaults requested

The **configurable proposal** under "Neue Kraken-TTP-Einstellungen" is:

| Rule | Default | UI |
|---|---|---|
| TTP activation | 0% (immediately armed at entry) | 0%, 0.5%, 1%, 2%, 3% |
| TTP trail | 1.5% | 0.5–4.0% step 0.5% |
| Action at trailing-stop hit | Flip to opposite side, then WAIT | Flip or close |
| Close/partial ratio (close mode only) | 100% | 50% or 100% |

**Critical distinction:** The Kraken `flip_engine.py` flips **when the trailing stop is hit**, not when TTP is first armed. Following a TTP-induced flip it enters WAIT (only swing SL); a same-direction IMBA impulse re-arms TTP, while an opposite IMBA impulse flips and arms TTP immediately. A full reversal must first close 100% of the original position; therefore the 50/100% close share is intentionally disabled in the Flip mode.

The new Kraken settings are stored in the normal v13 JSON preset (`krakenTtp`) and restored from it. They **do not change historical daily factors, generated charts, or sweep results** until a distinct intraday backtest produces signals and trade-level equity curves for those parameter values. The dashboard labels this limitation.

## Before the Kraken variant can enter the numeric sweep

1. Reconstruct 462 IMBA signal streams from the exact same Renko sensitivity and box grid and provide intraday price timestamps.
2. Run the **Kraken flip state-machine rules**, including WAIT swing SL, partial exits where applicable, fees, slippage, bar sequencing and realistic execution.
3. Export real daily factors and matching metadata for the requested TTP settings; the existing `YEAR_GZIP`, `META` and prefix return arrays must be regenerated or extended.
4. Only then enable the new TTP variants as fully computed candidate pools in Top-N / stop / correlation / automatic sweeps.
5. Validate TradingView alerts and venue fills independently; the close-based existing flip engine alone is not enough to prove intrabar parity.

## Build, verification and limitations

`app_v13.js` contains the extension (source also kept as `ttp_research_extension.js`). The unchanged `upgrade_trade_rules.py` can swap the script into the existing standalone v13 HTML, preserving its compressed market data.

Completed: JS syntax parse, controlled synthetic 462-pair matching and pool switching, default Kraken settings and JSON serialization. Not completed: actual full HTML run against the gzip-embedded 2022–2026 data, verification that the full set of 462 TTP matches exists, and genuine intraday Kraken-0%-activation historical returns. No financial results are claimed for that uncomputed mode.

## Renko-%-Kopplung zu bestehenden Strategie-Pools (6 November)

Problem: `original462`, `top100`, `balanced100`, `rawYear*` sowie die historischen paarigen 462er IMBA/TTP-Pools enthalten zunächst nur USD-Renko-IDs. Ein Wechsel von USD zu %-Renko filterte sämtliche Kandidaten weg und führte ohne erklärende Fehlermeldung zu null Trades.

Fix im Forschungsbranch: Sobald %-Renko aktiv ist, wird jeder Kandidat eines spezifischen USD-Pools auf **bereits vorhandene**, echte %-Renko-META-Einträge mit exakt derselben `s` (Sensitivity), `family` und `exitKey` (Exit-Programm) abgebildet. Jede passende %-Boxgröße bleibt ein unabhängiger, wirklich getesteter Kandidat. Im universellen `all4158`- oder Rasterpool wird nichts umgemappt; dessen originale %-Strategien werden direkt genutzt. Manuelle Pools können ihre vorhandenen %-Einträge beibehalten. Keine Kurve wird interpoliert, durch feste USD/Preis-Konversion ersetzt oder frei erfunden. Beim Filterergebnis null wird nun die Ursache und eine konkrete Bedienalternative angezeigt. Der %-Boxfilter bietet nur vorhandene %-Boxwerte an.

**Grenzen:** Dies ist eine **Gen-Kopplung**, keine identische TradingView-/Ausführungskurve zu einer USD-Renko-Strategie. Für einen ausgewählten Tag kann selbst bei vorhandenem Pool null Positionen aktiv sein (fehlender positiver vergangener Ranking-Score, aktive Stops oder strikter Korrelationsfilter). Syntaktisch geprüft, Matching-Logik an Kontrollbeispielen geprüft; vollständiger Browser-/Renditetest mit dem separaten 24,5-MB-Historien-HTML bleibt offen. Originalwerte im USD-Modus werden durch den Fix nicht geändert.
