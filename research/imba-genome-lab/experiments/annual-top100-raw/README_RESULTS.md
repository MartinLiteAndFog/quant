# IMBA | Jährliche Top-100-Raw-Familien, 2022–2026

**Forschungsstand: 06. November 2026.** Datensatz endet am **04. November 2026**.

## Ziel

Ist es besser, die 100 Top-Strategien des Vorjahres zu verwenden, die 2022er Top-100 dauerhaft weiterzutragen oder die Familien im Jahresverlauf zu akkumulieren? Die 2022er Top-100 waren bereits archiviert; ihre erneute Auswahl aus der identischen Rohpopulation stimmt **100/100** überein.

## Definition: gleicher 4.158er-Suchraum in jedem Trainingsjahr

- 462 feste USD-Renko-Parameterpaare × neun echte Exit-Programme = **4.158** Backtests pro Trainingsjahr.
- Für 2022, 2023, 2024, 2025 werden die 100 **höchsten vollständigen 1×-Jahresrenditen** gewählt; keine Tradezahl-, Korrelations- oder Drawdown-Qualitätsgewichtung in dieser Top100-Raw-Erstellung.
- Nach dem Ranking werden **ausschließlich ab dem Folgejahr** Bots gehandelt.
- Ohne prozentuale Renko- oder Candle-Strategien, um die 2022er Raw-Methode exakt vergleichbar zu halten. Zusätzliche Parameterfelder sind zukünftige getrennte Experimente.
- 0,06 % Gebühren je Orderseite in den zugrunde liegenden Bot-Tagesrenditen; **nicht zusätzlich** im Portfolio abgezogen.

## Welche Gene wurden im Training ausgewählt?

|   training_year |   none |   tp |   ttp |   both |   best_2022_ref_overlap |   profitable_of_4158 |
|----------------:|-------:|-----:|------:|-------:|------------------------:|---------------------:|
|            2022 |     59 |   31 |     0 |     10 |                     100 |                 3067 |
|            2023 |    100 |    0 |     0 |      0 |                      10 |                 3103 |
|            2024 |      0 |   67 |     7 |     26 |                       1 |                 2168 |
|            2025 |     12 |   42 |    16 |     30 |                       4 |                 2577 |

Überlappungen: 2022→2023 nur **10/100**, 2023→2024 **0/100**, 2024→2025 **6/100**. Die Marktlandschaften wechseln wesentlich.

## Portfolio-Regeln (gleiche Engine in allen Varianten)

- 1.000 € Startkapital pro unabhängigem Testjahr (zweite Auswertung: einmal 1.000 € für 2023–2026).
- Monatlich aus der **Reserve** intern zuführen: `min(Reserve, max(100 €, 20 % Reserve))`.
- Alle 14 Tage maximal sechs Gewinner anhand der **letzten 14** bzw. **30** vollständigen Kalendertage wählen; nur positive Kandidaten. Auswahl erfolgt zum Tagesstart ohne Blick in den laufenden Tag.
- Gemeinsamer Handelstopf auf ausgewählte Slots verteilt, 10× auf tägliche Bot-1×-Renditen **einmal**; negative tägliche Faktoren bei −100 % begrenzt.
- Pro Bot individueller 30-%-Drawdown-Stopp auf den 10×-Slotindex, Reset bei neuer 14-Tage-Auswahl. Kein überlappender gemeinsamer Portfolio-Stopp.
- Bei kumulierten +100-%-Gewinnmeilensteinen 50 % des Gewinnüberschusses intern in die Reserve transferieren.
- Keine externen Kapitalzuschüsse, kein Wetterfilter. Keine Intraday-Liquidationsprüfung; TradingView-Parität nicht nachgewiesen.

## Vergleich, separater 1.000-€-Start in jedem Jahr

**14-Tage-Ranking, 30-%-Stopp – Browser-Engine direkt ausgeführt** (Euro, Endvermögen):

|   year |   carry_2022 |   annual_rotate |   accumulate_past |   original462 |
|-------:|-------------:|----------------:|------------------:|--------------:|
|   2023 |      5492.67 |         5492.67 |           5492.67 |        193.15 |
|   2024 |       581.03 |          481.33 |            611.65 |         20.69 |
|   2025 |        13.76 |           62.04 |             91.11 |       1283.28 |
|   2026 |       227.72 |          956.21 |            683.99 |       1783.38 |

**30-Tage-Ranking, 30-%-Stopp** (Euro, Endvermögen):

|   year |   carry_2022 |   annual_rotate |   accumulate_past |   original462 |
|-------:|-------------:|----------------:|------------------:|--------------:|
|   2023 |        67.3  |           67.3  |             67.3  |         33.42 |
|   2024 |       781.21 |          479.87 |            614.44 |          3.98 |
|   2025 |        53.39 |           24.5  |              8.52 |       1278.91 |
|   2026 |      3493.61 |         1147.01 |            386.95 |        908.21 |

*`carry_2022`*: immer 2022er Raw-100. *`annual_rotate`*: nur die Top-100 des unmittelbar vorausgehenden Kalenderjahres. *`accumulate_past`*: alle bereits bekannten Jahrgänge in einem Pool, identische Strategien dedupliziert. *`original462`*: klassische IMBA ohne TP/TTP als Kontrolle.

**Cohort-2023 behalten** (14-Tage-Ranking): 2024: **481,33 €**, 2025: **490,83 €**, 2026: **1.265,58 €**. Das übertrifft die reine jährliche Rotation 2025/26, rettet aber 2024/25 ebenfalls nicht.

## Durchgehendes Depot, 2023–2026, 14-Tage-Ranking

|   year |   carry_2022 |   annual_rotate |   accumulate_past |   original462 |
|-------:|-------------:|----------------:|------------------:|--------------:|
|   2023 |     5492.67  |        5492.67  |          5492.67  |       193.15  |
|   2024 |      380.448 |         373.773 |          3173.78  |         0.002 |
|   2025 |        0.806 |           1.727 |           198.366 |         0     |
|   2026 |        0.031 |           0.256 |            10.188 |         0     |

Das ist eine **andere** Frage als die separat gestarteten Jahresbacktests. Der massive Drawdown 2024/25 erschöpft auch die anfangs erfolgreichen Depots. Die jährliche Familienrotation allein löst das Problem nicht.

## Interpretation

1. **2023**: 2022er Top100 können stark sein (+~449 % auf ein 1.000-€-Jahresdepot bei 14 Tagen); bei 30-Tage-Ranking fallen sie dagegen drastisch. Die Ranking-Länge ist entscheidend.
2. **2024**: Neue 2023er Top100 (**481 €**) schneiden schlechter ab als 2022er Weiterführen (**581 €**), allerdings gewinnt die kombinierte Familie (**612 €**).
3. **2025**: 2024er Top100 mit vorwiegend TP (**62 €**) schlagen 2022er (**14 €**), beide verlieren viel Kapital; der ältere **2023er Jahrgang** ist mit **491 €** deutlich besser.
4. **2026**: 2025er Top100 (**956 €**) schlagen bei 14-Tage-Ranking 2022er (**228 €**). Aber mit 30-Tage-Ranking gewinnt der 2022er Jahrgang sehr deutlich (**3.494 € vs. 1.147 €**). Keine robuste allgemeine Überlegenheit der jährlichen Rotation.
5. **Das echte Problem ist Familienwahl/Regime-Mismatch**, nicht einfach zu wenige Bots. Nur nach Vorjahressiegern umschichten ist ebenfalls eine auf der Vergangenheit optimierte Wahl.

## Verifikation und Einschränkungen

- **Alle eigenständigen Jahreszahlen** oben stammen aus der tatsächlich ausgeführten Browser-/JavaScript-Portfolio-Engine des Genome Parameter Lab v12. Keine Portfolio-Pseudowerte aus der farbigen Heatmap.
- **2022er Familienidentität 100/100** mit dem alten Top100-Raw-Preset verifiziert.
- Python-Ledger und Browser weichen in wenigen Fällen wegen nahezu gleicher Parameter-Rankings ab (z. B. Original462 2026). Die **Browser-Ergebnisse haben Vorrang**.
- Der v12-Forschungsdatensatz ist ein neues, intern konsistentes Signalmodell und **nicht vollständig identisch** mit der alten archivierten Original-462-Kurve (15.588 € im alten 2026-Test). Zahlen aus unterschiedlichen Engines nicht direkt gleichsetzen.
- Backtesting am selben historischen Bereich einschließlich nachträglich definierter Familien ist keine unabhängige Out-of-Sample-Bestätigung für reale 10×-Futures.

## Reproduzierbarkeit

- `run_annual_families.py`: Erstellen der Familien aus den Tagesrenditen, alternatives Python-Ledger, Balance-Checks.
- `browser_validate_annual.py`: unabhängige Jahrgänge 2023–2026 mit dem **originalen JS-Ledger**.
- `browser_chained_annual.py`: gleiche Kontobuchhaltung, ohne jährlichen Reset des Vermögens.
- Als Quelldatensatz wird `imba_genome_parameter_lab_v12_familien_transparent.html` benötigt, in dem die 7.326 Botkurven 2022–2026 eingebettet sind; nur der ursprüngliche 4.158er Fix-Renko-Raum wird verglichen.