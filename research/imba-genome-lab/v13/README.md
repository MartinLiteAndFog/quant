# QUANT Genome Parameter Lab v13 — Abschluss der offenen UI- und Chart-Punkte

**Stand:** 06.11.2026. **Historie:** täglich 2022–04.11.2026. **Code:** Forschungsbranch `research/imba-genome-lab-2026-11-05`, keine Live-Änderungen.

## Was konkret implementiert wurde

1. **Die drei echten Screenshots sind nun jeweils ein auswählbares Preset.** Das Preset stellt das Jahr 2026, USD-Renko, alle berechneten Boxgrößen, den korrekten Kandidatenpool, Rückblick und individuellen Stopp ein und aktualisiert die 10×-Portfolio-Engine. Jede Kontrollrendite wurde direkt in Chromium mit dem Original-JavaScript-Ledger verifiziert. Das 2026er Forschungskonto beginnt dabei jeweils **separat mit 1.000 €**.
2. **Zusätzlicher Raw-2022-Vergleich:** explizit als Annahme markiert; aus den Screenshots ist KEIN eigener Raw-2022-Parametersatz abzulesen (beide 2022er Top-100-Fotos sind Balanced).
3. **Interaktive Research-Charts direkt in der HTML-App:** 2026er drei-Büschel-Kapitalverlauf, Sharpe-Vergleich sowie die 2023–2026er Top-100-Jahresfamilienrotation (durchgehendes Kapital). Diese Charts sind fest zu den *dokumentierten* Forschungsparametern gerechnet; Änderungen am aktuellen Einzel-Portfolio-Filter verändern nicht rückwirkend die fixen Forschungsdiagramme. Der normale Portfolio-HUD reagiert weiterhin auf Einstellungen.
4. **Raw-Top100-Familien der Trainingsjahre 2022, 2023, 2024 und 2025** als eigene Kandidatenpools. Trainingsjahr und Testjahr sind getrennt: Familie aus 2025 wird erst ab 2026 angeboten. Auswahl derselben Familie in ihrem Trainingsjahr führt absichtlich zu null Kandidaten, um **Lookahead** zu verhindern.
5. **Korrelationsgrenze für die tatsächliche Top-6-Auswahl:** Aus/0,85/0,90; 28 *vollständige vorangegangene* tägliche ungehebelte Renditen, Pearson-Korrelation, Greedy-Auswahl unter den profitablen Kandidaten. Standardmäßig **strikt**: 0–6 Bots sind möglich. Optionales Auffüllen ist ausdrücklich als Lockerung der Grenze bezeichnet; es kann sie verletzen. Kein Tagespreis des gerade beginnenden Handelstages wird für die Auswahl verwendet.
6. **Weiter vorhanden aus v12:** vier 3D-Oberflächen gleichzeitig oder wählbare Exit-Gen-Familien, tatsächliche Kandidatenzahlen pro Familie, Marker für ausgewählte/handelnde/gestoppte Bots, 7/10/14/30/90 Tage Ranking, Einzelstopps, $-Renko/%-Renko/Candles, 2022–2026, einklappbare Seitenleiste und Panels, zeitliche Wiedergabe, manuell gespeicherte echte Bot-Punkte.

## 2026: Original-Screenshot-Einstellungen (jeweils 1.000 € separat)

| Quelldatei (im Git als PNG archiviert) | Im Screenshot ausgewählt | Top-6-Ranking | Individueller Stopp | 10×-Endvermögen 04.11.2026 | Status |
|---|---|---:|---:|---:|---|
| `screenshot_1_balanced30_stop30.png` (09:39:49) | 2022 Top 100 Balanced | 30 Tage | 30 % | **7.242,87 €** | Chromium reproduziert |
| `screenshot_2_original14_stop20.png` (09:41:04) | Original 462 | 14 Tage | 20 % | **9.367,05 €** | Chromium reproduziert |
| `screenshot_3_balanced30_stop25.png` (09:41:58) | 2022 Top 100 Balanced | 30 Tage | 25 % | **10.654,97 €** | Chromium reproduziert |

Die Bilder enthalten zwei Varianten von Balanced; **Top100 Raw kommt nicht separat auf den Bildern vor**. Für das Dreierportfolio wird daher zusätzlich eine ausdrücklich als Vergleichsannahme bezeichnete Raw-Einstellung (30 Tage/30 % Stopp) verwendet.

## Gemeinsames 2026er Portfolio: genau drei Familien

| Depot | Kapital 01.01.2026 | Top-6-Ranking | Einzeldrawdown-Stopp | Vermögen 04.11.2026 |
|---|---:|---:|---:|---:|
| Original 462 | 333,33 € | 14 Tage | 20 % | 4.322,76 € |
| 2022 Raw Top100 | 333,33 € | 30 Tage | 30 % | 823,22 € |
| 2022 Balanced Top100 | 333,34 € | 30 Tage | 25 % | 4.684,41 € |
| **Summe** | **1.000 €** | getrennte Verwaltung | getrennte Verwaltung | **9.830,39 €** |

**Mix-Sharpe 2,266**, maximaler Drawdown **−73,02 %**. Rückschauwerte, keine externe Validierung. Die Kennzahlen verwenden Tagesrenditen des *gesamten Depotvermögens*, risikofreien Zins 0 und annualisieren mit √365.

Ein separater fairer Vergleich startet jede der drei Einzelfamilien mit **1.000 €**, nicht mit einem kleineren Drittel; deren Sharpes: Original 462 **2,345**, Raw Top100 **1,629**, Balanced (25 %) **2,308**. Der Mix **verbessert die Sharpe nicht gegenüber Original 462 alleine** und hat sogar höheren maximalen Drawdown. Die interne monatliche Mindestzuweisung von 100 € führt bei einem kleinen Starttopf zu nichtlinearen Effekten; Skalieren der Einzelrechnungen ist darum falsch.

## Korrelations-Regel (nicht "negative Korrelation")

- `0,85` und `0,90` sind **maximal erlaubte positive Pearson-Korrelationen** zweier ausgewählter Bots; Antikorrelation wird nicht vorausgesetzt.
- Das Fenster umfasst die 28 vollständigen Tage *vor* der Rotation. Nur Kandidaten mit positiver vergangener Ranking-Rendite werden betrachtet.
- Die alte Korrelationserforschung 2023–2026 beruhte auf einer teilweise anderen Bot-Population und Depot-Engine. Ihre früheren Ergebnisse sind **nicht** ohne Weiteres mit dieser App vergleichbar.
- 2026 kurzer Chromium-Kontrolllauf, Top100 Raw, 30-Tage-Ranking, Stopp 30 %, 1.000 € Start: Filter Aus 3.493,61 €, strikt ≤0,85 2.883,68 €, strikt ≤0,90 3.271,67 €. Alle tatsächlich ausgewählten Paare hielten die Grenze ein (238 bzw. 241 Kontrollpaare). Nicht aus diesen Werten auf generell bessere Robustheit schließen.

## Chartdateien und Daten

- `2026_drei_bueschel_mix_kapitalkurve.svg`: echte 3× eigenständige 333er-Depotkurven + Summe; Vergleich mit dem bisherigen 30-%-Balanced-Mix.
- `2026_sharpe_drawdown_vergleich.svg`: Sharpe und Max-DD für die drei Einzelfamilien mit je 1.000 € sowie ein 1.000-€-Mischportfolio.
- `2023_2026_jahresfamilien_chained.svg`: durchgehendes Kapital 2023–2026 mit jährlicher Rotation, Beibehalten und Akkumulation der Raw-Top100, *logarithmische Skala*.
- `screenshot_mix_2026_daily.csv`, `screenshot_mix_2026_sharpe.csv`, `screenshot_preset_evidence_2026.csv`: prüfbare 2026er Ergebnisse.
- `screenshots_2026_presets.json`: die exakten Einstellungen mit Screenshots als Herkunft.
- `chart_data.json`: feste Plotly-Daten für die eingebetteten Research-Charts.
- `replay_screenshot_presets.py`, `make_charts.py`, `build_v13.py` und `test_*.py`: reproduzierbarer Code und Checks.

## Was weiterhin NICHT implementiert ist

- **Einen vollständig neuen, nie gerechneten Bot an beliebigem 3D-Punkt automatisch intraday neu backtesten.** 20×20/30×30 setzt derzeit nur Filter auf *vorab berechnete* Strategien. Punktklicks setzen vorhandene Bot-Strategien, auch wenn sie erst zu einer XY-Zelle zusammengefasst wurden. Eine neue Genkombination ohne vorhandene Kurve wird **nicht** mit erfundener Performance gehandelt.
- Neue flexible Gene wie RSI/ADX gehören in eine spätere echte Backtest-/Signal-Engine, nicht in die bestehende reine HTML-Visualisierung.
- Keine TradingView-exakte Renko-, Funding-, Intrabar-Liquidations- oder Live-Execution-Parität. 0,06 % pro Orderseite sind in 1×-Tagesfaktoren berücksichtigt; der 10×-Tageshebel bleibt ein Forschungsmodell.
- Andere 2026er Einstellungen wurden rückblickend getestet. Große Renditen können optimistisch und überangepasst sein.

## QA

- `node --check` für die integrierte JavaScript-App ohne Syntaxfehler.
- Chromium/Playwright: alle drei Screenshot-EUR-Werte auf Cent-Niveau reproduziert; Raw100 Zusatzvariante; Jahresfamilie 2025→2026 verfügbar, 2025→2025 gesperrt; Kapital/Sharpe/Jahresfamilien-Charts sichtbar, keine JavaScript-Pageerrors.
- Korrelations-Test: Rückblicksdaten nur vor Selection Day; ausgewählte Paare überschreiten strikt weder +0,85 noch +0,90.
- Kein Live-Deployment, kein Push auf `main`.