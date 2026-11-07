# QUANT 0.43 — Intraday-Kern: Ereignisreihenfolge, Stopps, Kontobuch und Rechenwiederverwendung

**Stand:** 2026-11-07. **Status:** Offline-Forschungsprototyp, keine Live-Änderung.

## Ziel
Nur den minütlichen Rechenkern und seine Worker-Cache-Semantik verbessern; keine zusätzliche Labor-UI. Die v0.42-Anordnung, 3D-Heatmaps, Bot-Kurven, Tagesarchiv, Top-N-Auswahl und manuelle Walk-Forward-Diagnose bleiben erhalten.

## Korrigierte Modell- und Abrechnungsfehler

1. **Globaler Zyklus-Drawdown-Stopp:** In 0.41 wurde ein bereits im 1m-OHLC-Minutenbereich unterschrittener Schwellenwert erst zum Minuten-Close geschlossen. In 0.43 löst der Stopp bei der modellierten Preis-Schwellenkreuzung innerhalb der Minute aus. Stopp-priorität: Depot-Fix-/Trailing-Schutz, dann globaler Zyklus-Stopp, dann Bot-Stopps und gegebenenfalls TP.
2. **Gemeinsames NAV mit gemischten Long-/Short-Sleeves:** Schwellenwertprüfung berücksichtigt, dass auf 0 begrenzte Einzel-Sleeves die Depot-Eigenkapitalfunktion stückweise linear machen. Alle potenziellen Knickpunkte im 1m-Preisband werden geprüft; die erste Schwellenkreuzung von der Minuten-Eröffnung aus wird numerisch bestimmt. Das ist weiterhin **keine Tick-Folge**.
3. **Minutenkerze berührt Insolvenz und TP:** Der konservative Forschungsfall priorisiert den modellierten Kapitalverlust vor TP. Vorher konnte ein profitabler Ausstieg verbucht werden, obwohl dieselbe Kerze bereits die Modell-Insolvenzschwelle erreicht hatte.
4. **Buchungsreihenfolge:** Die Fill-Markierung des Ausstiegs speichert den tatsächlichen Depotwert nach erfolgter Nullstellung einer insolventen Einzelposition. Die Minuten-/Tages-Depotsamples werden nach Gewinnmitnahmen und Gebühren erfasst.
5. **Transparenz am Abschnittsende:** `audit.beforeClose` (bewertete offene Positionen), `audit.closeoutCost` und `audit.hypotheticalCloseout` (hypothetische Schlussglattstellung samt Exitgebühren und Slippage). `closeAtEnd:true` bucht optionale tatsächliche Schluss-Trades im Rechenkern. Standard bleibt `false`. Das bisherige UI hat **noch keinen eigenen Schalter** für diesen neuen Kernparameter.

## Worker, Ranking und Sweep-Performance

- Der erste zu verwendende Minutenindex wird per Binärsuche bestimmt; das UTC-Datum/Monat wird nur beim Tageswechsel neu formatiert.
- `summaryOnly:true` für große Worker-Sweeps: keine langen 15-minütlichen Chartreihen für jede Konfiguration. Jahresauswertungen verwenden weiterhin tägliche Snapshots und minutenbasierte Drawdown-Verarbeitung.
- **Identische Eingangs-Gene** (Renko-Modus, Boxgröße, Sensitivity, Signalauflösung) nutzen ein gemeinsames IMBA-Signalset auch über mehrere TP-/TTP-Familien. **Rankings sind bewusst getrennt**, da unterschiedliche Exit-Regeln verschiedene Tagesrenditen produzieren.
- **Cachefehler behoben:** Eine vorangegangene kürzere `prepare(end)`-Auswertung hatte einen nur bis zu diesem Datum reichenden Ranking-Cache erzeugt, obwohl der Cache-Schlüssel das Enddatum nicht berücksichtigte. 0.43 erzeugt Rankings über den gesamten aktuell geladenen Datensatz, mit neuen Ranking-Schlüsseln. Signal-Caches werden nicht mit veralteten Portfolioergebnissen verwechselt; `CASE_VERSION='0.43.0'`.
- Für identische Konfigurationen liefert `summaryOnly` denselben Endwert wie der vollständige Minutenlauf; die Ausgaben umfassen weniger Kurvenpunkte, nicht weniger Handelsminuten.

## Reproduzierte Tests
- `test_quant_043_core.js`: €1.000, 10×, 20 % globaler und fester Depotstopp → €800 innerhalb 1m beim modellierten SOL-Preis von $98; Insolvenz vor gleichzeitigem TP; periodischer Closeout (mit Gebühr/Slippage); gebuchte Kurve exakt gleich Abschluss-NAV; keine ungeklärte Ledgerdrift, feste vs manuelle Auswahl identisch.
- `test_quant_043_worker.js`: Zwei Bots mit gleicher Signalgene aber verschiedenen TP/TTP-Ausstiegen → **1** Signaldatensatz, **2** Rankings; später längeres Datumsfenster nutzt **2/2** vollständige Rankings. Zwei wiederholte Portfolio-Sweep-Fälle → **2/2 Cachetreffer**, identischer Endwert in Summary/Detail.
- Chromium v0.43 mit **36.000 echten SOL-1m-Zeilen**: fünf Bots, Trade- und Kontobuchvergleich, drei Intraday-Renko-Boxreihen, **3/3 Cachetreffer** bei wiederholtem Box-Sweep, keine JS-Seitenfehler.
- Chromium v0.43 mit **36.000 echten SOL-1m-Zeilen**: zwei Bot-Gene in einem realen Walk-Forward-Abschnitt 1.–14.01.2024, **€966,3164** End-NAV, vier Fills, Diagnose zeigt exakt dieselben Positionen/Zeiten und Buchungen.
- Tagesarchiv 2026: **462 Heatmap-Kandidaten** und die historische **€9.367,04925136168**-Modellreferenz unverändert.
- Punktueller Mikrobenchmark auf 86.400 Minuten/5 Bots: 0.41 ~163 ms, 0.43 ~111 ms, 0.43 Summary ~127 ms; Summary-Kurve 1 statt 5.761 Punkte. Keine belastbare 32k-Sweep-Durchsatzmessung.

## Einschränkungen und nächste Arbeiten
- Keine genaue Intraminuten-OHLC-Hoch-/Tief-Reihenfolge; konservative Reihenfolge kann von echten Fills abweichen. Kosten, Funding, Exchange-Liquidationen, TradingView-IMBA-/Renko-Signale und Live-Ausführungen nicht exakt paritätsverifiziert.
- Der 32.000-Konfigurationen-Jahressweep und 2022–2026-OOS über alle Gene sind weiterhin *nicht* vollständig unter Last geprüft.
- Browser-Origin kann lokale IndexedDB-Persistenz einschränken, QRC-Export bleibt nötig; der isolierte Worker-Test nutzte Arbeitsspeicher-Fallback.

## Release-Artefakte (Chat-Download, **noch nicht auf GitHub als Binärdateien hochgeladen**)
- `QUANT_IMBA_Dashboard_0.43_Intraday_Core_Offline.html` — SHA-256 `3be17a253ec50b5907356041f4e1a81f6f7767f954ab95ede6848e7cc2d04fb4`.
- `QUANT_IMBA_Dashboard_0.43_Intraday_Core_Offline.zip` — SHA-256 `ada292d8ae6560e636c6e1fd84a2842f3a1c3d79e2393996638fb19697da4312`.

Das ZIP enthält die ausführbare Offline-HTML-App, den inspizierbaren `quant_intraday_core_043.js`, `quant_research_worker_043.js`, Kern-/Worker-/Browser-Regressionstests und README. Kein Railway-Service wurde geändert.
