# QUANT 0.40 — Ein gemeinsames Forschungs-UI: Tagesarchiv und Intraday-Archiv
Datum: 2026-11-06. **Forschungsprototyp, keine Live-Handelsfreigabe.**

## Entscheidung

Das zusätzliche Intraday-Labor ist nicht der zentrale Workflow. Die bisherigen **Genome/3D-Heatmaps, Bot-Kurven, Handelsregeln-Sweep und manueller Walk-Forward-Player** bleiben die Hauptoberfläche. Ein Schalter bestimmt die Datenquelle. Das Tagesarchiv und seine erfolgreichen, aber noch nicht TradingView-paritätsverifizierten Modelle bleiben vollständig erhalten.

- **Tagesarchiv:** identische gespeicherte Tagesfaktoren wie bisher. Referenz 2026: 462 Original-IMBAs, Top 6, 14-Tage-Rückblick, 20% Einzelstopp, historische Kapital- und Gewinnstaffel, €1.000 Startkapital → €9.367,04925136168 Endkapital (erneut in Chromium 0.40 verifiziert). Keine Aussage über tatsächliche Live-Fill-Parität.
- **Intraday (Standard):** 1m lokale OHLCV, bestätigte Renko/IMBA 1m/5m Signale, 1x Ranking, gemeinsames Portfolio mit Minute-OHLC Ausführung, Gebühren, Kapitalregeln, Stopps und Regel-/Manual-WF-Simulationsläufen im Worker.
- **Keine vorbereiteten Intraday-Gene:** die bisherige Heatmap bleibt sichtbar als **Tagesarchiv-Vorschau**, ausdrücklich gekennzeichnet und nicht als Minutenrendite verkauft.
- **Vorbereitete Gene:** dieselbe Heatmap zeigt nur wirklich rekonstruierten Minutenstrategien. Einzelflächen mit nur einer Achsenstufe werden als 3D-Punkte statt als unrenderbare Surface dargestellt. Ursprüngliche Parameter-/Familienfilter gelten weiterhin.

## 0.40 Codepfade

- `quant_intraday_core_040.js`: gemeinsames Portfoliokontobuch; Buchungsabgleich `initial + realizedGross + unrealized - fees + modeled liquidation corrections = final`. Ein-/Ausstiegsgebühren von Teiltrades werden zugeordnet, aktuelle Tages-Snapshots enthalten Reserve, aktives Kapital, gewählte/aktive Bot-IDs, Stopps und Transfers.
- `quant_research_worker_040.js`: Marktdaten als Float64-Transferarray; caching `signalSets`, `rankings`, `portfolioCases`, `sweepRuns` in versioniertem IndexedDB und Export als QRC. Der erneut identische Backtest nutzt gespeicherte Fälle, wobei Daten, Gene, Engine und Regelparameter Schlüsselbestandteile sind.
- `quant_040_unified.js`: bestehender großer Handelsregeln-Sweep-Button ruft den Minuten-Worker mit den bekannten Top N-/Ranking-/Gewinn-/Stopps-/Budget-/Risiko-Regelvarianten auf. Die manuellen WF-Buttons rufen denselben Portfoliokern mit zeitlich gelatchten Bot-Gruppen auf. Die genauen Bot-Gene werden nicht nachträglich anhand zukünftiger Daten ersetzt.
- `quant_intraday_ui_040.js`: das frühere Sonderlabor bleibt für Trades und Risiko-Diagnostik verfügbar, ist aber kein separater Standard-Rechenweg.

### Reproduzierter UI-Defekt

In 0.39/ursprünglichem 0.40 wurde `eligible()` im Intraday-Modus bereits beim Laden auf `quant40Have()` begrenzt. Ohne vorberechnete Signale hatte die Oberfläche daher **0 Heatmap-Zellen**. Gleichzeitig versuchte ein früher Render-Callback vor `loadYear()` auf noch nicht geladene F-/DATES-Matrizen zuzugreifen. 0.40 behebt beides durch Archivvorschau, Render-Guards und danach echten Minuten-Gen-Filter. 3D-Einzellinien/Punkte werden eigens gezeichnet.

### Abrechnung: zuletzt positive Trades vs negative Depotkurve

Die Liste der zuletzt geschlossenen Trades ist **keine abgeschlossene Jahres-/Abschnitts-Gewinnabrechnung**: sie zeigt nur letzte Teil-/Vollschließungen. Gesamtportfolio-NAV berücksichtigt frühere realisierte Verluste, offene Positionsverluste, Einstiegs- und Ausstiegsgebühren, Reserven und Transfers. 0.40 hat einen expliziten Kontobuch-/PNL-Abgleich und eine separate Liste der größten realisierten Verlusttrades; keine naive Summe der letzten positiven Einträge.

### Walk-Forward

Der manuelle 14-Tage-/Monats-Player nutzt jetzt denselben minutenbasierten Portfolio-Kern wie die Hauptberechnung, replayt bekannte Abschnitte mit dem gemeinsamen Kapital ab Jahresbeginn und übernimmt neu gewählte Bot-Gruppen erst zu Abschnittsbeginn. Er kann **anders als der automatische Top-N-Sweep** abschneiden, denn der manuelle Player handelt die tatsächlich ausgewählten Bots, während die automatische Regelwahl nur ex ante qualifizierte, positive Ranking-Kandidaten auswählt.

## QA / Regressionsnachweise
- Chromium: 462 archivierte Zellen; 462 klar markierte Archivvorschau-Zellen im ungeladenen Intraday-Modus; mit 36.000 echten SOL-1m-Kerzen 3 Minuten-Gene auf 2 belegten Parameterzellen; automatische Quellumschaltung und UI; **keine JavaScript-Fehler**.
- Direkter kurzer Minutensimulationsfall: 36 realisierte Trades, rechnerischer Depotbuchungsrest ca. -1.14e-13 EUR; 2 Regelvarianten erneut mit **2/2 Cache-Hits**.
- Der **alte Handelsregeln-Sweep-Button** wurde mit einer bewusst auf 2 Fälle begrenzten Regelmatrix im echten Worker getestet; Ergebnis 2 Zeilen. Der vollständige 32.000er-Jahressweep ist **noch NICHT** durchgerechnet/validiert.
- Manuelle 14-Tage-Neuwahl im Minutenledger: 1 Abschnitt im definierten Testfall 966,32 EUR; ohne JS-Fehler.
- Original Tagesarchiv 2026: €9.367,04925136168 unverändert.
- Dependency-free Node-Test `test_quant_040_core.js`: Trade-Ausführung, Cent-Ledger-Parität und manuelle WF-Auswahl nach Regelüberschreibung bestanden.
- Headless testet 2D-Heatmap und die konstruierten 3D-Spuren, **keinen** vollständigen interaktiven WebGL-3D-Stresstest.

## Grenzen

- OHLC-Minuten enthalten keine eindeutige intraminütliche High-Low-Abfolge; pessimistische Stopp-vor-TP-Annahme, keine exakte Tick-/TradingView-/Exchange-Fill-Parität oder Funding-/Liquidationssimulation.
- Der alte **rollierende Regel-OOS-Sweep** und separate **Box-Sweep** sind noch nicht komplett auf den Intraday-Kern migriert, Tagesarchiv-Funktionen bleiben verfügbar. Die vollständige Genom-Rekonstruktion für alle 7.326 Fälle/Jahre und der 32.000er-Intraday-Sweep benötigen zusätzliche Laufzeit- und Regressionstests. UI-Limit: 600 Gene je Vorberechnung.
- Aktuelle Browserpersistenz ist lokal pro Origin/Profil; Cache regelmäßig exportieren. Permanente IndexedDB-Persistenz unter allen Offline-file://-Origins noch nicht vollständig verifiziert.
- Keine Railway-Services/Live-Trading-Einstellungen geändert.

## Offline-Artefakt (noch NICHT auf GitHub hochgeladen)
- `QUANT_IMBA_Dashboard_0.40_Gemeinsames_Intraday_Archiv_Offline.html`: SHA256 `c1693dc4e86716e31da9109ba9f21e30015c4e2c3dfa92617bc4f6905fd4404e`
- `QUANT_IMBA_Dashboard_0.40_Gemeinsames_Intraday_Archiv_Offline.zip`: SHA256 `23ea51b1a1cd2413342982f4b8a4b3da11a9cf2c542b0375ba5af74806520c29`

Die App ist als ChatGPT-Datei verfügbar; der GitHub-Konnektor kann den lokalen 15-MB-ZIP-Anhang nicht direkt lesen/übertragen. Dieses Markdown ist eine Dokumentation, **kein** Upload der App.

## Nächste Schritte Richtung 1.0
1. Intraday-**Box-Sweep** und rollierenden OOS-Regel-Sweep gleichwertig an denselben Worker anschließen.
2. 32.000er-Intraday-Jahressweep mit Persistenz, Fortsetzung, Cachetreffern, Overflow-/Abbruch- und Quotenmanagement unter Last messen.
3. Echte TradingView-Signale/Fills gegen die Reproduktion tradeweise auditieren; vollständige 2022–2026 OOS-, Gebühren-, Slippage- und Crash-Periodenregression.
4. Mehr-Asset synchronisierte Zeitreihe statt Single-SOL-Komfortannahme und getrenntes serverseitiges Vorberechnungsformat für große Genome.
