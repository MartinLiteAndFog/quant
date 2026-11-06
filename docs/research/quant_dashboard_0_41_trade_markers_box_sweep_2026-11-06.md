# QUANT 0.41 — Minutentrades in der Depotkurve, Renko-Box-Sweep über denselben Worker
Stand: 2026-11-06. **Offline-Forschungsprototyp. Kein Deployment, keine validierte TradingView- oder Exchange-Tick-Parität.**

## Fehleranalyse anhand des Nutzer-Screenshots
Der Screenshot eines alten Intraday-Diagnoselaufs zeigt fünf markierte Bots, etwa 84.960 Handelsminuten und folgende gerundete *gesamte Bot-Sleeve-Renditen*:

| Bot | Ohne Zusatzschutz | Mit Schutz |
|---|---:|---:|
| S15_R0.20_ttp_3_1p5 | -66.01% | -57.59% |
| S15_R0.20_tp6 | -22.20% | +4.17% |
| S5_R0.40_ttp_3_1p5 | +1.53% | +23.68% |
| S20_R0.16_tp6_ttp_3_1p5 | -87.38% | -80.59% |
| S5_R0.40_tp6 | +166.51% | +248.29% |

Bei fünf gleichwertig zugeordneten Start-Sleeves à EUR 200 (gesamt EUR 1.000) folgt:
- Unprotected: `(-66.01 - 22.20 + 1.53 - 87.38 + 166.51)/5 = -1.51%`, ca. EUR 984.90; Screenshot EUR 984.88 aufgrund ungerundeter Originalwerte.
- Protected: `(-57.59 + 4.17 + 23.68 - 80.59 + 248.29)/5 = +27.592%`, ca. EUR 1,275.92; Screenshot EUR 1,275.93 aufgrund ungerundeter Originalwerte.

**Kein Widerspruch zwischen profitablen jüngsten Exit-Zeilen und einem starken früheren Depotverfall.** Die bislang angezeigten letzten 20–25 Exit-Fills sind nicht alle historischen Trades und berücksichtigen nicht isoliert ältere Verluste, Kosten und noch offene Mark-to-Market-Verluste. Eine profitable Teilglattstellung macht nicht automatisch eine langfristig profitable Strategie.

## Codeänderungen v0.41
- `quant_intraday_core_041.js`: `fills[]` enthält **jede Eröffnung und Schließung/Teilglattstellung** mit UTC, Bot-ID, Richtung, gehandelten Größen, Preisen und dem Gesamtdepot-NAV direkt nach dem Event. Der bestehende `trades[]`-Verlauf bleibt für realisierte Exit-PnL verfügbar. Je Bot gibt es summierte Gewinne/Verluste realisierter Exits und offene Netto-Beiträge. Das Audit prüft `initial + realizedGross + unrealized - fees + liquidationAdjustments = final`.
- `quant_041_trade_markers.js`: Plotly-SVG-Marker (kein WebGL-Zwang) im bestehenden Depotchart und in der optionalen Trade-Diagnose: Einstieg blau, profitabler Exit grün, verlustreicher Exit rot. Dropdown alle Exits / Ein- und Ausstiege / aus. Klick auf Ledgerzeile fokussiert das Event ±8 Stunden. Für mehr als 3000 Fills werden die dargestellten Marker ausgedünnt; das CSV ist weiterhin vollständig.
- `quant_intraday_ui_041.js`: Ledger mit allen Exits (bis zu 1000 gleichzeitig im DOM), den letzten 20 und den größten Verlusten; je-Bot-Zusammenfassung und die klassische Gewinnbrücke. „Markierte Bots“ bleibt nach Datenmoduswechsel selektiert, statt aufgrund einer nicht passenden Tages-Vorschau herauszufallen.
- `quant_041_unified.js`: Hauptdepot und manueller Intraday-Walk-Forward zeigen dieselben Fill-Marker aus dem minutengenauen Ledger. Bereits existierende Tagesarchiv-/3D-Sweep-Oberfläche und historische Daten werden nicht entfernt.
- **Renko-Box-Sweep**: Der alte `#q35SweepRun` ruft im Intraday-Modus den identischen Minuten-Worker auf. Alle für den aktuellen Gene-/Familienpool verfügbaren Boxreihen werden **einmal** vorbereitet. Die Top-N-Auswahl je Boxreihe berücksichtigt ausschließlich deren Kandidaten via `universeIds`, sonst würde man andere Boxgrößen hineinmischen. Die Bewertungszeiträume und Ziele aus dem vorhandenen UI bleiben erhalten. Ergebnisse werden mit fallabhängigem Cache wiederverwendet.
- Versionierte Fall-Schlüssel: `CASE_VERSION=0.41.0`; Signalkatalog-Key bleibt `VERSION=0.40.0` für Wiederverwendung bestehender teurer Renko-/IMBA-Signale. Der 0.40-Portfoliocache wird **nicht** fälschlich als 0.41-Ergebnis ausgegeben.

## Walk-Forward-Definition
`NAV = freeCash + reserve + vault + sum(max(0,botCash + side*qty*(price-entry)))`. Der Orderwert basiert auf je-Bot-allokiertem Kapital × Hebel; Gebühren und Slippage werden beim Ein-/Ausstieg verbucht. Der manuelle Walk-Forward replayt bei jeder Abschnittserweiterung die gesamte bisher bekannte Liste der ausgewählten Botgruppen **aus dem gleichen Ausgangskapital** und latched die Handelsregeln zu den jeweiligen Abschnittsgrenzen. Automatisches Top-N arbeitet dagegen nur mit **bis zum Vortag abgeschlossenen positiven** Rankingrenditen. Unterschiedliche Bot-Auswahl, Starttage, Kapitalquote, Stoppregeln oder Signalauflösung führen daher zu anderen Kurven. In einem vollständigen Gleichheitstest mit exakt identischen Eingaben waren die 0.41 Fixed-Portfolio- und Manual-WF-NAVs identisch.

## Verifizierte Tests
- `node test_quant_041_core.js`: 6 echte Exit-Events/14 Entry+Exit-Events im synthetischen Test; alle Fill-NAVs endlich; rechnungsgenaue Summe je Bot; kein Leakage fremder Box-Kandidaten; Buchungsdrift ~2.3e-13 EUR; manueller WF identisch mit Fixed-Auswahl bei gleichen Regeln.
- `python test_quant_041_browser.py`: Chromium mit 36.000 echten SOL-Minutenkerzen, fünf markierten Genes; 5 realisierte Exits, 10 Entry+Exit-Marker, Klick-Zoom im Chart; Bestands-Endwert EUR 886.97 im Testzeitfenster (nicht mit dem Nutzer-Screenshot von ca. 84.960 Minuten verwechseln); 3 Renko-Boxreihen im vorhandenen Box-Sweep; **3/3 Cache-Treffer bei identischer Wiederholung**; keine JS-Seitenfehler. Das Testfenster hatte keine positiven ex-ante Top-N-Kandidaten, daher 0 Transaktionen und EUR 1.000 Endkapital in dessen automatischem Box-Sweep. Diese Nullergebnisse sind korrekt als „keine Trades“ markiert.
- `python test_quant_041_archive.py`: 462 ursprüngliche Tagesarchiv-Heatmap-Zellen und unveränderte Archiv-2026-Modellreferenz EUR 9,367.04925136168 aus EUR 1,000. Kein Browser-JavaScriptfehler.

## Fehlende Arbeiten
- Der vollständige rollierende Intraday-OOS-Regelsweep und der 32.000er-Jahresstresstest sind **nicht** abgeschlossen; auch eine vollständige Genomrekonstruktion und 2022–2026-Walk-Forward-Parität aus echten Minute-Daten fehlen.
- Minute OHLC liefert nicht die exakte Hoch-/Tief-Reihenfolge, Tickfolge, Börsenfills, Funding und Liquidationen. Die Signale/Backtests sind nicht TradingView-verifiziert. Keine Gewinne als live bestätigt ausgeben.
- Tagesarchiv bleibt als expliziter Berechnungsmodus einschließlich sämtlicher alten Presets erhalten. Es wurde kein Railway-Service geändert.

## Offline-Releasedateien (nur im Chat, nicht als GitHub-Binärdateien committet)
- `QUANT_IMBA_Dashboard_0.41_Trade_Marker_BoxSweep_Offline.html` SHA256: `b91bc2bbe2edb5cdb434c51834c9753841f26fa4a6950124e41070ef7c8e0ea7`
- `QUANT_IMBA_Dashboard_0.41_Trade_Marker_BoxSweep_Offline.zip` SHA256: `865f10c887958b66362e4efa4f0449cf1b9fa6929899d156a7e2e4d6eb903626`

Der GitHub-Konnektor erlaubt hier nicht den direkten Upload der lokalen großen HTML-/ZIP-Datei als GitHub-Release. Die ZIP-Datei ist in der aktuellen Chatkonversation verifiziert verfügbar.
