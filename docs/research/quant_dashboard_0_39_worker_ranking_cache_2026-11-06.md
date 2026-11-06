# QUANT 0.39 — Worker, Forschungsdatenbank und wiederverwendbare Ranking-Signale
**Stand:** 2026-11-06  
**Status:** Offline-Forschungsprototyp; NICHT live ausrollen.

## Architekturentscheidung
Ein einziges QUANT-UI bleibt maßgeblich. Intraday soll künftig die reguläre Datenbasis werden; der separate Tagesarchiv-Modus mit historischen Presets, Heatmaps, früheren Top-N-Rechnungen und Ergebnissen wird **nicht entfernt**. Die technische Migration erfolgt schrittweise, ohne die alten Forschungsresultate als bestätigte Live-Renditen umzudeuten.

### Datenfluss
1. `OHLCV market source` — bereits vorhandenes lokales QDP/CSV.GZ-Archiv aus QUANT 0.34; 1m OHLCV, Metadaten und Qualitätsprüfung, IndexedDB `quant-local-market-data-v1`.
2. `signalSets` — gleiche SOL/BTC-Gene (IMBA sensitivity, Renko-Brick USD/% oder candle, Signal-Timeframe), Signalereignisse aus bestätigten 1m/5m-Quellbars. Pro Gen und exakt identischem Datenfenster einmal rekonstruiert.
3. `rankings` — 1× Ranking-/P&L-Tagesstände auf denselben Signalereignissen, separat nach Gebühren-/Slippage-Annahmen gespeichert. Keine zukünftigen Tagesstände in Top-N-Rankings.
4. `portfolioCases` — vollständiger Kapital-, Ranking-, Stopp-, Take-Profit- und Zeitfenster-Vektor; nur Ergebniskennzahlen werden standardmäßig dauerhaft gespeichert, nicht für jede Variante Millionen NAV-Punkte.
5. `sweepRuns` — einzelne Forschungsvergleiche als Manifest einschließlich Umfang und Cachetreffern.

Der zweite Speicher ist versioniert als **IndexedDB `quant-research-cache-v1`**, Schema 1. Die im Browser eingebetteten Offline-Skripte erzeugen einen **Blob Web Worker**, in dem 1m OHLCV, Renko/IMBA-Signale, Ranking und Portfoliofälle gerechnet werden. Ein Transferable `Float64Array` vermeidet die teure Kopie eines zweiten vollständigen OHLCV-Rohblocks. Der Cache-Schlüssel enthält Engineversion 0.39.2, Datenmanifest, untersuchten Zeitraum/Signalbar und den zusätzlichen FNV32-Inhaltschecksum des geladenen OHLC-Blocks. FNV32 ist ein schnelles Änderungsmerkmal, **kein kryptografischer Herkunftsnachweis**. Ein Wechsel von Top-N, Kapitalbudget, Portfolio-Stopp oder Gewinnmitnahme soll Signale **nicht erneut generieren**.

### Portabilität
Der Cache persistiert in einer geeigneten Browser-Origin samt Profil. Er ist NICHT in der HTML-Datei gespeichert. Die UI zeigt Objektzahlen und, falls zugänglich, die Browser-Speicherquote. Nutzer können alle vier Stores komprimiert als `QUANT_research_0.39_YYYY-MM-DD.qrc.gz` exportieren und wieder importieren. Das Markt-QDP bleibt ein getrenntes Originaldatenarchiv. Kann IndexedDB wegen privaten/opaque Browser-Origins nicht verwendet werden, zeigt die UI **Arbeitsspeicher** an; dann ist ein QRC-Export vor dem Schließen unerlässlich.

## Was in 0.39 tatsächlich umgestellt ist
- Im Intraday-Modus: Signalberechnung, 1× Ranking und gemeinsame Portfolio-Ausführung von 0.38 laufen im Worker.
- Vorbereitete Signale werden pro Gen wiederverwendet. Der 16-Fälle-Depot-Schutz-Sweep läuft ebenfalls im Worker; Ergebnisse werden pro Fall gespeichert und bei erneutem Durchlauf wiederverwendet.
- Die alten Tagesarchive und der bisherige vollständige mehrdimensionale Sweep sind erhalten.
- Nicht umgestellt: der vollständige 32.000+-Varianten-Intraday-Sweep, der historische manuelle Walk-Forward-Player und die vollständige mehrjährige 3D-Renko-Genom-Rekonstruktion. Diese Arbeiten sind **nicht als fertig zu deklarieren**.
- Nicht validiert: TradingView-/Renko-Tickparität, minütliche Ausführungsreihenfolge innerhalb OHLC, Funding, echte Exchange-Fills, vollständige Intraday-Liquidation.

## Browser-Regression 0.39.2
Mit **28.800 echten SOL-1m-Kerzen** und vier markierten Bots:
- Worker-Portfolio: €801,3658 aus €1000 unter den gewählten Testparametern (kein vergleichbarer Performancebeleg für andere Historien).
- 16 von 16 Fällen des Depot-Schutz-Sweeps berechnet; zweiter identischer Sweep: alle 16 Fälle aus dem Cache.
- Cache enthält 4 Signals, 4 Rankings, 16 Portfolios und 2 Sweep-Manifeste.
- Export → gzip QRC (5.046 Byte Testarchiv) → Import: 26 Einträge erfolgreich wiederhergestellt.
- Tagesarchiv-Umschaltung funktioniert; keine JavaScript-Laufzeitfehler.
- Die Headless-Test-Origin war `about:blank` ohne persistente IndexedDB-Berechtigung; der Worker meldete korrekt **Arbeitsspeicher**. Dauerhafte IndexedDB-Persistenz über Browserneustarts ist **noch nicht unabhängig nachgewiesen**.

## Noch zu erledigen (0.40+)
1. Deterministische Fenster-Snapshots für die Renko-Trendzustände, sodass überlappende Monate/Jahre ohne falschen warmup erneut kombiniert werden können.
2. Vollständige 4/5/6 Top-N-/14-Tage-/TP-/Kapitalregeln-Sweepparameter mit demselben Intraday-Kern und persistenter Falltabelle verbinden; Fortschritt/Abbruch/Wiederaufnahme ohne UI-Block.
3. Große Genom-Pools und 2022–2026 über ein chunkbasiertes Worker- oder optional lokales Python-Precompute-Verfahren verarbeiten, mit getrennten Markt- und Signalversionen.
4. Gebühren, Reversal und TradingView-Alerts/Exchange Fills gegen die Minutensignale paarweise auditieren; Out-of-sample Jahre und fallweise Drawdown-/Slippage-Stresstests.
5. QRC-Import in dauerhaftem normalem Browserprofil und Storage-Quota/LRU unter realen großen Datenpaketen testen.
6. Alte Archivpreset-Benchmarks mit neuem Minutenledger vergleichen, ohne Archivzahlen zu überschreiben.

## Offline-Release 0.39
- `QUANT_IMBA_Dashboard_0.39_Research_Worker_Cache_Offline.html`
- `QUANT_IMBA_Dashboard_0.39_Research_Worker_Cache_Offline.zip`
- Enthaltene Quellen: `quant_research_worker_039.js`, `quant_research_bridge_039.js`, `quant_intraday_ui_039.js`, `quant_intraday_core_038.js`, README.

Die HTML- und ZIP-Dateien sind **Downloads in der ChatGPT-Konversation, noch keine GitHub-Dateien**. Der GitHub-Konnektor kann die lokale ZIP nicht als Release-Asset entgegennehmen. Kein Live-Railway-Service wurde geändert.
