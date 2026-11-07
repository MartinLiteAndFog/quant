# QUANT 0.44 — lokaler Float32 Research-Datenspeicher

Datum: 2026-11-07. Status: **lokaler Forschungsprototyp, nicht live deployed**.

## Verifizierte Implementierung

Der Offline-App liegt ein kleiner, auf `127.0.0.1:8765` gebundener Python-HTTP-Dienst bei (nur Python-Standardbibliothek). Ein Mac-Starter `Start_QUANT.command` legt `~/QuantData` an, importiert einmal die vorhandene SOLUSDT CSV.GZ aus `~/Downloads` (Original bleibt erhalten) und öffnet die gewohnte QUANT-HTML-App. Ein separates `Prepare_Signale.command` kann neun übliche USD-Renko-/IMBA-Genkombinationen vorberechnen. **Der Starter wurde in der Linux-Testumgebung nicht auf macOS selbst ausgeführt.**

- Datenbank: `~/QuantData/meta/research.sqlite3` (SQLite, WAL), Tabellen `datasets`, `artifacts`, `sweep_cases`, `sweep_runs`, `families`, `settings`.
- Binärer Cache: `~/QuantData/cache`, zlib-komprimierte Little-Endian Float32 OHLCV (UTC uint32 Sekunden + 5×float32, 24 B/Bar), Renko-Bricks, IMBA-Signalereignisse. **8 GiB Artefakt-Quota, nicht vorreservierter SSD-Speicher; SQLite/WAL/temporäre Daten zählen nicht zur Quota.** Cash-Abrechnungen künftig Integer-Cent statt Float32-Akkumulation.
- Zeitintervalle: 1m, 5m, 15m, 30m, 1h, 4h, 1d, UTC-aligned. Datenversion und Source-SHA256 in Artefakt-Keys.
- Wiederverwendung: Renko/IMBA per Symbol, Datenversion, Zeitintervall, Boxgröße und Sensitivity; auch Jahre ohne Brick/Signal erhalten valide leere Cache-Partitionen.
- Adaptiver Sweep: **Planer**, 30 % grober Core, restliches Budget Detail/Diversität; eine echte 5.000er-Ausführung mit gezielter Gewinnerregion-Verfeinerung und minutengenauen Equity-Curves ist noch nicht implementiert. Strategie-Familien derzeit **nur nach Gen-Nähe**, nicht nach Renditekorrelation.
- Browser: Die lokale SOL-Datenquelle wird in die bestehende QUANT-Datenpipeline eingehängt; die bisherige Tagesarchiv-Ansicht bleibt verfügbar.

## Messungen mit dem tatsächlich vorhandenen SOLUSDT-Paket

- Quelle: 3.276.965 Minutenkerzen, 2020-08-11 bis 2026-11-04, 1.424 vermutlich fehlende Minuten, keine doppelten/ungültigen Zeilen.
- Sieben vorberechnete OHLCV-Zeitauflösungen: **58.865.978 Byte** komprimiert, 49 Jahresscheiben.
- 2025 1m: 525.600 Bars, 7.256.602 Byte; 5m: 105.120 Bars, 1.634.418 Byte.
- Zusätzliche 5m USD-Renko 0.1/0.2 und IMBA sensitivity 5/15: insgesamt 65.397.638 Byte Cache im Test. Wiederholte identische Signalberechnung: 7 Jahrescache-Treffer.
- Test `test_quant_044_store.py`: Import, UTC-Zeitzonen, 7 Auflösungen, Null-Brick- und Null-Signal-Caching, Fallcache, grobe Genfamilien, Sweep-Planer erfolgreich.
- Browser-Smoke-Test in Chromium mit simuliertem lokalen HTTP-Endpunkt und echtem SOL-Januar-2025-Binärblock: SOL-Datenquelle standardmäßig gewählt; kein JavaScript-Fehler. Direkter Loopback-Chromium-Aufruf war in der Sandbox durch eine Administratorregel blockiert; **Mac-Launcher und echte lokale Browser-HTTP-Verbindung sind noch nicht end-to-end verifiziert**.

## Noch offene Kernaufgaben

1. Ein gemeinsames minütliches Strategiearchiv für die komplette 3D-Heatmap, Top-N, Handelsregeln-Sweep und Walk-Forward. Der alte Tagesarchiv-Modellwert von EUR 9.367,05 darf nicht als Intraday-Wert ausgegeben werden.
2. Renko-%/ATR und andere Signalfamilien, Signal-Parität Python ↔ JavaScript ↔ TradingView und die korrekte Reihenfolge bei Minute-OHLC-Stopp/TP.
3. Persistente Float32-Minutenkurven je ausgewähltem Sweep-Fall, adaptiver Core-/Detail-/Diversitäts-Ausführer mit 5.000er Budget, Ergebnisfamilien nach **Trade- und Equity-Ähnlichkeit**, OOS/Walk-Forward-Validierung.
4. Echte macOS-Launcher-/Browser-/Storage-Tests, Cache-Quoten-/Eviction-Stresstests, optionaler lokaler Hintergrundprozess.

## Release

`QUANT_0.44_Lokale_Datenbank.zip` (Chat-Download, **nicht** ins GitHub-Repo hochgeladen), SHA256 `d855cc3e0d6ebc94e177ac9ef664fbc03c9c55a2fe0493618aec0162133bd1c7`. Enthält `dashboard.html`, `quant_store.py`, `quant_server.py`, `quant_local_bridge_044.js`, beide Mac-Starter, README und Regressionstest. Kein Railway-/Live-Trading-Code verändert.
