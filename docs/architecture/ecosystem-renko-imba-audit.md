# ECO-19b — 5-Minuten-Renko / IMBA-Kandidatensignale mit striktem TradingView-Paritäts-Gate

**Stand:** Forschungsdiagnostik; **kein handelbarer, überprüft TV-paritärer Bot**. Auf dem gestapelten Forschungsbranch wird aus dem ECO-19a-Streaming-SQLite-Index die bereits existierende Repository-Logik `src/quant/features/renko.py::renko_from_close` und `src/quant/strategies/imba.py::compute_imba_signals` aufgerufen. **Keine neu erfundene IMBA-Einstiegsformel.**

Die Datei `docs/RENKO_TRADINGVIEW_PARITY.md` dokumentiert die wichtige Repo-Erkenntnis vom 07.10.2026:
- Traditionelles Renko dreht erst nach **zwei** Boxen um; bei mehreren Bricks aus einer Quelle erhalten die Bricks auf TradingView künstliche **1-Millisekunden-Zeitverschiebungen**.
- Die betroffene TV-SOL-Konfiguration nutzt **5m Quellen-OHLCV** und Traditional Renko **0,1**. Direkt aus 1m-Daten rekonstruierte Renko-Steine sind ***nicht*** gleichwertig.
- Selbst bei 5m statt 1m bleiben **Renko-Wicks** kritisch; die vorhandene `renko_from_close`-Funktion ist **close-driven** und erzeugt keine exakt gesicherten TV-Wicks. In der dokumentierten historischen Vergleichsperiode standen 5.322 TV-Bricks / 99 IMBA-Änderungen nur 5.279 lokal rekonstruierten 5m-Bricks / 105 IMBA-Änderungen gegenüber (Lookback 50). Das ist **keine bewiesene Signalparität**.

## Was dieses Paket konkret tut

1. `source_5min_bars(index, start, end)`: liest **nur** das angegebene Zeitfenster aus dem SQLite-1m-Index. Alle fünf enthaltenen 1m-Open-Stempel müssen strikt vorhanden sein und in einer UTC-5m-Quelle liegen. Jede Quelle wird **erst zu ihrer realen 5m-Schließzeit** bekannt; Hoch/Tief/Volumen/Close werden auf dieser Ebene korrekt resampelt, Lücken führen zum Fail-Closed. Default maximal 10.080 Minuten pro Fenster, also eine Woche (konfigurierbar), keine 2,6 Mio. Kerzen in RAM.
2. `candidate_imba_signals(index, start, trading_start, end, strategy_id, box=0.1, lookback=50)`: verwendet nach einer expliziten **Warmup-Periode** dieselben alten Renko-/IMBA-Funktionen. Da die Renko-Referenz noch keinen TV-Wick-Abgleich hat, werden die Bricks **mit Body-High/Low** in die alte IMBA-Funktion gegeben und streng als `ImbaCandidate` markiert. Eine naive Weitergabe an ECO-18 als echte Zielpositionssignale **findet ausdrücklich nicht statt**.
3. Jede Mutation/Quelle bekommt eine eindeutige Forschungs-ID; exakte X/Y-Strategiegenparametervalidierung erfolgt später über den ECO-04-Genvertrag. Mehrere 1-ms-verschobene Renko-Bricks können während **derselben noch ungeschlossenen** 5m-Quellbar entstehen: sie sind nicht separat handelbar. Der Kandidat für den Schlusszeitpunkt ist nur die **letzte** dort erkennbare IMBA-Richtungsänderung. Historische 1-ms-Brick-Stempel werden **niemals als Informations-/Order-Verfügbarkeit** interpretiert.
4. `compare_confirmed_alerts(batch, reference)`: nimmt eine **extern bereitgestellte** Referenzliste `reference_id,direction,confirmed_5m_close_at`, misst exakt gepaarte 5m-Bucket-Zeitpunkte und Richtungen, falsche Vorzeichen, fehlende und zusätzliche Signale. Ein 100%-Match beweist erst einen Match **zu dieser gelieferten Referenz**, nicht automatisch Quellenidentität, korrekte Wicks, gesicherte Funding-Daten, TradingView-Äquivalenz aller Marktphasen oder OOS-Gewinn. Entsprechend bleiben `full_TradingView_parity_established=False` und `safe_to_claim_oos_alpha=False` **immer**, bis eine gesonderte umfassende unabhängige Verifikation implementiert wurde.
5. `scripts/ecosystem_renko_imba_audit.py` exportiert optional eine JSON-Diagnose mit Referenzvergleich. Er exportiert **keine TargetSignal-Orders**, keine handelbaren Positionen, kein Geld und keine Webseite.

## Forschungsaufruf (nur wenn echte Daten vorhanden)

```bash
PYTHONPATH=src python scripts/ecosystem_renko_imba_audit.py \
  --database /PFAD/ZU/sol_1m_research.sqlite \
  --start 2026-09-15T00:00:00Z \
  --trading-start 2026-09-20T00:00:00Z \
  --end 2026-09-22T00:00:00Z \
  --strategy-id imba-candidate-50 \
  --box 0.1 --lookback 50 \
  --reference-alerts /PFAD/ZU/tradingview_5m_signale.csv \
  --output /PFAD/ZU/imba-paritaetsanalyse.json
```

**Hinweis:** Eine Woche als Defaultfenster bedeutet, dass der gesamte Demo-Zeitraum inkl. Warmup **maximal 7 Tage** umfasst. Für längere Vergleiche explizites Fenstermemorybudget erhöhen; das ist keine 5-Jahres-Streaming-Backtestengine. `--reference-alerts` ist optional und der Aufruf nur ein Platzhalter. Keine reale TV-Exportdatei wurde in diesem Branch als Referenz hochgeladen oder verifiziert. Ohne Referenz sind Anzahl und Zeitpunkt der lokalen Kandidaten lediglich Forschungsmerkmale.

Der Referenz-CSV-Stempel muss **tatsächlicher Quell-Bar-Schluss** und nicht bloße Webhook-Empfangszeit sein. Wenn nur Empfangszeit bekannt ist, zuerst einen belegbaren Event-zu-Quellbar-Abgleich bauen; nicht per beliebig großem Zeit-Toleranzfenster Signalparität behaupten.

## Tests, verbleibende Blocker

```bash
PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_renko_imba_audit.py -v
```

Synthetische Regressionen: vollständige 5m-Aggregation von 5×1m, richtige Nicht-Vorverfügbarkeit bei 1ms-Brick-Offsets, mindestens ein tatsächlich erzeugtes IMBA-Flipsignal unter kontrollierter Preisfolge, identische Kandidaten-/Referenzpaare als reine Paar-Diagnostik, Gap-Rejection, fehlende/falsche/zeitlich verschobene Alerts, unvollständiger Warmup und CLI-Export mit allen `verified=False`-Kontrollen.

**Als Nächstes:** echten TradingView-Kraken-SOLUSD.PM-5m-Renko-Export (Bricks incl. High/Low/Wicks und echte Alert-/Signalübersicht) sowie passende Venue-1m-Daten aus genau derselben Quelle identifizieren. Wick-Rekonstruktion oder geprüfte Referenz-Pine-Portierung als separates (und gegen Originalexports getestetes) Modul. Erst bei historisch geprüfter Strategie-Signalparität pro Genom ECO-18-Brokerzielsignale erzeugen; anschließend fees/funding/SL/TP und echte unabhängige OOS-Tests, bevor Evolution einen Wertbeitrag beanspruchen darf.

**Nie** aus synthetischen Fixture-Signalen OOS-Glaubwürdigkeit, KuCoin-Live-Leverage oder reale Gewinne ableiten.
