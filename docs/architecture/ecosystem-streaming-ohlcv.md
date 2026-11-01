# ECO-19a — Echtes SOL-Minuten-CSV/GZIP verlustfrei und speicherschonend indexieren

**Implementiert:** `src/quant/ecosystem/market_data/streaming.py`, `scripts/ecosystem_stream_ohlcv.py`, `tests/ecosystem/test_streaming_ohlcv.py`.

**Wichtig:** Dies ist **ein Importer, keine tatsächlich geladene SOL-Historie, keine fertige TradingView-Parität und kein Trading-Backtest**. Ein überprüfbarer kompletter Fünfjahresdatensatz wurde noch nicht bereitgestellt. Im Repo existiert `quant.strategies.imba` sowie `quant.features.renko`, und die Dokumentation `docs/RENKO_TRADINGVIEW_PARITY.md` zeigt eine wichtige zuvor erkannte Abweichung (0.1-Box Renko auf TradingView 5-Minuten-Quelle, nicht 1-Minute). Der neue Importer ersetzt diese Strategie-/Signalprüfung **nicht**.

## Ablauf

Der Import liest echte, vom Nutzer bereitgestellte UTC-*Open-Timestamp*-Kursdaten zeilenweise und persistiert sie **atomar** in eine eigene SQLite-Datei. Die Quelldatei wird nicht geändert. Bei Fehlern (unvollständige GZIP-Datei, ungültige OHLCV-Preise/Volumen, Out-of-Order, Doppeltimestamp, inkompatibles Intervall, ungültiger Lizenz-/Markt-Provenienz-Eintrag oder unvereinbare Abdeckungsgrenzen) wird der Import abgebrochen und die temporäre Datenbank entfernt. Für ein vorhandenes Ziel besteht striktes Overwrite-Verbot. Über eine SQL-Primary-Key-Zeitindexfunktion können einzelne UTC-Fenster gelesen werden, ohne Jahre an Kerzen in Python-Arbeitsspeicher zu laden.

Zwei Formate:
- `canonical`: CSV mit Kopfzeile `timestamp,open,high,low,close,volume`. Timestamp als ISO 8601 **mit Zeitzone**, Intervall-Open, keine Close- oder Trade-Zeitstempel.
- `binance_klines`: **nur CSV-Layout** mit genau zwölf Spalten ohne Header, Spalte 0 in UTC Epoch-Millisekunden und Spalte 6 als Kerzen-Endmillisekunde. Ein gleich benanntes Layout beweist weder Börse noch Markt und setzt *keinesfalls automatisch* Futures/Spot oder Gebührenwerte; dafür sind gesonderte Provenienzangaben zwingend.

`Provenance` (aus ECO-02 #37) ist ein explizites JSON-Objekt mit Quelle, Lizenz, exaktem Symbol, Marktart (`spot` oder `linear_futures`), Kontraktmultiplikator, Takt, UTC-Open-Semantik und `synthetic`-Wahrheitswert. Der Importer kann weder eine Lizenz noch die historische Datenqualität verifizieren. SourceFile-SHA256 wird über alle **originalen** CSV-/GZIP-Bytes, Candle-SHA256 über normalisierte Zeit-/OHLCV-Datensätze gebildet. Derselbe Input erhält denselben Canonical Dataset-Fingerprint. Herkunfts-/Hash-Manifest wird in der SQLite-Datenbank eingebettet, nicht in irgendeiner unkontrollierten externen Dateibenennung.

Gaps werden **niemals gefüllt**: Summen `missing_bars`/`gap_runs` sowie die ersten 64 zusammenhängenden Lücken sind im Manifest. `--reject-gaps` stoppt den gesamten Import. `--expected-start`/ `--expected-end` (exklusiv) erlauben auch die **vorgegebenen** Randperioden auf fehlende Kerzen zu prüfen; ohne diese werden ausdrücklich nur beobachtete Grenzen ausgewiesen. Unvollständige UTC-Tage werden gemeldet. Höchstens 4096 Datensätze bleiben in der Ingest-Warteschlange. `read_window_as_dataset(..., max_bars=10080)` erzeugt für **ein** eng begrenztes Forschungsfenster eine kompatible ECO-02-`Dataset`, wobei dieser Fenster-Hash nicht mit dem Gesamt-Datenhash verwechselt werden darf.

## Beispiel CLI (nur nach echtem Datenbezug)

```bash
PYTHONPATH=src python scripts/ecosystem_stream_ohlcv.py \
  --source /PFAD/ZU/ECHTER_SOL_1m.csv.gz \
  --layout canonical \
  --provenance /PFAD/ZU/sol_source.json \
  --expected-start 2021-11-01T00:00:00Z \
  --expected-end 2026-11-01T00:00:00Z \
  --output /PFAD/ZU/sol_1m_research.sqlite
```

**Nur als Platzhalter!** Der Pfad existiert derzeit nicht nachgewiesenermaßen. `--reject-gaps` zusätzlich verwenden, wenn ein komplett lückenfreies Rohdatenfenster verlangt wird; sonst Lücken dokumentieren und Replay bei positionstragenden Gaps abbrechen.

Beispiel für die **explizite vom Dateneigner zu bestätigende** Provenienz:

```json
{
  "source": "GENAUER_ORIGINALANBIETER_UND_DOWNLOAD",
  "license": "EXAKTE_ZULÄSSIGE_NUTZUNG",
  "symbol": "EXAKTES_BÖRSENINSTRUMENT",
  "market_type": "linear_futures",
  "contract_multiplier": "1",
  "interval_seconds": 60,
  "synthetic": false,
  "timezone": "UTC",
  "timestamp_semantics": "open"
}
```

**Nicht die Beispiel-Provenienz als tatsächlich wahr einreichen:** Hier fehlen konkrete Venue-Bestätigung, Kursquelle, Lizenz und Kontraktspezifikation.

## Tests und Grenzen

```bash
PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_streaming_ohlcv.py -v
```

Tests prüfen GZIP-Dekodierung, deterministische SHA256, OHLCV-Validität, 12-Spalten-Epoch-Millisekunden-Layout, Dubletten und unsortierte Daten, Lücken-/Randprüfung, atomare Löschung bei abgeschnittenem GZIP, Indexfenster und **40.000 synthetische Minuten** mit begrenztem Spitzen-Arbeitsspeicher sowie ein externes CLI-Testkommando. **Der Testumfang stellt keine gemessene 5-Jahres-Verarbeitungsgeschwindigkeit dar**; Laufzeit-/Platzbedarf einer echten 2,6-Millionen-Bar-Datei sind anschließend separat zu vermessen.

**Nächste konkrete Integrationsaufgaben:**
1. Den echten historischen Originaldatensatz (Quell-URL, Lizenz, Venue-Kontrakt, Funding, Dauer, SHA256) bereitstellen. Ohne diesen keine realen PnL-Behauptungen.
2. Aus `quant.features.renko.renko_from_close` nur unter nachgewiesener 5-Minuten-Quellperiodizität die korrekte Renko-0.1-Struktur überprüfen; **Renko-Wicks** und Kraken SOLUSD.PM TradingView-Signalparität sind gemäß bestehender Reponotiz noch nicht gesichert. Nicht aus Renko-Markerpreisen füllen.
3. Einen *separaten* kausalen 5m-zu-Renko-zu-IMBA-Achsgen-Signaladapter mit eindeutiger Quelle, Bar-Schlusszeitpunkt und Vergleich gegen echte TradingView-Webhooks bauen; erst dann an ECO-18 Markt-Fills weitergeben.
4. ECO-18 Replay ist für eine `Dataset`-Woche materialisiert; ein **fortlaufender** inkrementeller Paper-MTM-Executor für 5 Jahre muss die Fenster mit offenem Positions-/Stop-/Fundingzustand durchreichen. Keinen 5-Jahres-Datensatz in ein großes `normalize()` zurückladen.
5. OOS-/Walk-forward inkl. unabhängiger Parameter-Optimierungsgrenzen und multiple Testkorrektur abgrenzen; kein Mutationserfolg aus rückblickendem Training behaupten.

Keine Produktions-, Kraken-, KuCoin-, TradingView- oder Railway-Änderung.
