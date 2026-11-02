# TradingView → Cloudgarden: unabhängiger Renko-MTF-RSI und Ausführungs-Audit

**Stand: 2. November 2026 · Research-only, keine Live-Orders und keine Cloudgarden-Kapitalbuchung.**

## Prüfbasis

- Binance SOLUSDT 1-Minuten-Historie, gzip: SHA-256 `c4f9b9f85d00072f8e367a4bd23978ec92c3bce0d326fe3d17b2c26b5f3d1c9c`.
- TradingView-Export `BINANCE_SOLUSDT, 5 (5)(1).csv`: SHA-256 `795f77224dddc873d6b445440b3f03512e2b7274b029ba4d23a1f1c098dd3771`.
- Zeitzonenkorrigiertes Seed-/Trade-Manifest v2: SHA-256 `58713335df03de608b3e40348dd6838ed802098deb3a95d9b8b9597ea5acf70f`.
- Datenabdeckung bis **17.10.2026 10:29:59 UTC** für vollständig verfügbare 30-Minuten-Binance-Quellkerzen. Alle späteren TV-Daten aus den Vergleichen ausgeschlossen.
- **Die Originaldaten liegen ausschließlich in Quant (privat), nicht im GitHub-Repo.**

## 1. Renko-basierter RSI(14) unabhängig vom Export berechnet

Traditionelle Renko-Konstruktion aus echten Binance-Schlusskursen: 0,10-USDT-Box, 1-Box-Fortsetzung, 2-Box-Umkehr, mehrere Bricks pro Quellkerze mit +1-ms-Indizes. Der 5-Minuten-Chart braucht vor dem 07.09.2026 H1-Quellen und danach 5-Minuten-Quellen; die unabhängigen 15m- und 30m-RSI-Feeds verwenden durchgehend 15- bzw. 30-Minuten-Quellkerzen. Auf jeder Serie: Wilder-RSI(14) mit SMA-Seeding.

| TradingView-RSI-Zeile | Exakt (absoluter Fehler < 10⁻⁶) | Verglichene Zeilen | MAE (RSI-Punkte) |
|---|---:|---:|---:|
| 5m | 10 074 | 10 225 | 0.000600 |
| 15m | 10 219 | 10 225 | 0.001876 |
| 30m | 10 217 | 10 225 | 0.003155 |

Alle 15m-/30m-Nichtmatches konzentrieren sich auf die bekannte 12.-Oktober-Renko-Zeitverschiebung; sie sind keine pauschalen RSI-Formelfehler. Historische chartseitige Werte können vor den Quellkerzen-Schluss datiert sein.

## 2. Historischer Chart-RSI versus strikt bestätigter RSI

**Dies sind zwei unterschiedliche Zeitmodelle:** (a) historischer TradingView-Chartwert zum exportierten Brick-Zeitstempel, (b) RSI-Wert wird erst **nach Abschluss der zugrundeliegenden 30m-Binance-Quellkerze** freigegeben; ein 5m-Signal gilt frühestens am Ende seiner 5m-Quellkerze als bekannt. Der zweite Modus ist eine bewusst konservative Annahme, kein gesicherter Befund über das Live-Verhalten des geschützten RSI-Indikators.

- **756** BINANCE-SOL-Entry-Zeitstempel aus sechs Varianten liegen im verfügbaren Datenfenster; davon 30 mit deaktiviertem RSI-Filter.
- 726 Einträge mit aktivem Filter: Der **chartseitige** RSI besteht **726/726** (ausgewählte TradingView-Trades; Auswahl-Effekt), der **bestätigte** RSI nur **353/726**. Die übrigen **373** scheitern im streng kausalen Modell.
- Aktivierter Filter, nach Richtung: Long 79/446 bestätigt, Short 274/280 bestätigt. Sechs Strategien teilen zahlreiche Ereignisse; 726 sind **keine unabhängigen Trades**.
- Dies ist **kein Beweis für Lookahead/Repainting im Original-Skript**, dessen Quelle geschützt ist. Es ist ein Warnsignal, das vor einem ehrlichen Trading-Backtest geprüft werden muss.

## 3. Referenzorders mit realen Binance-1m-Preisen neu bewertet

Das Referenz-Entry/Exit-Timing stammt weiterhin aus dem XLSX; die neue Berechnung **generiert die Orders nicht selbst**. Für jedes Referenzereignis wird unter der Annahme `Quell-5m-Kerze zuerst abschließen` der Open der ersten folgenden vollständigen 1m-Marktkerze verwendet. Futures-Marktpreis und Gebührenmodell können abweichen; hier Binance SOLUSDT (verfügbare 1m-Quelle) und original dokumentierte 0,06% Kommission pro Seite, **ohne zusätzliche Slippage**. Referenzmenge bleibt dieselbe (keine Equity-/Sizing-Reoptimierung).

- **746** geschlossene Binance-Referenztrades können auf beide folgenden 1m-Opens abgebildet werden; **5** würden Ein-/Ausstieg zur selben frühestmöglichen Minute benötigen; **12** haben keine vollständige Rohdatengrundlage.
- Durchschnittliche absolute Abweichung: **13,58 bp Entry**, **18,15 bp Exit**. Kein Spread oder Orderbuch/Slippage berücksichtigt.
- Beispiel: Short 07.09.2026 **02:35 UTC** bei TradingView **106,10 USDT** (synthetischer Renko-Preis), frühester hypothetischer realer 1m-Open **02:40 UTC / 106,90 USDT**; Differenz **+75,4 bp**. Das für Renko benutzte 5m-Schlussdatum ist erst um 02:40 vollständig bekannt.

| Referenzvariante | Vergleichbare Trades | TV-Netto¹ | Zeitfixiertes 1m-Repricing¹ |
|---|---:|---:|---:|
| Binance SOL · 17 Oct | 85 | +336.03 | -72.02 |
| Binance SOL · 18 Oct A | 191 | +1 314.66 | -338.40 |
| Binance SOL · 18 Oct B (RSI aus) | 30 | -4.48 | -45.80 |
| Binance SOL · 18 Oct C | 189 | +2 457.32 | +62.47 |
| Binance SOL · 18 Oct D | 166 | +6 033.17 | +421.69 |
| Binance SOL · Baseline | 85 | +336.03 | -72.02 |

¹ Die Zahlen sind **ausschließlich** auf die jeweils vergleichbaren, zu originalen Zeitpunkten fixierten Trades bezogen. Sie sind **weder ein unabhängig ausgeführter Backtest noch ein Netto-Portfolioresultat**, da die Handelssignale, Marktpreise nach einem anders verlaufenden Portfolio und Reihenfolge/Stopmechanik nicht neu erzeugt wurden. Die sechs Varianten überlappen. Niemals summieren und als unabhängige Gewinne interpretieren.

## 4. Einordnung der noch ausstehenden Exit-Parität

- Tradingade Strategy Builder hat `Trailing TP %`, `TTP % = 0`, `Dev % = 0.8`, `SL % = 0.9` (die SOL-KuCoin-Variante hat 1.1% SL). Die XLSX-Signale `SL`, `TP`, `long`, `short` enthalten deren eigene Fill- und Flip-Logik. **Diese Exit-State-Machine ist hier noch nicht unabhängig reproduziert.**
- Der interne TradingView-Broker simuliert Trades bei Renko-Zeitstempeln und Renko-Kursen; real sind diese Kurse nicht zwangsläufig nach Signalbestätigung handelbar. Siehe TradingView-Dokumentation: https://www.tradingview.com/support/solutions/43000480330-the-renko-chart-shows-incorrect-values-and-or-is-constantly-being-recalculated/
- Der `Multiple Time Frame RSI 1m 5m 15m 30m` Originalindikator ist ein geschütztes TradingView-Skript: https://www.tradingview.com/script/l2Q6Dq6A-Multiple-Time-Frame-RSI-1m-5m-15m-30m/. Die numerische Darstellung wird fast exakt rekonstruierbar, nicht sein interner `request.security`-Aufruf nachgewiesen.
- KuCoin SOL/BTC-Varianten sind mangels jeweils venue-gerechter 1m-Daten in diesem Schritt **nicht** realpreislich nachgerechnet.

## Code, Tests und Wiederholung

```sh
python3 -m unittest discover -s apps/cloudgarden/scripts -p "test_verify_tv_renko_mtf_and_fills.py"
python3 apps/cloudgarden/scripts/verify_tv_renko_mtf_and_fills.py \
  --raw-binance-gz "/private/Quant/SOLUSDT (3).csv(1).gz" \
  --tv-renko-csv "/private/Quant/BINANCE_SOLUSDT, 5 (5)(1).csv" \
  --seed-manifest "/private/Quant/quant_tv_first_plants_corrected_utc_v2.json" \
  --out-dir "/private/Quant/reports/MTF_2026-11-02"
```

Der Code enthält keine Exchange-Secrets, keinen Webhook-Client, keine Orders und schreibt lediglich lokale Berichte. Siehe `tradingview_renko_mtf_audit_summary.json`, `tradingview_mtf_renko_rsi_rows.csv`, `tradingview_mtf_rsi_entry_gate.csv`, `tradingview_market_1m_fixed_event_reprice.csv`.

## Nächster überprüfbarer Schritt

Tradingade SL-/TTP- und Richtungswechselzustände als deterministische, **quellzeitpunktbezogene** State-Machine portieren und eigenständig aus dem geklärten EW/RSI-Eventstrom Trades erzeugen. Erst danach dieselben abgeschlossenen Orders und echten 1m-Fills mit dem Cloudgarden-Ledger verbinden; keine historische XLSX-Zukunft in Fitness/Genetik/Vitalität einspeisen.