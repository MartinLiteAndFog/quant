# ECO-18 – Kausaler OHLCV-/Signal-Replay → Simulationsuhr → Evolution

**Status:** Forschungsadapter auf `codex/ecosystem-market-replay-bridge` (Stack auf [ECO-17 #52](https://github.com/MartinLiteAndFog/quant/pull/52)). Keine produktive Börsenverbindung. Dies ist ein **Paper-Fill-/Netto-PnL-Adapter**, kein Nachweis einer erfolgreichen SOL-Strategie.

## Eingangsdaten

1. `Dataset` aus [ECO-02 #37](https://github.com/MartinLiteAndFog/quant/pull/37), aus quell-/lizenzdokumentiertem, normalisiertem UTC-open-OHLCV (idealerweise 1-Minuten-Kerzen). Der Datensatz bringt einen SHA-256-Inhaltsfingerabdruck und Lückenbericht. Bei der Projektimplementierung ist bisher **kein** 5-Jahres-SOL-Minuten-CSV als importiert nachgewiesen; die Tests verwenden vollständig als **synthetic** markierte Kerzen.
2. Externe `TargetSignal(signal_id, strategy_id, target_quantity, signal_at, available_at, received_at)`. **Weder Signale noch profitable Trades werden aus OHLCV erfunden.** Das ist eine numerische **Zielpositionsgröße**: Long >0, Short <0 (nur bei ausgewiesenem Linear-Futures-Kontrakt), Flat =0. `available_at` darf nie vor `signal_at` liegen. Früheste Ausführung am **ersten Kerzenopen strikt nach verfügbarer Signalzeit und Empfang**. Gleichzeitige widersprüchliche Zielwechsel werden abgelehnt. Absolute Zielpositionsgröße erfordert einen expliziten externen Research-Cap; sie ist nicht durch den Pflanzenbewuchs automatisch finanzierbar.
3. Optionale ausdrücklich datierte und unabhängig bekannte `FundingTick(funding_id, at, known_at, rate, mark_price)` mit gültiger Futures-Provenienz; positiver Zinssatz belastet Long und begünstigt Short. Kein blind angenommener Funding-Zeitplan. Aus Sicherheitsgründen werden Funding-Veröffentlichungen, die erst am Folgetag bekannt werden, **abgelehnt**, bis ein expliziter verzögerter Settlement-Adapter existiert.
4. `ReplayConfig` mit Spread/Slippage/Fees, zulässiger Höchstposition, Datensatz-Vollständigkeit, maximal toleriertem Tagesend-Marktalters und Fill-Bestätigungszeit. Konkrete bps-Werte von ECO-02 sind **Annahmen**, keine gemessenen Futures-Orderbook-Kosten.

## Geldfluss / Bewertung

Der Adapter `replay_market(dataset, signals, funding=(), config=())` produziert `ReplayOutput` (Datensatz-/Signal-Fingerprints, `PaperFill`-Nachweise, `ReplayDay.events` und positionale Endzustände).

- Für jeden Fill: Richtung und Menge aus dem **Delta zum letzten Ziel**, ECO-02-`paper_fill` mit bezahltem Halbspread + adverser Slippage und getrenntem Fee.
- **Exakte Mark-to-Market-Änderung am Fill-Open:** `old_signed_quantity × contract_multiplier × (next_open_reference - last_reference) - filled_signed_delta × multiplier × (fill_price - next_open_reference)`. Danach negative `fee` als *separates* ECO-03-`IntradayEvent`. Spread und Slippage sind bereits im Ausführungspreis enthalten und werden **nicht** erneut als externe Kosten abgezogen.
- Zum Tagesende wird nur der **letzte bereits geschlossene, im selben UTC-Tag tatsächlich bekannte** Kerzen-Close verwendet (23:58-Open → 23:59-Close; niemals die 23:59-Kerzen-Schließung am **Folgetag**). Wird gleichzeitig ein neuer Kerzen-Open gehandelt, hat das alte, nun bekannte Close **Vorrang vor dem neuen Fill**; so erhält die gerade erst eröffnete Position niemals den alten Kerzengewinn.
- Geschlossene logische Trades werden nicht mit Fillzahlen verwechselt. Ein erstmaliger Entry ist noch kein abgeschlossener Trade; schlichte Teilreduzierungen werden nicht als unabhängige Gewinner gezählt. Eine vollständige Schließung oder Richtungsumkehr zählt als abgeschlossener Trade.
- Funding ist ein separat datierter positiver oder negativer NAV-Eintrag und wird einmal pro Funding-ID verbucht. Externe Cashflows fehlen bewusst: die Marktdaten-Pipeline darf keinen zusätzlichen Kapitalfluss erfinden.
- **Lücken bei offener Position** und zu alte Tagesendbewertung führen zu einem harten Replay-Fehler, nicht zu interpoliertem Risiko oder nachträglicher PnL-Erfindung. Provenienz bleibt im Export.

## Verbindung zur Ökologie

`apply_replay_to_evolution(initial_state, market_output, weather_by_day=..., opportunities_by_day=..., free_margin_by_day=..., config=...)` reicht die Tagesereignisse aus ECO-02 an ECO-17 `step_evolution` weiter. Drei Informationen sind **verpflichtend extern**: kausales Market-Weather, **unabhängig ex ante definierte** Gelegenheiten je Tag/Strategie und realistisch validierte freie Margin. Fehlende Marktgelegenheiten dürfen **nicht** aus erfolgreichen Trades oder Vision-DNA abgeleitet werden, sonst wäre Dormanzschutz statistisch manipulierbar. Für neue Mutationskinder existieren keine automatisch generierten Markt-Entscheidungen: ihr Signalmodul und ihr historischer Forward-Test sind getrennt zu implementieren.

WICHTIG: Das Replay von **vorgegebenen Zielgrößen** ist noch **nicht** durch die dynamische ECO-17-Positionsgrößen-/Margin-Intents risikofreigegeben. Es kann historische PnL eines vorgegebenen Signals rekonstruieren und die resultierenden Tage an die Ökologie geben, ist **aber kein investierbarer, geschlossener Evolutionsbacktest**, solange Volumen-, Lot-, Venue-/Liquidationsregeln, historische Signal-Generation je Genom, Orderbook-Slippage, Portfolio-Allocations, OOS und Multipletest-Korrektur fehlen. `dataset.provenance.synthetic=False` ist nur Marktdaten-Provenienz, **niemals automatisch** `oos_verified=True`.

## Offline-Eingabe / direkt nutzbares CSV-CLI

```bash
PYTHONPATH=src python scripts/ecosystem_market_replay.py \
  --bars /data/SOLUSDT_1m.csv \
  --signals /data/signals.csv \
  --provenance /data/solusdt-provenance.json \
  --output /data/paper-replay
```

`signals.csv` benötigt genau diese Spalten: `signal_id,strategy_id,target_quantity,signal_at,available_at,received_at`. Zeitwerte mit expliziter Zeitzone, keine nachträgliche Outcome-Sortierung. `provenance.json` verwendet das vorhandene ECO-02-`Provenance`-Schema (u.a. Quelle, Lizenz, `symbol`, `market_type`, `contract_multiplier`, Intervall, `synthetic`). Optional `--funding funding.csv` (`funding_id,at,known_at,rate,mark_price`), `--start`, `--end`; `--allow-partial-days` **nur für Fixtures**. CLI ist offline und liest Eingaben nur; Ausgabe: `manifest.json`, `paper_events.jsonl`, `paper_fills.jsonl`, `daily_positions.jsonl` mit Datensatz-/Signalfingerabdruck und expliziten Warnfeldern `verified_oos=false`, `capital_risk_cleared=false`, `live_orders_sent=0`. Der CLI-Export allein schaltet **keine** Positionen in der Wiese frei.

## Tests / Hand-off

```bash
PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_market_replay.py -v
```

Testet:
- Übernacht-Long mit +10/-10-Preisbewegung, Spread/Slippage/Fees jeweils nur einmal und ECO-03-NAV-Konsistenz;
- Short-Futures plus getrennte Funding-Zahlung;
- Signal-/Receive-Zeitverzug, erste nächste Open-Preis-Ausführung;
- Duplikat-/Gleichzeitigkeit- und unbekannte Daten-Rejection;
- Lücken innerhalb einer offenen Position, unvollständige UTC-Tage, Futures-vs.-Spot-Kontraktunterscheidung;
- Pflichtübergabe von Chancen und free_margin an ECO-17, deren 90/10-Reserve und deterministischen Snapshot.

**Nächste Agenten-Aufgaben:** historische SOL-Minutenquelle + Lizenz/Dateihash nachweisbar importieren; echte Pine-/TV-Signaläquivalenz pro Genom implementieren und gegen Alerts/Fills validieren; OOS-Walk-forward strikt vom Parametertuning abtrennen; Paper-Account Positions-Ledger/Risikogrenzen **vor** Signalausführung verbindlich integrieren. Keinen realen Kraken-/KuCoin-Executor deployen.
