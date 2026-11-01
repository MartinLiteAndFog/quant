# ECO-17 – Deterministische evolutionäre Population (nur synthetische Paperwelt)

**Status:** integrierter Research-Reducer auf einem eigenen Branch über ECO-16. **Keine echten Orders, keine KuCoin-/Kraken-/Railway-Befehle, keine echten SOL-Backtestdaten.** Finanzwerte kommen ausschließlich aus explizit zugeführten, vor Tagesende bekannten inkrementellen Handels-PnL-, Gebühren-, Funding- und Cashflow-Ereignissen. Synthetische Beispiel-Trades dienen nur dem Nachweis der Mechanik.

## Integration, keine zweite Equity

`start_evolution(simulation_id, first_day, equity, seed, config)` übernimmt die drei Mutterpflanzen und 20 bereits kapitalisierten Keimlinge der ECO-16-Beispielwelt. Es ergänzt pro Strategie unveränderlich versionierte Trading-Gene, funktionale Morphologie, Wurzeln/Blätter, Generation, Mutter/Vater/Biene, eindeutige Heimatkachel und eine rein optische Referenz (für ECO-15-VisualDNA-Adapter; beeinflusst **niemals** Tradingresultate).

`step_evolution(previous, day, intraday_events, weather, ...)` verwendet die bestehende ECO-03-Clock für genau einen UTC-Tag und die Phase-Reihenfolge:

1. **Metrics**: Nach Tagesende werden nur *tatsächlich eingespeiste* PnL-/Fee-/Funding-Deltas dem jeweiligen offenen Starttags-Budget zugeordnet; Cashflows sind **kein** Bot-Alpha. Separat vom Ereignisproducer gelieferte Zahl der Trades und **ex ante** Chancen fließen in die ECO-07-Fitness/Vitalitäts-/Credibility-Bewertung. Aktuelle Ergebnisse sind erst für **zukünftige** Saaten nutzbar. Keine zukünftigen Trading-Bar-Daten.
2. **Capital**: ECO-05 bucht 90% des **einen** globalen USDT-Paper-NAV höchstens als Budget-Entitlements (10% Reserve). `closing`-Pflanzen bleiben **im Escrow** und werden ohne bestätigten Abschluss nicht auf andere Pflanzen verteilt. Verbleibendes Budget wird pro handelbarer Pflanze zugeteilt, unter Beachtung offener und reservierter Margin und des extern gemeldeten free_margin. Keine zusätzlichen Wasser-/Regionalkapital-Pools.
3. **Ecology**: ECO-06 berechnet täglich Root-Depth/-Spread, Blatt-/Verzweigungsfläche, Kaktus-Sprossrinde (leaf_count >=1), Sonne und verzögerten Grundwasserzugang. Intended Margin und Leverage sind **Vorschläge**, keine echten Orders, und werden zusätzlich durch reale genetische Maximalhebelschranken und beobachtete freie Margin gedeckelt.
4. **Breeding**: ECO-08 erzeugt bei hinreichendem Paper-Evidenzfenster (in der Demo explizit niedrige, **nicht als OOS behauptete** Reifegrenze) local-/hive-Kandidaten aus den echten IMBA-Sensitivity-/TTP-Parametergenen, nicht aus Screen-Koordinaten. Die Elternkapitalaufteilung ist eine exakte Decimal-Umbuchung im selben Konto. Ein Hive-Research-Kind bleibt immer unfinanziert. Genealogy und ein Versions-/VisualDNA-Verweis werden gespeichert. Die Nachwuchspflanze handelt **niemals rückwirkend am Tag ihrer Geburt**.
5. **Lifecycle**: ECO-09 greift ausschließlich Pflanzen unter 70%-Vitalität an, mit Rinden-Evidenz, Probezeit und Kaktus-Dormanzschutz. Ein Fraßereignis lautet zunächst `closing_requested`, gibt **0 Budget** frei und erzeugt **keine** Flatten-Order. ECO-09 archiviert nur, wenn externe Reconciliation bereits am **Vortag** im Zustand `closing` war und `positions_flat`, `orders_cancelled`, `fills_reconciled`, `free_margin_confirmed` sowie `used_margin=reserved_margin=0` nachweist. Archivierte Genome/Trades bleiben unterirdisch als Daten erhalten, ohne sichtbare Pilze. Kapitalanspruch wird erst danach verfügbar, aber NAV steigt nie dadurch.
6. **Snapshot/Replay**: Der ECO-03-Clock-Checkpoint enthält die vollständige dynamische Population, deren Gene, Kapital, Wasser, Metrics-Historie, Hive und Archive. Gleiche Eingaben, Datum, Seeds und Config müssen zu identischen Bytes führen, auch nach Neustart. User-Interface und VisualDNA sind **nicht** Teil von Profitberechnung oder Triggerlogik.

## Nutzung und Testlauf

Auf Branch `codex/ecosystem-evolution-cycle` mit Python 3.11 und `PYTHONPATH=src`:

```bash
python -m unittest discover -s tests/ecosystem -p test_evolution_cycle.py -v
```

Die Tests simulieren 30 Tage mit gemeldeten synthetischen Ergebnis-/Chancen-Ereignissen, mehr als eine Generation, Kapitalbegrenzung und Parent-Budget-Umbuchung; prüfen Restart/Replay, Mutationen auf realen Genachsen, OOS-neutralen Hive, Rinde und Dormanz, keine freie Entitlement-Freigabe vor externem confirmed-flat, Safety-Hardcaps und VisualDNA-Neutralität. Ein fehlendes oder falsches Konto-, Wetter-, Seed-, Trade- oder Reconciliation-Ereignis schlägt fehl.

## Grenzen / offene Arbeit

- Der initiale 3×20-Fall und alle Nachkommen bleiben in **einem** synthetischen SOL-/IMBA-Kachelraum und einem virtuellen USDT-Paper-Account. Die Multi-Asset-Tile-Welt #41 und die echte SOL-1m-Ausführung #37 sind **noch nicht** angebunden. Auch echte OOS-Evidenz, Mehrfachtestkorrektur und Handelstageskalender fehlen; die Demo-Reifegrenze dient nur reproduzierbaren Physik-/Accounting-Tests.
- ECO-07 rechnet mit cashflow-korrigierten Tagesrenditen aus der Eventquelle; bislang keine standardisierte, echte Markt-Price-/Fill-to-TWR-Bridge. Es werden daher **keine** profitablen Ergebnisse über die Strategie oder die Evolution behauptet.
- Die zugekoppelten Prerequisite-Source-Dateien aus #48 (Metrics), #50 (Bees), #49 (Predator) liegen als **temporäre, identische Entwicklungs-Abhängigkeiten** im PR-Branch, damit CI testen kann. Vor finalem Merge getrennte PRs reviewen, rebasen und diese Kopien deduplizieren. Gleiches gilt für den Stack auf #47.
- `visual_dna_ref` ist bewusst eine rein optische, stabile externe Referenz; vollständiges vererbbares VisDNA aus #45 muss später über validierte ECO-01 Snapshot-Contracts übertragen werden, statt zwei konkurrierende unabhängige Genmodelle entstehen zu lassen.
- `closing` ist keine echte Exchange-Handelsaktion. Die reale Positionsabwicklung und Venue-spezifische Risiko-/Stop-Synchronisierung gehören ausschließlich in einen **separaten** geprüften Live-Executor und sind hier nicht freigegeben.
- Nächster wissenschaftlicher Schritt: Walk-forward/OOS aus realer unabhängiger SOL-Preisreihe, mit Gebühren/Slippage/echten Signalmodulen; Entwicklung von Fortpflanzung/Selektion experimentell gegen reine zufällige Genmutation vergleichen.
