# ECO-06 — Tägliche Pflanzenphysiologie (nur Simulation)

`quant.ecosystem.ecology.physiology` berechnet einen **reinen** Tagestick: `ecology_day(previous_water, utc_day, signed_global_NAV, plants, observed_weather, config)`. Kein Handels-Executor, keine Geldbewegungen. Alle Konten sind vorher in **eine reconciled NAV** überführt; sie wird nicht durch die Ökologie selbst zusammenaddiert.

## Vererbbare funktionale Gene und Kaktus-Regel

`root_growth` (Wachstumsausgabe), `root_depth` (vertikale Richtung), `root_spread` (laterale Richtung) können beide gleichzeitig stark sein. Damit wächst mehr Wurzelmasse auf Kosten unmittelbar möglicher Kapitalauslastung, ohne Equity zu zerstören. `branching` und `leaf_count` (ganze Zahl **mindestens 1**) bilden mit `leaf_size` die Photosynthesefläche. Mehr Verzweigung und Blätter => mehr effektive Blattfläche, Sonnenaufnahme und mögliche Exposure, aber **immer** unter globalen und gen-/portfolioseitigen Hard-Caps.

**Kaktus:** Genau bei `leaf_count == 1` und `photosynthetic_rind == true` kann der **grüne Spross / die grüne Stammoberfläche** als funktionales Blatt wirken; `rind_efficiency` steuert seine Sonnenaufnahme. `bark_strength` (Fraß-/Schutzwirkung) ist ein *separater* Faktor: eine dicke Rinde darf nicht automatisch enorme Photosynthese oder ungeprüfte Leverage erzeugen. Die phänotypische Darstellung (Kaktus ohne sichtbare Blätter, Rippen/Dornen) wird vom Visual-Genom zusätzlich bestimmt. Blattuntergrenze 1 kann niemals durch Mutation/Pruning unterschritten werden.

`dormancy` reduziert den eigenen Kapitalbedarf; `reproduction_budget` und `protective_stop` sind unabschaltbare finanzielle/Sicherheits-Policies. Die Ausführung darf diese von Mutation nicht verschwinden lassen.

## Kapital/Sonne/Wasser/Zeitskala

- 90% des **globalen NAV** sind höchstens bereits zugeteilte `entitlements`; die 10% Reserve sind kein zweiter Pool. Biomasse, Wurzeln, Blätter und „Wachstumsenergie“ sind **physiologische Spielgrößen**, kein echter Abzug vom Konto.
- `intended_margin` und `intended_notional` sind **unverbindliche** Simulationsvorschläge. ECO-05 muss später zusätzlich freie Margin, realisierte Positionen, Gebühren, Minimum-Lots und Kontoübertragungen prüfen. Die Summe von `entitlement` wird gegen 90%-NAV geprüft; keine profitable Pflanze erhält dadurch automatisch zusätzliches Geld.
- `sun` ist ein 0..1-Wert mit kausalen `observed_at` und `known_at`. Der tägliche Zustand darf nur Daten bis UTC Tagesende verwenden; später ggf. Signalzeitfenster innerhalb des Tages statt Tagesendwert für intraday Leverage.
- Wasserziel `d_target(E)=d_min+(d_max-d_min) exp(-k max(E,0)/E_ref)`. `d_min>0` verhindert Kontakt zur Erdoberfläche. Der tatsächliche Pegel nähert sich dem Ziel täglich mit getrennten `rise_speed/fall_speed`-Alphas. `hysteresis_depth` schützt gegen häufige Grenzübertritte; ein tiefer Wurzelbaum braucht zusätzlich `bonus_qualified` (aus ECO-07 Fitness/Evidenz). Optionales `max_water_access_fraction` begrenzt die Empfänger, erzwingt aber niemals das Auffüllen einer Mindestpopulation.

## Grenzen, ausdrücklich kein Produktivbetrieb

Diese Formeln sind **Hypothesen für Experimente** und keine nachgewiesene risikoadjustierte Handelslogik. Das Sonnenlicht beeinflusst im aktuellen Prototyp sowohl Auslastung als auch Leverage; diese doppelte Hebelwirkung sollte als Ablation im Experimentrahmen ECO-12 überprüft und ggf. ausgeschaltet werden. Wetter auf Tagesende für Intraday-Exposure nicht vorzeitig verwenden. Eine Game-Leverage-Intent führt nie selbst Venue-Orders aus. Die Genfamilie ist für funktionales Wachstum vorgesehen, ein **separates rein optisches Genom** beeinflusst lediglich Rendering/Variation und niemals Kapitallimits, Portfolio-Risiko oder Alpha-Fitness.

## Tests

`python -m unittest discover -s tests/ecosystem -p test_ecology.py -v` prüft: 90/10, keine Doppelallokation, Wassertiefe asymptotisch stets positiv, Auf-/Abstieg, Hysterese, echte Verzweigungs-/Blattflächeneffekte, einblättrigen Kaktus mit grüner Rinde, Trennung von Bark und Photosynthese, Risikocaps, kausales Wetter, geschützte Sicherheit, hohe gleichzeitige vertikale/laterale Root-Gene, schwache Pflanzen und Dormanz. ECO-01/ECO-03-Adapters müssen später bei der Integration diese Tageswerte in `SimulationSnapshot` übernehmen.
