# Trading-Ecosystem / Evolution Game — Umfassender Hand-off und Multi-Agent-Plan

**Stand:** 2026-11-01 · **Status:** Produktkonzept + Architektur-Backlog, *kein* produktives Trading-Spiel · **Sprache:** Deutsch.
**Repository:** [MartinLiteAndFog/quant](https://github.com/MartinLiteAndFog/quant) · **Integrationsbranch:** `feat/family-hedge-ledger-20261101` · **Draft-PR:** [#22](https://github.com/MartinLiteAndFog/quant/pull/22) · **Master-Tracker:** [#36](https://github.com/MartinLiteAndFog/quant/issues/36).
**Wichtig:** Die Gespräche beschreiben viele **neue Regeln, nicht vorhandene Implementierungen**. Aktuell gibt es eine **synthetische Demo**. Nichts an diesen Konzepten autorisiert Live-Orders, Änderungen an Railway oder KuCoin, automatisch erstellte TradingView-Alarme oder Transfers.

## 0. Die Idee in einem Absatz

Wir bauen ein interaktives, beschleunigbares Evolutionsspiel im Stil von *The Bibites*, das über einem historischen (angestrebt: fünf Jahre SOL-Minuten) Markt läuft. **Oberirdische Pflanzen sind finanzierte Tradingstrategien**, deren individuelle Genome Strategiebausteine, Risikoverhalten und Physiologie definieren. **Unterirdisch** forschen sehr viele zusätzliche Genome in isoliertem Paper Trading; deren Daten werden aggregiert, nicht als Millionen Pilze gerendert. **Bienen** kombinieren und verbreiten Gene lokal oder über einen zentralen Bienenstock; **Fresser** entfernen schwache, angreifbare Pflanzen; **Wurzeln** steuern die Kapitalzugangspriorität; **Blätter** Kapitalnutzung und Hebel; **Sonne** das globale Risikoregime; **Rinde** begründet längere Schonung seltener/zyklischer Strategien. **Grundwasser ist die Equity sämtlicher zugeordneter Accounts**, nicht ein zweiter Geldtopf. Tägliche ökologische Aktualisierung, Minuten-/signalbasierte Tradingausführung, statistisch prüfbare Evolution, optional skalierte Parallel-Rechnungen und später eine schöne 3D-Welt. Das Ziel ist Forschung an robusten Strategien, **kein Versprechen selbststabilisierender Profitabilität**.

## 1. Reale Basis, die bereits im Repo existiert (geprüft 2026-11-01)

- [`apps/fleet-desktop/src/components/EvolutionMeadow.tsx`](../../apps/fleet-desktop/src/components/EvolutionMeadow.tsx): Fleet Cockpit (React/Tauri), rotierbare orthografische SVG-Simulationswiese, Auswahl/Filter, Bienen-Modi, Gen-Eingabe; derzeit keine echte 3D-Engine, kein realer Performancefeed.
- [`apps/fleet-desktop/src/lib/evolutionMeadow.ts`](../../apps/fleet-desktop/src/lib/evolutionMeadow.ts): deterministische synthetische Strategie-/Profitlandschaft, drei Roots + neun Töchter, Mutation, Fitnesscluster, visualisierte Biomasse. X = IMBA sensitivity, Y = TTP %, Z / Stängel bisher Nettorendite/visuelle Höhe; andere Parameter sind im Genom verborgen. Browser-Wächter für ca. 450 sichtbare Pflanzen, keine Unterwelt-Millionengrenze.
- [`apps/fleet-desktop/src/lib/geneLibrary.ts`](../../apps/fleet-desktop/src/lib/geneLibrary.ts): numerische und Bool-Gene; gespeicherte Bibliothek in Browser-`localStorage`; 5 bestehende Bienenarten `mixer`, `mosaic`, `explorer`, `collector`, `pruner`; 5/10%-Abwurf optionaler Gene. **Achtung:** `stop_loss_pct` ist derzeit *optional*, finanzielle/Sicherheits-Policies fehlen als geschützter Typ.
- [`src/quant/execution/family_mutation_meadow.py`](../../src/quant/execution/family_mutation_meadow.py): rein deterministische 5–15 Kandidaten/Samen-Vorschläge für 2 Parameter; keine Orders, kein neues Pine-Signal.
- [`src/quant/execution/family_monthly_evaluation.py`](../../src/quant/execution/family_monthly_evaluation.py): monatliche, vorläufige Scores und virtuelle Budgetvorschläge. **Noch nicht** tagesbasierte Ökologie oder multi-horizon Fitness.
- [`src/quant/execution/family_ingress.py`](../../src/quant/execution/family_ingress.py): opt-in Shadow-Router, ein `/webhook/family` und **ein gemeinsamer `FAMILY_WEBHOOK_TOKEN`**, Identifikation pro `strategy_id`/`event_id`. Keine Live-Ausführung.
- [`src/quant/execution/family_durable_ledger.py`](../../src/quant/execution/family_durable_ledger.py), [`family_order_planner.py`](../../src/quant/execution/family_order_planner.py), [`family_hedge_preflight.py`](../../src/quant/execution/family_hedge_preflight.py): bestätigte virtuelle Fills, reine Orderplanung und lesender Hedge-Mode-Preflight als Vorarbeiten; **kein** produktiver multi-bot-sicherer Single-Writer.
- [`docs/plans/2026-11-01-evolution-meadow.md`](2026-11-01-evolution-meadow.md): früheres Design und die bestehenden Demo-Regeln. **Dieses neue Dokument überschreibt nur das konzeptionelle Zielbild, nicht automatisch den funktionierenden Code.**
- [`apps/fleet-desktop/package.json`](../../apps/fleet-desktop/package.json): `npm run build` = TypeScript-Check + Vite-Build; GitHub Actions für Evolution Meadow, Family-Ledger und PostgreSQL-Integration existieren in PR #22.

**Bewusst abweichender Altbestand:** Biomasse bislang pro Monat max +36% / −25%; neue Vorgabe ist **täglicher** Ecologyschritt. Alte Root-Finanzierung: 3 Familien teilen 90% Equity, 10% Reserve, initiale Töchter erhalten 2% des *Familienbudgets*, nicht 2% der Gesamtequity. Diese Pilotregel **bis zu einem ausdrücklich freigegebenen Cutover beibehalten**. Gen-Bibliothek führt Schemas, keine ausführbaren beliebigen Strategiebausteine.

## 2. Fachliche Grundregeln und explizit revidierte Ideen

### Fest vereinbart / Zielvorgaben

1. Täglicher Spieltick für Biomasse, Wurzeln, Blätter, Rinde/Vitalität, Geburt, Jagd und Grundwasserschwankung. Trades laufen auf ihrer echten Minuten-/Signalzeit, nicht nur einmal am Tag.
2. Pflanzen haben komplette Genome mit vielen sichtbaren und verborgenen Genen. Die 2 Bodenachsen können später beliebige Gene zeigen; fehlende Werte dürfen Pflanzen nicht heimlich unsichtbar machen.
3. Pflanzen entstehen als *klein kapitalisierte Samen/Keimlinge* (in der Simulation) und wachsen beziehungsweise schrumpfen langsam. Sie verschwinden nicht nach ein bis zwei Verlustmonaten.
4. Nachkomme entnimmt sein Startbudget von der Mutter-/Familien-Zuteilung (Pilot zurzeit 2% des Familienbudgets; im Evolutionsspiel auch bounded Reproduktionsquote z.B. 8%, 10%, 15%). Keine Geldschöpfung.
5. Weggefressene Pflanzen verschwinden grafisch. Ihre Historie/Genome bleiben im **unsichtbaren Research-Archiv** und dürfen später erneut papergetestet werden. **Keine sichtbaren Pilze**, kein zweites virtuelles Leben als dauerhafte Individualfigur.
6. Bienen können lokal bestäuben, zentral im Hive Genome sammeln und neu kombinieren oder beides; fünf bestehende Bienenarten sind ein Startpunkt, kein hartes Naturgesetz. Mittelung, Gen-Neuzugang und zufälliges Abwerfen optionaler Gene erlaubt.
7. Fresser haben eine einstellbare obere Jagd-Vitalitätsgrenze (Start: unter 70%; 70–80% Schonzone; über 80% geschützt). Gewichtet-zufällige, schwächsten- und altersorientierte Fresser möglich. Bark/Alter/Evidenz/seltene Opportunitäten modulieren Jagdbarkeit; harte Risikogrenzen nie.
8. Genetische Struktur darf komplex sein. **Keine obligatorische Komplexitätsstrafe**; optional später Slider, Default exakt 0.
9. Sicherheits- und finanzrelevante Gene/Policies (Risk Caps, verpflichtender Schutz-Stop, Budgetkonsistenz, Reproduktion) dürfen **nie entfernt oder deaktiviert** werden; Werte allenfalls innerhalb autorisierter Grenzen mutieren.
10. Wurzelwachstum seitlich und nach unten genetisch getrennt; hoher Wert in beiden Richtungen möglich, aber mit Wachstums-/Kapitalnutzungs-Trade-off und zeitlichem Aufbau.
11. Grundwasser repräsentiert globales Equity/NAV, **kein zusätzliches Kapital**. Maximal 90% als verteilbare Budgets, 10% Reserve. Der Spiegel nähert sich mit wachsender Equity asymptotisch der Erdoberfläche und erreicht sie niemals; tägliche Glättung/Hysterese einstellbar.
12. Blätter/Blattfläche steuern persönliche Kapitalauslastung und persönliche Leverage (unter Limits), Sonne den globalen Risiko-/Leverage-Multiplikator, Verschattung später Portfolio-Korrelation/Signalüberlappung.
13. Fitness, gegenwärtige Vitalität und statistische Evidenz sind *drei getrennte Zustände*. Der ehemalige Champion soll Kapital verlieren, wenn sein Edge verschwindet; Kaktus/Fishing-Stratege darf bei wirklich ausbleibender Gelegenheit lange ruhen.
14. Die Evolution soll nicht nur bessere Strategien, sondern bessere Suchverfahren (Bienen/Genkombination/Selektoren) entwickeln. Nur OOS-Evidenz darf erfolgreichen „Honig“ erzeugen.
15. Nicht automatisch Produktions-Railway/KuCoin/TradingView verändern. Ein eigenes Shadow-/Paper-Spiel ist erstes Ziel.

### Ausdrücklich verworfen beziehungsweise ersetzt

- **Individuell dargestellte Pilze für verstorbene Pflanzen**: verworfen, unüberschaubare visuelle Populationszahl. „Myzel“ allenfalls als aggregierte Forschungsdatenbank/Netzwerkmetapher.
- **Separater Komposthaufen/Extra-Kapitalpool**: ersetzt durch **Equity = Grundwasser**; abgestorbene Pflanzen setzen nach sicherem Positionsabschluss *bereits vorhandenes* Kapital frei, nicht neues.
- **Zweiter 75/25-Wasserpool zusätzlich zur 90/10-Regel**: verworfen durch die einfachere neue Analogie: 90% gesamte Equity zuteilbar, 10% Reserve. 90% nicht automatisch in offenen Trades.
- **Wurzelwachstum als echter Kapitalverbrauch/Kosten**: nur Opportunitätskosten / vorübergehend geringere Tradingkapitalnutzung, kein ausgebuchtes Equity.
- **Zwangsläufiges Aussterben nach 1–2 schlechten Monaten**: verworfen; Fresser, Biomasse, Evidenz, Dormanz und Rinde regulieren.
- **Pflanzenklasse als fest verdrahteter Handelsalgorithmus**: vermeiden. Gras/Blume/Busch/Kaktus/Orchidee als Phänotyp aus Genom, Komplexität, Zeit, Wurzel-/Blattverhältnis.

### Noch bewusst experimentelle, NICHT endgültige Zahlen

- Jagdlimit 70% Vitalität; Schutzband 70–80%; Schonfrist; Wirksamkeit von Rinde.
- Start-Mutationswahrscheinlichkeit optionaler Gene 5–10%; 5–15 Samen beim bisherigen Meadow-Vorschlag.
- Reproduktionsbudget der neuen simulierten Gene 8/10/15%; erste Pilot-Töchter **fix 2% Familienbudget**.
- 90/10 Grundwasser-Equity. Ursprung ist die alte Sicherheitsreserve, nun zentrale Bilanzregel.
- Pegel-Anstieg/Fall, Hysterese, 2 bis 80% Zugangsanteil, Sonne-/Blatt-Hebelkurven, Fitnessgewichte, Biomasse-Grenzen.
- „Monatliche Ernte/Blütenfall“ als visuelles Profit-Ereignis, obwohl darunter täglicher Tick.
- Ohne Messung keine konkrete Millionenstrategien/Minute-/FPS-Leistung versprechen.

## 3. Zwei Ebenen — Oberwelt und Unterwelt

**Oberwelt (sichtbar):** 3 Anfangseltern + 9 manuell konfigurierte Töchter als heutige Fleet-Pilotgruppe; später unterschiedlich viele echte oder simulierte finanzierte Pflanzen. Jede Pflanze hat stabile ID, Stammbaum, Genome-Version, Strategie-Code-Version, Budgetberechtigung, PnL/Fees/Funding, Biomasse, Blattfläche, Wurzeltiefe, Wurzelbreite, Rinde, Lebensphase, Vitalität und Fitnessevidenz. Oberwelt zeigt nur lebende beziehungsweise nicht archivierte Pflanzen. Kapitalisierte Pflanzen benötigen sinnvolle Mindestordergröße; ein zu kleiner Keimling darf als virtuelles Budget existieren, aber keine zu kleine Venue-Order erzwingen.

**Unterwelt (nicht pro Genome als Pilz gerendert):** Millionen mögliche Genome/Seeds, Genbibliothek, Paper-Trades, komplette historische Versuchsergebnisse, Cache/Kompression, Archiv der gefressenen Pflanzen, Hive-Wissenspool, Validierungs-Pipeline. UI als Schnitt durch Boden mit Wurzelgeflecht, Cluster/Heatmap, Rankings, Datenbank-/Forschungslayer; Kandidaten nur bei Auswahl einzeln öffnen.

**Grenzübergang:** `proposed -> screened -> backtested -> out_of_sample_verified -> paper_forward -> eligible_for_sowing -> capitalized -> dormant -> archived`. Die letzte Stufe kann erneut zu `proposed`/Test werden; ein einzelner guter Monat reicht nicht automatisch. Hintergrundleistungen niemals durch künftig erst bekannte Daten belohnen.

## 4. Kapitalmodell — exakt ein Bilanzkreis

### Definitionsgleichungen

- `E_t = sum(unique account NAV / cashflow-corrected equity)`: eindeutige, nach Währung und Zeit reconciled Equity. Ohne Cross-Account-FX/Transferdaten keine „einfach zusammengerechneten“ Phantomwerte.
- `R_t = 0.10 * E_t`: nicht handelbare Reserve im Design (bei negativem NAV fail closed).
- `D_t = 0.90 * E_t`: höchstens budgetierbarer Betrag.
- `sum_i entitlement_i(t) <= D_t`: Summe aller Budgetansprüche inkl. Eltern, Kinder und Bonus. **Keine zweite Zuteilung nur deshalb, weil ein Tiefwurzler Grundwasser berührt.**
- `used_margin_i <= realizable_margin_i <= entitlement_i`; freie verfügbare Margin, offene Positionen und geplante Orders dürfen nur **einmal** berücksichtigt werden.
- `notional_exposure_i = margin_used_i * effective_leverage_i`, unter globalen Limits und exchange-spezifischen Minima.
- Neue Samen erhalten Budget **durch Umbuchung** aus Mutter-/Familien-Entitlements; Geburt erhöht nicht die Equity.
- Fresser dürfen Budget erst nach bestätigt geschlossenem Positionszustand/Orders-Reconciliation neu freigeben. Keine automatische Liquidation allein wegen neuer Pegel-/Gen-Entscheidung.

**Beispiel:** `E=100 USDT`, `R=10`, `D=90`. Drei Roots könnten im heutigen Pilot je `30` Familienbudget führen; Tochter mit 2% davon erhält `0.6 USDT` internes Budget (ggf. zu wenig für eine reale Order). 20 evolutionäre Kinder in einer Simulation teilen sich Entitlements innerhalb derselben `90` statt zusätzlich zu existieren. Ein Kaktus mit Budgetberechtigung 20 USDT nutzt vielleicht nur 2 USDT Margin; die verbleibenden 18 sind **nicht automatisch separat frei**, solange sie als exklusive Entitlement reserviert sind. Der Kapital-Allocator muss die Rechte für neue Orders und die tatsächliche Margin sauber unterscheiden und darf nicht gleichzeitig denselben Betrag mehreren garantiert zusagen.

**Vorherige Frage „wie vermeiden wir totes Kapital?“**: Zuteilung kann täglich *unbenutzte künftige Orders-Budgets* an geeignete Pflanzen umpriorisieren, ohne offene Positionen zu schließen. Die 90% sind ein **Budgetlimit**, keine Pflicht, jederzeit 90% Exposure zu haben. Wenn keine sicheren Handelsgelegenheiten bestehen oder Min-Lots nicht erreicht werden, bleibt Liquidität liegen. Höhere Nutzungsquote darf kein Risikoerzwingen sein.

### Grundwasserstand

Vorschlag, **nur beispielhafte Parametrisierung**:
`d_target(E) = d_min + (d_max-d_min) * exp(-k * max(E,0)/E_ref)`, mit `d_min > 0` und `d_max > d_min`. `d` ist die Tiefe *unter* Erdoberfläche: steigendes NAV -> kleineres `d` -> Wasser steigt, aber nie bis zum Boden. `k, E_ref, d_min, d_max` per Config.
`d_t = d_(t-1) + alpha_up/down * (d_target - d_(t-1))`; `alpha_up`/ `alpha_down` täglich separat, plus Hysterese in Zugangsstatus (kleine Pegelwechsel führen nicht zu täglichen Flip-Flops). Alle Rechenwerte sind eigene Spielkoordinaten, nicht wirkliche Meter.
**Wichtig:** Pegelhöhe ist nur Zugangs-/Prioritätsregel; wirkliche `D_t`-Budgetmenge ist separat bilanziert. Hoher Pegel kann nie die 10%-Reserve überschreiben. „Mindestens 2 / höchstens 80% Pflanzen erreichen Wasser“ als optionales, zu testendes Regelwerk, *keine* automatische Kapitalberechtigung statistisch unzureichender Pflanzen.

### Wurzelaufnahme und Wurzelkosten

- `root_growth`, `root_depth`, `root_spread` eigenständige, versionierte Gene; tiefe **und** breite Wurzeln möglich.
- Effektive physische Wurzelgröße entsteht nur schrittweise aus Biomasse, Zeit, Investitions-/Wachstumsenergie und Genen; ein neuer Keimling ist nicht sofort Tiefwurzler trotz starken Genwerts.
- **Opportunitätskosten**: hohe Wachstumsinvestition senkt die für *eigene kurzfristige Trades* aktiv nutzbare Kapitalquote; Eigenkapital wird dadurch nicht vernichtet, sondern der Allokator darf unbenutzte Kapazität verteilt halten. Laterale Wurzeln konkurrieren um durch Fraß **freigegebene** lokale Ansprüche; tiefe Wurzeln erhalten Zugang/Priorität zum allgemeinen Grundwasser-Budget.
- Pflanzen, die nahe beieinander liegen, teilen lokale freigewordene Mittel proportional zu `root_weight_i / sum(root_weight)`; Kausalität: Koordinaten sind genetischer Parameterraum, Nähe ist eine **Modellentscheidung**, nicht reale Marktgeografie.
- Auszahlungen/Budgets unterliegen Fitness-/Evidenz-Gates und Sicherheitslimits. Gene sind kein Recht, Geld zu erzwingen.

## 5. Photosynthese = Exposure / Leverage

- `sun_t`: globaler, ausschließlich aus zum Zeitpunkt t **bekannten** Marktdaten abgeleiteter Risiko-Multiplikator (Volatilität, beobachtbares Regime, Stress, Korrelationen), oder reiner Game-Parameter für Experimente.
- `leaf_area_i`: genetisch beeinflusste und mit Biomasse gewachsene Blattfläche; steuert Kapitaleinsatz/Nutzungsquote `U_i in [0,1]` und gegebenenfalls persönlichen Leverage-Modifier.
- `L_i = clip(L_global(t) * leaf_modifier_i, allowed_risk_bound_i, venue_limit)`; kein Blattgen darf `L_global` für alle anderen Bots selbst erhöhen.
- `Exposure_i = actually_allocated_capital_i * utilization_i * leverage_i`; eventuelle doppelte Wirkung der Blattfläche auf U und L ist **expliziter Versuchsparameter**, nicht unbemerkt doppelte Hebelung.
- `shade_i` aus Strategie-Signalüberlappung / realer zeitabhängiger Renditekorrelation (nicht nur Bildschirmnähe); optional später selektive Reduktion zusammenfallender Exposures.
- Kaktus = wenige Blätter, tiefe Wurzeln, geringe gewöhnliche Margin-Nutzung, kapitalsparender Schlaf, seltenes großes Signal; Gras = hohe Blattnutzung, schnelle Bewertung, flach; großer Baum = langfristig beide Systeme gewachsen.
- **Globale Hard-Caps** (max Position/Leverage/Margin/Drawdown, venue constraints) sind nicht genetisch ausschaltbar.

## 6. Genbibliothek / Vererbung / Varianten

**Genarten:** (a) `immutable_safety_policy` ohne Abschaltung; (b) `required_bounded_financial` z.B. reproduction_budget oder allokations-/Risikopolicy, immer vorhanden und nur kontrolliert mutierbar; (c) `optional_strategy_gene` z.B. RSI-Perioden, Sensitivität, TTP, zusätzliche Filter, exit-bezogene Tuningparameter. Pflichtgene fehlen nie. **`stop_loss_pct` in alter Demo ist zu bereinigen**: Sicherheits-Stop-Pflicht von optionalem Stop-Tuning trennen. Komplexitäts-Malus Default = 0; optional Slider.

**Strategiegenom** ist Mischung aus Parameterwerten und versionierten, ausführbaren oder interpretierbaren **Strategiemodulen**. User kann neue Module hinzufügen, aber beliebiger Python-/Pine-Quelltext darf nicht unkontrolliert in Executor oder App ausgeführt werden. Kennung, Version, Inputs, erlaubter Datenzugriff, Risikovertrag und Tests pro Modul. Bestehende TradingView-Signale sind zunächst Ground Truth für Live-Pilot; Parität einer eigenen Strategie-Engine separat testen.

**Bienen heute vorhanden:** Mixer (numerischer Mittelwert), Mosaic (pro Gen ein Elternteil), Explorer (größere Mutation), Collector (fehlende Gene aufnehmen), Pruner (5/10% optionale Gene löschen). Neue Forschung vergleicht lokale Pollination (Samen an Pflanzenstandort), zentralen Hive (Wissen sammeln, neue Kandidaten im Stock), hybrid (beides) und Random-only ohne Bienen als Kontrollgruppe. Eine Biene bekommt eigene mutierbare Suchparameter (keine unendliche Typentabelle nötig) und Honig als validierten OOS-Nutzen ihrer späteren Kinder. Stammbaum Mutter/Vater/Biene/Genom-/Codeversion festhalten; kein Loops-Replay ohne Seed.

## 7. Vitalität, Fitness, Glaubwürdigkeit, Rinde und Fresser

Es braucht **drei getrennte Skalen**:
- **Langfristige Fitness**: netto und risikobereinigt (Profit/TWR, Sharpe, Drawdown, Tail Loss, Kapitaleffizienz), über mehrere Marktregime, mit OOS-Evidenz.
- **Aktuelle Vitalität**: EWMA der jüngeren Gelegenheit-abhängigen Performance, aktueller Drawdown, Wasser-/Tradingaktivität, Erholung seit Peak, gegen eigene Historie gemessener Edge-Drift, Market-Regime-Passung. **Kein Trade ist nicht automatisch schlecht**.
- **Credibility/Uncertainty**: effektive Anzahl *unabhängiger* Trades, Messdauer, Datenqualität, Autokorrelation, Kostenklarheit, Konfidenz/Fehlerbänder; nicht eine willkürliche positive Rendite. Viele Backtests implizieren Multiple-Comparison-Bias.

Weitere separat auswertbare Größen: Ulcer Index / Time Under Water, CVaR/Expected Shortfall, vol-regime conditional return, trade opportunity utilization, turnover, correlation & marginal portfolio utility. Eher nicht alles linear in einen 0–100-Score mischen; intern Komponenten normalisiert und Scoreformel versioniert, Gewichte als Experiment.

**Rinde `bark_strength`**: genetischer Modifikator der Jagdgefährdung und des Evaluationshorizonts, wirksam nur in Relation zu glaubwürdig belegter langfristiger Robustheit. **Kann nie** globale Drawdown-/Loss-Stops umgehen. **Dormanz** schützt vor falsch gemessener Inaktivität: gute Signale fehlten -> okay; passende Signale gab es und Strategie versagte -> Vitalitätsverlust.

**Fresser:** schwächstenorientiert, probabilistisch-zufällig **nur unter einer Jagdschwelle** und alters-/erholungsorientiert. Ausgangspunkt: >80% Vitalität sicher geschützt, 70–80 Schonzone, <70% grundsätzlich Kandidat; Mindestalter/-Evidenz, konfigurierbare Hysterese und Bark-Schutz. Fressen heißt nicht „alte Position sofort killen“: Lebenszyklus/Zuteilungsstop, sichere Bestandsbehandlung, bestätigte freie Budgets, dann Visual entfernen. Archiv enthält Gene und Berichte, nicht Millionen sichtbare Pilze. Auch ungewöhnliche hochprofitable, selten handelnde Büsche/Kakteen müssen durch lange Horizonte angemessen bewertet werden.

## 8. Wachstum, Aussehen, Visualisierung

Biomasse täglich aus realisiertem/unrealisiertem, korrekt netto markiertem PnL, geglätteter Vitalität und Growth-Allocation; nicht jeden Tag +36% von Monatsdemo übernehmen. Negative Monate -> langsames Schrumpfen, niemals direkt verschwinden. **Logarithmische visuelle Skalierung** verhindert, dass ein seltener Superstar die Szene sprengt. Nach monatlicher Gewinnmitnahme/Ernte kann eine Blüte sichtbar fallen; realisierte Gewinne fließen nur einmal in die Bilanz, nicht als zweites Cash.

**Phänotyp:** 2 aktive Strategieparameter: Gras; mehr Gene: Blüten-/Busch-/Orchideen-Anmutung; tiefe Wurzeln + wenig Blätter + Rinde: Kaktus; komplexe alte Pflanze: Baum. Kein festes Gesetz: Genotyp und Biomasse/Alter/Wurzeln bestimmen Darstellung; Komplexität selbst zunächst **ohne Fitnessstrafe**. Farben: Familie und Vitalität; Größe: Biomasse, Flower/Blossom: Fitness/Verlässlichkeit, Höhe kann Profit zeigen; getrennte Kanäle nötig, damit „hoher Profit“ nicht 3× sichtbar kodiert wird.

3D-Grundfläche mit dynamisch wählbaren Genachsen, Profit-/Fitness-Höhenskalierung, korrekte kategoriale/fehlende Werte. Bei Achsenwechsel **nicht** alle Pflanzen ohne betreffendes Gen ausblenden; Filter/Fächer/Bucket. Zeig Wurzel-Schnitt, Pegel, Schatten, Bienenflüge und Fresser. LOD/Instancing und Cluster-Berechnung: 1000 lebende Pflanzen darstellbar, Millionen Forschungskandidaten nur aggregiert. Grafik 30/60 FPS ist eigene Aufgabe, nicht gekoppelt an millionenfache Paper-Mikroticks.

## 9. Zeit, Marktdaten, Statistik und Compute

**Kausalität:** Spiel-Evolution täglich, Trades realistisch mit Candle-/Tick-/Signal-Timing und kein Future-Leak. Fünf Kalenderjahre SOL 1m ≈ 2.63 Mio Minuten-Bars *bei ununterbrochener Verfügbarkeit*; genaue Quelle, Handelssymbol, Funding, Kontraktkonvention, fehlende Bars, Delistings, Zeitzone sind offen und müssen dokumentiert werden. Nicht so tun, als sei ein angefragter 5-Jahres-Datensatz bereits vorhanden.

**Pipeline:** historische Preisquelle -> QA/normalisierte Bars -> validierte Strategie-Ausführung -> Ausführungsmodell (Zeitpunkt, reale Marktpreise, Slippage, Spread, Gebühren/Funding) -> täglicher NAV+Opportunity-Snapshot -> Ökologie/Selektion/Reproduktion -> Event-Log -> UI/Analyse.

**Skalierung:** Millionen *vorgeschlagene* Genome sind einfacher als Millionen vollständig auf 2.63 Mio Bars backgetestete Varianten. Mehrstufiger Filter, deduplizierte Konfigurationen, Caches, Batch/Parallel CPU, deterministisches Seed, echte RAM/Throughput-Benchmarks auf MacBook Pro. Rust optional bei Messvorteil; nicht als unbegründeter Rebuild-Zwang. Keine Prognosen zur Hardwareleistung ohne Mac-/RAM-Angaben und Benchmark.

**Experimente:** vier identisch finanzierte/zeitgleich getestete Welten: A random-only, B lokal, C Hive, D hybrid. Über viele Seeds replizieren; gleiche Compute-/Marktdaten-Budgets; purged/walk-forward und finales echtes OOS. Score-Ablation für Wurzel-/Blatt-/Wetter-/Jäger-/Bark-Mechanismen und Komplexitätskosten-Slider (default null). Statistik/Mehrfachtests dokumentieren; eine attraktive Simulationswiese ist kein Beweis von Alpha.

## 10. Sicherheits- und Produktionsgrenze

- Dieses Projekt ist **nicht** die Freigabe, die existierende 3×3 Blaupunkt-Familie live umzubauen. Produktive Legacy-Route bleibt `/webhook/tv-execute` für `blaupunkt`; andere 11 Pilot-Profile brauchen manuell konfigurierte TradingView Alerts für den gemeinsamen zukünftigen `/webhook/family`; Familienrouter derzeit Shadow.
- Initiale IDs im [Pilot roster](../../src/quant/execution/family_pilot_roster.py): `blaupunkt` + `d01..d03`, `blaupunkt-p02` + Töchter, `blaupunkt-p03` + Töchter (12 gesamt). Shared webhook-token, individuelle strategy_id/event_id. `FAMILY_ADMIN_TOKEN` separat; keine Secrets in Issues/Docs.
- Kritische Live-Gates: KuCoin Hedge Mode tatsächlich und lesend verifizieren; ein einziger autoritativer, hedge-aware Order-Writer, virtual-position Attribution, per-Bot protective SL/TP (reduce-only), Absturz-Reconciliation, idempotente Fills, Funding/Fees und Cashflow-TWR; altes Cancel-All-Verhalten ersetzen; Mengen-Minimum für SOLUSDTM beachten (im bestehenden Setup 0.1 als Stückelungs-/Mindestgrenze prüfen); keine Autoflatten bei Gate-Wechsel oder Budget-Neuberechnung.
- Zuteilungen dürfen nur zukünftige Entries und nachweislich freie Margin beeinflussen. Keine automatischen echten Konto-Transfers, Änderung des Position Mode oder Deployment ohne explizite Betreiberfreigabe.
- Globaler Kontostopp und Risikocaps haben immer Priorität vor evolutionär mutierten Leverage-, Blatt- oder Bark-Werten.

## 11. Software-Architektur und Multi-Agent-Schnittstellen

**Empfohlene isolierte Paketstruktur (neue Ordner, keine implizite Existenzbehauptung):**

```text
src/quant/ecosystem/
  market_data/   # historische Daten und paper fills
  sim_clock/     # Tages-Engine, RNG, Snapshot, Replay
  capital/       # GroundwaterAccounting; Budgets, Reserve, NAV
  ecology/       # Wachstum, Wurzeln, Blätter, Wasser, Sonne
  metrics/       # Fitness/Vitalität/Evidenz/Opportunity
  breeding/      # Bienen, Hive, Genvererbung und lineage
  lifecycle/     # Fraß, Bark, Dormanz, Archiv
  search/        # Candidate Queue, Screening, Compute
  experiments/   # A/B/C/D, Walk-forward, Reports
  adapters/      # read-only Shadow API Brücke
apps/fleet-desktop/src/lib/ecosystemContracts.ts
apps/fleet-desktop/src/components/EvolutionEcosystem*.tsx
benchmarks/ecosystem/
tests/ecosystem/
docs/architecture/
docs/experiments/
```

**Einen stabilen Vertrag zuerst abstimmen:** `StrategyGenome`, `GenePolicy`, `MarketSnapshot`, `TradeFill`, `DailyPlantState`, `GroundwaterState`, `AccountNAV`, `CapitalEntitlement`, `BeeAction`, `PredationEvent`, `MetricWindow`, `ResearchCandidate`, `SimulationEvent`, `SimulationSnapshot`. Adapter schützen Python/TS-Namensgebung und Versionen. Events auditierbar, replaybar, deterministic. Kein Agent schreibt in fremden Paketordner, bevor Schnittstelle abgesprochen wurde.

**Reihenfolge des Tages (Vorschlag, später vertraglich fixieren):**
1. Tageskerzen/Signale zeitlich korrekt paper-ausführen und Fills/PnL/Funding erfassen.
2. Marktwert/NAV + Opportunity + Risk + historische Metrikfenster aktualisieren.
3. Reserve 10%, Budgets, Grundwassersoll/-hysterese, nachweislich freie Margin und Entitlements abstimmen.
4. Biomasse, Wurzeln, Blätter, Dormanz, Sonnen-/Schatten-Effekte täglich wachsen lassen.
5. Bestäubung, Hive, Mutationen, Keimlinge inklusive *umbuchungsfinanzierter* kleiner Budgets.
6. Predation/Archiv/Lifecycle und nur sicher freigewordene Mittel verteilen.
7. Audit-Event-Log + statischen Snapshot persistieren; UI liest ihn, führt aber keine Orders aus.
Keine Tagesstufe darf Zufallszahlen aus unspezifizierter globaler Quelle verwenden; kein Double Spend zwischen Schritt 3 und 6.

## 12. GitHub-Arbeitspakete für eigenständig arbeitende Agenten

**Zentrales Epic: [#36](https://github.com/MartinLiteAndFog/quant/issues/36)**. Jeder folgende Issue enthält den ausführlichen selbständigen **Agenten-Prompt**, Dateibesitz, Abhängigkeiten und Akzeptanztests. Hier steht eine ebenfalls direkt verwendbare Kurzfassung:

| Paket | GitHub | Unabhängiger Arbeitsauftrag (kurzer Agenten-Prompt) | Welle |
|---|---|---|---|
| ECO-01 – Datenverträge | [#23](https://github.com/MartinLiteAndFog/quant/issues/23) | „Lege versionierte Typen, Tages-Eventreihenfolge und Contract-Tests an; trenne NAV, 90/10-Reserve, Budgets, Margin und Exposure. Keine Live-Writes.“ | A/P0 |
| ECO-02 – SOL Data & Fills | [#24](https://github.com/MartinLiteAndFog/quant/issues/24) | „Normalisiere fünf Jahre SOL-Minuten/OHLCV mit Provenienz/Gaps und baue Kosten-, Funding-, Slippage- und zeitkorrektes Paper-Fill-Modell mit Fixtures. Keine erfundene TV-Parität.“ | A/P0 |
| ECO-03 – Tages-Engine | [#25](https://github.com/MartinLiteAndFog/quant/issues/25) | „Baue pure Tages-Tick-Engine, Event-Log, Seed-Replay, Pause/Resume und getrennte intraday Signale; implementiere deduplizierte snapshots und Tests.“ | B/P0 |
| ECO-04 – geschützte Gene | [#26](https://github.com/MartinLiteAndFog/quant/issues/26) | „Migriere geneLibrary auf protected finanzielle/Risiko-Policies plus optionale Strategie-Module; Stop-Loss Pflichtpolicy von SL-Tuning unterscheiden; Gene-Dropout nur optional, Genkosten Default 0.“ | A/P0 |
| ECO-05 – Equity/Grundwasser | [#27](https://github.com/MartinLiteAndFog/quant/issues/27) | „Implementiere konservierendes Decimal-Groundwater-Ledger, 90/10-Schutz, Familien-/Kinderansprüche, freies vs reserviertes Kapital, reale Transfer-Gates und Kapitaltests.“ | B/P0 |
| ECO-06 – Pflanzenökologie | [#28](https://github.com/MartinLiteAndFog/quant/issues/28) | „Implementiere tägliche Biomasse, Wurzel-Tiefe/Breite, Wasser-Asymptote, Trägheit/Hysterese, Sonnen-/Blatt-Risiko und Schatten, inklusive Kaktus-/Gras-Fixtures.“ | B/P1 |
| ECO-07 – Metriken | [#29](https://github.com/MartinLiteAndFog/quant/issues/29) | „Berechne Fitness, aktuelle Vitalität, Credibility, Sharpe, DD, Recovery, CVaR, Edge-Drift, Opportunity-normalisierte Inaktivität und Diversifikation mit OOS-/kleine-N-Schutz.“ | A/P0 |
| ECO-08 – Bienen/Hive | [#30](https://github.com/MartinLiteAndFog/quant/issues/30) | „Führe lokale, zentrale und hybride Bestäubung mit fünf Bienenalgorithmen, Mutation, geschützten Genen und nachvollziehbarer Lineage zusammen; evaluiere Nachwuchs später OOS.“ | B/P1 |
| ECO-09 – Fresser/Archiv | [#31](https://github.com/MartinLiteAndFog/quant/issues/31) | „Baue Lebenszyklus, Mindestbeobachtung, bark-durch-Evidenz, Vitalitätsschutz 70/80 und gewichtete Zufallsjäger; archivieren statt Pilze rendern; Restkapital sicher freigeben.“ | B/P1 |
| ECO-10 – Compute | [#32](https://github.com/MartinLiteAndFog/quant/issues/32) | „Baue paralleles Screening, Deduplikation, Daten-Splits und reproduzierbare Mac-Benchmarks; messen vor Rust-Rewrite; nicht Millionen Vollbacktests versprechen.“ | A→B/P1 |
| ECO-11 – 3D-Cockpit | [#33](https://github.com/MartinLiteAndFog/quant/issues/33) | „Baue Ober-/Unterwelt-UI mit Phänotypen, Wurzeln, Pegel, Bienen, Fressern, LOD, editierbaren Achsen und fehlenden-Gen-Buckets; nutze klar markierte Sim-Fixtures.“ | C/P1 |
| ECO-12 – Statistik | [#34](https://github.com/MartinLiteAndFog/quant/issues/34) | „Vergleiche Random-only, lokal, Hive und hybrid mit gleichem Budget/Compute, vielen Seeds, Walk-forward/OOS, risk net PnL und Ablation der Ecosystem-Slider.“ | C/P2 |
| ECO-13 – Shadow Bridge | [#35](https://github.com/MartinLiteAndFog/quant/issues/35) | „Entwirf read-only validierte ResearchCandidate→Strategy Registry→TV-Alert→Family Shadow Brücke, gegen Tests verifizierte Go-live-Gates; keine Venue-Writes.“ | C/P2 |

### Standard-Prompt, den jeder Agent vor seinem Paket erhalten soll

> Arbeite im Repo `MartinLiteAndFog/quant`, ab Branch `feat/family-hedge-ledger-20261101`. Lies vollständig `docs/plans/2026-11-01-trading-ecosystem-handoff.md` und dein eigenes GitHub-Issue, einschließlich Abhängigkeiten und Dateibesitz. Erstelle einen eigenen Branch und fokussierten Draft-PR zurück auf den Integrationsbranch. Implementiere **nur dein Paket**, liefere Versionierung, dokumentierte Annahmen, Determinismus/Seed-Fixtures und Tests; markiere nicht entschiedene Parameter als konfigurierbar. Prüfe den realen Code vor Änderungen, erfinde keine vorhandenen APIs. Nimm keine Änderungen an Live-KuCoin, TradingView, Railway, Secrets, Positionen, Orders, Auto-Transfers oder `main` vor. Konfliktende Schnittstellen vorher abstimmen. Schreibe im PR, welche Akzeptanzkriterien erfüllt sind und welche ungelöst bleiben.

### Abhängigkeiten und parallele Zusammenarbeit

- **Welle A (parallel nach schnellem ECO-01-Contract-Entwurf):** #23, #24, #26, #29, #32-Gerüst. #23 sollte als erstes integriert werden; andere nutzen zunächst Mocks/Fixtures.
- **Welle B:** #25 auf #23, #27 auf #23, #28 auf #23+#27, #30 auf #23+#26+#27, #31 auf #23+#29. Separate Paketpfade minimieren Konflikte.
- **Welle C:** #33 darf sofort mit UI-Mocks anfangen, verbindet aber erst nach #28; #34 nach Daten/Metriken/Bienen/Search; #35 als letztes und ausschließlich read-only.
- **PR-Regel:** Ein Issue = ein Verantwortlicher = ein Branch/PR. Keine parallelen Umbauten derselben Dateien ohne Abstimmung. Kernverträge nur durch ECO-01 oder gezielte Review. Status/Links ins Master-Issue #36.
- **Tests:** kleine pure Tests bei jedem Agenten + bestehende Family Tests/PG-Integration/TS Build nicht brechen; neue Verhaltensregeln über Property-Tests, damit Equity, Pflichtgene, Stammbaum und Determinismus invariant bleiben.

## 13. Messbare Definition of Done für einen ersten spielbaren Prototyp

1. Historische SOL-Daten mit Datenqualitätsbericht oder klar gekennzeichnete synthetische Demo-Daten.
2. Reproduzierbare Simulation über mindestens 1 Woche bis mehrere Monate (später 5 Jahre) mit täglichen Pflanzen-Updates und korrekter Intraday-Paper-Execution.
3. 100-USDT-Testwelt mit maximal 90 USDT Gesamtbudget, 10 USDT Reserve, drei Eltern, 20 Kinder; Kapital bleibt bei Geburt, Fraß, Profit und Verlust konsistent.
4. Drei reproduzierbare Phänotypen (Gras, Busch, Kaktus) durch Gene, nicht durch Sonderregeln, inklusive Rinde/Dormanz und Blatt-/Wurzel-Trade-off.
5. Fresser fressen niemals geschützte Vitale; fehlende Trades allein töten seltene Strategien nicht; archivierte Strategien verschwinden visuell.
6. Lokal/Hive/Hybrid unterscheidbar und in einem kontrollierten Experiment gegenüber rein zufälliger Mutation testbar.
7. Fleet UI zeigt Oberwelt und Forschungs-Unterwelt aggregiert; Achsenwechsel verliert keine Pflanzen; Performancekennzahlen klar „synthetisch“ oder „empirisch“ markiert.
8. Keine Strategie erhält echtes Kapital, bis später explizit genehmigte KuCoin-Hedge-/Orderwriter-/Ledger-Gates nachweisbar passiert sind.

## 14. Entscheidungen, die der Nutzer später per Regler testen kann

- Globaler 90/10-Reserveschutz (live zunächst hard-coded; Simulation darf Parameterexperimente ausführen, Safety-Floor getrennt).
- `groundwater_scale`, `groundwater_min_depth`, `water_rise_speed`, `water_fall_speed`, `hysteresis`, eventuell 2–80%-Zugangsquote.
- `root_growth`, `root_depth`, `root_spread`, Blatt-/Wurzel-Wachstumstradeoff, `leaf_area`, globale Sun-/Risk-Multiplier, Exposure-Caps und Verschattungskurve.
- `predator_vitality_ceiling` (Start 70%), 70–80% Schonzone, Bark-Wirkung, Lebensdauer/Evidence-Gates, Jagdhunger/Zufallsgewicht.
- `vitality_half_life_days`, multi-horizon Fitness, rare-opportunity mode, Confidence-Floor, Benchmark-/OOS-Window.
- `reproduction_budget` in bounded GenePolicy, Samenanzahl, Bee-Breed, local/hive/hybrid, Gene-Add-/Drop-Rate.
- `complexity_penalty_weight=0` als Standard; optional erhöhen und testen.
- Grafik: Pflanzenklassifizierung, Biomasse-Log-Skala, Bodenschnitt, Render-Limit/Cluster, 2 Achsengene.

## 15. Offene Fragen und Risiken (nicht als beschlossen ausgeben)

1. Datenquelle und Echtheit: Gibt es den vollständigen fünfjährigen SOL Futures Datensatz? Kommen 1m-Exchange-Bars inklusive funding/mark? Strategien mit Pine/Renko brauchen semantische Signalparität.
2. Realer MacBook-Pro-Chip/RAM und Benchmark-Ziel noch nicht bekannt. Millionenkandidaten ≠ millionenfache Vollbacktests.
3. Braucht das Spiel eine native Rust-Engine? Erst nach Messung entscheiden; die aktuelle Python-/TS-Basis existiert bereits.
4. Formel der Vitalität, Fitnessgewichte, OOS-Konfidenz und Bark-Evidenz muss empirisch kalibriert werden; 70/80% sind vorläufige UI-Werte.
5. Pegelkurve, Trägheit, Hysterese und zu- bzw. abnehmender Zugang sind Parametertests, keine bewiesene Marktwirkung.
6. Welche Gene werden real executable Strategy Modules, welche nur Phänotyp-/Metastrategie-Gene? **Keine direkte Arbitrary-Code-Ausführung.**
7. Exakte Mehrkonto-Nettoequity/Transfers ist im Simulation-Portfolio möglich, in realen Subaccounts ohne Freigabe nicht.
8. „Niedrige Kapitalnutzung“ versus „hoher Kapitalzugang“ muss über **nicht-exklusive Entitlements** und real-time freie Margin geregelt werden; sonst werden Mittel implizit doppelt reserviert.
9. Die Wiese könnte auf fünf Jahre SOL overfitten: deshalb historische Split-Grenzen, Multiple Tests, echte Kosten und niemals einen einzigen erfolgreichen Lauf als Beweis präsentieren.
10. Der Architektur-Backlog ist unabhängig von Freigabe und Migration des bestehenden Blaupunkt Live-Bots.

**Nächster sauberer Start:** [#23 Contracts](https://github.com/MartinLiteAndFog/quant/issues/23), [#24 Marktdaten](https://github.com/MartinLiteAndFog/quant/issues/24), [#26 geschützte Gene](https://github.com/MartinLiteAndFog/quant/issues/26) und [#29 Metriken](https://github.com/MartinLiteAndFog/quant/issues/29) parallel beginnen; #27 Kapitalbuch anschließend mit #28 Ökologie zusammenbringen; #33 UI kann mit Fixtures parallel starten.

## 16. NEUE ENTSCHEIDUNG: Kategorisierte zweidimensionale *Gen-Parameter-Wiesen* (ersetzt universelle Verhaltenskarte)

**Entscheidung des Nutzers:** Die Standard-Landschaft darf **keine** PCA/UMAP-/Verhaltensähnlichkeitskarte sein. Die Position der Pflanzen muss durch **zwei echte genetische, strategisch relevante, mutierbare Parameter** bestimmt sein, weil die räumliche Aussaat gerade diese Parameter fine-tunen soll. Mehrdimensionale Genome sind erlaubt, aber die sichtbare Wiese bleibt pro Kategorie zweidimensional.

### Karten-/Kategorieprinzip

- Eine **Kategorie** (auch Wiese/Map/Parameterraum) definiert `category_id`, `strategy_module_family` bzw. Kompatibilitätsregel, zwei konkrete Gen-Parameter `axis_x_gene_id` und `axis_y_gene_id`, Wertebereiche, Schritte und stabile Versionskennung. Beispiel: IMBA-Sensitivität × TTP-Abstand; RSI-Periode × RSI-Entry-Grenze; Trend-Following-Stärke × Zeithorizont (nur falls beides tatsächlich Strategieparameter sind, nicht nachträglich erfundene Behavioral-Metriken); SL-Tuning × Exit-Trailing-Parameter.
- Ein Organismus ist **nur auf der Kategorie-Wiese sichtbar, deren beide Achsengene er tatsächlich trägt** und deren Strategie-/Modulfamilie dazu passt. Ein Bot, der weder Sensitivität noch TTP besitzt, erscheint ausdrücklich **nicht** auf der IMBA×TTP-Wiese – das ist gewollt, nicht ein Visualisierungsfehler.
- **Jede Strategie muss mindestens einer Kategorie zugewiesen werden**. Fehlt ein passendes zweidimensionales Achsenpaar, wird ein neues relevantes, tatsächlich ausführbares Parameterpaar als Kategorie angelegt bzw. die Strategiedefinition um ein fachlich sinnvolles zweites *echtes* Tuning-Gen erweitert; bis dahin ist die Strategie **unzugeordnet** und als Daten-/Konfigurationsfehler sichtbar, nicht lautlos aus der Gesamtpopulation verschwunden. Keine Erfindung von Parametern oder zwingende Änderung einer Strategie allein aus Darstellungsgründen.
- Ein Organismus kann **mehrere kompatible Kategorieansichten** besitzen, aber es gibt pro Organismus **eine kanonische Heimat-Kategorie** für Aussaat, Wurzeln, lokale Nachbarschaft und Predation. Weitere Ansichten sind Visualisierungsprojektionen **derselben eindeutigen strategy_id**, keine vervielfältigten Bots, Budgets, Equity oder Nachkommen. Die Entscheidung der Heimatkategorie ist versioniert und bei Aussaat festgelegt, änderbar nur durch explizite Migration.
- Kategorie-Switching im Fleet Cockpit wählt Wiese und Filter; die andere Pflanzenpopulation entwickelt sich **im Hintergrund weiter**, auch wenn gerade nicht gerendert. Die globale Equity/Grundwasserbuchhaltung und Sicherheitsreserve bleiben *über alle Kategorien hinweg* dieselbe, ohne pro Kategorie Kopien.
- Bienen, die lokal Samen streuen, mutieren bevorzugt die **beiden Achsengene ihrer Heimatwiese**. Übrige Gene können dabei ebenfalls (separat konfigurierbar) mutieren, aber die kindliche Position `(x,y)` ist das wirkliche Ergebnis der Achsgenänderung. Kein bloß dekoratives Samenflugdiagramm. Genewechsel, die das Kind für eine andere Wiese qualifizieren, brauchen explizites Kategorie-Transfer-/Neuzuordnungsereignis; kein unbemerktes Verschwinden. Hive-/Fernflug-Bienen dürfen kategorienübergreifend Gene sammeln, aber erzeugen nur bei gültiger Genom-/Strategiemodulkombination handelbare Kandidaten.
- Statistische Parameterflächen sind nur **unter transparenten Bedingungen interpretierbar**: Nebenparameter, Strategiemodul, Datensplit und Ausführungsmodell können Fitness-Cluster verfälschen. UI muss Filter, Genom-Details, Facets und Vergleich gleicher Nebenparameter anbieten und ggf. **„heterogene Nebenparameter“** anzeigen.
- Karten sollen einzeln anzeigbare Populationszahlen, Fitness-Cluster, Forschungs- und Kapitalstatistik haben; alle Karten zusammen zeigen exakt die deduplizierte Gesamtpopulation.
- Ein optionaler späterer **Node-/High-D-Graph** ist denkbar als **separates Forschungswerkzeug**, ersetzt aber nicht die normale schöne 2D-Wiese und ihre mutierbaren strategischen Achsen.

### UI- und Forschungs-Akzeptanztests

1. Strategie `A` besitzt Sensitivität+TTP und erscheint auf der IMBA-Wiese. Strategie `B` besitzt nur RSI+SL-Tuning und erscheint ausschließlich auf einer dazu passenden Wiese; das Umschalten zwischen beiden ist möglich.
2. `A` kann mehrere passende Karten haben, aber bei Aggregation und Kapitalbuchhaltung genau **einmal** erscheinen. Eigentümliche lokale Wurzeln/Seed-Koordinaten bleiben unverändert, wenn lediglich die Ansicht gewechselt wird.
3. Eine neue Strategie mit noch keinem definierten Achsenpaar erzeugt einen **klar ausgewiesenen unzugeordneten Status** und einen Workflow „Kategorie erstellen“, statt ungesehen zu verschwinden.
4. Bienenmutationen verändern reale `axis_x_gene`/`axis_y_gene`-Werte der Nachkommen; deren sichtbare Position folgt deterministisch diesen mutierten Genen.
5. Wechseln des Filters beendet keine Simulation, löscht keine Pflanzen und verändert keine globale Equity. Unterschiedlich große Wiesen sind ausdrücklich gewollt.
6. Keine separate Kategorien-Eigenkapitalrechnung, falls sie nicht durch tatsächliche Allokation/Transfer im kanonischen Ledger gedeckt ist.

**Vorrangregel:** Dieser Abschnitt ersetzt die frühere Formulierung in §8 und Issue #33, wonach jede Wiese alle Strategien mit einem „Gen nicht vorhanden“-Bucket darstellen müsse. **Kategorien dürfen Populationen bewusst filtern**; das Gesamtsystem muss jedoch jede Strategie in mindestens einer Kategorie oder als sichtbaren Zuordnungsfehler erfassen. Er ersetzt ebenso ausdrücklich die zwischenzeitlich diskutierte Verhaltenskarte als Standard-Weltgeometrie.
