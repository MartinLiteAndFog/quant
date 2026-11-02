# Trading-Ecosystem / Cloudgarden — neuer Git-Audit und Agenten-Hand-off

**Stand 02.11.2026 · MartinLiteAndFog/quant · Forschungs-/Dokumentationsbranch: feat/family-hedge-ledger-20261101 (Draft PR #22).**

**Aufgabe dieses Dokuments:** Eigenständig lesbarer, aktueller Arbeitsstand und Übergabe an Nachfolgeagenten. Die ursprünglichen produkt- und spielmechanischen Entscheidungen bleiben ausführlich in docs/plans/2026-11-01-trading-ecosystem-handoff.md (§§0–26). Dessen spätere Statusaussagen sind inzwischen teilweise überholt. **Dieses Dokument ist für den aktuellen Git-Stand maßgeblich**, bis ein nachfolgender Audit neue Commits prüft.

**Kurzurteil:** Kernmodule, Spielvisualisierung, separate Python-Evolution und eine neue spielbare Cloudgarden-Anwendung existieren. Laut Cloudgarden-Dokumentation ist erstmals ein historischer SOL-Paper-Trading-Lauf mit vererbbaren Strategiemodulen durchgängig ausgeführt worden. Der dokumentierte 140-Tage-Lauf ist jedoch stark negativ. Die einzelnen PRs sind weiterhin ungemergte Drafts; eine einheitliche, unabhängig validierte, profitabel evolvierende Handelsengine existiert **noch nicht**. Keine Forschung freigibt reale Orders.

## 1. Evidenzstatus und Git-Inventur

**A – unabhängig direkt via GitHub geprüft:** PR-Titel, Status, Basis- und Head-Branches, geänderte Dateien, ausgewählte aktuelle Source-Dateien, Vertrags-/Feature-Dokumentationen, am jeweiligen PR-HEAD zugeordnete GitHub-Action-Runs.

**B – in Git/PR belegt, aber hier nicht unabhängig reproduziert:** manuell behauptete Unit-Tests, Cloudgarden npm-Test-/Build-Protokolle und Safari/Desktop-Checks, lokale Parquet-Quelle und tatsächlicher 140-Tage-Marktreplay. Die Original-Marktdaten sind nicht in diesem GitHub-PR enthalten.

**C – noch nicht nachgewiesen:** Rechte und Venue-/Instrument-Identität der historischen SOL-Quelle, echte KuCoin-/Kraken-/TradingView-Renko-Wick-/MTF-/TP2-/Flip-Parität, venuekalibriertes Funding und Stop-Ausführung, wissenschaftliche Out-of-sample-Überlegenheit evolutionärer Suche gegenüber Random, Live-Execution-Sicherheit.

**GitHub-Status am Stichtag:** Exakt 22 relevante Projekt-PRs (#22, #37, #38, #40, #41, #42, #43, #45, #47, #48, #49, #50, #52, #53, #55, #56, #57, #58, #59, #60, #61, #62). **Alle 22 OPEN + DRAFT; keiner gemergt.** Bei **17** ist mindestens ein abgeschlossener GitHub-PR-Workflow am aktuellen HEAD **grün**. Die HEAD-Abfrage ergab für **#37, #58, #59, #60, #62** **keinen PR-Workflow-Run**; deren lokal dokumentierte Tests sind damit nicht widerlegt, aber auch nicht als GitHub-PR-CI verifiziert. Keine der beschriebenen Entwicklungsänderungen wurde durch dieses Audit auf main gemergt.

### Inventar nach Arbeitspaket

| Arbeitspaket | PR | Konkreter Stand | HEAD-PR-CI |
|---|---|---|---|
| Fleet-/Family-Ausgangspunkt | [#22](https://github.com/MartinLiteAndFog/quant/pull/22) | 3×3-Draft-Pilot, synthetische EvolutionMeadow, Gene Library, Family Shadow | grün, 3 Workflows |
| ECO-01 | [#57](https://github.com/MartinLiteAndFog/quant/pull/57) | versionierte Python-/TS-Kontrakte, Fixtures, Decimal/UTC/90:10 | grün |
| ECO-02 | [#37](https://github.com/MartinLiteAndFog/quant/pull/37) | OHLCV-Normalisierung, konservativer next-open Paper-Fill | kein Run ermittelt |
| ECO-03 | [#42](https://github.com/MartinLiteAndFog/quant/pull/42) | deterministische UTC-Clock, Events, Replay | grün |
| ECO-04 | [#38](https://github.com/MartinLiteAndFog/quant/pull/38) | geschützte Pflicht-Safety-/Finanzgene | grün |
| ECO-05 | [#40](https://github.com/MartinLiteAndFog/quant/pull/40) | NAV/Grundwasser, 90/10-Reserve, Margin, Decimal-Schutz | grün |
| ECO-06 | [#43](https://github.com/MartinLiteAndFog/quant/pull/43) | Wurzeln, Blätter, Sonne, Trägheit, Kaktusphysiologie | grün |
| ECO-07 | [#48](https://github.com/MartinLiteAndFog/quant/pull/48) | Fitness, Vitalität, Credibility, Risikomessung/Opportunitäten | grün |
| ECO-08 | [#50](https://github.com/MartinLiteAndFog/quant/pull/50) | fünf Bienenarten, lokale/Hive-/Hybrid-Genmutationen | grün |
| ECO-09 | [#49](https://github.com/MartinLiteAndFog/quant/pull/49) | Fresser, Rinde+Evidenz, Dormanz, confirmed-flat-Archivierung | grün |
| ECO-10 | [#60](https://github.com/MartinLiteAndFog/quant/pull/60) | deduplizierende Research-Candidate-Queue und synthetische Compute-Smokes | kein Run ermittelt |
| ECO-11 | [#61](https://github.com/MartinLiteAndFog/quant/pull/61) | Fleet-SVG-Spielwelt mit synthetischen Pflanzen/Unterwelt | grün |
| ECO-12 | [#59](https://github.com/MartinLiteAndFog/quant/pull/59) | A/B/C/D-Experiment-Harness, Walk-forward, OOS-Gate | kein Run ermittelt |
| ECO-13 | [#58](https://github.com/MartinLiteAndFog/quant/pull/58) | read-only Family-Shadow-ResearchCandidate-Adapter | kein Run ermittelt |
| ECO-14 | [#41](https://github.com/MartinLiteAndFog/quant/pull/41) | mehrere echte Parameterwiesen-Tiles, Asset-Regionen als Demo | grün |
| ECO-15 | [#45](https://github.com/MartinLiteAndFog/quant/pull/45) | vererbbare optische DNA, Baum-/Kaktus-Phänotypen | grün |
| ECO-16 | [#47](https://github.com/MartinLiteAndFog/quant/pull/47) | 3 Eltern + 20 Keimlinge; 30–365 synthetische Paper-Tage | grün |
| ECO-17 | [#52](https://github.com/MartinLiteAndFog/quant/pull/52) | Tageskreislauf Metrics→Kapital→Ökologie→Bienen→Fresser→Archiv | grün |
| ECO-18 | [#53](https://github.com/MartinLiteAndFog/quant/pull/53) | exogene Zielsignale, echte Kerzen-Fills, MTM/Fee/Funding→ECO-17 | grün |
| ECO-19a | [#55](https://github.com/MartinLiteAndFog/quant/pull/55) | GZIP/CSV-Streaming→SQLite-Index mit Gaps und SHA256 | grün |
| ECO-19b | [#56](https://github.com/MartinLiteAndFog/quant/pull/56) | vollständige 1m→5m-Quellkerzen, IMBA/Renko-Kandidaten, Referenzvergleich | grün |
| **Cloudgarden** | **[#62](https://github.com/MartinLiteAndFog/quant/pull/62)** | **eigenständige React-/TS-App, historische SOL-Paper-Sessions, vererbbare IMBA-/EW-Module** | **kein Run ermittelt** |

Ein grüner Teilworkflow ist keine erfolgreiche wissenschaftliche Gesamtvalidierung und nicht gleichbedeutend mit einem Merge oder Go-live. Alte Handoff-/Epic-Checklisten können noch den Status vor PR #57–#62 enthalten.

## 2. Neues Schlüsselprojekt: Cloudgarden (#62)

**GitHub-Source geprüft:** apps/cloudgarden/README.md; HISTORICAL_TRADING_SPEC.md; PAPER_REPLAY_SPEC.md; src/{simulation,historicalGarden,paperReplay,sourceStrategies,strategySignals,HistoricalTrading}.ts(x); scripts/{export_sol_history.py,verify_history.mjs}. Cloudgarden ist eine eigene Anwendung unter apps/cloudgarden und **nicht bloß** die bisherige Fleet-Wiese.

### Was die Anwendung jetzt kann

- **SOL Trading** als Hauptmodus: tageweise historische Preiskerzen lesen, auf allen finanzierten Pflanzen deren unterschiedliche **vererbbare Tradingmodule** ausführen, Target-Signale, strikt spätere Open-Fills, Kosten und Netto-PnL verbuchen, **ein** Paper-NAV/90:10-Ledger fortschreiben und danach Ökologie, Geburt und Fraß ausführen.
- Tages-, Wochen-, Saison- und Play/Pause-Steuerung; historische UTC-Datums-/Equity-Ansicht, per-Pflanze-Gene, Ausführungen, Kosten, Fill-/Reject-Anzahl. Browserlokales Versioned Save/Restore und eigener Desktop-Launcher. Synthetic-Meadow und Understory bleiben getrennte und ausdrücklich künstliche Demo-Tabs.
- Start: **100 virtuelle USDT**, davon **10 Reserve/90 Budget**, **3 Eltern × 20 USDT** und **20 einmalige Keimlingsansprüche × 1,50 USDT**. Einmalige Keimlingsansprüche werden bei Geburt umgebucht statt Geld zu erschaffen; Fraß gibt Budget erst nach Archivierung im Paper-Spiel wieder frei.
- Startfamilien: IMBA Trend, invertierter IMBA-Countertrend und IMBA×EW. Eigenständige SourceGenome-Gene für Channel-Lookback, ATR, bestätigte EW-Pivots, Bewegungs-/Retracementfilter, EW-Exit und verpflichtenden Stop-Close-Trigger. Lokal/Hive/Hybrid/Random erben diese begrenzt. **Es sind Candle-Research-Varianten, keine nachgewiesenen exakten TradingView-/Renko-/MTF-/Strategy-Builder-Strategien.**
- Modellannahmen: **1× simulierte Long/Short-Position**, Zielgröße aus **halber** Pflanzenzuteilung am bekannten Tageseröffnungs-Preis, **täglich flat**, Stop durch abgeschlossenen Kerzenschluss ausgelöst. Keine garantierten Schutzorder-Preise; Modellkosten **6 bps Fee + 2 bps Spread + 1 bp Slippage**, keine historischen Funding-Zahlungen. Die live relevanten TP2/Flip-Legs werden damit **nicht** exakt reproduziert.

### Historische Marktdaten: der alte Status muss korrigiert werden

Die Cloudgarden-Dokumentation benennt inzwischen eine lokal vorhandene, als immutable bezeichnete Parquet-Quelle:

- **Pfad:** research/solusdt_robust_growth_v1/immutable/SOL-USDT_1m_20260711T092900Z.parquet
- **Dokumentierter Original-SHA256:** fa3e689ddbc42b35b89f66db4240521a330896e2d89fc2d2df8624bc749fda64
- **Bericht:** 3.109.186 UTC-Minutenkerzen, 11.08.2020 06:00 bis 11.07.2026 09:29 UTC; **10 Lücken, 1.424 fehlende Minuten**, keine Duplikate.
- scripts/export_sol_history.py erstellt täglich gehashte JSON-Assets, deren genaue Bytes der Browser vor dem Replay prüft. Der Export ist in apps/cloudgarden/public/market/ absichtlich **Git-ignored**.
- **Einschränkung des Audits:** Sowohl die .parquet-Originaldatei als auch public/market/manifest.json wurden via GitHub-Pfadabfrage auf Branch #62 **nicht gefunden**. Ein über GitHub unabhängig ausgeführter 140-Tage-Lauf war daher **nicht möglich**. SHA/Zeilenanzahl/Testbericht sind **Quelltext- und Entwicklerdokumentation, keine erneute Prüfung der Originalbytes in diesem Audit**.
- **Provenienz ungelöst:** Venue, Spot-vs-Futures-Produkttyp, Contract-Multiplier für den Originalmarkt, Lizenz/Weitergaberecht und Funding sind ausdrücklich **unverified**. Das verwendete Linear-Futures-Instrument mit 1× Exposition ist eine Forschungsannahme, keine bekannte KuCoin-/Kraken-Kontraktparität.

**Korrektur der früheren Handoffs:** Die Aussage „keinerlei längere reale SOL-Datei vorhanden“ ist seit #62 **nicht mehr richtig**. Richtig ist: **Ein großer lokaler historischer Datensatz wird dokumentiert und verarbeitet, aber das Original liegt nicht in Git und seine konkrete Markt-/Rechts-Provenienz ist noch ungeklärt**.

### Dokumentierter historischer 140-Tage-Test

PR #62 und HISTORICAL_TRADING_SPEC.md nennen für **01.01.2026–20.05.2026**:

| Messgröße | dokumentierter Wert |
|---|---:|
| NAV Beginn | 100,0000 Paper-USDT |
| NAV Ende | **31,3791 Paper-USDT** |
| Veränderung relativ zu Beginn | **−68,6209 %** |
| Ausgeführte Paper-Fills | **28.078** |
| Summe modellierter Fees | **45,9937 Paper-USDT** |
| verbrauchte einmalige Seedling-Entitlements | **20/20** |
| Performance-/Benchmark-Urteil | **not_comparable** |

Diese **konkrete Forschungsvariante verliert massiv**. Ihre Durchführung ist ein wichtiger Integrations- und Accounting-Test, aber **keine** positive Strategie- oder Evolutionsvalidierung. Hohe Ausführungshäufigkeit und modellierte Kosten gehören in eine gesonderte Ursachenanalyse; ohne Kosten-Ablation nicht behaupten, dass ausschließlich Fees die Verluste verursachen.

PR #62 behauptet außerdem **86 lokal erfolgreiche Tests**, einen TS/Vite-Produktionsbuild, eine historische HTTP-/Desktop-/Safari-Probe. Das Testskript ist im Git vorhanden (npm test, npm run build, npm run verify:history -- 140). Eine **GitHub Actions PR-CI für #62 fehlt am geprüften HEAD**, deshalb im nächsten Paket ergänzen.

### Ausführung (nur bei rechtmäßig vorhandenem lokalen Parquet-Original)

    # im Repo-Root, nur mit echtem lokalen Archiv
    .venv/bin/python apps/cloudgarden/scripts/export_sol_history.py
    cd apps/cloudgarden
    npm install
    npm test
    npm run build
    npm run dev
    npm run verify:history -- 140

Laut README nutzt Vite lokal http://127.0.0.1:5184. Bei fehlender Quelle nicht durch generierte Bars „echte“ Historie vortäuschen.

## 3. Die anderen seit dem alten Hand-off neu fertiggestellten Entwicklungs-PRs

**#57 ECO-01: Python-/TS-Contract v1.0.** src/quant/ecosystem/contracts.py, apps/fleet-desktop/src/lib/ecosystemContracts.ts, Schema-/Fixtures. Enthält eindeutige Accounts/Plants/Events, UTC-Kausalität, Decimal-Strings, immutable Safety-Gene, NAV/90:10/Free-/Used-/Reserved-Margin, Fitness/Vitalität/Evidenz als getrennte Felder. CI Fleet grün. **Cloudgarden erfüllt noch nicht automatisch dieselbe Austausch-ABI.**

**#60 ECO-10 Screening/Compute.** Deterministische deduplizierte Kandidatenqueue mit Seed, Split-, Evaluator-Version und optionalen Workern; **100/1.000/10.000 synthetische** Kandidaten-Smokes. Beispielmeldung: Apple M3 Pro, 10.000 Vorschläge, 5.141 unique; Messung betrifft den kleinen synthetischen 1.024-Samplescreen, **keinen** Millionen-Fünfjahres-SOL-Backtest. Kein zugeordneter PR-Actions-Run.

**#61 ECO-11 Welt-Ansicht.** apps/fleet-desktop/src/components/EcosystemGameWorld.tsx mit **36 synthetischen** Pflanzen, parametrischer 2D-Wiese, Wurzeln, Bienen, Unterweltaggregat, lokalem Tick. Dokumentierter Rendering-Cap 1.200 Knoten; Fleet-CI grün. Noch kein echter Cloudgarden-/Python-View-Model-Anschluss.

**#59 ECO-12 Experiment-Harness.** A Random, B Local, C Hive, D Hybrid; gleiche Seeds/Evaluationsbudgets, purged Walk-forward, finaler unberührter OOS-Tail, gepaarte Vergleiche/Korrektur, Default-Genkomplexitätsmalus **0**. SyntheticEvaluator ist nur Protokolltest und läuft **nicht** den tatsächlichen Cloudgarden-/Python-Fill-/Fraß-Engine-Pfad. Ohne echte Evidenz **not_comparable**. Lokale Tests im PR berichtet, kein HEAD-Actions-Run.

**#58 ECO-13 Shadow-Adapter.** Versionierte, **read-only** ResearchCandidate-Konfiguration für nachweisbar OOS-qualifizierte Kandidaten, manueller Secret-freier TradingView-Alert-Entwurf, Idempotenz-Fixture. Keine Registrierung, kein Venue-/Family-Write oder Kapitaltransfer. 11 lokale Tests berichtet, kein HEAD-Actions-Run. **Kein Go-live.**

## 4. Was noch ein einheitliches Produkt verhindert

### 4.1 Drei unterschiedliche Wahrheiten und ein gemeinsamer Vertrag fehlen

**Python ECO-01/03/05/06/07/08/09/16/17/18/19** nutzt eigene Decimal-/Wire-Contracts, dynamische Population und Metrics-/Event-Causalität.

**Fleet-Tile-/VisualDNA-/GameWorld-Prototypen (#22/#41/#45/#61)** sind Frontend-Demos mit eigenen synthetischen Zuständen, teils echten 2D-Achsgenwerten und wertvoller Morphologie, aber nicht durchgehend an die tatsächlichen historischen Day-/Fill-Records gebunden.

**Cloudgarden #62** besitzt als **eigenständige React-/TS-App** einen eigenen SimulationWorld-/CapitalLedger-Typ, separate SourceGenome-Gene, Habitat-Fitness, BrowserlocalStorage und historischen Paper-Daily-Flat-Runner.

Das ist **nicht automatisch falsch**, aber es gibt noch keine überprüfte **eine kanonische Genome-/Balance-/Event-/Evidence-Semantik** zwischen allen Projekten. Besonders gefährlich: Cloudgarden heißt **fitness/evidence** im Sinne von synthetischem Habitat und Beobachtungszeit; ECO-07 meint empirische Risiko-/Opportunity-/OOS-Evidenz. Eine ungeprüfte Feldübernahme könnte synthetische Biologie als echte statistische Trading-Fitness ausgeben.

**Empfehlung – noch keine Produktentscheidung:** ECO-01 v1.0 als gemeinsamen Austauschvertrag verwenden; einen einzigen verantwortlichen Account-/NAV-/Entitlement-Writer pro Simulationswelt bestimmen. Python als wissenschaftlichen Referenz-/Prüfpfad und Cloudgarden als spielbaren Research-Runner/Frontend behandeln; über versionierte, beidseitig getestete Adapter verbinden. Nicht ad hoc die beiden Ledgers zusammenzählen.

### 4.2 Tatsächliche Marktdaten sind noch keine Signal-/Venue-Parität

Cloudgarden handelt eigene **Candle-Research-Varianten mit Daily Flat 1×**. Reale historische TradingView- und Live-Bots können aber **Renko aus 5m-Quellbars**, TTP-/TP2-/Flip-Zustandsmaschinen, Gate-Latching, mehrtägige Positionen, TP1-Partial, SL-/TP-Reduce-only, Hedge-Mode- und Lot-Regeln besitzen.

Die vorhandene Datei docs/RENKO_TRADINGVIEW_PARITY.md dokumentiert Kraken SOLUSD.PM, 5m-Traditional-Renko, Box **0,1**, Lookback 50: **5.322 TV-Bricks / 99 IMBA-Flips** gegen **5.279 lokale 5m-Bricks / 105 Flips**. Renko-Wicks und Feed-/Anker-Parität sind ungelöst. ECO-19b #56 liefert daher **unverifizierte Research-Kandidaten**, keine als TV-paritätisch freigegebenen Handelsanweisungen. Ein historischer Paper-Fill auf echten Preisen **beweist** keine korrekte Strategieformel.

### 4.3 Die Rückkopplung „echt profitabel → mehr Nachkommen, schwach → Fraß“ ist nicht belegt

ECO-07 liefert empirische Netto-/Risk-/Evidenzmetriken; Cloudgarden-Pflanzen verwenden im Spiel Habitat-Fitness, Vigor und synthetische Evidence. Es ist **nicht nachgewiesen**, dass historische Marktperformance im tatsächlichen Cloudgarden-Loop kausal dieselbe Priorität, Saat, Allokation, Rindenwirkung und Predation kontrolliert wie ECO-07. Diese Verbindung ist zentraler Gegenstand der eigentlichen Forschungsfrage und derzeit eine wesentliche Integrationslücke.

### 4.4 Echte statistische Überlegenheit vs. Random ist offen

ECO-12 ist ein formaler A/B/C/D-Protokollbaukasten; ECO-10 ist ein Screening-Scheduler. Beides **kein** empirischer, mehrfachtestkorrigierter Vergleich mit echten pro-Genom-/pro-Kandidat-Fills über unabhängige Walk-forward-Folds. Es gibt derzeit **keinen** verifizierten Gewinner und keinen Nachweis, dass Bienen mehr bringen als gleich teure Zufallssuche. Die −68,6209 % des Cloudgarden-Piloten belegen dies ebenfalls **nicht**.

### 4.5 Frontend-Wiesen, Pflanzen-DNA und Multi-Asset-Regionen sind nur teilintegriert

Eine Wiese bleibt **eine 2D-Kachel mit genau zwei tatsächlich genutzten, strategisch relevanten Achsgenen**. Varianten ohne Achsenpaar brauchen eine eigene Kategorie, nicht eine falsche Projektion. Multi-Asset-Kacheln, tile-spezifische Gene, fokussierbarer Zoom, Unterweltaggregate/Myzelwissen, erbliche Form/Farbe/Drehung/Verzweigung, grüne Kaktus-Rinde und ein permanenter optischer Stammbaum sind in #41/#45/#61 **teilweise** entwickelt. In Cloudgarden #62 werden Formen aus Wurzel-/Blattgenen abgeleitet. **Ein gemeinsames realdatengebundenes, LOD-skaliertes Wiesenuniversum ist noch nicht vorhanden**, Unreal Engine 5 nicht implementiert.

### 4.6 Kapital, Sicherheit und Live-Übergang

**Unverrückbar:** eine Equity-Bilanz, maximal **90%** verfügbarer positiver NAV als Entitlements, mindestens **10%** Reserve, kein Doppelbudget für Grundwasser/Pilze/Sonne. Used/Reserved/Free Margin und Anteilseigner trennen. Gene dürfen Schutzstopps nicht entfernen. Fresser archivieren und verteilen nur nach confirmed-flat und reconcilierten offenen Orders/Fills; **keine** automatischen realen Close-/Cancel-all-Befehle.

Für jede spätere echte Order: per-Venue Lot/MinQty, Position-Mode, Single-Writer, Hedge-Side, Reduce-only-Protective-SLs, TP2/Flip-TTP-Parität, Stop-Ausführung, Funding und unabhängige Operatorfreigabe. Der bisherige Paper-Code erfüllt **keine** Live-Freigabe.

## 5. Abhängigkeiten und vorgeschlagener Integrationsplan

**Stack, der erst nach Review/Merge wieder dedupliziert werden muss:**

    main (separate existing production code)
      └─ feat/family-hedge-ledger-20261101          #22 Draft
         ├─ ECO-01 contracts                        #57 Draft
         ├─ ECO-02 paper data                        #37 Draft
         ├─ ECO-03/04/05/06/07/08/09               #42/38/40/43/48/50/49
         ├─ ECO-10/11/12/13/14/15                 #60/61/59/58/41/45
         ├─ ECO-16 paper world                      #47 Draft
         │  └─ ECO-17 evolution                     #52 Draft
         │     └─ ECO-18 replay                     #53 Draft
         │        └─ ECO-19a indexed history        #55 Draft
         │           └─ ECO-19b TV/IMBA audit       #56 Draft
         └─ Cloudgarden standalone                  #62 Draft

**Welle 0, parallel:** Agent A echte Daten-/CI-Reproduktion, Agent B ECO-01-/Cloudgarden-Contract-Adapter, Agent C TV-/Renko-/IMBA-/TP2-/Flip-Paritätsdiagnose. Diese Arbeitspfade müssen unabhängig sein.

**Welle 1 danach:** Agent D echte Performance → Pflanzen-Vitalität → Geburten/Fraß/Allokation; Agent F eine gemeinsame Tile-/VisualDNA-Ansicht; Agent G CPU/RAM/Streaming-Profiling.

**Welle 2 erst bei gültigem empirischen Runner:** Agent E echte OOS-A/B/C/D-Versuche; Agent H read-only Shadow- und Sicherheitsspezifikation. Ein Gewinner wird **nicht** automatisch produktiv gestellt.

## 6. Vollständig delegierbare Agentenprompts

### P0 · Agent A: echte Daten-Provenienz, CI und 140-Tage-Reproduktion

**Pfad-/PR-Owner:** apps/cloudgarden/scripts, apps/cloudgarden/README.md/HISTORICAL_TRADING_SPEC.md und neue CI-Dateien; #62, ECO-19a #55.

**Prompt:** „Prüfe das tatsächlich verfügbare lokale Original research/solusdt_robust_growth_v1/immutable/SOL-USDT_1m_20260711T092900Z.parquet. Rechne dessen Original-SHA256 selbst nach und vergleiche dokumentierten fa3e689d…749fda64, 3.109.186 Zeilen, 10 Gaps/1.424 Minuten, UTC-Grenzen. Lege Ursprung, Lizenz, genaue Börse und Spot/Futures-Kontrakt, Contract-Multiplier, Zeitbasis, Funding-Verfügbarkeit offen. Derivative dürfen nur mit erlaubter Lizenz verteilt werden. Installiere GitHub-CI für npm test + npm run build auf einer öffentlichen kleinen Synthetic-Fixture; den 140-Tage-Verify mit tatsächlicher privater Datei nur auf einem autorisierten Runner. Ein Original außerhalb des GitHub-Zugriffs ist ein expliziter Blocker, niemals Grund für erfundene Daten.“

**Abnahme:** Hash-/Metadatenbeleg oder sichtbarer Source-Blocker; eigene CI, reproduzierbare 140-Tage-Inputs und Ergebnismanifest mit 28.078-Fill-/31,3791-NAV-Abgleich; kein nicht autorisierter Daten-Upload.

### P0 · Agent B: EIN gemeinsamer Wire-Contract und konservierende Ledger-Semantik

**Owner:** nur neue Adapter/Golden-Fixtures im ECO-01/Cloudgarden-Integrationsbereich; #57, #40, #48, #52, #62.

**Prompt:** „Lege versionierte JSON-Austauschtypen und Testfixture zwischen ECO-01 Python/TS und Cloudgarden SimulationWorld/HistoricalSession fest: dataset_fingerprint, account_NAV/reserve/distributable, Entitlements, used/reserved/free margin, booked PnL, Currency, strategy/genome/code_version, Lineage, event_id, known_at und day_index. Benenne Habitat-Fitness und tatsächliche empirische TWR-/Vitalitäts-/OOS-Credibility strikt unterschiedlich. Golden-Test: Python→TS→Python, auch bei 1/10000-USDT-Rundung, Budgettransfer, Cashflow, negativer NAV, Replay. Verhindere zwei autoritative Writer auf dasselbe Kapital. Keine Änderung an echten Account- oder Webhook-Schnittstellen.“

**Abnahme:** CI-gedeckte Wire-Fixtures, idempotente Buchung, keine synthetische Evidenz-Promotion, 90/10 zu jedem Zeitpunkt, dokumentierte Contract-Version.

### P0 · Agent C: Original-Signal-/Fill-Parität nachweisen

**Owner:** separate research/parity-Pfade, ECO-19b #56, vorhandene quant.features.renko, quant.strategies.imba und originale Pine-/TradingView-Referenzen; #62 nur read-only Vergleich.

**Prompt:** „Besorge und prüfe den echten TradingView-Kraken SOLUSD.PM 5m Traditional-Renko-0.1-Export inkl. historischer Wicks und Webhook-Alertliste, plus Preisquelle vom exakt selben Instrument. Reproduziere 5.322 vs. 5.279 Bricks / 99 vs. 105 IMBA-Flips, erkläre jede verbleibende Abweichung. Implementiere kausale Wick-/Anker-/Multibrick-Zeitkorrektur mit Snapshot-Tests. Dann erst die echte versionierte Flip-TTP-/TP2-Leg-/TP1-/Stop-Logik als isoliertes Paper-Modul, echte verfügbare Preise und Fees statt Renko-Kunstpreise. Gatewechsel gelten nur für neue Legs, bestehende latched Exits nicht nachträglich ändern. Kein Legacy-Signal oder 1ms-Renko-Stempel darf sofort handelbar werden.“

**Abnahme:** expliziter Side-/Timestamp-/Stop-/TP-Abgleich mit unabhängigen Referenzereignissen. Ohne exakte Parität gilt machine-readable parity_verified=false, keine echten Backtest-Behauptungen.

### P1 · Agent D: tatsächliche Handelsfitness steuert Evolution

**Owner:** ECO-07/08/09/17-Adapter und Cloudgarden research integration; keine Produktivexekution.

**Prompt:** „Führe ausschließlich bis zum jeweiligen Tag bekannte, pro Genom ausgeführte Netto-TWR-, Drawdown-, Activity-, Opportunity-/Confidence-Scores auf eine verifizierte Population zurück. Ein echter Underperformer darf mehr Reproduktionsbudget verlieren oder häufiger von Fressern bedroht sein, unabhängig von hübscher Farbe; ein langzeitig robuster, selten handelnder Kaktus darf bei nachweislich fehlenden echten Gelegenheiten geschützt ruhen. Das Cloudgarden-Habitat-Fitness-Feld darf nur ökologisch/visuell bleiben. Jede Geburt kostet das Mutter-Entitlement, Hive-Research bekommt kein Kapital, Stopps bleiben Pflicht. Schließungswünsche geben kein Budget frei, bis confirmed-flat/Fills/Margin feststehen. Mehrgenerations-Tests und Crash-Replay.“

**Abnahme:** Gegenfaktischer Test: Zwei gleich gestaltete Genotypen mit verschieden kausal gemessenen Nettotrades entwickeln unterschiedliche Reproduktions-/Fraßverläufe. Keine Future-Fitness, keine doppelte Equity.

### P1 · Agent E: unabhängige OOS-Evolution gegen Random

**Owner:** ECO-12 #59 + ECO-10 #60 als Runner-Adapter, keine Veränderung historischer Trades nach Ergebniskenntnis.

**Prompt:** „Vergleiche A Random, B lokal, C Hive, D hybrid bei gleichen historischen Daten, Seeds, Startgenomen, Kosten und identischem Evaluationsbudget. Pre-register Walk-forward mit Purge/Embargo und einmaligem, unangetastetem finalem Holdout. Baselines: Cash/NoTrade, Buy-and-Hold und festgefrorene statische Strategie. Kennzahlen: Nettorendite, Worst-Drawdown, Tail-/CVaR, Turnover/Fees, Kapitaleffizienz, OOS-Confidence, Zahl unterschiedlicher geprüfter Genome und Compute. Korrigiere Multipletests und halte die gesamte Suchhistorie. Fehlende Feed-/Kontrakt-/Funding-Parität oder zu kleine unabhängige Stichprobe => not_comparable.“

**Abnahme:** reproduzierbarer A/B/C/D-Bericht mit negatives Ergebnis zulässig; keine unberechtigte Gewinnerselektion, keine OOS-Reuse.

### P1 · Agent F: eine echte Wiesen-Kachelwelt und vererbbare Optik

**Owner:** #41 TileWorld, #45 plantAppearance, #61 GameWorld, #62 UI; read-only State-Adapter.

**Prompt:** „Verbinde die realen Versioned-Snapshot-Strategien mit tatsächlichen zweidimensionalen Parameter-Kacheln. Jede Kategorie hat genau zwei echte Achsengene; nicht passende Pflanzen werden korrekt einer neuen Kategorie zugeordnet. Assetklassen und Märkte sind Regionen, nicht neue Geld-Pools. Zeige Pflanzen-Genealogy, Farb-/Dreh-/Verzweigungs-/Leaf-/Root-Gene, Kaktus mit mindestens einem photosynthetischen Rindenblatt, Sonne/Schatten/Wurzelzugang und vom empirischen Nettoresultat getrennte visuelle Biomasse. Unterwelt mit ausschließlich aggregierten Forschungskandidaten und Myzel-Wissensbrücken. Zoom auf markierte Kachel, Überblick über Landschaft und LOD. Keine UI-Interaktion darf Kapital oder Trades direkt buchen.“

**Abnahme:** Echte Genachsen und Statusevidenz im Tile-Snapshot, LOD-/Zoom-Tests; gleiche NAV unabhängig von Darstellung; keine millionenfach gerenderten Pilze. Unreal 5 erst nach Messung einer realen Renderergrenze.

### P2 · Agent G: Daten-/Genom-Skalierung und Langzeit-Replay

**Owner:** ECO-19a #55, ECO-18 #53, ECO-10 #60.

**Prompt:** „Scanne den 3,1-Millionen-Kerzen-Index blockweise, ohne 5 Jahre in ein einzelnes Python-/Browserarray zu laden. Trage offenen Renko-, Indikator-, Stop-, TP2-/Flip-, Funding- und Position-State über UTC-Tage hinweg fort. Cache nur identische Genome+Codeversion+Quellen-Hash+Fold+Kosten. Benchmark 10k/100k/1M Candidate-Proposals mit echtem Evaluator, erfasse gesamte Worker-RSS, Zeit, Fehler, Wiederanlauf und deterministische Wiederholbarkeit. Rust/Numba erst nach Profiling.“

**Abnahme:** überprüfbare realistische Laufzeiten/Speicherberichte, kein Memory-Swap-basierter Scheinbenchmark, nie heimlicher Datensatz-Future-Leak.

### P2 · Agent H: Shadow-Gates ohne Deployment

**Owner:** ECO-13 #58 + bestehende 3×3-Family-Files nur read-only.

**Prompt:** „Erstelle einen auditierbaren, manuellen Go-/No-Go-Gateplan für künftige OOS-geeignete ResearchCandidates: ein autoritativer Orderwriter, korrekter KuCoin Hedge Mode, Lot-/Notional-Minimum, per Bot Reduce-only-SL/TP, Fill/Crash-Reconciliation, event_id-Idempotenz, sichere bestehende Legs, kein cancel-all fremder Schutzorders, globale Risikolimits und 90/10. Ohne Betreiberfreigabe weder Alarm erstellen, noch Register-/Kontodaten ändern, noch Railway deployen.“

**Abnahme:** reine Konfigurationsvorschläge und Prüfberichte; keine realen Order-, Position- oder Transferaktionen.

## 7. Verantwortliche Reihenfolge und strenge Änderungsregeln

**Jetzt parallel A + B + C.** Erst nach tatsächlicher Preis-/Signal-/Contract-Evidenz D + F + G. **E erst nach integriertem empirischen Evaluator**, H als separate spätere read-only Sicherheitsprüfung. Jeder Agent: eigener klar abgegrenzter Branch, Draft-PR, CI, aktuelle Head-IDs prüfen und nur die eigenen Pfade bearbeiten.

**Stack-Abhängigkeit prüfen statt alles blind zu mergen:**

    main
      └─ feat/family-hedge-ledger-20261101   (PR #22 WIP)
         ├─ ECO-01 / 02 / 03 / 04 / 05 / 06 / 07 / 08 / 09
         ├─ ECO-10 / 11 / 12 / 13 / 14 / 15
         ├─ ECO-16 PR #47
         │   └─ ECO-17 PR #52
         │       └─ ECO-18 PR #53
         │           └─ ECO-19a PR #55
         │               └─ ECO-19b PR #56
         └─ Cloudgarden PR #62 (separat, nicht dieselbe Python-Engine)

Einige PRs enthalten temporär identische Abhängigkeitskopien. Vor Integration **rebasen und deduplizieren**, einen einzigen autoritativen Contract-/Capital-Writer wählen und **erneut** CI ausführen. Keine Live-Änderung als Nebeneffekt des Doc-Merges.

## 8. Definition of Done und ehrliches Gesamturteil

**Meilenstein 1: Spielbarer historischer Paper-Prototyp — weitgehend, laut lokalem Cloudgarden-Bericht erreicht.** Ein historischer Preis-→Signal-→Fill-→NAV-→Tageswachstum-Lauf mit mehreren Pflanzen wird dokumentiert, allerdings Daten-Original/CI hier nicht unabhängig reproduzierbar und Provenienz unverified.

**Meilenstein 2: Wissenschaftlich belastbare Strategie-Evolution — NICHT erreicht.** Echte gleichartige Venue-/TV-Parität, unabhängige kausale Netto-TWR-/Opportunity-basierte Selektion/Vererbung, Frozen Walk-forward/OOS und ein korrigierter Vergleich mit Random fehlen. Der aktuell berichtete Markt-Probelauf verliert 68,6209 % des Start-NAV.

**Meilenstein 3: Sicherer optionaler Livebetrieb — NICHT erreicht / NICHT beauftragt.** Erfordert Meilenstein 2, unabhängige Shadow-/Broker-/Hedge-/Stop-/Order-/Kapitallimits-Tests und **separate explizite Betreiberfreigabe**.

### Wichtige Referenzen

- Repo: https://github.com/MartinLiteAndFog/quant
- Ursprünglicher detaillierter Konzept-Hand-off: docs/plans/2026-11-01-trading-ecosystem-handoff.md (Branch feat/family-hedge-ledger-20261101).
- **Dieser neue Status-/Agenten-Hand-off:** docs/plans/2026-11-02-trading-ecosystem-git-audit-handoff.md (gleicher Branch).
- Master-Issue: https://github.com/MartinLiteAndFog/quant/issues/36
- Offene empirische SOL-Parität: https://github.com/MartinLiteAndFog/quant/issues/54
- Cloudgarden: https://github.com/MartinLiteAndFog/quant/pull/62
- Neuer ECO-01-Contract: https://github.com/MartinLiteAndFog/quant/pull/57
- Forschungs-Experimentprotokoll: https://github.com/MartinLiteAndFog/quant/pull/59
- Tile/VisualDNA: https://github.com/MartinLiteAndFog/quant/pull/41 / https://github.com/MartinLiteAndFog/quant/pull/45

**Betreiberregel:** Nichts auf main mergen, nichts auf Railway deployen, keine KuCoin-/Kraken-Trades, kein aktuelles Live-Stop-System anfassen, keine TradingView-Alarme erzeugen, keine Transfers oder Hedge-Mode-Wechsel ohne eigene ausdrückliche Freigabe. Forschungs-Drafts sind keine Handelsfreigaben.

## 10. Ergänzung vom 02.11.2026: Eigenständige Umsetzung von E und H parallel zu A/B/C

Der Benutzer hat **Agent A (historische Daten/Cloudgarden), B (kanonische Schnittstellen) und C (Signalparität)** bereits in getrennten Online-Arbeitssitzungen gestartet. **Diese drei Aufgaben bleiben im Zuständigkeitsbereich dieser Agenten.** Es wurden von diesem Bearbeiter deshalb weder das Dataset und Cloudgarden-`historicalGarden.ts` noch die ECO-01-Wire-Contracts noch TradingView-/Renko-Signalmodule geändert.

### E – Wissenschaftliche Evidenz-Firewall auf ECO-12 (Draft PR #64)

**[PR #64](https://github.com/MartinLiteAndFog/quant/pull/64)** auf Branch `codex/eco12-evidence-firewall-20261102`, **stacked auf ECO-12 [PR #59](https://github.com/MartinLiteAndFog/quant/pull/59)**, ist **GitHub-CI-grün** (Workflow `ECO-12 Evidence Firewall and Experimental Reproducibility`).

**Entdeckte methodische Fehlermöglichkeit:** Die ursprüngliche ECO-12-v1-Auswertung verwendete verschiedene Zufalls-Seeds auf **ein und demselben finalen Markt-OOS-Zeitfenster**, berechnete darauf z-Tests/95%-Intervalle/Holm und ließ theoretisch ein positives `beats` zu, wenn das Eingabeobjekt `source_kind="empirical"` und der Evaluator eine andere ID als `SyntheticEvaluator` behauptete. **30 Such-Seeds auf derselben historischen Kurve sind aber nicht 30 unabhängige Marktstichproben.** Dieses Pseudoreplikationsproblem kann falsch kleine p-Werte erzeugen.

**Implementiert:** `src/quant/ecosystem/experiments/core.py` → Protokoll **`eco12.v1.1`**, `tests/test_ecosystem_experiments.py` (einschließlich bewusst gefälschter `empirical`-/`verified`-Eingaben), `experiment.example.json`, Dokumentation und Actions-Workflow. Der jetzige Harness berichtet gepaarte Seed-Mittelwerte plus **ausdrücklich deskriptive** Seed-Streubänder, aber **immer** `p_value_two_sided=null`, `p_value_holm=null`, inferenzielle Konfidenzintervalle `null`, `independent_market_holdouts=1`, `inferential_interval_valid=false` und `not_comparable`. Auch ein externer Evaluator oder angeblich empirischer Datensatz umgeht das **nicht**. Er schreibt weder Trades noch Daten.

**Wissenschaftlicher nächster Schritt:** Ein eigenständiges **v2-Experiment mit tatsächlich unabhängigen Out-of-sample-Marktsegmenten**, glaubwürdiger Datensatz-/Venue-/Signal-/Funding-Provenienz, Block-/Cluster-Abhängigkeit, vorab definierter Mehrfachtestfamilie und einmaliger Holdout-Verwendung. Erst dann sind valide positive/negative Signifikanzurteile zulässig. **Das ist ausdrücklich noch nicht implementiert und benötigt die Ergebnisse A/B/C und historische Daten.**

### H – Expliziter 16-Gate-NO-GO-Sicherheitsbericht für Family Shadow (Draft PR #66)

**[PR #66](https://github.com/MartinLiteAndFog/quant/pull/66)** auf Branch `codex/eco13-shadow-safety-gates-20261102`, **stacked auf ECO-13 [PR #58](https://github.com/MartinLiteAndFog/quant/pull/58)**, ist **GitHub-CI-grün** (Workflow `ECO-13 Read-only Shadow Safety Gate Checks`).

**Problem:** ECO-13 bereitete bewusst eine read-only-Konfiguration aus einem behaupteten `eligible_for_sowing`-/`beats`-ResearchCandidate mit `out_of_sample=true` vor. Diese Eingabefelder sind **keine unabhängig verifizierten** wissenschaftlichen oder Börsen-Nachweise und dürfen niemals als automatische Live-Freigabe gelesen werden.

**Implementiert:** `src/quant/ecosystem/adapters/shadow_audit.py`, `tests/ecosystem/test_shadow_audit.py`, `docs/plans/ecosystem-shadow-safety-gates.md` und Actions-CI. Der Bericht bindet jede behauptete Gateprüfung an einen **SHA-256 des exakten immutable Snapshot** (Änderung der Strategie-/Genomversion macht alte Nachweise ungültig); jeder grüne Claim braucht benannten Prüfer, UTC-Datum, externen Artefaktbeleg und dessen SHA-256. **Das Modul prüft die behaupteten Artefakt-Bytes nicht selbst** und kontaktiert weder GitHub noch Venue/TradingView. Fehlt ein Gate oder ist es abgelaufen, widersprüchlich, dupliziert oder für ein anderes Genom, bleibt `NO_GO`.

**Alle 16 Gates:** Originalquelle/Lizenz; Markt-/Kontrakt-/Multiplier; TradingView-Parität; echte OOS-/Multipletests; Fees/Slippage/Funding; Netto-PnL/MTM; read-only Hedge-Mode; ein autoritativer Orderwriter; eigene Reduce-only-TP/SL; kein Cross-Bot-Cancel-all; aktuelle Mindestlotgröße; Order-/Fill-/Position-Reconciliation; Crash-Idempotenz; Account-Risk/90:10; read-only Shadow Replay; getrennte Operator-Cutover-Prozedur.

**Wichtig:** Auch bei **16/16 behauptet grünen** Attestations meldet das Modul nur `claims_complete_needs_independent_review` und **ausnahmslos** `cutover="NO_GO"`, `live_routing_authorized=false`, `order_sent=false`, `artifact_hashes_unverified=true`. **Kein Live-Test, keine Registrierung oder Operator-Freigabe.**

### Handoff/Integrationsanweisung und derzeitiger kritischer Pfad

- In beiden PRs wurden nur **eigene Stack-Branches** beschrieben und geändert, keine Agent-A/B/C-Arbeitsdateien und keine Produktion. Vor Merge **#59 → #64** und **#58 → #66** separat prüfen und bei Bedarf rebasen; nicht direkt nach `main` oder in aktive Railway-Dienste mergen.
- **E:** Mechanische Vergleichs-/Sicherheits-Tests abgeschlossen; tatsächlicher OOS-Test bleibt bis zur unabhängigen Markt-/Strategie-Beweisführung gesperrt.
- **H:** Formal sichere, read-only Gate-Checkliste abgeschlossen; echte Belege und operatorseitige Abnahme stehen noch aus.
- **D:** Eine nachgewiesene kausale Verbindung von echter pro-Genom-Rendite, Vitalität, Bienen-/Predator-/Kapitalentscheidung wäre der nächste große wissenschaftliche Mehrwert, **wartet sinnvollerweise auf B+C**. Die bestehenden synthetischen Cloudgarden-`fitness/evidence`-Begriffe nicht als reale ECO-07-Werte wiederverwenden.
- **F:** 2D-Tiles/VisualDNA-Frontend ist teilweise in #41/#45/#61 vorhanden; Integration mit den tatsächlichen Cloudgarden-/Python-Snapshots folgt erst nach Schemaabgleich von **B**.
- **G:** Der 5-Jahres-State-/Millionen-Candidate-Performancepfad benötigt den echten Datensatz bzw. dessen nachgewiesenes Formatinventar von **A**. Reine synthetische Screening-Benchmarks können unabhängig weitergeführt werden, sind aber kein historisches Alpha.

**Aktualisierte Bewertung:** Wir haben einen technisch enger abgesicherten **Forschungsprozess**, nicht die wissenschaftlich nachgewiesene Strategieevolution und nicht die Freigabe für echte Börsenpositionen.

## 11. Agent-F-Teilpaket ebenfalls umgesetzt: Kategorievorschläge für zweidimensionale Wiesen (Draft PR #67)

**[Draft PR #67](https://github.com/MartinLiteAndFog/quant/pull/67)**, Branch `codex/eco14-catalog-meadow-proposals-20261102` **stacked auf ECO-14 [#41](https://github.com/MartinLiteAndFog/quant/pull/41)**. **GitHub-CI grün** (neuer Workflow `ECO-14 Read-only New Two-Gene Meadow Proposals` und bestehende `Fleet Evolution Meadow`-CI).

**Datenunabhängiger, bereits implementierter Teil:** `apps/fleet-desktop/src/lib/meadowCatalogProposals.ts`, `tests/meadowCatalogProposals.smoke.mjs`, `docs/architecture/eco14-missing-meadow-proposals.md`, Workflow `.github/workflows/eco14-meadow-proposals.yml`. Es wurde weder die Cloudgarden-Strategien-/Kapital-Engine noch das zurzeit von Agent B bearbeitete Contract-Modul verändert.

- **Nicht zuordenbare Pflanze verschwindet nicht:** Pflanzen mit `homeTileId=null` werden gegen bereits vorhandene kompatible Zwei-Gen-Wiesen getestet. Besteht eine solche, meldet das Modul **nur die vorhandene Kategorie**, statt eine zweite finanzierte Pflanze zu erfinden.
- Fehlt sie, werden **nur katalogisierte echte, numerische Strategie-Genpaare** betrachtet: Modul-ID/-Version, eindeutige Pair-ID, X-/Y-Gen, normalisierte begrenzte Parameterbereiche, separate `approvedStrategySpecId`. Es wird **nicht** willkürlich ein Paar aus irgendeinem numerischen Genom genommen. Bekannte Safety-/Finanz-/Optikgene werden als Achse abgewiesen.
- Kandidaten desselben Assets und Genpaars erzeugen nur **einen** stabilen Wiesen-Vorschlag. Gleiche Genpaare auf SOL, WTI/Öl oder anderen Assets bleiben **getrennte Regions-/Symbol-Kategorien**, ohne Querverrechnung oder neues Kapital. Freie Slots werden stabil auf einem 4×4-Kachelraster gesucht, mit Kollisionsschutz auch auf globalen Weltkoordinaten. Ist die Region voll oder fehlt ein registriertes echtes Genpaar, bleibt die Pflanze **sichtbar als offener Zuordnungsfall** (`no_free_region_slot` oder `no_registered_two_gene_pair`).
- **Alles ist rein lesend:** Vorschläge melden `proposalOnly=true`, `createsFundedPlants=false`, `transfersCapital=false`. Es wird **kein** `homeTileId`, Markt-Trade, Marginkonto, Genome-Fitness oder Water Budget geändert. Eine neue Kategorie wird erst nach Schema-/Modulfreigabe materialisiert.
- GitHub-CI testet Eingabereihenfolge, zwei tatsächliche numerische Achsen, Gruppierung, fehlende Gene, bestehende Wiese, Öl-Region getrennt von SOL, volle Region, verbotene Optik-/Finanz-Gene und vorhandenes ECO-14-Navigations-/TypeScript-/Vite-Build.

**Folgeschritt nach Agent B:** Das tatsächliche Cloudgarden-/ECO-01-Genom- und Versionsschema durch einen read-only Adapter auf die bereits existierende `TileWorld`-Schnittstelle übertragen, `approvedStrategySpecId` wirklich verifizieren und Vorschläge erst dann im UI zur Auswahl anbieten. Die Funktion ist **kein** Live-Trading-Kategorien-Autocreation-System. Keine lokalen Originaldaten benötigt.
