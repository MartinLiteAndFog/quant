# ECO-08 — Bienen, lokale Pollination, Hive und vererbte Strategieparameter

Pure deterministische Genforschung, kein TradingExecutor. Fünf Starttypen wie im bisherigen Fleet-Cockpit: mixer (Zahlenmittel), mosaic (Gen-Mosaik), explorer (breitere Mutation), collector (optionale Gene sammeln), pruner (optional 10% löschen). Mutierende Bienen-Suchgene: Radius, Mutationsstärke, optionale Add/Drop-Rate und Hive-Verbreitung.

Eine Kachel definiert genau zwei notwendige reale Strategieparameter als Achsen. Jedes lokale Kind erhält diese Gene aus der Kombination/Mutation. Seine neue Position ist exakt die Normierung der mutierten X/Y-Gene. Kein kosmetisches Saatenverteilen ohne Parameteränderung.

GeneSpec kennt notwendige immutable safety policies (protective_stop muss True bleiben), notwendige begrenzte finanzielle Gene (max_leverage, max_margin_fraction, reproduction_budget), genau zwei notwendige axis genes und weitere optionale Handelsgene. Schutz-/Finanzgene können niemals wegfallen, auch nicht durch pruner. Weitere optionale Gene dürfen in kontrollierter Rate hinzugefügt oder entfernt werden, sofern im Katalog vorhanden; keine neuen unvalidierten Codes/Tradingmodule.

Lokaler Modus erzeugt einen SeedProposal für eine kompatible Elternpaarung innerhalb derselben Heimatkachel und des genetischen Suchradius. Central/Hive erzeugt ausschließlich research_only Kandidaten; sie dürfen auch kompatible Gene aus anderen Tiles derselben Modul-/Gene-Semantik kombinieren, aber **nie unmittelbar Kapital erhalten**. Hybrid kombiniert beide Modi; die Hive-Wahrscheinlichkeit ist evolvierbar.

Kapital ist KEIN Nebenprodukt der Bestäubung. In einem separaten, gen-/limitgebundenen reinen fund_local_seeds-Schritt kann eine externe Paper-Buchführung die lokalen Kinder durch exakte Decimal-Umbuchung aus der Mutter-Entitlement finanzieren, begrenzt durch reproduction_budget und bereits gebundene Margin. Globale Kapitalmenge bleibt unverändert, keine Live-Fills, Transfers oder Konto-/FX-Überbrückung. Jede Strategie-/Genom-ID ist eindeutig, Mutter/Vater/Biene/Tag ergeben nachvollziehbare Lineage und Seed-ID. Über mehrere Accounts muss ECO-05 später die Eigentümergrenzen zusätzlich prüfen.

Bienenhonig basiert ausschließlich auf nachträglich separat gelieferten unabhängig geprüften OOS-Evaluationen, niemals auf unvalidierter Trainingsfitness oder erzeugten Forschungskandidaten. Die Bewertungsfunktion wird in ECO-12 statistisch abgesichert.

Testkommando: PYTHONPATH=src python -m unittest discover -s tests/ecosystem -p test_bees.py -v.

**Nächster Schritt:** ECO-01/04 sichere Wire-Adapter zu den bereits implementierten TS-genLibrary-/Pydantic-Schema-Branches, dann die Bienen-Reduktion in ECO-03/16 phasenrichtig verbinden und nur kausal beobachtete Eltern/Fitnessdaten zulassen. Kein Live-Trading aktivieren.
