# ECO-14b — Keine Pflanze verschwindet wegen eines fehlenden 2D-Genpaares

**Standalone, rein lesendes Teilpaket von Agent F.** Basis: ECO-14 [PR #41](https://github.com/MartinLiteAndFog/quant/pull/41); dieses Paket ergänzt nur `apps/fleet-desktop/src/lib/meadowCatalogProposals.ts`. Es erfordert weder lokale SOL-Parquets noch TradingView-Alert-Daten und verändert keine Quell-/Tageskapital-Buchung.

Eine Wiese ist eine **echte 2D-Kachel**, deren Achsen zwei explizit definierte, strategisch relevante **numerische** Genparameter derselben Strategie-/Modulversion sind. Eine unbekannte Pflanzenstrategie, die bereits eine kompatible Wiese hat, bekommt **nur** einen `existingTile`-Hinweis. Für fehlende passende Kategorien erstellt `proposeMissingMeadows(world,approvedCatalog)` lediglich **Vorschläge**; es erfindet niemals aus zwei zufälligen Zahlen die angeblich sinnvollste Strategie. Die Pair-Catalog-Einträge enthalten Modul-ID, Modulversion, Pair-ID, X/Y-Gen-Specs und eine separate Quelle `approvedStrategySpecId`, die von einem späteren Menschen/Contract-Prozess zu prüfen ist.

## Genaues Verhalten

1. Nur `homeTileId=null` wird betrachtet. Pflanzen mit vorhandenem gültigen Zuhause bleiben unverändert. Kein `homeTileId` wird automatisch versetzt, keine Performance oder Finanzdaten verändert.
2. Für bereits vorhandene kompatible Wiesen werden die IDs in `existingMatches` berichtet. Ein passender Sekundärblickpunkt erzeugt **keine** zweite finanzierte Pflanze.
3. Für Pflanzen ohne passende Kategorie werden genehmigt registrierte, nach `moduleId/moduleVersion/pairId` stabile Zweigen-Paare geprüft. Es müssen **beide** Achs-Gene im Strategie-Genome als finite Zahlen in ihren wirklichen, nicht extrapolierten Grenzen existieren. Bekannte rein optische oder Policy-/Finanzgene wie `visual_color`, `root_depth`, `max_leverage`, `reproduction_budget` oder `protective_stop` sind ausdrücklich ausgeschlossen.
4. Mehrere Pflanzen gleichen Asset-/Modul-/Genpaar-Typs werden zu **einem** Vorschlag zusammengefasst, selbst bei vertauschter Eingabereihenfolge. Gleiche Genpaare auf BTC, SOL oder Öl erzeugen separate Tile-Vorschläge **für das eigene Asset und seine Asset-Class-Region**.
5. Visualisierungsvorschläge bekommen stabile freie Rasterkoordinaten nur innerhalb eines 4×4-Regionsbereichs. Das Modul prüft auch *globale* Weltkoordinaten gegen belegte andere Regionen. Wenn eine Region voll ist, werden die Strategie-IDs ausdrücklich als `no_free_region_slot` gemeldet – niemals heimlich überzeichnet.
6. Für nicht registrierte/ungeeignete Gene folgt `no_registered_two_gene_pair`; das ist ein sichtbarer Forschungsbedarf, keine stillschweigende Entfernung von Trades/Pflanzen.
7. Jeder Vorschlag ist `proposalOnly=true`, `createsFundedPlants=false` und `transfersCapital=false`. Das eigentliche Hinzufügen in die Welt, die Gene-Registry-Freigabe, die Kapitalverteilung und ein späterer dynamischer World-Snapshot gehören **nicht** zu diesem datenunabhängigen Teilpaket und warten auf die Schnittstellen von Agent B.

## Tests

```bash
cd apps/fleet-desktop
node --experimental-strip-types tests/meadowCatalogProposals.smoke.mjs
node --experimental-strip-types tests/tileWorld.smoke.mjs
npm run build
```

Prüft zwei ungeordnete neue SOL-Pflanzen in **einer** neuen Kategorie, bereits kompatible bestehende Wiese ohne neues Kapital, eine unterausgestattete Forschungsstrategie mit nur einem Gen als ungelöst, getrennte Ölregion, doppelten Katalog, verbotene reine Policy-/optische Gene, gefüllte 16-Raster-Region und deterministische Eingabereihenfolge. **Reine statische Tests**, keine echte Wiesen-, Order-, PnL-, Unreal- oder Marktdatenverbindung.

## Nächster echter Agent-F-Schritt

Nach Agent B: den versionierten ECO-01-/Cloudgarden-Contract über einen **read-only** Adapter an `TileWorld` binden; `approvedStrategySpecId` gegen die konkrete Modulversion und strategische Parameterbeschreibung verifizieren, und erst dann Vorschläge im UI anzeigen und nach Zustimmung zu gültigen Regionen hinzufügen. Die Umwelt-/Kapital-/Genompositionen dürfen beim Umschalten von Kacheln nicht verändert werden. **Rendering niemals als Quelle für neue Handelsentscheidungen oder Geldvermehrung nutzen.**
