import { KRAKEN_STARTER_PROFILES } from "./krakenProfiles.ts";

export type GeneCandidate = { name: string; source: string; parameters: Record<string, unknown> };
export type GeneEntry = GeneCandidate & { id: string; createdAt: string; notes: string; snippet: string };
export type GeneLibraryStore = { schemaVersion: 1; entries: GeneEntry[] };
export type CuratedGeneModule = GeneCandidate & { id: string; category: string; description: string; snippet: string; caveat: string; license: string };
export const GENE_LIBRARY_STORAGE_KEY = "cloudgarden.gene-library.v1";
export const GENE_LIBRARY_LIMITS = { entries: 100, name: 120, source: 500, notes: 4000, snippet: 20000, bytes: 1200000 } as const;
export type GeneCausality = { inputs: string; lookback: string; confirmationLag: string; observableAt: string; executionAt: string; retrospectivePlot: string; verification: "tested" | "declarative"; futureRequired: false };
export type AtomicGene = CuratedGeneModule & { causality: GeneCausality; kind: "signal"|"confirmation"|"exit"|"risk"|"function" };
const causal = (inputs:string,lookback:string,lag="0 abgeschlossene Kerzen",verification:GeneCausality["verification"]="tested"):GeneCausality => ({inputs,lookback,confirmationLag:lag,observableAt:"Erst am Schluss der letzten benötigten Bestätigungskerze",executionAt:"Frühestens am folgenden beobachteten Minuten-Open",retrospectivePlot:"Ein historischer Markierungspunkt ist niemals der Verfügbarkeitszeitpunkt.",verification,futureRequired:false});
const atom = (id:string,name:string,kind:AtomicGene["kind"],description:string,parameters:Record<string,unknown>,causality:GeneCausality):AtomicGene => ({id,name,kind,category:kind,description,parameters,causality,source:kind==="function"?"Cloudgarden · deklarativer Verknüpfungsentwurf":"tradingview/sol_imba_strategy.pine",snippet:description,caveat:kind==="function"?"Als Verknüpfung dokumentiert und validiert; noch kein frei ausführbarer Strategiegraph.":"Wirksam nur in ausdrücklich implementierten und getesteten Strategierezepten; keine TradingView-Paritätszusage.",license:"Lokale Quellregeln; Herkunft erhalten."});
export const CURATED_GENE_MODULES: readonly AtomicGene[] = [
 atom("imba_channel","IMBA-Signal","signal","Abgeschlossene Renko-Proxy-Beobachtungen wechseln die Kanalrichtung.",{lookback:172,upper:.236,lower:.786},causal("Abgeschlossene 5-Minuten-Kerzen","20–500 bestätigte Renko-Bars")),
 {...atom("ew_pivots","EW-Signal · bestätigter Swing","signal","Ein Pivot wird erst nach seiner rechten Bestätigung verfügbar; niemals am historischen Pivot handeln.",{pivotLength:13,atrLength:14,minMoveAtr:3},causal("Abgeschlossene Hochs/Tiefs und ATR","2 × Pivotlänge + ATR","pivotLength rechte Kerzen")),source:"EW_v3_2_MTF_AUDIT.pine · krakenEw.ts",license:"MPL-2.0 · Elliott Wave Advanced Multi-Degree Analyzer v3.2 MTF Edition"},
 atom("ew_confirmation","EW-Bestätigung","confirmation","Ein Einstieg ist nur erlaubt, wenn die bereits bestätigte EW-Richtung übereinstimmt.",{required:true},causal("Verfügbares Einstiegssignal und bestätigter EW-Bias","Letzter bestätigter Swing","Verzögerung des EW-Signals")),
 atom("ttp","Trailing Take Profit","exit","Verfolge den günstigsten bereits abgeschlossenen Schlusskurs und löse bei Rücklauf aus.",{trailPct:.5},causal("Position und abgeschlossene Schlusskurse","Seit Einstieg")),
 atom("stop_loss","Stop-Loss · verpflichtend","risk","Ein Schutzstopp ist in jeder Strategie zwingend vorhanden und darf nicht durch Bedingungen abgeschaltet werden.",{maxDistancePct:2.5,mandatory:true},causal("Einstieg und bereits scharfgestellter Schutzstopp","Aktuelle Position")),
 atom("partial_tp","Teilgewinn","exit","Reduziere die bestehende Position bei bestätigtem Gewinnziel.",{targetPct:2.8,closePct:50},causal("Position und abgeschlossene OHLC-Kerze","Seit Einstieg")),
 atom("risk_cap","Kapitalgrenze 1×","risk","Begrenze Positionsnotional auf verfügbares Budget; notwendige Reduktionen sind eigene Ausführungen.",{maximumMultiple:1},causal("Vorab zugewiesenes Budget und beobachteter Open","Vor Handelstag eingefrorenes Budget")),
 ...(["and","or","not","threshold","cooldown"] as const).map(id=>atom(id,({and:"UND",or:"ODER",not:"NICHT",threshold:"Nur ab Schwelle",cooldown:"Wartezeit"})[id],"function",({and:"Alle verfügbaren Eingänge müssen wahr sein.",or:"Mindestens ein verfügbarer Eingang muss wahr sein.",not:"Negiere einen verfügbaren Eingang; ein fehlender Wert bleibt unbekannt.",threshold:"Prüfe eine Schwelle ausschließlich gegen bereits beobachtete Werte.",cooldown:"Erlaube eine Entscheidung erst nach einer vorab festgelegten vergangenen Wartezeit."})[id],{},causal("Nur bereits verfügbare Eingangswerte","Kein Zukunftszugriff; Verfügbarkeit = spätester Eingang","Abhängig von Eingängen","declarative"))),
];
export type StrategyGeneNode = { id:string; geneId:string; inputs:readonly string[] };
export type StrategyRecipe = { id:string; name:string; nodes:readonly StrategyGeneNode[]; entryNode:string; stopNode:string; execution:"compiled_profile"|"design_only" };
const recipe = (id:string,name:string,ew:boolean,execution:StrategyRecipe["execution"]):StrategyRecipe => ({id,name,execution,nodes:[{id:"entry",geneId:ew?"ew_pivots":"imba_channel",inputs:[]},{id:"ttp",geneId:"ttp",inputs:["entry"]},{id:"stop",geneId:"stop_loss",inputs:[]},{id:"cap",geneId:"risk_cap",inputs:[]}],entryNode:"entry",stopNode:"stop"});
export const STRATEGY_RECIPES:readonly StrategyRecipe[] = [
 ...KRAKEN_STARTER_PROFILES.map(p=>({...recipe(p.id,p.name,false,"compiled_profile"),nodes:[...recipe(p.id,p.name,false,"compiled_profile").nodes,{id:"partial",geneId:"partial_tp",inputs:["entry"]}]})),
 recipe("ew_ttp","EW-Einstieg + TTP + Schutzstopp",true,"design_only"),
 {id:"imba_ew_confirmation",name:"IMBA + EW-Bestätigung + TTP + Schutzstopp",execution:"design_only",nodes:[{id:"imba",geneId:"imba_channel",inputs:[]},{id:"ew",geneId:"ew_pivots",inputs:[]},{id:"confirmation",geneId:"ew_confirmation",inputs:["imba","ew"]},{id:"entry",geneId:"and",inputs:["imba","confirmation"]},{id:"ttp",geneId:"ttp",inputs:["entry"]},{id:"stop",geneId:"stop_loss",inputs:[]}],entryNode:"entry",stopNode:"stop"},
];
/** Validate graph topology and compulsory risk independently from execution support. No arbitrary code is evaluated. */
export function validateStrategyRecipe(recipe:StrategyRecipe,genes:readonly AtomicGene[]=CURATED_GENE_MODULES):void {
 if(!recipe||!Array.isArray(recipe.nodes)||!recipe.nodes.length||recipe.nodes.length>64)throw new RangeError("Ungültiger Strategiegraph.");
 const definitions=new Map(genes.map(g=>[g.id,g])),nodes=new Map(recipe.nodes.map(n=>[n.id,n]));
 if(nodes.size!==recipe.nodes.length||!nodes.has(recipe.entryNode)||nodes.get(recipe.stopNode)?.geneId!=="stop_loss"||nodes.get(recipe.stopNode)!.inputs.length)throw new RangeError("Unbedingter Stop-Loss und Einstieg sind verpflichtend.");
 const visiting=new Set<string>(),done=new Set<string>();
 const visit=(id:string)=>{if(done.has(id))return;if(visiting.has(id))throw new RangeError("Zyklus im Strategiegraph.");const n=nodes.get(id),g=n&&definitions.get(n.geneId);if(!n||!g||g.causality.futureRequired!==false)throw new RangeError("Unbekanntes oder zukunftsabhängiges Gen.");if(!Array.isArray(n.inputs)||n.inputs.length>16)throw new RangeError("Ungültige Geneingänge.");if(g.id==="not"&&n.inputs.length!==1)throw new RangeError("NICHT benötigt genau einen Eingang.");if(["and","or"].includes(g.id)&&n.inputs.length<2)throw new RangeError("UND/ODER benötigen mindestens zwei Eingänge.");visiting.add(id);n.inputs.forEach(visit);visiting.delete(id);done.add(id);};
 recipe.nodes.forEach(n=>visit(n.id));
}

function boundedText(value: unknown, max: number, field: string, allowEmpty = false): asserts value is string {
  if (typeof value !== "string" || value.length > max || (!allowEmpty && !value.trim())) throw new RangeError(`${field}: ungültige Länge.`);
}
function validateParameters(parameters: unknown): asserts parameters is Record<string, unknown> {
  if (!parameters || typeof parameters !== "object" || Array.isArray(parameters)) throw new RangeError("Parameter müssen ein JSON-Objekt sein.");
  let count = 0;
  const visit = (value: unknown, depth: number): void => {
    if (++count > 2000 || depth > 6) throw new RangeError("Parameterstruktur ist zu groß.");
    if (value === null || typeof value === "boolean") return;
    if (typeof value === "number" && Number.isFinite(value)) return;
    if (typeof value === "string" && value.length <= 4000) return;
    if (Array.isArray(value) && value.length <= 200) { value.forEach((item) => visit(item, depth + 1)); return; }
    if (typeof value === "object" && value && Object.getPrototypeOf(value) === Object.prototype && Object.keys(value).length <= 100) {
      for (const [key, item] of Object.entries(value)) {
        if (key.length > 120 || ["__proto__", "prototype", "constructor"].includes(key)) throw new RangeError("Ungültiger Parameterschlüssel.");
        visit(item, depth + 1);
      }
      return;
    }
    throw new RangeError("Parameter dürfen nur endliche JSON-Werte enthalten.");
  };
  visit(parameters, 0);
}
export function parseGeneLibrary(raw: string): GeneLibraryStore {
  if (typeof raw !== "string" || raw.length > GENE_LIBRARY_LIMITS.bytes) throw new RangeError("Die Genbibliothek ist zu groß.");
  const store = JSON.parse(raw) as GeneLibraryStore;
  if (!store || store.schemaVersion !== 1 || !Array.isArray(store.entries) || store.entries.length > GENE_LIBRARY_LIMITS.entries) throw new RangeError("Unbekanntes oder beschädigtes Bibliotheksformat.");
  const ids = new Set<string>();
  for (const entry of store.entries) {
    if (!entry || typeof entry !== "object") throw new RangeError("Ungültiger Geneintrag.");
    boundedText(entry.id, 160, "ID"); boundedText(entry.name, GENE_LIBRARY_LIMITS.name, "Name");
    boundedText(entry.source, GENE_LIBRARY_LIMITS.source, "Quelle"); boundedText(entry.notes, GENE_LIBRARY_LIMITS.notes, "Notizen", true);
    boundedText(entry.snippet, GENE_LIBRARY_LIMITS.snippet, "Snippet", true);
    if (ids.has(entry.id) || typeof entry.createdAt !== "string" || !Number.isFinite(Date.parse(entry.createdAt))) throw new RangeError("Doppelte ID oder ungültiges Datum.");
    ids.add(entry.id); validateParameters(entry.parameters);
  }
  // Clone a whitelist; unexpected imported fields never become application data.
  return { schemaVersion: 1, entries: store.entries.map(({ id, name, source, parameters, notes, snippet, createdAt }) =>
    ({ id, name, source, parameters: JSON.parse(JSON.stringify(parameters)), notes, snippet, createdAt })) };
}
export function addGeneEntry(library: GeneLibraryStore, entry: GeneEntry): GeneLibraryStore {
  validateParameters(entry.parameters);
  return parseGeneLibrary(JSON.stringify({ schemaVersion: 1, entries: [entry, ...library.entries] }));
}
export function loadGeneLibrary(storage: Pick<Storage, "getItem">): { library: GeneLibraryStore; error: string | null } {
  try {
    const raw = storage.getItem(GENE_LIBRARY_STORAGE_KEY);
    return { library: raw === null ? { schemaVersion: 1, entries: [] } : parseGeneLibrary(raw), error: null };
  } catch (error) {
    return { library: { schemaVersion: 1, entries: [] }, error: `Gespeicherte Bibliothek konnte nicht geladen werden: ${error instanceof Error ? error.message : "Speicher nicht verfügbar."}` };
  }
}
export function saveGeneLibrary(storage: Pick<Storage, "setItem">, library: GeneLibraryStore): void {
  const validated = parseGeneLibrary(JSON.stringify(library));
  storage.setItem(GENE_LIBRARY_STORAGE_KEY, JSON.stringify(validated));
}
