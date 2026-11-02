import test from "node:test";
import assert from "node:assert/strict";
import { CURATED_GENE_MODULES, GENE_LIBRARY_STORAGE_KEY, addGeneEntry, loadGeneLibrary, parseGeneLibrary, saveGeneLibrary, STRATEGY_RECIPES, validateStrategyRecipe, type GeneEntry } from "./geneLibrary.ts";
const entry: GeneEntry = { id:"my-source",name:"Mein Kraken-Modul",source:"Lokale Quelle",parameters:{lookback:172,enabled:true,nested:{ttp:.5}},notes:"Eigene Beobachtung",snippet:"plain <script>alert('not executed')</script>",createdAt:"2026-11-02T12:00:00.000Z" };
function memory() { const data=new Map<string,string>();return { data,getItem:(key:string)=>data.get(key)??null,setItem:(key:string,value:string)=>{data.set(key,value);} }; }
test("gene library persists exact bounded data and snippets remain plain text",()=>{
 const storage=memory(),library=addGeneEntry({schemaVersion:1,entries:[]},entry);saveGeneLibrary(storage,library);assert.deepEqual(loadGeneLibrary(storage),{library,error:null});assert.equal(loadGeneLibrary(storage).library.entries[0]!.snippet,entry.snippet);assert.equal(library.entries[0]!.parameters.lookback,172);assert.ok(storage.data.has(GENE_LIBRARY_STORAGE_KEY));
});
test("invalid or unavailable saved libraries are reported and preserved",()=>{
 const storage=memory();storage.setItem(GENE_LIBRARY_STORAGE_KEY,'{"schemaVersion":9,"entries":[]}');const before=storage.getItem(GENE_LIBRARY_STORAGE_KEY);assert.ok(loadGeneLibrary(storage).error);assert.equal(storage.getItem(GENE_LIBRARY_STORAGE_KEY),before);assert.ok(loadGeneLibrary({getItem(){throw new Error("disabled");}}).error);assert.throws(()=>saveGeneLibrary({setItem(){throw new Error("quota");}},{schemaVersion:1,entries:[]}),/quota/);
});
test("import validation rejects duplicates, oversized snippets, nonfinite and prototype keys",()=>{
 assert.throws(()=>parseGeneLibrary(JSON.stringify({schemaVersion:1,entries:[entry,entry]})),/Doppelte/);
 assert.throws(()=>addGeneEntry({schemaVersion:1,entries:[]},{...entry,snippet:"x".repeat(20001)}),/Snippet/);
 assert.throws(()=>parseGeneLibrary('{"schemaVersion":1,"entries":['+JSON.stringify(entry).replace('"lookback":172','"__proto__":{"polluted":true}')+']}'),/Parameterschlüssel/);
 assert.throws(()=>addGeneEntry({schemaVersion:1,entries:[]},{...entry,parameters:{value:()=>42}}),/JSON|Parameter|length/i);
});
test("unknown imported fields are discarded and curated modules retain exact source and MPL provenance",()=>{
 const r=parseGeneLibrary(JSON.stringify({schemaVersion:1,entries:[{...entry,unexpected:"discard"}]}));assert.equal("unexpected" in r.entries[0]!,false);const ew=CURATED_GENE_MODULES.find(x=>x.id==="ew_pivots")!;assert.match(ew.license,/MPL-2.0/);assert.ok(STRATEGY_RECIPES.find(x=>x.id==="kraken_17_2_close"));assert.ok(!CURATED_GENE_MODULES.find(x=>x.id==="kraken_17_2_close"));
});

test("recipe graph validates atomics and rejects absent mandatory stop and cycles",()=>{for(const r of STRATEGY_RECIPES)validateStrategyRecipe(r);const r=STRATEGY_RECIPES[0]!;assert.throws(()=>validateStrategyRecipe({...r,nodes:r.nodes.filter(n=>n.id!=="stop")}),/Stop/);assert.throws(()=>validateStrategyRecipe({...r,nodes:r.nodes.map(n=>n.id==="entry"?{...n,inputs:["ttp"]}:n)}),/Zyklus/);assert.throws(()=>validateStrategyRecipe({...r,nodes:r.nodes.map(n=>n.id==="entry"?{...n,geneId:"future_peak"}:n)}),/Unbekannt/);assert.match(CURATED_GENE_MODULES.find(g=>g.id==="ew_pivots")!.causality.confirmationLag,/rechte/);});
