import assert from 'node:assert/strict';
import {DEFAULT_GENES, BEE_BREEDS, normalizeGenome, breedGenomes, migrateLibrary, validateLibrary, DEFAULT_COMPLEXITY_PENALTY_WEIGHT} from '../src/lib/geneLibrary.ts';
const legacy = DEFAULT_GENES.filter(g => !g.policy);
const migrated = migrateLibrary(legacy);
assert.equal(migrated.length, DEFAULT_GENES.length);
assert.deepEqual(migrateLibrary(migrated), migrated);
assert.equal(DEFAULT_COMPLEXITY_PENALTY_WEIGHT, 0);
const genes = normalizeGenome({sensitivity:17.2,ttp:0.65}, migrated);
assert.equal(genes.protective_stop,true);
assert.throws(()=>normalizeGenome({...genes,protective_stop:false},migrated));
assert.throws(()=>normalizeGenome({...genes,sensitivity:NaN},migrated));
assert.throws(()=>normalizeGenome({...genes,adx_gate:1},migrated));
assert.throws(()=>normalizeGenome({...genes,evil_code:1},migrated));
for (const protectedGene of DEFAULT_GENES.filter(g=>g.required)) {
  assert.throws(()=>validateLibrary(migrated.filter(g=>g.id!==protectedGene.id)));
}
assert.throws(()=>validateLibrary(migrated.map(g=>g.id==='protective_stop'?{...g,required:false}:g)));
function seeded(seed) { return () => { seed=(Math.imul(seed,1664525)+1013904223)>>>0; return seed/2**32; }; }
for (let seed=0;seed<100;seed++) for (const bee of BEE_BREEDS) {
  const parent={...genes,stop_loss_pct:1.8,rsi_length:14};
  const child=breedGenomes(parent,parent,migrated,bee.id,seeded(seed));
  assert.deepEqual(child,breedGenomes(parent,parent,migrated,bee.id,seeded(seed)));
  assert.equal(child.protective_stop,true);
  for (const def of migrated.filter(g=>g.required)) {
    assert.ok(Object.hasOwn(child,def.id));
    if(def.kind==='number') assert.ok(child[def.id]>=def.min && child[def.id]<=def.max);
  }
}
const pruned=breedGenomes({...genes,stop_loss_pct:1.8},genes,migrated,'pruner',()=>0.01);
assert.ok(!Object.hasOwn(pruned,'stop_loss_pct'));
assert.equal(pruned.protective_stop,true);
assert.throws(()=>breedGenomes(genes,genes,migrated,'mixer',()=>NaN));
console.log('Protected genes: legacy migration, 500 deterministic offspring, fail-closed values OK');
