import assert from "node:assert/strict";
import {
  DEFAULT_GENES, addGene, breedGenomes, defaultGenome, geneCount,
  growBiomass, normalizeGene, validateLibrary, visualPlantHeight, BEE_BREEDS,
} from "../src/lib/geneLibrary.ts";

// Schema, gene safety, parent/offspring inheritance; no API or exchange.
const lib = addGene(DEFAULT_GENES, {
  id:"volatility_window", label:"Volatility window", group:"Signal",
  kind:"number", min:2, max:40, step:2, defaultValue:14, required:false,
});
assert.equal(lib.length, DEFAULT_GENES.length + 1);
assert.equal(BEE_BREEDS.length, 5);
assert.throws(() => addGene(lib, lib[0]), /doppelt/);
assert.throws(() => validateLibrary(lib.filter(g=>g.id!=="sensitivity")), /Pflichtgen/);
assert.equal(normalizeGene(lib.at(-1), 19), 20);
const left = defaultGenome(lib,16,0.5);
const right = {...defaultGenome(lib,20,1.0), volatility_window:18};
const rand = () => 0.5;
const mixed = breedGenomes(left,right,lib,"mixer",rand);
assert.equal(mixed.sensitivity,18);
assert.equal(mixed.ttp,0.75);
assert.equal(mixed.volatility_window,18);
for (const bee of BEE_BREEDS) {
  const offspring = breedGenomes(left,right,lib,bee.id,rand);
  assert.ok(offspring.sensitivity >= 12 && offspring.sensitivity <= 25);
  assert.ok(offspring.ttp >= 0.25 && offspring.ttp <= 1.65);
  assert.ok(geneCount(offspring) >= 2);
}
// 5%/10% pruning never removes mandatory two coordinate genes.
const pruned = breedGenomes(left,right,lib,"pruner",()=>0.01);
assert.ok("sensitivity" in pruned && "ttp" in pruned);
assert.ok(!("stop_loss_pct" in pruned));
// Value growth is bounded per MONTH and smoothly logarithmic in the view.
assert.equal(growBiomass(1,0.36),1.36);
assert.equal(growBiomass(1,1.6),1.36);
assert.equal(growBiomass(1,-0.2),0.8);
assert.ok(growBiomass(growBiomass(1,-0.2),-0.2)>0);
assert.ok(visualPlantHeight(1e9)<40);
assert.ok(visualPlantHeight(2)>visualPlantHeight(1));
assert.throws(()=>growBiomass(1,Number.NaN),/Ungültig/);
console.log("Gene library: 5 bees, gene crossover/pruning, required genes and smooth growth OK");
