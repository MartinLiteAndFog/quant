import assert from "node:assert/strict";
import {
  randomVisualDNA, randomFunctionalMorphology,
  breedVisualDNA, breedFunctionalMorphology,
  classifyMorphotype, potentialLightArea, validateFunctionalMorphology,
  validateVisualDNA, visualColor, VISUAL_PRESETS,
} from "../src/lib/plantAppearance.ts";

function seeded(seed) {
  let x=seed>>>0;
  return () => {
    x+=0x6d2b79f5;
    let t=x;
    t=Math.imul(t^(t>>>15),t|1);
    t^=t+Math.imul(t^(t>>>7),t|61);
    return ((t^(t>>>14))>>>0)/4294967296;
  };
}
const a=VISUAL_PRESETS.map(p=>randomVisualDNA(seeded(100+VISUAL_PRESETS.indexOf(p)),p));
const b=VISUAL_PRESETS.map(p=>randomVisualDNA(seeded(100+VISUAL_PRESETS.indexOf(p)),p));
assert.deepEqual(a,b,"same seed -> identical appearance");
assert.notEqual(a[0].hue,a[1].hue,"different plants retain visual individuality");
assert.match(visualColor(a[0]), /^hsl\(/);
assert.equal(classifyMorphotype(randomFunctionalMorphology(seeded(12),"cactus")),"cactus");
assert.equal(classifyMorphotype(randomFunctionalMorphology(seeded(13),"shrub")),"shrub");

for (let k=0;k<500;k++) {
  const rand=seeded(13+k);
  const left=randomVisualDNA(rand,"flower"),right=randomVisualDNA(rand,"cactus");
  const child=breedVisualDNA(left,right,rand);
  const morphA=randomFunctionalMorphology(rand,"cactus");
  const morphB=randomFunctionalMorphology(rand,"shrub");
  const morphChild=breedFunctionalMorphology(morphA,morphB,rand);
  assert.equal(validateVisualDNA(child),child);
  assert.equal(validateFunctionalMorphology(morphChild),morphChild);
  assert.ok(morphChild.leaf_count>=1);
  assert.ok(child.petalCount>=3&&child.petalCount<=12);
  assert.ok(child.hue>=0&&child.hue<360);
}
const cactus={...randomFunctionalMorphology(seeded(30),"cactus"),leaf_count:1,leaf_size:0,
  photosynthetic_rind:true,rind_efficiency:0.7};
const naked={...cactus,photosynthetic_rind:false};
assert.ok(potentialLightArea(cactus,1)>0,"green stem doubles as photosynthetic leaf");
assert.equal(potentialLightArea(naked,1),0,"bark armor alone creates no light");
assert.equal(cactus.bark_strength,naked.bark_strength);
assert.throws(()=>validateFunctionalMorphology({...cactus,leaf_count:0}),/Physiologie/);
assert.throws(()=>validateVisualDNA({...a[0],hue:400}),/optische/);
assert.throws(()=>potentialLightArea(cactus,Number.NaN),/Biomasse/);
console.log("Visual DNA: seeded reproducibility, inheritance and one-leaf cactus physiology OK");
