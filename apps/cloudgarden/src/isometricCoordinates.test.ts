import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';

test('isometric planes preserve exact anchors and render 200 coincident specimens without hiding them', () => {
  // Isolate the CSS import hook in a child process: other contract tests retain ordinary module loading.
  const scene = new URL('./IsometricGarden.tsx', import.meta.url).href;
  const simulation = new URL('./simulation.ts', import.meta.url).href;
  const script = `
    import { registerHooks } from 'node:module';
    import assert from 'node:assert/strict';
    registerHooks({ load(url, context, next) {
      return url.endsWith('.css') ? { format: 'module', source: 'export default {};', shortCircuit: true } : next(url, context);
    }});
    const React = (await import('react')).default;
    const { renderToStaticMarkup } = await import('react-dom/server');
    const { createSimulationWorld } = await import(${JSON.stringify(simulation)});
    const { gardenPlaneCoordinates, IsometricGarden } = await import(${JSON.stringify(scene)});
    const world = createSimulationWorld();
    const plant = world.plants[0];
    assert.deepEqual(gardenPlaneCoordinates(plant, 'root-leaf'), {
      xValue: plant.genes.root, yValue: plant.genes.leaf, x: plant.genes.root * 420, y: plant.genes.leaf * 420,
    });
    assert.deepEqual(gardenPlaneCoordinates({ ...plant, genes: { ...plant.genes, water: 0, sun: 1 } }, 'water-sun'),
      { xValue: 0, yValue: 1, x: 0, y: 420 });
    assert.deepEqual(gardenPlaneCoordinates({ ...plant, tradingTraits: { lookbackBars: 120, entryThresholdBps: 1, exitThresholdBps: 0 } }, 'trading'),
      { xValue: 120, yValue: 1, x: 420, y: 0 });
    for (const [root, leaf] of [[0,0], [1,1], [0,1], [1,0], [.5,.5]]) {
      const active = Array.from({ length: 200 }, (_, index) => ({ ...plant,
        genes: { ...plant.genes, root, leaf }, id: 'plant-' + String(index).padStart(3, '0'), name: 'Plant ' + index,
      }));
      const dense = { ...world, plants: [...active, { ...plant, id: 'retired', status: 'archived', name: 'Archived specimen' }] };
      const markup = renderToStaticMarkup(React.createElement(IsometricGarden, {
        world: dense, selectedId: active[0].id, onSelect() {}, running: false, motionEnabled: false, capitalBaselineUsdt: 150,
      }));
      assert.equal((markup.match(/data-anchor-x=/g) || []).length, 200);
      assert.ok(!markup.includes('Archived specimen'));
      const anchors = [...markup.matchAll(/data-anchor-x="([^"]+)" data-anchor-y="([^"]+)"/g)];
      assert.ok(anchors.every((match) => Number(match[1]) === root * 420 && Number(match[2]) === leaf * 420));
      const crowns = [...markup.matchAll(/data-crown-center-x="([^"]+)" data-crown-center-y="([^"]+)"/g)]
        .map((match) => [Number(match[1]), Number(match[2])]);
      assert.equal(crowns.length, 200);
      assert.ok(crowns.every(([x,y]) => x >= -440 && x <= 440 && y >= -80 && y <= 500));
      for (let i = 0; i < crowns.length; i++) for (let j = 0; j < i; j++) {
        assert.ok(Math.hypot(crowns[i][0] - crowns[j][0], crowns[i][1] - crowns[j][1]) >= 47 * .42 - 1e-8);
      }
      assert.match(markup, /150.00 USDT Kapitalbasis/);
      assert.match(markup, /data-motion="off"/);
      assert.match(markup, /data-running="false"/);
    }
    console.log('exact-anchors-and-200-crowns-verified');
  `;
  const output = execFileSync(process.execPath, ['--import', 'tsx', '--input-type=module', '-e', script],
    { encoding: 'utf8', cwd: new URL('..', import.meta.url), timeout: 15_000 });
  assert.match(output, /exact-anchors-and-200-crowns-verified/);
});
