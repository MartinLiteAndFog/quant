import test from "node:test";
import assert from "node:assert/strict";
import { KRAKEN_STARTER_PROFILES, getKrakenProfile } from "./krakenProfiles.ts";
test("exact whitelisted Kraken17.2 source settings and distinct starter identities", () => {
  const p = getKrakenProfile("kraken_17_2_close"), r = getKrakenProfile("kraken_17_2_reverse");
  assert.equal(p.channelLookback, Math.round(p.originalSensitivity * 10)); assert.equal(p.channelLookback, 172);
  assert.deepEqual([p.ttpTrailPct,p.partialTpPct,p.partialClosePct,p.ctTp1Pct,p.ctTp1ClosePct,p.hardSlPct], [.5,2.8,50,3.15,95,2.5]);
  assert.deepEqual({ ...r, id:p.id,name:p.name,sourceFiles:p.sourceFiles,reverseWaitSl:false },p);
  assert.deepEqual(KRAKEN_STARTER_PROFILES.map(p=>p.id),["kraken_legacy","countervariante","kraken_17_2_close","kraken_17_2_reverse","imbatp"]);
  assert.equal(getKrakenProfile("imbatp").mode,"TF"); assert.equal(getKrakenProfile("imbatp").tfTp1Pct,3.2);
  assert.equal(p.sourceSizing.leverage,10); assert.equal(p.paperOverrides.maxNotionalMultiple,1);
  assert.doesNotMatch(JSON.stringify(KRAKEN_STARTER_PROFILES), /webhook|token|https?:|alert_id/i);
  assert.throws(()=>getKrakenProfile("missing"),/Unknown/);
});
