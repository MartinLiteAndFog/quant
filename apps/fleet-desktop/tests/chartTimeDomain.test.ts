import assert from "node:assert/strict";
import test from "node:test";

import { asUnixSeconds, rangeWindowUnix } from "../src/lib/chartTimeDomain.ts";

test("asUnixSeconds leaves seconds alone and floors ms", () => {
  assert.equal(asUnixSeconds(1_700_000_000), 1_700_000_000);
  assert.equal(asUnixSeconds(1_700_000_000_500), 1_700_000_000);
});

test("24h window anchors to clock t1, not data span", () => {
  const t1 = 1_700_000_000;
  const win = rangeWindowUnix(24, t1 + 999, t1);
  assert.ok(win);
  assert.equal(win.t1, t1);
  assert.equal(win.t0, t1 - 24 * 3600);
});

test("7d and 30d use hour spans", () => {
  const t1 = 1_700_000_000;
  assert.equal(rangeWindowUnix(168, t1, t1)?.t0, t1 - 168 * 3600);
  assert.equal(rangeWindowUnix(720, t1, t1)?.t0, t1 - 720 * 3600);
});

test("all (hours 0) returns null so caller can use full history", () => {
  assert.equal(rangeWindowUnix(0, 1_700_000_000, 1_700_000_000), null);
});

test("without clock, window uses nowSec", () => {
  const now = 1_700_000_000;
  const win = rangeWindowUnix(24, now, null);
  assert.deepEqual(win, { t0: now - 24 * 3600, t1: now });
});
