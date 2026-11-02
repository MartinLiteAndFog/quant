// Usage: from apps/cloudgarden: node --import tsx scripts/verify_tv_seed_reference.mjs /path/to/manifest.json
import fs from "node:fs";
import { auditTradingViewReferenceManifest } from "../src/tradingViewSeedCatalog.ts";
const input = process.argv[2];
if (!input) {
  console.error("Usage: node --import tsx scripts/verify_tv_seed_reference.mjs <reference_manifest.json>");
  process.exit(2);
}
try {
  const payload = JSON.parse(fs.readFileSync(input, "utf8"));
  const report = auditTradingViewReferenceManifest(payload);
  console.log(JSON.stringify(report, null, 2));
} catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
}
