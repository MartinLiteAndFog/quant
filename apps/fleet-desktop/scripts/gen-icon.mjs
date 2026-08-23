// Fleet's canonical app-icon source is the existing transparent pixel robot.
// Keep this helper deterministic so regenerating Tauri assets can never
// silently replace the product branding with an unrelated generated symbol.
import { copyFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = fileURLToPath(
  new URL("../src/assets/fleet-robot-logo-transparent-v2.png", import.meta.url),
);
const out = process.argv[2] || "src-tauri/icons/source-robot.png";

copyFileSync(source, out);
console.log(`copied Fleet robot icon to ${out}`);
