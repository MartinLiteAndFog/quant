/**
 * CLI for cross-language ECO-01 <-> Cloudgarden golden tests.
 * Strictly reads a JSON snapshot from stdin and writes validated JSON to stdout.
 */
import { readFileSync } from "node:fs";
import { validateWire } from "./ecoWireBridge.ts";

const input = JSON.parse(readFileSync(0, "utf8")) as unknown;
process.stdout.write(JSON.stringify(validateWire(input)) + "\n");
