import {build} from 'esbuild';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const here=dirname(fileURLToPath(import.meta.url));
await build({entryPoints:[resolve(here,'unreal_bridge.mjs')],outfile:resolve(process.argv[2]??'/tmp/cloudgarden-unreal-runtime/bridge.mjs'),bundle:true,platform:'node',format:'esm',target:'node22'});
