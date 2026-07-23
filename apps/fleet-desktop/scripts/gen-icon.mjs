// Renders the Fleet Cockpit "HUD reticle" app icon to a 1024px master PNG.
// Pure Node (analytic AA + zlib PNG encoder) — no image deps. The master is
// then fed to `tauri icon` to generate every platform size.
import { deflateSync } from "node:zlib";
import { writeFileSync } from "node:fs";

const S = 1024;
const C = S / 2;

// Palette (Fleet Cockpit HUD): warm charcoal tile, gold reticle.
const CHARCOAL = [0x15, 0x16, 0x1b];
const GOLD = [0xd4, 0xa8, 0x4b];
const GOLD_DIM = [0x8a, 0x6f, 0x36];

const clamp = (x, a, b) => (x < a ? a : x > b ? b : x);
const AA = 1.3; // ~1px edge softness

// coverage 1 inside, 0 outside, given a signed distance (neg = inside)
const cov = (sd) => clamp(0.5 - sd / AA, 0, 1);

function sdRoundRect(px, py, half, r) {
  const qx = Math.abs(px - C) - (half - r);
  const qy = Math.abs(py - C) - (half - r);
  const ox = Math.max(qx, 0);
  const oy = Math.max(qy, 0);
  return Math.hypot(ox, oy) + Math.min(Math.max(qx, qy), 0) - r;
}

function sdRing(d, R, hw) {
  return Math.abs(d - R) - hw;
}

// signed distance to a radial segment along axis angle `a`, from r0 to r1 (band)
function sdRadialTick(px, py, ang, r0, r1, hw) {
  // rotate point into tick-local frame (x along the axis)
  const dx = px - C;
  const dy = py - C;
  const cx = Math.cos(ang);
  const cy = Math.sin(ang);
  const u = dx * cx + dy * cy; // along axis
  const v = -dx * cy + dy * cx; // perpendicular
  const mid = (r0 + r1) / 2;
  const halfLen = (r1 - r0) / 2;
  const au = Math.abs(u - mid) - halfLen;
  const box = Math.hypot(Math.max(au, 0), Math.max(Math.abs(v) - hw, 0)) +
    Math.min(Math.max(au, Math.abs(v) - hw), 0);
  return box;
}

// gold coverage = union (max) of all reticle primitives at (px,py)
function goldCoverage(px, py, d) {
  let g = 0;
  const add = (sd, alpha = 1) => {
    g = Math.max(g, cov(sd) * alpha);
  };
  // outer main ring
  add(sdRing(d, 300, 10));
  // center dot
  add(d - 24);
  // inner gap-crosshair arms (N/E/S/W)
  for (let k = 0; k < 4; k++) {
    const ang = (k * Math.PI) / 2;
    add(sdRadialTick(px, py, ang, 70, 205, 9));
    // outer stubs crossing the ring
    add(sdRadialTick(px, py, ang, 272, 348, 10));
  }
  return g;
}

// dim gold: faint inner ring + 45deg minor ring ticks (instrument detail)
function goldDimCoverage(px, py, d) {
  let g = 0;
  const add = (sd, alpha = 1) => { g = Math.max(g, cov(sd) * alpha); };
  add(sdRing(d, 150, 3), 0.5); // faint inner ring
  for (let k = 0; k < 4; k++) {
    const ang = Math.PI / 4 + (k * Math.PI) / 2; // 45,135,225,315
    add(sdRadialTick(px, py, ang, 300, 330, 6), 0.7);
  }
  return g;
}

// RGBA buffer
const buf = Buffer.alloc(S * S * 4, 0);

function over(dst, off, rgb, a) {
  if (a <= 0) return;
  const da = dst[off + 3] / 255;
  const outA = a + da * (1 - a);
  for (let i = 0; i < 3; i++) {
    const s = rgb[i] / 255;
    const dcol = dst[off + i] / 255;
    const outC = (s * a + dcol * da * (1 - a)) / (outA || 1);
    dst[off + i] = Math.round(outC * 255);
  }
  dst[off + 3] = Math.round(outA * 255);
}

const tileHalf = S / 2; // full-bleed tile
const tileRadius = 190; // macOS-style rounded corners

for (let y = 0; y < S; y++) {
  for (let x = 0; x < S; x++) {
    const px = x + 0.5;
    const py = y + 0.5;
    const off = (y * S + x) * 4;
    const d = Math.hypot(px - C, py - C);

    // 1) charcoal rounded-rect tile
    over(buf, off, CHARCOAL, cov(sdRoundRect(px, py, tileHalf, tileRadius)));
    // 2) dim detail then bright reticle
    over(buf, off, GOLD_DIM, goldDimCoverage(px, py, d));
    over(buf, off, GOLD, goldCoverage(px, py, d));
  }
}

// ---- minimal PNG encoder (truecolor + alpha, 8-bit) ----
function png(width, height, rgba) {
  const raw = Buffer.alloc((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 4 + 1)] = 0; // filter: none
    rgba.copy(raw, y * (width * 4 + 1) + 1, y * width * 4, (y + 1) * width * 4);
  }
  const idat = deflateSync(raw, { level: 9 });

  const crcTable = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();
  const crc32 = (b) => {
    let c = 0xffffffff;
    for (let i = 0; i < b.length; i++) c = crcTable[(c ^ b[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length, 0);
    const t = Buffer.from(type, "ascii");
    const body = Buffer.concat([t, data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body), 0);
    return Buffer.concat([len, body, crc]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type RGBA
  ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", ihdr),
    chunk("IDAT", idat),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const out = process.argv[2] || "src-tauri/icons/source-reticle.png";
writeFileSync(out, png(S, S, buf));
console.log(`wrote ${out} (${S}x${S})`);
