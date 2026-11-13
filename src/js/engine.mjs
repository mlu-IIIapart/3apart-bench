// 3apart shared engine (JavaScript). The single source of the protocol constants,
// the input generator and the reference placement. Every runner imports from here.
// Changing anything in this file changes the protocol: see PROTOCOL.md first.

export const NL = 24;            // number of bands
export const CAP = 0.9;          // band edges run from +CAP to -CAP on the z axis
export const MARGIN = 0.12;      // 12% margin inside each band
export const SEED = 20261112;    // fixed seed for all synthetic inputs
export const EDGES = Float64Array.from({ length: NL + 1 }, (_, k) => CAP * (1 - 2 * k / NL));

// mulberry32. State after k draws is seed + k * 0x6D2B79F5 (mod 2^32), so other
// languages can generate the same stream in parallel (see src/py/engine.py).
export function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0; let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Draw order per marker: band, time, value.
export function inputs(n, seed = SEED) {
  const r = rng(seed), layer = new Int32Array(n), t = new Float64Array(n), v = new Float64Array(n);
  for (let i = 0; i < n; i++) { layer[i] = Math.floor(r() * NL); t[i] = r(); v[i] = r(); }
  return { n, layer, t, v };
}

// The disclosure's place() + sphere_xyz(), line for line. Writes x,y,z per marker into out.
export function place(n, layer, t, value, out, off = 0) {
  for (let i = off; i < off + n; i++) {
    const phi = Math.PI * (2 * t[i] - 1), l = layer[i], zt = EDGES[l], zb = EDGES[l + 1];
    const z = zb + (zt - zb) * (MARGIN + (1 - 2 * MARGIN) * value[i]);
    const th = Math.acos(z < -1 ? -1 : z > 1 ? 1 : z), st = Math.sin(th);
    out[3 * i] = st * Math.sin(phi); out[3 * i + 1] = -st * Math.cos(phi); out[3 * i + 2] = Math.cos(th);
  }
}

export function stats(ms) {
  const a = Float64Array.from(ms).sort(), n = a.length, q = p => a[Math.min(n - 1, Math.ceil(p * n) - 1)];
  let s = 0; for (const x of a) s += x;
  return { runs: n, min: a[0], median: n % 2 ? a[n >> 1] : (a[n / 2 - 1] + a[n / 2]) / 2, mean: s / n, p95: q(0.95), max: a[n - 1] };
}

// Share of markers whose height on the sphere is outside their own band.
export function outsideShare(n, layerAt, xyzAt) {
  let k = 0;
  for (let i = 0; i < n; i++) {
    const [x, y, z] = xyzAt(i), r = Math.hypot(x, y, z) || 1, h = z / r, l = layerAt(i);
    if (!(h <= EDGES[l] && h >= EDGES[l + 1])) k++;
  }
  return k / n;
}

export function timeOrderIdx(d) {
  const ord = new Int32Array(d.n); for (let i = 0; i < d.n; i++) ord[i] = i;
  return ord.sort((p, q) => d.layer[p] - d.layer[q] || d.t[p] - d.t[q]);
}

// Share of time-consecutive pairs in a band that run the same way (counter-clockwise) around the globe.
export function timeOrder(ord, layer, xyAt) {
  let ok = 0, tot = 0;
  for (let k = 1; k < ord.length; k++) {
    const p = ord[k - 1], q = ord[k];
    if (layer[p] !== layer[q]) continue;
    tot++;
    const a = xyAt(p), b = xyAt(q), d = Math.atan2(b[1], b[0]) - Math.atan2(a[1], a[0]);
    if (Math.sin(d) > 0 || Math.abs(d) < 1e-12) ok++;
  }
  return tot ? ok / tot : 1;
}

// Binary inputs file shared by every engine (little-endian):
//   magic "3AIN" | uint32 version=1 | uint32 n | uint32 seed | int32[n] band | float64[n] time | float64[n] value
export function encodeInputs(d, seed = SEED) {
  const head = 16, buf = new ArrayBuffer(head + d.n * (4 + 8 + 8)), dv = new DataView(buf);
  [0x33, 0x41, 0x49, 0x4e].forEach((b, i) => dv.setUint8(i, b));
  dv.setUint32(4, 1, true); dv.setUint32(8, d.n, true); dv.setUint32(12, seed >>> 0, true);
  new Int32Array(buf, head, d.n).set(d.layer);
  const bytes = new Uint8Array(buf);
  bytes.set(new Uint8Array(d.t.buffer, d.t.byteOffset, 8 * d.n), head + 4 * d.n);
  bytes.set(new Uint8Array(d.v.buffer, d.v.byteOffset, 8 * d.n), head + 12 * d.n);
  return bytes;
}

export function decodeInputs(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  if (magic !== '3AIN') throw new Error('not a 3apart inputs file');
  const n = dv.getUint32(8, true), seed = dv.getUint32(12, true), head = 16;
  // copy into fresh, aligned buffers (Node's Buffer.slice returns a view, not a copy)
  const copy = (off, len) => { const c = new Uint8Array(len); c.set(u8.subarray(head + off, head + off + len)); return c.buffer; };
  return { n, seed, layer: new Int32Array(copy(0, 4 * n)), t: new Float64Array(copy(4 * n, 8 * n)), v: new Float64Array(copy(12 * n, 8 * n)) };
}
