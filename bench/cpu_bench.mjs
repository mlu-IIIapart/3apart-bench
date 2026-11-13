// CPU benchmark in Node, one core: 3apart geometry vs the d3-force-3d physics layout.
// Usage: node bench/cpu_bench.mjs --who <name> [--machine <name>] [--plan quick|standard|full]
//   quick     a few seconds     (CI and a first look)
//   standard  about 2 minutes   (default; what most runs should use)
//   full      about 40 minutes  (the server protocol on the page: 100 runs per size, 10M markers, 100k physics)
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { NL, CAP, MARGIN, SEED, inputs, place, stats, outsideShare, timeOrder } from '../src/js/engine.mjs';
import { buildPhys, addMarkerAndResettle, TICKS } from '../src/js/physics.mjs';
import { ROOT, arg, machine, who, commit, save } from '../tools/machine.mjs';

const PLANS = {
  quick:    { geo: [[1e3, 50], [1e4, 50], [1e5, 30]], arrival: [20, 1e5], tick: [[1e3, 20]], layout: [[1e3, 1, 1]] },
  standard: { geo: [[1e3, 100], [1e4, 100], [1e5, 100], [1e6, 20]], arrival: [100, 1e6], tick: [[1e3, 50], [1e4, 10]], layout: [[1e3, 3, 1], [1e4, 1, 1]] },
  full:     { geo: [[1e3, 100], [1e4, 100], [1e5, 100], [1e6, 100], [1e7, 100]], arrival: [100, 1e6], tick: [[1e3, 100], [1e4, 100], [1e5, 100]], layout: [[1e3, 100, 100], [1e4, 10, 10], [1e5, 1, 1]] },
};

const person = who(), mach = machine(), planName = arg('plan', 'standard'), plan = PLANS[planName];
if (!plan) { console.error(`Unknown plan "${planName}". Use quick, standard or full.`); process.exit(2); }
const d3v = JSON.parse(fs.readFileSync(path.join(ROOT, 'node_modules/d3-force-3d/package.json'))).version;
const log = s => process.stderr.write(s + '\n');
const fmt = ms => ms < 1 ? `${(ms * 1000).toFixed(1)} µs` : ms < 1000 ? `${ms.toFixed(2)} ms` : `${(ms / 1000).toFixed(2)} s`;

const out = {
  schema: 1, kind: 'cpu-node', who: person, machine: mach, at: new Date().toISOString(), commit: commit(), plan: planName,
  versions: { node: process.version, d3_force_3d: d3v },
  protocol: { seed: SEED, bands: NL, cap: CAP, margin: MARGIN, physics_ticks: TICKS },
  geo: {}, arrival_ns: null, physics: {},
};

log(`3apart CPU benchmark · ${mach.name} · ${mach.cpu} · plan ${planName}`);

// Geometry: one placement per run, after warm-up.
for (const [n, runs] of plan.geo) {
  const d = inputs(n), xyz = new Float64Array(3 * n), ms = [];
  for (let w = 0; w < (n < 1e6 ? 20 : 2); w++) place(n, d.layer, d.t, d.v, xyz);
  for (let k = 0; k < runs; k++) { const t0 = performance.now(); place(n, d.layer, d.t, d.v, xyz); ms.push(performance.now() - t0); }
  const s = stats(ms);
  s.ns_per_marker = s.median * 1e6 / n;
  s.outside = outsideShare(n, i => d.layer[i], i => [xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]]);
  if (n <= 1e6) s.order = timeOrder(Int32Array.from({ length: n }, (_, i) => i).sort((p, q) => d.layer[p] - d.layer[q] || d.t[p] - d.t[q]), d.layer, i => [xyz[3 * i], xyz[3 * i + 1]]);
  out.geo[n] = s;
  log(`  geometry   ${String(n).padStart(9)} markers  median ${fmt(s.median)}  (${s.ns_per_marker.toFixed(0)} ns/marker, ${s.outside} outside band)`);
}

// One new marker among 100,000: each run is the mean over many single-marker placements.
{
  const [runs, per] = plan.arrival, n = 1e5, d = inputs(n), xyz = new Float64Array(3 * n), us = [];
  for (let k = 0; k < runs; k++) {
    const t0 = performance.now();
    for (let j = 0; j < per; j++) place(1, d.layer, d.t, d.v, xyz, j % n);
    us.push((performance.now() - t0) * 1000 / per);
  }
  const s = stats(us);
  out.arrival_ns = { runs, per_run: per, median: s.median * 1000, min: s.min * 1000, max: s.max * 1000 };
  log(`  one new marker among 100,000: ${(s.median * 1000).toFixed(0)} ns (median of ${runs} runs × ${per.toLocaleString()} arrivals)`);
}

// Physics: one tick, a full layout (build + 300 ticks), and re-settling after one new marker.
for (const [n, runs] of plan.tick) {
  const b = buildPhys(inputs(n), 7); b.sim.tick(10); const ms = [];
  for (let k = 0; k < runs; k++) { b.sim.alpha(0.5); const t0 = performance.now(); b.sim.tick(); ms.push(performance.now() - t0); }
  (out.physics[n] ||= {}).tick = stats(ms);
  log(`  physics    ${String(n).padStart(9)} markers  one tick ${fmt(out.physics[n].tick.median)}`);
}
for (const [n, runs, arrivals] of plan.layout) {
  const d = inputs(n), ms = [], outside = [], arr = []; let order = null, ticks = null;
  for (let k = 0; k < runs; k++) {
    let t0 = performance.now(); const b = buildPhys(d, 11 + k); b.sim.tick(TICKS); ms.push(performance.now() - t0);
    outside.push(outsideShare(n, i => b.nodes[i].layer, i => [b.nodes[i].x, b.nodes[i].y, b.nodes[i].z]));
    if (k === 0) order = timeOrder(b.ord, d.layer, i => [b.nodes[i].x, b.nodes[i].y]);
    if (k < arrivals) { t0 = performance.now(); ticks = addMarkerAndResettle(b, n, 5 + k); arr.push(performance.now() - t0); }
    log(`  physics    ${String(n).padStart(9)} markers  layout ${k + 1}/${runs}: ${fmt(ms[k])}`);
  }
  Object.assign(out.physics[n] ||= {}, { layout: stats(ms), outside: stats(outside), order, arrival: arr.length ? stats(arr) : null, arrival_ticks: ticks });
}

const g = out.geo[1e5], p = out.physics[1e4] || out.physics[1e3];
if (g && p && p.layout) {
  const n = out.physics[1e4] ? 1e4 : 1e3;
  log(`\n  geometry 100,000: ${fmt(g.median)}   physics layout ${n.toLocaleString()}: ${fmt(p.layout.median)}`);
}
log(`  saved ${save(person, mach, 'node', out)}`);
