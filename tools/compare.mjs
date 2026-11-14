// Collect every result file under results/ into one comparison: printed, and written to results/SUMMARY.md.
// Usage: node tools/compare.mjs
import fs from 'node:fs';
import path from 'node:path';
import { ROOT } from './machine.mjs';
import { positive, measuredMedian, numpyXL, browserXL, browserCold, nativeXL, legacyGeometryLabel } from './result-scope.mjs';

const dir = path.join(ROOT, 'results');
const files = [];
(function walk(d) { for (const e of fs.readdirSync(d, { withFileTypes: true })) { const p = path.join(d, e.name); if (e.isDirectory()) walk(p); else if (e.name.endsWith('.json')) files.push(p); } })(dir);

const ms = v => !positive(v) ? '—' : v < 1 ? `${(v * 1000).toFixed(0)} µs` : v < 1000 ? `${v.toFixed(2)} ms` : v < 60000 ? `${(v / 1000).toFixed(2)} s` : `${(v / 60000).toFixed(1)} min`;
const med = measuredMedian;
const day = s => (s || '').slice(0, 10);
const who = r => r.who || '—';
const mname = r => (r.machine && r.machine.name) || '—';
const hw = r => { const m = r.machine || {}, e = r.env || {}; return [m.model, m.cpu, e.gpu].filter(Boolean).join(' · ') || '—'; };

const rows = { numpy: [], node: [], engine: [], browser: [], numpyXL: [], browserXL: [], browserCold: [], nativeXL: [] };
for (const f of files.sort()) {
  // A malformed evidence file must fail CI instead of silently disappearing.
  const r = JSON.parse(fs.readFileSync(f, 'utf8'));
  const rel = path.relative(dir, f);
  if (r.experiment === 'native-metal-stock-trig-fp32' && r.n === 100_000_000) {
    const summary = nativeXL(r), parts = rel.split(path.sep);
    rows.nativeXL.push([parts[0], parts[1], r.device, path.basename(rel).slice(0, 10),
      summary.valid ? 'all 300M XYZ in every CPU/GPU trial' : 'FAILED/incomplete',
      ms(summary.singleMs), `${ms(summary.multiMs)} (${r.worker_count} workers)`,
      ms(summary.gpuMs), ms(summary.materializedMs), rel]);
  } else if (r.kind === 'browser-xl-tiled') {
    const summary = browserXL(r);
    rows.browserXL.push([who(r), mname(r), hw(r), day(r.at), r.provenance?.implementation ?? 'unknown',
      summary.valid ? 'all 300M XYZ; final deterministic output' : 'excluded: earlier, failed or incomplete protocol',
      ms(summary.residentMs), ms(summary.readbackMs), `${r.geometry?.raw?.length ?? 0}/${r.geometry?.resident_draw_plus_full_readback?.raw?.length ?? 0}`, rel]);
  } else if (r.kind === 'browser-xl-cold-retained') {
    const summary = browserCold(r);
    rows.browserCold.push([who(r), mname(r), r.trials?.[0]?.env?.gpu ?? '—', day(r.at), r.provenance?.implementation ?? 'unknown',
      summary.valid ? 'all 300M XYZ in each of 3 outputs' : 'FAILED/incomplete', ms(summary.coldMs), rel]);
  } else if (r.kind === 'numpy-xl-e2e') {
    const summary = numpyXL(r);
    rows.numpyXL.push([who(r), mname(r), hw(r), day(r.at), r.variant,
      r.plan.markers.toLocaleString('en-US'), summary.valid ? 'all XYZ: final resident + every cold output' : 'FAILED/incomplete',
      ms(summary.residentMs), ms(summary.coldMs), `${r.plan.resident_runs}/${r.plan.cold_runs}`, rel]);
  } else if (r.kind === 'numpy') {
    const c = r.claim_check || {}, s = r.sweep || {};
    rows.numpy.push([who(r), mname(r), hw(r), day(r.at), ms(c.median), ms(c.max), `${c.above_12 === 0 && c.nodes_outside_own_band === 0 ? 'holds' : 'does not hold'}`, ms(med(s['1000000'])), ms(med(s['10000000'])), ms(med(s['100000000'])), rel]);
  } else if (r.kind === 'cpu-node') {
    const g = r.geo || {}, p = r.physics || {};
    rows.node.push([who(r), mname(r), hw(r), day(r.at), r.plan, ms(med(g['100000'])), ms(med(g['1000000'])), r.arrival_ns ? `${r.arrival_ns.median.toFixed(0)} ns` : '—', ms(med(p['1000'] && p['1000'].layout)), ms(med(p['10000'] && p['10000'].layout)), rel]);
  } else if (r.kind === 'engine') {
    const s = r.sizes || {};
    rows.engine.push([who(r), mname(r), r.engine, r.language || '—', r.device || '—', day(r.at), r.verified ? 'yes' : 'NO', ms(med(s['100000'])), ms(med(s['1000000'])), ms(med(s['10000000'])), rel]);
  } else if (r.kind === 'browser' || r.kind === 'published-page') {
    const g = (r.gpu || {}), c = (r.cpu || {}), e = (g.estimates || {})['100000000'];
    const estimate = e && positive(e.physics_layout_ms) && !e.error && e.passed !== false &&
      med(g.layout?.['10000000']) !== null ? `≈ ${ms(e.physics_layout_ms)} (est.)` : '—';
    rows.browser.push([who(r), mname(r), hw(r), (r.env && r.env.browser) || '—', day(r.at), r.emulated ? 'emulated' : 'real', ms(med(g.geo && g.geo['10000000'])), ms(med(g.geo && g.geo['100000000'])), legacyGeometryLabel(g.geo?.['10000000']), ms(med(g.layout && g.layout['1000'])), ms(med(g.layout && g.layout['10000'])), ms(med(g.layout && g.layout['10000000'])), estimate, ms(med(c.geo && c.geo['100000'])), rel]);
  }
}

const table = (title, head, body) => body.length ? `## ${title}\n\n| ${head.join(' | ')} |\n|${head.map(() => '---').join('|')}|\n${body.map(r => `| ${r.join(' | ')} |`).join('\n')}\n` : null;
const md = [
  '# Benchmark results across machines',
  '',
  `Generated by \`node tools/compare.mjs\` from ${files.length} result files. Do not edit by hand.`,
  '',
  'Timings have different boundaries and numerical precision. Zero, nonfinite and failed timing series are not measurements. Historical browser geometry timings without exhaustive XYZ checks remain explicitly unverified. See [XL timing and validation protocol](../XL_BENCHMARK.md).',
  '',
  table('Batched NumPy FP64: retained host output', ['Who', 'Machine', 'Hardware', 'Date', 'Variant', 'Markers', 'Validation', 'Resident placement + materialization', 'Cold input generation through retained output', 'Resident/cold samples', 'File'], rows.numpyXL),
  table('100M tiled WebGL FP32: resident inputs', ['Who', 'Machine', 'Hardware', 'Date', 'Variant', 'Validation', 'Resident placement', 'Placement + streaming RGBA readback', 'Placement/readback samples', 'File'], rows.browserXL),
  table('100M tiled WebGL FP32: cold application pipeline', ['Who', 'Machine', 'Hardware', 'Date', 'Variant', 'Validation', 'Fresh input generation through retained RGBA output', 'File'], rows.browserCold),
  table('100M native FP32: resident inputs, full buffers', ['Who (path)', 'Machine (path)', 'Device', 'Date (path)', 'Validation', 'CPU single', 'CPU multicore + barrier', 'Metal placement', 'Metal placement + retained XYZ copy', 'File'], rows.nativeXL),
  table('Disclosure re-check (NumPy, one core): 100,000 markers in 9–12 ms', ['Who', 'Machine', 'Hardware', 'Date', 'Median', 'Worst', 'Claim', '1M markers', '10M markers', '100M markers', 'File'], rows.numpy),
  table('CPU, Node, one core: geometry vs physics', ['Who', 'Machine', 'Hardware', 'Date', 'Plan', 'Geometry 100k', 'Geometry 1M', 'One new marker', 'Physics layout 1k', 'Physics layout 10k', 'File'], rows.node),
  table('Engines (verified against the reference placement)', ['Who', 'Machine', 'Engine', 'Language', 'Device', 'Date', 'Verified', '100k', '1M', '10M', 'File'], rows.engine),
  table('Historical browser runner: GPU (WebGL2) and browser CPU', ['Who', 'Machine', 'Hardware', 'Browser', 'Date', 'GPU', 'GPU geometry 10M', 'GPU geometry 100M', '10M geometry validation', 'GPU physics layout 1k', 'GPU physics layout 10k', 'GPU physics layout 10M', 'GPU physics 100M', 'CPU geometry 100k', 'File'], rows.browser),
].filter(x => x !== null).join('\n');
fs.writeFileSync(path.join(dir, 'SUMMARY.md'), md + '\n');
process.stdout.write(md + '\n');
