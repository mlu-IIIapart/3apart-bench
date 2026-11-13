// Benchmark and verify any placement engine that follows engines/README.md.
// Usage:
//   node tools/engine_bench.mjs --who <name> --name <engine-name> --cmd "<command that runs your engine>"
//        [--sizes 1000,10000,100000,1000000] [--runs 100] [--machine <name>] [--no-save]
// Example:
//   node tools/engine_bench.mjs --who joel --name rust-simd --cmd "./engines/rust-simd/target/release/place"
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { inputs, encodeInputs, place, outsideShare, timeOrder, timeOrderIdx, stats, NL, CAP, MARGIN, SEED } from '../src/js/engine.mjs';
import { arg, machine, who, commit, save } from './machine.mjs';

const TOL = { f64: 1e-9, f32: 5e-4 };   // max abs difference from the reference placement, per coordinate
const person = who(), mach = machine();
const name = arg('name'), cmd = arg('cmd');
if (!name || !cmd || name === true || cmd === true) { console.error('Need --name <engine-name> and --cmd "<command>". See engines/README.md.'); process.exit(2); }
const sizes = String(arg('sizes', '1000,10000,100000,1000000')).split(',').map(Number);
const runsArg = arg('runs');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), '3apart-engine-'));
const log = s => process.stderr.write(s + '\n');

const out = { schema: 1, kind: 'engine', engine: name, cmd, who: person, machine: mach, at: new Date().toISOString(), commit: commit(),
  protocol: { seed: SEED, bands: NL, cap: CAP, margin: MARGIN }, sizes: {}, verified: true };
log(`engine ${name} · ${mach.name}`);

for (const n of sizes) {
  const d = inputs(n), inFile = path.join(tmp, `in-${n}.bin`), xyzFile = path.join(tmp, `xyz-${n}.bin`), jsonFile = path.join(tmp, `out-${n}.json`);
  fs.writeFileSync(inFile, encodeInputs(d));
  const runs = runsArg ? +runsArg : n >= 1e7 ? 10 : 100;
  const r = spawnSync(`${cmd} --inputs "${inFile}" --runs ${runs} --warmup ${n >= 1e6 ? 2 : 10} --xyz "${xyzFile}" --json "${jsonFile}"`, { shell: true, stdio: ['ignore', 'inherit', 'inherit'] });
  if (r.status !== 0) { log(`  ${n}: engine exited with status ${r.status}`); out.verified = false; out.sizes[n] = { error: `exit ${r.status}` }; continue; }
  const rep = JSON.parse(fs.readFileSync(jsonFile, 'utf8'));
  const prec = rep.precision === 'f32' ? 'f32' : 'f64';
  const raw = fs.readFileSync(xyzFile), got = prec === 'f32' ? new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4) : new Float64Array(raw.buffer, raw.byteOffset, raw.byteLength / 8);
  const ref = new Float64Array(3 * n); place(n, d.layer, d.t, d.v, ref);
  let maxErr = got.length === ref.length ? 0 : Infinity;
  if (isFinite(maxErr)) for (let i = 0; i < ref.length; i++) { const e = Math.abs(got[i] - ref[i]); if (!(e <= maxErr)) maxErr = e; }
  const outside = isFinite(maxErr) ? outsideShare(n, i => d.layer[i], i => [got[3 * i], got[3 * i + 1], got[3 * i + 2]]) : 1;
  const order = isFinite(maxErr) && n <= 1e6 ? timeOrder(timeOrderIdx(d), d.layer, i => [got[3 * i], got[3 * i + 1]]) : null;
  const ok = maxErr <= TOL[prec] && outside === 0;
  if (!ok) out.verified = false;
  const s = stats(rep.ms);
  out.sizes[n] = { ...s, ns_per_marker: s.median * 1e6 / n, precision: prec, max_abs_err: maxErr, outside, order, verified: ok };
  out.language ||= rep.language; out.device ||= rep.device; out.threads ||= rep.threads;
  log(`  ${String(n).padStart(9)} markers  median ${s.median.toFixed(3)} ms  ${(s.median * 1e6 / n).toFixed(1)} ns/marker  max err ${maxErr.toExponential(1)}  ${ok ? 'verified' : 'FAILED verification'}`);
}
fs.rmSync(tmp, { recursive: true, force: true });
if (!arg('no-save')) log(`  saved ${save(person, mach, `engine-${name}`, out)}`);
if (!out.verified) { log('  This engine does not match the reference placement. Its times are saved but marked unverified.'); process.exit(1); }
