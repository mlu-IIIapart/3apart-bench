const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const checkerModule = { exports: {} };
vm.runInNewContext(fs.readFileSync(require('node:path').join(__dirname, '../web/xl-support.js'), 'utf8'), { module: checkerModule });
const { begin, auditChunk, finish } = checkerModule.exports;
const edges = Float64Array.from({ length: 25 }, (_, k) => .9 * (1 - 2 * k / 24));
const z = (edges[3] + edges[4]) / 2, y = -Math.sqrt(1 - z * z);
const ref = Float64Array.of(0, y, z), good = Float32Array.of(0, y, z, 1), layer = Int32Array.of(3);
const cases = [];
function one(name, edit, gate) {
  const a = begin(1), out = good.slice(), r = ref.slice(); edit(out, r);
  auditChunk(a, 0, 1, out, r, layer, edges); const result = finish(a);
  assert.equal(result.passed, false, name);
  assert.equal(result.checks[gate], false, name); cases.push(name);
}
{
  const a = begin(3), pair = Float32Array.from([...good, ...good]), pairRef = Float64Array.from([...ref, ...ref]);
  auditChunk(a, 0, 2, pair, pairRef, Int32Array.of(3, 3), edges);
  // Final partial chunk: unused trailing buffer bytes deliberately poisoned.
  pair.fill(NaN); pair.set(good);
  auditChunk(a, 2, 1, pair, ref, layer, edges);
  const result = finish(a); assert.equal(result.passed, true); assert.equal(result.checked_xyz_coordinates, 9); cases.push('valid contiguous chunks and partial final buffer');
}
one('GPU NaN is rejected', out => out[0] = NaN, 'all_xyz_finite');
one('GPU Infinity is rejected', out => out[1] = Infinity, 'all_xyz_finite');
one('FP64 NaN is rejected', (out, ref) => ref[0] = NaN, 'all_xyz_finite');
one('over-tolerance coordinate is rejected', out => out[0] = .00051, 'fp32_reference_tolerance');
one('outside-band marker is rejected', out => out[2] = -.9, 'zero_outside_assigned_band');
one('unwritten alpha is rejected', out => out[3] = 0, 'every_output_written');
{
  const a = begin(2); auditChunk(a, 0, 1, good, ref, layer, edges);
  assert.equal(finish(a).passed, false); assert.equal(finish(a).checks.full_coverage, false); cases.push('incomplete output is rejected');
}
for (const [name, start] of [['skipped chunk is rejected', 1], ['duplicate chunk is rejected', 0]]) {
  const a = begin(2);
  if (start === 0) auditChunk(a, 0, 1, good, ref, layer, edges);
  assert.throws(() => auditChunk(a, start, 1, good, ref, layer, edges), /Non-contiguous/);
  assert.equal(finish(a).passed, false); cases.push(name);
}
{
  const a = begin(1); assert.throws(() => auditChunk(a, 0, 1, good.subarray(0, 3), ref, layer, edges), /shorter/);
  assert.equal(finish(a).passed, false); cases.push('short readback buffer is rejected');
}
{
  // Ensure normalization matches stock band predicate, rather than raw z bounds.
  const a = begin(1), out = good.slice(); for (let k = 0; k < 3; k++) out[k] *= 2;
  auditChunk(a, 0, 1, out, Float64Array.from(out.subarray(0, 3)), layer, edges);
  assert.equal(finish(a).outside_band, 0); cases.push('band predicate uses normalized height');
}
const { planTiles } = checkerModule.exports;
for (const n of [1, 4095, 4096, 4097, 25_000_000, 25_000_001, 100_000_000]) {
  for (const maxTex of [4096, 16384]) {
    const p = planTiles(n, maxTex); let first = 0, bytes = 0;
    for (const t of p.tiles) {
      assert.equal(t.first, first); first += t.count; bytes += 2 * t.bytes_per_texture;
      assert.ok(t.width <= maxTex && t.height <= maxTex);
      assert.ok(t.width * t.height >= t.count && t.width * t.height - t.count < t.width);
      assert.ok(t.bytes_per_texture <= 512 * 1024 ** 2);
    }
    assert.equal(first, n); assert.equal(bytes, p.resident_texture_bytes);
  }
}
const xl = planTiles(100_000_000, 16384);
assert.equal(xl.tiles.length, 4); assert.equal(xl.resident_texture_bytes, 3_200_253_952);
for (const n of [0, -1, 100_000_001, NaN, 1.5]) assert.throws(() => planTiles(n, 16384));
cases.push('tile coverage, padding, dimensions and memory budgets at row/tile boundaries');
const page = fs.readFileSync(require('node:path').join(__dirname, '../web/index.html'), 'utf8');
const checkMethod = page.slice(page.indexOf('  check(phase'), page.indexOf('  // time `runs`'));
const check = vm.runInNewContext('({' + checkMethod + '}).check');
for (const error of [0x500, 0x501, 0x502, 0x505, 0x506]) {
  assert.throws(() => check.call({ gl: { NO_ERROR: 0, OUT_OF_MEMORY: 0x505, getError: () => error, isContextLost: () => false } }, 'test'), /WebGL error/);
}
cases.push('all GL errors including the observed allocation error fail closed');
for (const script of page.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) new vm.Script(script[1]);
cases.push('page scripts parse');
process.stdout.write(JSON.stringify({ passed: true, tests: cases.length, cases }, null, 2) + '\n');
