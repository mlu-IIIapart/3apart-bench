const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
let scope;
test.before(async () => { scope = await import('../tools/result-scope.mjs'); });

test('failed and zero-duration series cannot become measured timings', () => {
  for (const value of [undefined, { median: 0 }, { median: Infinity }, { median: -1 },
    { median: 1, min: 0 }, { wall: { median: 5 }, error: 'out of memory' },
    { median: 5, passed: false }]) assert.equal(scope.measuredMedian(value), null);
  assert.equal(scope.measuredMedian({ wall: { median: 5, min: 4 } }), 5);
});

test('historical timing-only outputs are not promoted to full validation', () => {
  assert.equal(scope.legacyGeometryLabel({ wall: { median: 1 } }), 'timing only; full XYZ not established');
  assert.equal(scope.legacyGeometryLabel({ wall: { median: 0 } }), 'missing/invalid');
});

test('sample median rejects missing samples, nonfinite and zeros', () => {
  assert.equal(scope.sampleMedian([4, 1, 3, 2], 4), 2.5);
  assert.equal(scope.sampleMedian([], 0), null);
  for (const samples of [[1], [1, NaN], [1, 0]]) assert.equal(scope.sampleMedian(samples, 2), null);
});

test('CPU full-output gate rejects a corrupt coordinate or an unvalidated cold output', () => {
  const file = path.join(__dirname, '../results/joel/lorium/2026-11-14T003655-numpy-xl-e2e.json');
  const original = JSON.parse(fs.readFileSync(file));
  assert.equal(scope.numpyXL(original).valid, true);
  for (const mutate of [
    r => { r.resident.validation.nonfinite_output_coordinates = 1; },
    r => { r.cold_pipelines[0].validation.xyz_coordinates_checked--; },
    r => { r.cold_pipelines.pop(); },
    r => { r.resident.samples[0].placement_and_complete_output_materialization_ms = 0; },
    r => { r.source_unchanged = false; },
    r => { r.resident.validation.tolerance = 1; },
    r => { r.cold_pipelines = {}; },
    r => { r.resident.samples = {}; },
    r => { r.cold_pipelines[0] = null; },
  ]) {
    const r = structuredClone(original); mutate(r); assert.equal(scope.numpyXL(r).valid, false);
  }
});

test('browser and native summaries reject incomplete validation and malformed timings', () => {
  const read = file => JSON.parse(fs.readFileSync(path.join(__dirname, '../results/joel/', file)));
  for (const [file, check, corrupt] of [
    ['lorium/2026-11-14T005207-browser-xl-tiled.json', scope.browserXL, r => { r.geometry.validation.alpha_not_one = 1; }],
    ['lorium/2026-11-14T005359-browser-xl-cold.json', scope.browserCold, r => { r.trials[1].validation.checked_markers--; }],
    ['aureon/2026-11-14T003901-native-metal-xl.json', scope.nativeXL, r => { r.cpu_multi_trials[1].validation.nonfinite = 1; }],
  ]) {
    const r = read(file); assert.equal(check(r).valid, true); corrupt(r); assert.equal(check(r).valid, false);
  }
  assert.equal(scope.browserCold({trials: {}}).valid, false);
  assert.equal(scope.nativeXL({gpu_trials: {}, cpu_single_trials: 'bad'}).valid, false);
  const cold = read('lorium/2026-11-14T005359-browser-xl-cold.json'); cold.trials[0] = null;
  assert.equal(scope.browserCold(cold).valid, false);
  const native = read('aureon/2026-11-14T003901-native-metal-xl.json'); native.gpu_trials[0] = null;
  assert.equal(scope.nativeXL(native).valid, false);
});
