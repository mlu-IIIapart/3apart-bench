// Presentation gates, not a replacement for exhaustive source/output review.
export const positive = value => typeof value === 'number' && Number.isFinite(value) && value > 0;
const values = (items, key) => Array.isArray(items) ? items.map(item => item?.[key]) : null;

export function sampleMedian(samples, expected) {
  if (!Number.isInteger(expected) || expected < 1 || !Array.isArray(samples) ||
      samples.length !== expected || !samples.every(positive)) return null;
  const sorted = [...samples].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function measuredMedian(measurement) {
  if (!measurement || measurement.error || measurement.passed === false || measurement.status === 'failed') return null;
  const stat = measurement.wall ?? measurement;
  // A zero minimum reveals failed/quantized samples even if the median is positive.
  if ('min' in stat && !positive(stat.min)) return null;
  return positive(stat.median) ? stat.median : null;
}

export function numpyValidation(v, n) {
  return v?.passed === true && v.markers_checked === n && v.xyz_coordinates_checked === 3 * n &&
    ['nonfinite_inputs', 'invalid_input_markers', 'nonfinite_output_coordinates',
      'nonfinite_reference_coordinates', 'coordinates_over_tolerance', 'outside_band', 'invalid_radius']
      .every(key => v[key] === 0) && v.tolerance === 1e-9 &&
    Number.isFinite(v.max_abs_error) && v.max_abs_error <= 1e-9;
}

export function numpyXL(r) {
  const n = r.plan?.markers, resident = r.resident;
  const valid = r.passed === true && r.status === 'passed' && r.source_unchanged === true &&
    Number.isInteger(n) && n > 0 && numpyValidation(resident?.validation, n) &&
    Array.isArray(r.cold_pipelines) && r.cold_pipelines.length === r.plan.cold_runs &&
    r.cold_pipelines.every(p => numpyValidation(p?.validation, n));
  const residentMs = sampleMedian(values(resident?.samples, 'placement_and_complete_output_materialization_ms'), r.plan?.resident_runs);
  const coldMs = sampleMedian(values(r.cold_pipelines, 'end_to_end_ms'), r.plan?.cold_runs);
  const complete = valid && residentMs !== null && coldMs !== null;
  return { valid: complete, residentMs: complete ? residentMs : null, coldMs: complete ? coldMs : null };
}

export function browserValidation(v, n) {
  return v?.passed === true && v.expected_markers === n && v.checked_markers === n &&
    v.checked_xyz_coordinates === 3 * n && v.finite_xyz_coordinates === 3 * n &&
    ['nonfinite_xyz_coordinates', 'nonfinite_reference_coordinates', 'alpha_not_one', 'outside_band'].every(k => v[k] === 0) &&
    Array.isArray(v.coverage_errors) && v.coverage_errors.length === 0 &&
    Number.isFinite(v.max_abs_error) && v.max_abs_error <= 5e-4 && v.tolerance === 5e-4;
}

export function browserXL(r) {
  const g = r.geometry, n = g?.allocation?.markers;
  const valid = r.passed === true && r.status === 'passed' && g?.passed === true &&
    n === 100_000_000 && g.uploaded_markers === n && browserValidation(g.validation, n);
  const residentMs = sampleMedian(g?.raw, 10);
  const readbackMs = sampleMedian(g?.resident_draw_plus_full_readback?.raw, 10);
  const complete = valid && residentMs !== null && readbackMs !== null;
  return { valid: complete, residentMs: complete ? residentMs : null, readbackMs: complete ? readbackMs : null };
}

export function browserCold(r) {
  const n = r.scope?.markers, samples = values(r.trials, 'pipeline_ms');
  const valid = r.passed === true && r.status === 'passed' && n === 100_000_000 &&
    r.scope.repetitions === 3 && Array.isArray(r.trials) && r.trials.length === 3 &&
    r.trials.every(t => t?.passed === true && t.uploaded_markers === n &&
      t.allocation?.retained_host_output_bytes >= 16 * n && browserValidation(t.validation, n));
  const coldMs = sampleMedian(samples, 3), complete = valid && coldMs !== null;
  return { valid: complete, coldMs: complete ? coldMs : null };
}

export function nativeXL(r) {
  const n = r.n;
  const checked = trials => Array.isArray(trials) && trials.length === r.repeats && trials.every(t => {
    const v = t?.validation;
    return v?.pass === true && v.actual_coordinates === 3 * n && v.expected_coordinates === 3 * n &&
      v.nonfinite === 0 && v.coordinates_over_5e_4 === 0 && v.outside_band === 0 &&
      Number.isFinite(v.max_abs_error_vs_raw_fp64) && v.max_abs_error_vs_raw_fp64 <= 5e-4;
  });
  const fields = {
    gpuMs: sampleMedian(values(r.gpu_trials, 'resident_encode_submit_complete_ms'), r.repeats),
    materializedMs: sampleMedian(values(r.gpu_trials, 'resident_plus_full_output_copy_ms'), r.repeats),
    singleMs: sampleMedian(values(r.cpu_single_trials, 'placement_ms'), r.repeats),
    multiMs: sampleMedian(values(r.cpu_multi_trials, 'placement_including_barrier_ms'), r.repeats),
  };
  const valid = n === 100_000_000 && r.repeats === 10 && r.cpu_single_multi_bit_identical === true &&
    [r.gpu_trials, r.cpu_single_trials, r.cpu_multi_trials].every(checked) && Object.values(fields).every(positive);
  return { valid, ...Object.fromEntries(Object.entries(fields).map(([k, v]) => [k, valid ? v : null])) };
}

export function legacyGeometryLabel(measurement) {
  if (measuredMedian(measurement) === null) return 'missing/invalid';
  // The original large runner exported timings without an exhaustive XYZ check.
  return 'timing only; full XYZ not established';
}
