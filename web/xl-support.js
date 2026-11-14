/* XL allocation planning and full final-output validation; stock FP32 tolerance. */
(function (root) {
  'use strict';
  const TOLERANCE = 5e-4;
  function planTiles(n, maxTextureSize, markersPerTile = 25_000_000) {
    if (!Number.isSafeInteger(n) || n < 1 || n > 100_000_000 ||
        !Number.isSafeInteger(maxTextureSize) || maxTextureSize < 1 ||
        !Number.isSafeInteger(markersPerTile) || markersPerTile < 1) throw Error('Invalid XL allocation request');
    const width = Math.min(4096, maxTextureSize, n);
    // Stay below 512 MiB per texture, including row padding. ANGLE can cap a
    // single texture below MAX_TEXTURE_SIZE squared even with ample free RAM.
    const maxRows = Math.min(maxTextureSize, Math.floor(512 * 1024 * 1024 / (16 * width)));
    const capacity = Math.min(markersPerTile, width * maxRows), tiles = [];
    for (let first = 0; first < n; first += capacity) {
      const count = Math.min(capacity, n - first), height = Math.ceil(count / width);
      tiles.push({ first, count, width, height, bytes_per_texture: width * height * 16 });
    }
    const textureBytes = tiles.reduce((total, tile) => total + 2 * tile.bytes_per_texture, 0);
    if (textureBytes > 4 * 1024 ** 3) throw Error('XL resident texture budget exceeds 4 GiB');
    return { strategy: 'tiled-resident-rgba32f-v1', markers: n, tiles,
      texture_count: tiles.length * 2, resident_texture_bytes: textureBytes,
      texture_budget_bytes: 4 * 1024 ** 3, per_texture_budget_bytes: 512 * 1024 ** 2,
      allocated_texels_per_side: textureBytes / 32, excluded_padding_texels_per_side: textureBytes / 32 - n };
  }
  function begin(expectedMarkers) {
    return { expected_markers: expectedMarkers, expected_xyz_coordinates: expectedMarkers * 3,
      checked_markers: 0, checked_xyz_coordinates: 0, finite_xyz_coordinates: 0,
      nonfinite_xyz_coordinates: 0, nonfinite_reference_coordinates: 0,
      alpha_not_one: 0, outside_band: 0, max_abs_error: 0,
      max_abs_error_by_axis: [0, 0, 0], sum_squared_error: 0,
      max_unit_norm_error: 0, worst_coordinate: null, first_failures: [],
      band_counts: Array(24).fill(0), coverage_errors: [], tolerance: TOLERANCE };
  }
  function auditChunk(audit, start, count, rgba, ref, layers, edges) {
    if (!Number.isInteger(start) || !Number.isInteger(count) || start !== audit.checked_markers || count <= 0 || start + count > audit.expected_markers) {
      audit.coverage_errors.push({ start, count, expected_start: audit.checked_markers });
      throw new Error('Non-contiguous, duplicate or out-of-range output chunk');
    }
    if (rgba.length < count * 4 || ref.length < count * 3 || layers.length < count) {
      audit.coverage_errors.push({ start, count, error: 'short buffer' });
      throw new Error('Output/reference/input buffer is shorter than claimed coverage');
    }
    for (let i = 0; i < count; i++) {
      const l = layers[i];
      if (!Number.isInteger(l) || l < 0 || l >= 24) throw new Error('Invalid reference input band');
      audit.band_counts[l]++;
      let failed = false;
      if (rgba[4 * i + 3] !== 1) { audit.alpha_not_one++; failed = true; }
      for (let k = 0; k < 3; k++) {
        const value = rgba[4 * i + k], reference = ref[3 * i + k];
        audit.checked_xyz_coordinates++;
        if (!Number.isFinite(value)) { audit.nonfinite_xyz_coordinates++; failed = true; }
        else audit.finite_xyz_coordinates++;
        if (!Number.isFinite(reference)) { audit.nonfinite_reference_coordinates++; failed = true; }
        if (Number.isFinite(value) && Number.isFinite(reference)) {
          const error = Math.abs(value - reference);
          audit.sum_squared_error += error * error;
          audit.max_abs_error_by_axis[k] = Math.max(audit.max_abs_error_by_axis[k], error);
          if (error > audit.max_abs_error) { audit.max_abs_error = error; audit.worst_coordinate = { marker: start + i, axis: 'xyz'[k], gpu: value, reference }; }
          if (error > TOLERANCE) failed = true;
        }
      }
      const norm = Math.hypot(rgba[4 * i], rgba[4 * i + 1], rgba[4 * i + 2]);
      // Identical band predicate to stock outsideShare(), including its radius fallback.
      const h = rgba[4 * i + 2] / (norm || 1);
      if (!(h <= edges[l] && h >= edges[l + 1])) { audit.outside_band++; failed = true; }
      if (Number.isFinite(norm)) audit.max_unit_norm_error = Math.max(audit.max_unit_norm_error, Math.abs(norm - 1));
      if (failed && audit.first_failures.length < 8) audit.first_failures.push({ marker: start + i, layer: l, gpu: Array.from(rgba.subarray(4 * i, 4 * i + 4)), reference: Array.from(ref.subarray(3 * i, 3 * i + 3)) });
    }
    audit.checked_markers += count;
  }
  function finish(audit) {
    const full = audit.checked_markers === audit.expected_markers && audit.checked_xyz_coordinates === audit.expected_xyz_coordinates && audit.coverage_errors.length === 0;
    const checks = {
      full_coverage: full,
      all_xyz_finite: audit.nonfinite_xyz_coordinates === 0 && audit.nonfinite_reference_coordinates === 0,
      every_output_written: audit.alpha_not_one === 0,
      fp32_reference_tolerance: audit.max_abs_error <= TOLERANCE,
      zero_outside_assigned_band: audit.outside_band === 0
    };
    return { ...audit, rms_error: Math.sqrt(audit.sum_squared_error / Math.max(1, audit.checked_xyz_coordinates)),
      outside_share: audit.outside_band / Math.max(1, audit.checked_markers), checks,
      passed: Object.values(checks).every(Boolean) };
  }
  const api = { begin, auditChunk, finish, planTiles, TOLERANCE };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.FullGpuAudit = api;
})(globalThis);
