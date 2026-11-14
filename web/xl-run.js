/* Dedicated geometry run: avoids a preceding large physics workload. */
(() => {
  const card = document.createElement('div'); card.className = 'card'; card.id = 'xl-geometry';
  card.innerHTML = `<h3>100-million-marker geometry</h3>
    <p>Run geometry on its own, using four resident input/output tile pairs (about 3 GiB). Every final XYZ coordinate is checked against the original formula. Ten placements are timed after three warmups.</p>
    <p>Placement timing excludes input preparation and full output validation. Ten additional placements measure compute plus full output readback; the final output is validated separately. This run does not test physics or angular ordering.</p>
    <button id="xl-run" type="button">Run 100 million markers</button>
    <button id="xl-stop" type="button" hidden>Stop</button>
    <button id="xl-download" type="button" hidden>Download 100M result</button>
    <p id="xl-status" class="status" role="status" aria-live="polite">Ready.</p>
    <pre id="xl-summary" style="white-space:pre-wrap"></pre>`;
  document.getElementById('gpu-card').after(card);
  const get = id => document.getElementById(id), run = get('xl-run'), stop = get('xl-stop'), download = get('xl-download'), status = get('xl-status');
  let active = false, abort, result;
  const sha = async s => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s))), b => b.toString(16).padStart(2, '0')).join('');
  new MutationObserver(() => { run.disabled = active || get('run').disabled; }).observe(get('run'), { attributes: true, attributeFilter: ['disabled'] });
  stop.onclick = () => abort && abort.abort();
  run.onclick = async () => {
    if (active || ctl) return;
    active = true; abort = new AbortController(); run.disabled = true; get('run').disabled = true; stop.hidden = false; download.hidden = true;
    status.className = 'status'; get('xl-summary').textContent = ''; hero.pause(true);
    let g; const started = performance.now();
    result = { kind: 'browser-xl-tiled', schema: 1, at: new Date().toISOString(), passed: false, machine: { name: get('mname').value.trim() || null }, who: get('who').value.trim() || null };
    try {
      if (!['localhost', '127.0.0.1', '[::1]'].includes(location.hostname) || DB) throw Error('Use the local offline benchmark for this experimental implementation');
      const read = async file => { const r = await fetch(file, { cache: 'no-store' }); if (!r.ok) throw Error(`Cannot read ${file}`); return r.text(); };
      const manifest = JSON.parse(await read('xl-manifest.json'));
      for (const [file, hash] of Object.entries(manifest.files)) if (await sha(await read(file)) !== hash) throw Error(`Source hash mismatch: ${file}`);
      result.provenance = manifest;
      result.runtime_source_hashes = { fragment_shader: await sha(FS_GEO), input_draw_at: await sha(drawAt.toString()), fp64_reference: await sha(place.toString()), xl_method: await sha(GpuBench.prototype.geoXL.toString()), timing_method: await sha(GpuBench.prototype.timed.toString()) };
      g = new GpuBench(); g.check('context setup'); result.env = { ...deviceInfo(g), user_agent: navigator.userAgent };
      if (/SwiftShader|llvmpipe|software|emulat/i.test(g.renderer)) throw Error('Hardware GPU is required');
      result.geometry = await g.geoXL(100_000_000, 10, abort.signal,
        i => { status.textContent = `Timed placement ${i + 1} / 10; full output validation follows`; },
        message => { status.textContent = message; });
      result.passed = result.geometry.passed;
      result.status = result.passed ? 'passed' : 'failed';
      status.textContent = result.passed ? 'PASS — all 100,000,000 markers / 300,000,000 XYZ coordinates validated.' : 'FAIL — see result.';
      const r = result.geometry;
      get('xl-summary').textContent = `GPU: ${g.renderer}\nResident placement median: ${r.wall.median.toFixed(3)} ms\nGPU device median: ${r.gpu ? r.gpu.median.toFixed(3) + ' ms' : 'unavailable'}\nCompute + full output readback median: ${r.resident_draw_plus_full_readback.median.toFixed(3)} ms\nTexture memory: ${(r.allocation.resident_texture_bytes / 1024 ** 3).toFixed(3)} GiB in ${r.allocation.texture_count} textures\nChecked: ${r.checked.toLocaleString()} markers\nFinite XYZ: ${r.validation.finite_xyz_coordinates.toLocaleString()}\nMaximum absolute error: ${r.maxErr.toExponential(6)} (limit 5e-4)\nOutside assigned band: ${r.validation.outside_band}\nUnwritten outputs: ${r.validation.alpha_not_one}\nFull output validation: ${(r.check_ms / 1000).toFixed(2)} seconds`;
    } catch (e) {
      result.status = abort.signal.aborted ? 'stopped' : 'error'; result.error = String(e.message || e); if (e.validation) result.validation = e.validation;
      status.className = 'status err'; status.textContent = `No validated pass: ${result.error}`;
    } finally {
      if (g) { g.gl.deleteProgram(g.pGeo.p); g.gl.deleteProgram(g.pTick.p); g.gl.deleteVertexArray(g.vao); const lose = g.gl.getExtension('WEBGL_lose_context'); if (lose) lose.loseContext(); }
      result.total_wall_ms = performance.now() - started; result.completed_at = new Date().toISOString();
      active = false; run.disabled = false; get('run').disabled = false; stop.hidden = true; download.hidden = false; hero.pause(false);
    }
  };
  download.onclick = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2) + '\n'], { type: 'application/json' }), url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = `${result.at.replace(/:/g, '').replace(/\..+/, '')}-browser-xl-tiled.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
})();
