const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict'), path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../web/index.html'), 'utf8');
const moduleShim = { exports: {} };
vm.runInNewContext(fs.readFileSync(path.join(__dirname, '../web/xl-support.js'), 'utf8'), { module: moduleShim });
const audit = moduleShim.exports, events = []; let clock = 0;
const auditWrapper = { ...audit, auditChunk: (...args) => { events.push('audit'); return audit.auditChunk(...args); } };
const engine = source.slice(source.indexOf('const NL ='), source.indexOf('const yieldNow ='));
const gpuClass = source.slice(source.indexOf('class GpuBench'), source.indexOf('// ============================================================================ page'));
const ctx = vm.createContext({ FullGpuAudit: auditWrapper, performance: { now: () => ++clock }, yieldNow: async () => {}, sleep: async () => {}, Stopped: class extends Error {} });
const api = vm.runInContext(engine + '\n' + gpuClass + '\n({ GpuBench, drawAt, rng, place, EDGES })', ctx);
// Independent unbounded integer arithmetic checks the random-access stream at
// each real tile boundary and at the uint32 counter wrap.
const mask = 0xffffffffn;
function oracle(k) {
  let t = (20261112n + BigInt(k) * 0x6D2B79F5n) & mask;
  t = ((t ^ (t >> 15n)) * (t | 1n)) & mask;
  t = (t ^ ((t + (((t ^ (t >> 7n)) * (t | 61n)) & mask)) & mask)) & mask;
  return Number((t ^ (t >> 14n)) & mask) / 4294967296;
}
const random = api.rng(20261112);
for (let k = 1; k <= 3000; k++) assert.equal(api.drawAt(k), random());
for (const marker of [0, 4095, 4096, 24999999, 25000000, 49999999, 50000000, 74999999, 75000000, 99999999]) {
  for (const offset of [1, 2, 3]) assert.equal(api.drawAt(3 * marker + offset), oracle(3 * marker + offset));
}
for (const k of [4294967295, 4294967296, 4294967297]) assert.equal(api.drawAt(k), oracle(k));
assert.notEqual(api.drawAt(75000001), api.drawAt(1)); // reset-at-tile regression

async function exercise(swapTile = false) {
  events.length = 0; let texture, framebuffer, drawCount = 0, syncCount = 0, id = 0;
  const allTextures = [], outputs = [], deleted = [];
  const g = Object.create(api.GpuBench.prototype);
  const gl = { NO_ERROR: 0, OUT_OF_MEMORY: 0x505, getError: () => 0, isContextLost: () => false,
    clearBufferfv: () => framebuffer.output.data.fill(0), finish: () => {}, useProgram: () => {}, bindVertexArray: () => {}, activeTexture: () => {}, uniform1i: () => {}, uniform1fv: () => {}, viewport: () => {}, readBuffer: () => {},
    bindTexture: (_, t) => { texture = t; }, bindFramebuffer: (_, f) => { framebuffer = f; },
    drawArrays: () => {
      events.push('draw'); drawCount++;
      const n = texture.data.length / 4, L = new Int32Array(n), T = new Float64Array(n), V = new Float64Array(n), ref = new Float64Array(3 * n);
      for (let j = 0; j < n; j++) { L[j] = texture.data[4*j]; T[j] = texture.data[4*j+1]; V[j] = texture.data[4*j+2]; }
      api.place(n, L, T, V, ref);
      for (let j = 0; j < n; j++) framebuffer.output.data.set([ref[3*j], ref[3*j+1], ref[3*j+2], 1], 4*j);
    },
    readPixels: (x, y, w, rows, format, type, out) => {
      events.push('read');
      const selected = swapTile && framebuffer === outputs[1] ? outputs[0] : framebuffer;
      out.set(selected.output.data.subarray(y*w*4, (y+rows)*w*4));
    },
    deleteTexture: t => deleted.push(t), deleteFramebuffer: f => deleted.push(f) };
  Object.assign(g, { gl, maxTex: 2, tq: null, lost: false, pGeo: { p: {}, u: {} }, vao: {} });
  g.tex = (w, h) => { const t = { id: id++, data: new Float32Array(w*h*4) }; allTextures.push(t); return t; };
  g.fbo = output => { framebuffer = { output }; outputs.push(framebuffer); return framebuffer; };
  g.upload = (t,w,h,data,x,y) => t.data.set(data,y*w*4);
  g.sync = fb => { assert.ok(outputs.includes(fb)); syncCount++; events.push('sync'); };
  let timed = 0;
  const run = g.geoXL(14,10,null,i => { timed++; assert.equal(drawCount,(3+i+1)*4); assert.equal(syncCount,(i+2)*4); });
  if (swapTile) { await assert.rejects(run, /validation failed/); return; }
  const result = await run;
  assert.equal(result.passed,true); assert.equal(result.checked,14); assert.equal(result.validation.checked_xyz_coordinates,42);
  assert.equal(drawCount,92); assert.equal(syncCount,44); assert.equal(timed,10); assert.equal(result.raw.length,10); assert.equal(result.resident_draw_plus_full_readback.raw.length,10);
  assert.equal(events.filter(x=>x==='read').length,44); assert.equal(events.slice(0,events.indexOf('audit')).filter(x=>x==='read').length,41);
  assert.equal(deleted.length,12); assert.equal(result.scope.synchronization,'one pixel from every output framebuffer');
}
(async () => { await exercise(); await exercise(true); console.log(JSON.stringify({ passed:true, controls:['BigInt RNG oracle at real tile boundaries and wrap','sequential input equivalence','all four draws and all four synchronizations per trial','ten independent continuous full-readback samples before validation','full coverage with partial final tile','swapped tile data rejected','all resources released'] })); })().catch(e => { console.error(e); process.exitCode=1; });
