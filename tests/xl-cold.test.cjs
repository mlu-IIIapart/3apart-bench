// CPU-only regression test of the actual cold pipeline, with small surrogate buffers.
// No browser, hardware GPU, or benchmark-sized allocation is used.
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const dir=path.join(__dirname,'../web');
const html=fs.readFileSync(path.join(dir,'index.html'),'utf8');
const cold=fs.readFileSync(path.join(dir,'xl-cold.js'),'utf8');
const checkerModule={exports:{}};vm.runInNewContext(fs.readFileSync(path.join(dir,'xl-support.js'),'utf8'),{module:checkerModule});
const checker=checkerModule.exports;
const engine=html.slice(html.indexOf('const NL = 24'),html.indexOf('// share of markers whose height'));
const actual=vm.runInNewContext(engine+'\n({drawAt,place,NL,EDGES,EDGES32})');
const functionText=cold.slice(cold.indexOf('  async function one('),cold.indexOf("  get('cold-run').onclick"));
assert.ok(functionText.includes('const pipeline_ms=performance.now()-started;'));
const cases=[];
async function exercise(fault=null){
 let clock=0,id=0,contexts=0;const events=[],allIds=[],deleted=[];
 const event=(op,data={})=>events.push({op,clock,...data});
 const outputArrays=[];
 class FakeGPU{
  constructor(){contexts++;this.maxTex=7;this.renderer='Apple fake GL for CPU orchestration tests';this.pGeo={p:{},u:{}};this.pTick={p:{}};this.vao={};
   let input,fb;const outputs=[];
   this.gl={NO_ERROR:0,OUT_OF_MEMORY:0x505,
    bindTexture(_,x){input=x;},bindFramebuffer(_,x){fb=x;},useProgram(){},bindVertexArray(){},activeTexture(){},uniform1i(){},uniform1fv(){},viewport(){},readBuffer(){},
    drawArrays(){event('draw',{fb:fb.id});const n=input.data.length/4,L=new Int32Array(n),T=new Float64Array(n),V=new Float64Array(n),xyz=new Float64Array(n*3);for(let j=0;j<n;j++){L[j]=input.data[4*j];T[j]=input.data[4*j+1];V[j]=input.data[4*j+2];}actual.place(n,L,T,V,xyz);for(let j=0;j<n;j++)fb.data.set([xyz[3*j],xyz[3*j+1],xyz[3*j+2],1],4*j);},
    readPixels(x,y,w,h,f,t,out){event('readback',{fb:fb.id,values:out.length});outputArrays.push(out);const chosen=fault==='swap'&&fb===outputs[1]?outputs[0]:fb;if(fault==='unwritten'&&fb===outputs.at(-1))return;out.set(chosen.data.subarray(0,out.length));},
    deleteFramebuffer(){},deleteTexture(t){deleted.push(t.id);},deleteProgram(){},deleteVertexArray(){},getExtension(){return{loseContext(){event('release_context');}}}};
   this.tex=(w,h)=>{const t={id:id++,data:new Float32Array(w*h*4)};allIds.push(t.id);event('allocate_texture');return t;};
   this.fbo=t=>{outputs.push(t);fb=t;return t;};
   this.upload=(t,w,h,data,x,y)=>{event('upload');t.data.set(data,y*w*4);};
   this.check=phase=>{event('check',{phase});if(fault==='gl'&&phase==='cold full output materialization')throw Error('injected GL failure');};
  }
 }
 const validationEvents=[];
 const ctx={...actual,drawAt(...args){event('input_rng');return actual.drawAt(...args);},N:23,RS:128,GpuBench:FakeGPU,FullGpuAudit:{...checker,planTiles:(n,m)=>checker.planTiles(n,m,7),auditChunk(...a){event('validate');validationEvents.push(events.length);return checker.auditChunk(...a);}},deviceInfo:g=>({renderer:g.renderer}),Stopped:class extends Error{},yieldNow:async()=>event('yield'),performance:{now(){clock++;event('clock');return clock;}}};
 const one=vm.runInNewContext(functionText+'\none',ctx);
 const results=[];
 for(let i=0;i<(fault?1:3);i++){
  if(fault){await assert.rejects(one(i,{aborted:false},()=>{}),fault==='gl'?/GL failure/:/validation failed/);break;}
  const eventStart=events.length,contextBefore=contexts,outputsBefore=outputArrays.length;
  const r=await one(i,{aborted:false},()=>{});results.push(r);
  assert.equal(contexts,contextBefore+1);assert.equal(new Set(outputArrays.slice(outputsBefore)).size,4);assert.equal(r.passed,true);assert.equal(r.validation.checked_markers,23);assert.equal(r.validation.checked_xyz_coordinates,69);
  assert.equal(r.allocation.retained_host_output_bytes,4*7*16);assert.equal(r.chunks.at(-1).count,2);
  const trialEvents=events.slice(eventStart),firstClock=trialEvents.find(e=>e.op==='clock').clock;
  const lastRead=trialEvents.filter(e=>e.op==='readback').at(-1);
  assert.equal(trialEvents.filter(e=>e.op==='draw').length,4);assert.equal(trialEvents.filter(e=>e.op==='readback').length,4);
  const firstCheck=trialEvents.find(e=>e.op==='validate');
  assert.ok(lastRead.clock<firstClock+r.pipeline_ms);assert.ok(firstCheck.clock>firstClock+r.pipeline_ms);
  assert.ok(trialEvents.find(e=>e.op==='allocate_texture').clock>=firstClock);
  assert.ok(trialEvents.find(e=>e.op==='upload').clock<firstClock+r.pipeline_ms);
  assert.equal(trialEvents.filter(e=>e.op==='input_rng'&&e.clock<firstClock+r.pipeline_ms).length,69);
 }
 assert.equal(deleted.length,allIds.length);
 return {trials:results.length,contexts,draws:events.filter(e=>e.op==='draw').length,full_readbacks:events.filter(e=>e.op==='readback').length,checks:events.filter(e=>e.op==='validate').length,deleted_textures:deleted.length};
}
(async()=>{
const normal=await exercise();cases.push('Three independent cold contexts allocate/upload/draw/read retained host output, then validate 23 markers/69 coordinates including partial last tile; clocks exclude reference validation');
await exercise('swap');cases.push('Swapped retained tile data fails full comparison');
await exercise('unwritten');cases.push('Unwritten final retained host array remains NaN and fails finite/alpha/coordinate gates');
await exercise('gl');cases.push('Readback GL failure prevents successful trial and releases resources');
console.log(JSON.stringify({passed:true,controls:cases,normal}));
})().catch(error=>{console.error(error);process.exitCode=1;});
