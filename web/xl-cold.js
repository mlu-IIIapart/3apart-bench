/* Additive fresh-resource pipeline measurement. Resident v2 clocks stay unchanged. */
(() => {
  const N = 100_000_000, RS = 128, get = id => document.getElementById(id);
  const card = document.createElement('div'); card.className = 'card'; card.id = 'xl-cold';
  card.innerHTML = `<h3>100-million-marker full pipeline</h3>
    <p>Three fresh runs include input generation, memory allocation, upload, one placement, and the complete output retained in CPU memory. Each output is validated after its timing stops. Browser startup, source loading and shader setup are excluded.</p>
    <button type="button" id="cold-run">Run 3 full pipelines</button> <button type="button" id="cold-stop" hidden>Stop pipelines</button> <button type="button" id="cold-download" hidden>Download pipeline result</button>
    <p id="cold-status" class="status" role="status">Ready. About 4.47 GiB of texture and retained output storage, plus bounded working buffers.</p><pre id="cold-summary" style="white-space:pre-wrap"></pre>`;
  get('xl-geometry').after(card);
  let last = null, active = false;
  const sha = async s => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(s))), b => b.toString(16).padStart(2,'0')).join('');
  get('cold-stop').onclick = () => ctl && ctl.abort();
  async function one(trial, signal, stage) {
    const g = new GpuBench(), gl = g.gl, plan = FullGpuAudit.planTiles(N, g.maxTex), W = plan.tiles[0].width;
    const tiles = plan.tiles.map(t => ({...t, tin:null, tout:null, fb:null, host:null}));
    const env = deviceInfo(g), chunks = [], timings = {};
    const stop = () => { if (signal.aborted) throw new Stopped(); };
    const started = performance.now();
    try {
      if (/SwiftShader|llvmpipe|software|emulat/i.test(g.renderer)) throw Error('Hardware GPU is required');
      g.check('cold context setup');
      const buf = new Float32Array(W*RS*4);
      for (const t of tiles) {
        stop(); t.tin=g.tex(W,t.height,null); g.check('cold input allocation');
        t.tout=g.tex(W,t.height,null); g.check('cold output allocation'); t.fb=g.fbo(t.tout); g.check('cold framebuffer');
        t.host=new Float32Array(W*t.height*4); t.host.fill(NaN);
      }
      let uploaded=0;
      for (const t of tiles) for (let row=0;row<t.height;row+=RS) {
        stop(); const rows=Math.min(RS,t.height-row), first=t.first+row*W, count=Math.min(rows*W,t.count-row*W);
        if(first!==uploaded)throw Error('Cold input coverage mismatch'); buf.fill(0);
        for(let j=0;j<count;j++){const k=3*(first+j);buf[4*j]=Math.floor(drawAt(k+1)*NL);buf[4*j+1]=drawAt(k+2);buf[4*j+2]=drawAt(k+3);}
        g.upload(t.tin,W,rows,buf.subarray(0,W*rows*4),0,row);g.check('cold upload');uploaded+=count;
        stage(`pipeline ${trial+1}/3 · prepared ${uploaded.toLocaleString()} inputs`);await yieldNow();
      }
      if(uploaded!==N)throw Error('Incomplete cold input');
      timings.preparation_wall=performance.now()-started;
      const {p,u}=g.pGeo;gl.useProgram(p);gl.bindVertexArray(g.vao);gl.activeTexture(gl.TEXTURE0);gl.uniform1i(u.uIn,0);gl.uniform1fv(u.uE,EDGES32);
      for(const t of tiles){gl.bindTexture(gl.TEXTURE_2D,t.tin);gl.bindFramebuffer(gl.FRAMEBUFFER,t.fb);gl.viewport(0,0,W,t.height);gl.drawArrays(gl.TRIANGLES,0,3);}
      // All complete host arrays coexist. Each read goes directly into its final
      // retained array, and no reference computation occurs before the clock stops.
      for(const t of tiles){stop();gl.bindFramebuffer(gl.FRAMEBUFFER,t.fb);gl.readBuffer(gl.COLOR_ATTACHMENT0);gl.readPixels(0,0,W,t.height,gl.RGBA,gl.FLOAT,t.host);g.check('cold full output materialization');}
      const pipeline_ms=performance.now()-started;
      const validationStart=performance.now(), audit=FullGpuAudit.begin(N), L=new Int32Array(W*RS), T=new Float64Array(W*RS), V=new Float64Array(W*RS), ref=new Float64Array(3*W*RS);
      for(const t of tiles)for(let row=0;row<t.height;row+=RS){
        stop();const rows=Math.min(RS,t.height-row),first=t.first+row*W,count=Math.min(rows*W,t.count-row*W);
        for(let j=0;j<count;j++){const k=3*(first+j);L[j]=Math.floor(drawAt(k+1)*NL);T[j]=drawAt(k+2);V[j]=drawAt(k+3);}
        place(count,L,T,V,ref);FullGpuAudit.auditChunk(audit,first,count,t.host.subarray(row*W*4,(row+rows)*W*4),ref,L,EDGES);chunks.push({first,count,tile_first:t.first,row,rows});
        stage(`pipeline ${trial+1}/3 · validated ${audit.checked_markers.toLocaleString()} markers`);await yieldNow();
      }
      const validation=FullGpuAudit.finish(audit);timings.validation_after_pipeline=performance.now()-validationStart;
      if(!validation.passed)throw Object.assign(Error('Cold output validation failed'),{validation});
      return {trial,pipeline_ms,passed:true,env,allocation:{...plan,retained_host_output_bytes:plan.allocated_texels_per_side*16,staging_and_reference_bytes:buf.byteLength+L.byteLength+T.byteLength+V.byteLength+ref.byteLength},uploaded_markers:uploaded,chunks,validation,timings_ms:timings};
    } finally {
      for(const t of tiles){if(t.fb)gl.deleteFramebuffer(t.fb);if(t.tin)gl.deleteTexture(t.tin);if(t.tout)gl.deleteTexture(t.tout);t.host=null;}
      gl.deleteProgram(g.pGeo.p);gl.deleteProgram(g.pTick.p);gl.deleteVertexArray(g.vao);const lose=gl.getExtension('WEBGL_lose_context');if(lose)lose.loseContext();
    }
  }
  get('cold-run').onclick=async()=>{
    if(active||ctl||get('run').disabled)return;active=true;ctl=new AbortController();hero.pause(true);get('run').disabled=true;get('cold-run').disabled=true;get('cold-stop').hidden=false;get('cold-download').hidden=true;get('cold-summary').textContent='';get('cold-status').className='status';
    last={kind:'browser-xl-cold-retained',schema:1,at:new Date().toISOString(),who:get('who').value.trim()||null,machine:{name:get('mname').value.trim()||null},passed:false,trials:[],scope:{markers:N,repetitions:3,draws_per_trial:4,warmups:0,includes:['fresh input generation and FP32 packing','texture and host output allocation','input upload','one full placement','full retained CPU RGBA output','UI yielding during input preparation'],excludes:['browser/process startup','source loading','context and shader setup','FP64 reference and comparison'],every_trial_fully_validated:true,physics_tested:false,angular_order_tested:false}};
    try {
      if(!['127.0.0.1','localhost','[::1]'].includes(location.hostname)||DB)throw Error('Use the local offline benchmark');
      const read=async f=>{const r=await fetch(f,{cache:'no-store'});if(!r.ok)throw Error(`Cannot read ${f}`);return r.text();};const manifest=JSON.parse(await read('xl-manifest.json'));
      for(const [f,h]of Object.entries(manifest.files))if(await sha(await read(f))!==h)throw Error(`Source hash mismatch: ${f}`);last.provenance=manifest;
      for(let i=0;i<3;i++)last.trials.push(await one(i,ctl.signal,s=>get('cold-status').textContent=s));
      last.pipeline={...stats(last.trials.map(t=>t.pipeline_ms)),raw:last.trials.map(t=>t.pipeline_ms)};last.passed=last.trials.length===3&&last.trials.every(t=>t.passed);last.status='passed';
      get('cold-status').textContent='PASS — all 3 full pipelines validated, 100,000,000 markers each.';
      get('cold-summary').textContent=`Full pipeline median: ${last.pipeline.median.toFixed(3)} ms\nRaw trials: ${last.pipeline.raw.map(t=>t.toFixed(3)).join(', ')} ms\nAll 300,000,000 XYZ coordinates checked in each trial\nComplete host output retained per trial: ${(last.trials[0].allocation.retained_host_output_bytes/1024**3).toFixed(3)} GiB\nValidation runs after each continuous timing stops.`;
    }catch(e){last.status=ctl.signal.aborted?'stopped':'error';last.error=String(e.message||e);if(e.validation)last.failed_validation=e.validation;get('cold-status').className='status err';get('cold-status').textContent=`No complete three-trial pass: ${last.error}`;}
    finally{last.completed_at=new Date().toISOString();ctl=null;active=false;get('run').disabled=false;get('cold-run').disabled=false;get('cold-stop').hidden=true;get('cold-download').hidden=false;hero.pause(false);}
  };
  get('cold-download').onclick=()=>{if(!last)return;const url=URL.createObjectURL(new Blob([JSON.stringify(last,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download=`${last.at.replace(/:/g,'').replace(/\..+/,'')}-browser-xl-cold.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);};
})();
