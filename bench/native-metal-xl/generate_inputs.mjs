// Unmodified stock JS generator and FP64 placement are the independent reference.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {inputs, encodeInputs, place, NL, CAP, MARGIN, SEED} from '../../src/js/engine.mjs';
// Usage: node bench/native-metal-xl/generate_inputs.mjs 100000000 tmp/native-metal-xl
const dir=path.resolve(process.argv[3] || 'tmp/native-metal-xl');
const sizes=process.argv[2]?process.argv[2].split(',').map(Number):[1000,10000,100000,1000000,10000000];
fs.mkdirSync(path.join(dir,'inputs'),{recursive:true});
const manifest={generator:'unmodified src/js/engine.mjs inputs/encodeInputs/place',node:process.version,protocol:{bands:NL,cap:CAP,margin:MARGIN,seed:SEED},files:{}};
for(const n of sizes){
  if(!Number.isSafeInteger(n)||n<1||n>100000000)throw Error('invalid n');
  const d=inputs(n), ref=new Float64Array(n*3);place(n,d.layer,d.t,d.v,ref);
  for(const [name,bytes] of [[`${n}.3ain`,encodeInputs(d)],[`${n}.reference.f64`,Buffer.from(ref.buffer)]]){
    const dest=path.join(dir,'inputs',name), exists=fs.existsSync(dest), hash=crypto.createHash('sha256');
    if(exists && fs.statSync(dest).size!==bytes.length)throw Error(`preserve differing ${dest}`);
    const fd=fs.openSync(dest,exists?'r':'wx');
    try {
      const block=64*1024*1024;
      for(let offset=0;offset<bytes.length;offset+=block){
        const chunk=bytes.subarray(offset,Math.min(bytes.length,offset+block));
        if(exists){const old=Buffer.allocUnsafe(chunk.length);const got=fs.readSync(fd,old,0,old.length,offset);if(got!==old.length||!old.equals(chunk))throw Error(`preserve differing ${dest}`);}
        else {let written=0;while(written<chunk.length)written+=fs.writeSync(fd,chunk,written,chunk.length-written,offset+written);}
        hash.update(chunk);
      }
    } finally {fs.closeSync(fd);}
    manifest.files[`inputs/${name}`]={bytes:bytes.length,sha256:hash.digest('hex')};
  }
  process.stdout.write(JSON.stringify({generated:n})+'\n');
}
fs.writeFileSync(path.join(dir,`inputs-manifest-${sizes.join('-')}.json`),JSON.stringify(manifest,null,2)+'\n');
