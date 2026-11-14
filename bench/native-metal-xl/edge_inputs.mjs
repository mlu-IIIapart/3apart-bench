// Deterministic boundary fixture, using the unchanged independent JS FP64 formula.
import fs from 'node:fs';
import path from 'node:path';
import {encodeInputs,place} from '../../src/js/engine.mjs';
const dest=process.argv[2];if(!dest)throw Error('usage: edge_inputs.mjs OUTPUT_DIRECTORY');
fs.mkdirSync(dest,{recursive:true});
const times=[0,2**-24,.25,.5,.75,1-2**-24,1],values=[0,2**-24,.12,.5,1-2**-24,1];
const n=24*times.length*values.length,d={n,layer:new Int32Array(n),t:new Float64Array(n),v:new Float64Array(n)};
let i=0;for(let l=0;l<24;l++)for(const t of times)for(const v of values){d.layer[i]=l;d.t[i]=t;d.v[i]=v;i++;}
const ref=new Float64Array(3*n);place(n,d.layer,d.t,d.v,ref);
fs.writeFileSync(path.join(dest,'edges.3ain'),encodeInputs(d),{flag:'wx'});
fs.writeFileSync(path.join(dest,'edges.reference.f64'),Buffer.from(ref.buffer),{flag:'wx'});
console.log(JSON.stringify({n,reference:'unchanged src/js/engine.mjs place()',times,values,bands:24}));
