#!/usr/bin/env node
// Reference engine (JavaScript) implementing the engine contract in engines/README.md.
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { decodeInputs, place } from '../../src/js/engine.mjs';

const a = k => { const i = process.argv.indexOf('--' + k); return i < 0 ? null : process.argv[i + 1]; };
const d = decodeInputs(fs.readFileSync(a('inputs'))), runs = +a('runs'), warmup = +(a('warmup') ?? 5);
const xyz = new Float64Array(3 * d.n), ms = [];
for (let w = 0; w < warmup; w++) place(d.n, d.layer, d.t, d.v, xyz);
for (let k = 0; k < runs; k++) { const t0 = performance.now(); place(d.n, d.layer, d.t, d.v, xyz); ms.push(performance.now() - t0); }
fs.writeFileSync(a('xyz'), Buffer.from(xyz.buffer));
fs.writeFileSync(a('json'), JSON.stringify({ engine: 'js-reference', language: `JavaScript (Node ${process.version})`, precision: 'f64', device: 'cpu', threads: 1, ms }));
