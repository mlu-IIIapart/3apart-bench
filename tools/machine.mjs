// Machine and tester details stamped into every result file.
import os from 'node:os';
import fs from 'node:fs';
import path from 'node:path';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export function arg(name, fallback = null) {
  const i = process.argv.indexOf('--' + name);
  if (i < 0) return fallback;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith('--') ? true : v;
}

function sh(cmd) { try { return execSync(cmd, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim(); } catch { return null; } }

export function machine() {
  const cpus = os.cpus();
  let cpu = cpus[0] ? cpus[0].model.trim() : 'unknown';
  if (process.platform === 'darwin') cpu = sh('sysctl -n machdep.cpu.brand_string') || cpu;
  const model = process.platform === 'darwin' ? sh('sysctl -n hw.model') : null;
  return {
    name: arg('machine') || process.env.BENCH_MACHINE || os.hostname().replace(/\.local$/, ''),
    cpu, model, cores: cpus.length, ram_gb: Math.round(os.totalmem() / 2 ** 30),
    os: `${os.type()} ${os.release()} ${os.arch()}`,
  };
}

export function who() {
  const w = arg('who') || process.env.BENCH_WHO;
  if (!w || w === true) {
    console.error('Say who is running this: --who <your-name> (or set BENCH_WHO). It labels your results.');
    process.exit(2);
  }
  return String(w).toLowerCase().replace(/[^a-z0-9-]+/g, '-');
}

export const slug = s => String(s).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

export function commit() { return sh(`git -C "${ROOT}" rev-parse --short HEAD`); }

export function save(person, mach, kind, data) {
  const dir = path.join(ROOT, 'results', person, slug(mach.name));
  fs.mkdirSync(dir, { recursive: true });
  const stamp = new Date().toISOString().replace(/[:]/g, '').replace(/\..+/, '');
  const file = path.join(dir, `${stamp}-${kind}.json`);
  fs.writeFileSync(file, JSON.stringify(data, null, 1) + '\n');
  return path.relative(ROOT, file);
}
