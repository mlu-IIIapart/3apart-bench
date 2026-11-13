# 3apart Layout Benchmark

**Confidential: 3apart. Shared under NDA only. Synthetic data only; no patient or market data.**

This repo measures how fast 3apart geometry places markers on the Kosmos globe compared with a force-directed physics layout (d3-force-3d), and re-checks the invention disclosure's claim that 100,000 markers are placed in 9 to 12 ms (NumPy, one core).

Everyone runs the same code, the same inputs (fixed seed) and the same protocol on their own machine, saves a result file, and opens a pull request. `results/SUMMARY.md` puts every machine side by side.

## Run it on your machine

You need Node 20 or later and Python 3.10 or later with NumPy.

```bash
git clone git@github.com:<3apart-org>/3apart-bench.git
cd 3apart-bench
npm ci
python3 -m pip install -r requirements.txt

export BENCH_WHO=yourname           # labels your results
export BENCH_MACHINE=lorium         # a short name for this machine

npm run check                       # 10 seconds: your setup matches the reference placement
npm run disclosure                  # about 3 minutes: the 9–12 ms claim, on this machine
npm run cpu                         # about 2 minutes: geometry vs physics in Node, one core
npm run compare                     # rebuilds results/SUMMARY.md
```

Then the GPU: open `web/index.html` in Chrome or Safari, enter your name and machine name, press **Run on this machine** (about a minute), then **Download result** and put the file where the page says (`results/<you>/<machine>/`).

Commit your new files under `results/` plus the updated `results/SUMMARY.md`, and open a pull request.

For a fair run: plug the laptop in, close other apps, and don't use the machine while it runs.

## What's in here

| Path | What it is |
|---|---|
| `src/js/engine.mjs`, `src/py/engine.py` | The protocol constants, the input generator (identical in both languages) and the reference placement. **The single source of truth.** |
| `src/js/physics.mjs` | The physics baseline: d3-force-3d, tuned for band membership. |
| `bench/disclosure_check.py` | The disclosure's claim, re-checked with its protocol. |
| `bench/cpu_bench.mjs` | Geometry vs physics in Node: scale, one new marker, layout quality. Plans: `quick`, `standard`, `full`. |
| `web/index.html` | The browser runner: GPU (WebGL2) and browser CPU. |
| `engines/` | Other implementations of the placement (Rust, C++, GPU, …) and the contract they follow. |
| `tools/engine_bench.mjs` | Times any engine and **verifies its output against the reference** before its numbers count. |
| `tools/compare.mjs` | Builds `results/SUMMARY.md` from every result file. |
| `results/` | One folder per person, one per machine. `results/published/` holds the numbers on the published page. |

## Bringing your own engine or benchmark

See [engines/README.md](engines/README.md) for engines and [CONTRIBUTING.md](CONTRIBUTING.md) for new benchmarks. The short version: any language, any hardware, but it reads the shared inputs file, and its output has to match the reference placement before its times are compared.

## The live page

The published page on claude.ai is the same runner with a shared run history. The copy in `web/` works offline and doesn't save anything by itself; use **Download result**.
