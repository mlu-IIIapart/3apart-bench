# Protocol

Results from different people and machines can only be compared if everyone measures the same thing the same way. Everything below is fixed. **Changing any of it needs a pull request that both maintainers approve**, a bump of `schema` in the result files, and a note at the bottom of this file. Results made under an older protocol stay in the repo but are compared only with each other.

## Inputs

- 24 bands. Band edges run from z = +0.9 to z = −0.9 (`CAP = 0.9`), evenly spaced.
- Each marker has a band, a time and a value, drawn in that order from mulberry32 with seed `20261112`.
- Synthetic data only. No patient, market or customer data goes in this repo.
- Engines in other languages read the shared binary inputs file (format in `engines/README.md`) rather than re-implementing the generator, unless they verify they produce the same stream.

## 3apart geometry (the reference placement)

The disclosure's `place()` and `sphere_xyz()`:

- longitude φ = π(2t − 1): time runs once around the globe;
- height z inside the band with a 12% margin at each edge (`MARGIN = 0.12`): z = z_bottom + (z_top − z_bottom)(0.12 + 0.76·value);
- x = sin θ sin φ, y = −sin θ cos φ, z = cos θ, with θ = arccos(z).

`src/js/engine.mjs` and `src/py/engine.py` are the reference. They agree to within 1e-15.

**Open item:** the NumPy re-check should run the disclosure's own reference code unchanged. `src/py/engine.py` is a vectorised equivalent; when the disclosure's exact code is added under `reference/`, `bench/disclosure_check.py` must call it instead.

## Physics baseline

d3-force-3d **3.0.6** (pinned), tuned for this task so the comparison is against physics at its best:

- many-body charge −10 (Barnes–Hut, theta 0.9);
- links between markers that follow each other in time within a band, distance 30;
- radial force onto a sphere of radius R = 30·√(n / 4π), strength 1;
- a pull toward the middle of each marker's band, strength 1;
- random start on the sphere; alpha from 1 to 0.001 over 300 ticks (d3's default schedule).

One new marker: one link to its band, reheat to alpha 0.3, run until alpha < 0.001.

## What counts as a measurement

- **Geometry time** is one placement of all markers, after warm-up, wall clock, one thread, excluding input generation and file I/O.
- **Physics layout time** is building the simulation plus 300 ticks.
- **Disclosure re-check**: 100 runs, each the mean of 20 placements of 100,000 markers.
- Report the median and keep every run (min, p95, max in the files).
- An engine's times count only if its output **passes verification**: max abs difference from the reference ≤ 1e-9 (float64) or ≤ 5e-4 (float32), and 0 markers outside their band.
- Multi-threaded or GPU engines are welcome, but they're reported with their thread count and device and compared separately from single-core numbers.

## Quality checks

- **Inside the band**: share of markers whose height is outside their own band. Geometry must be 0.
- **Time order**: share of time-consecutive pairs in a band that run the same way around the globe. Geometry is 1 by construction; physics is about 0.5.

## What this does not show

A force-directed layout encodes graph relations; 3apart geometry encodes fixed bands. These numbers compare two ways to lay out a banded globe, not two solvers for the same problem. Benchmarks are measurements of this code on the machines named, not a guarantee of performance elsewhere.

## Changes

- schema 1 (2026-11-13): first version, matching the published page.
- schema 1, addition (2026-11-13): optional extra-large runs. Browser: GPU physics tick (3 runs) and one full layout at 10M markers, with the one-layout limit raised from 90 s to 6 min for that size only; GPU geometry at 100M markers (10 runs), inputs written to the GPU in slices and every position checked against the float64 formula. GPU physics at 100M is an **estimate** (300 × measured 10M tick × 10), stored under `gpu.estimates` and never as a measurement. NumPy: `--xl` adds 100M markers to the sweep (10 runs). Existing measurements are unchanged, so the schema stays 1.
