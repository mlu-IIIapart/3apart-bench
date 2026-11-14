# Native CPU and Metal placement

This additional macOS experiment uses the original JavaScript inputs and FP64
reference. It compares FP32 placement on one CPU thread, a persistent CPU worker
pool and native Metal compute, using full buffers rather than WebGL textures.
It does not replace the original benchmark or the raw/Rust detector workload.

From the repository root, with Xcode Command Line Tools installed:

```sh
node bench/native-metal-xl/generate_inputs.mjs 100000000 tmp/native-metal-xl
xcrun clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -fobjc-arc \
  -framework Foundation -framework Metal bench/native-metal-xl/native_bench.mm \
  -o tmp/native-metal-xl/native_bench
tmp/native-metal-xl/native_bench --self-test
tmp/native-metal-xl/native_bench --prepare bench/native-metal-xl/place.metal
tmp/native-metal-xl/native_bench tmp/native-metal-xl/inputs/100000000 \
  bench/native-metal-xl/place.metal tmp/native-metal-xl/100000000-NEW-RUN 16 10
```

Choose the worker count for the machine (16 on Lorium, 18 on Aureon). Use a new
output prefix for each run; existing results are preserved. Fixtures alone use
4.4 GB of disk, and final GPU/CPU outputs add 3.6 GB. The measured process needs
roughly 12 GB of memory at 100M. Run one large benchmark at a time per machine.

Three warmups precede ten trials per method. Every timed trial is followed by
an exhaustive 300M-coordinate FP64-reference check, outside its timing clock.
The checker rejects nonfinite values, coordinate errors above 5e-4 and band
violations. Single and multicore CPU final arrays must be bit-identical.

The resident GPU clock includes command encoding, submission and completion.
The continuous materialization clock also includes copying the complete packed
XYZ output into a retained host array. This is a CPU memcpy from shared Metal
memory, not a discrete-GPU PCIe transfer. Input generation, file reads, FP32
conversion, context/shader setup, allocation, input copying and validation are
excluded from both clocks. It is not a fresh-input end-to-end measurement.

Metal uses safe/precise functions; CPU compilation disables fast math and fused
contraction. Fixed GPU → single CPU → multicore order and per-trial validation
affect cache state and cadence. Native returns 1.2 GB packed FP32 XYZ; the WebGL
path reads 1.600127 GB padded RGBA. Compare these as implementations with stated
endpoints, not as an isolated measure of browser overhead.

## Native Metal validation modes

The historical `native_bench.mm` and its strict FP64 checker remain unchanged.
`optimized_bench.mm` records the math-function and parallel-checker comparison.
`validation_modes_bench.mm` adds full-output GPU comparison with exact CPU
revalidation of changed blocks, plus explicitly optional sparse checking.

### Build and run

Run from the repository root on macOS with Xcode command-line tools and Metal.
Use the physical CPU worker count appropriate to the machine (18 on the tested
M5 Max Aureon, 16 on the tested M4 Max Lorium).

```sh
mkdir -p tmp/native-validation
xcrun clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off \
  -fobjc-arc -framework Foundation -framework Metal \
  bench/native-metal-xl/validation_modes_bench.mm \
  -o tmp/native-validation/run

## Generate once. Retain the original FP64 inputs and independent JS reference.
node bench/native-metal-xl/generate_inputs.mjs 100000000 tmp/native-validation

## Full checks on every iteration, with GPU comparisons and exact CPU updates.
tmp/native-validation/run tmp/native-validation/inputs/100000000 \
  tmp/native-validation/full-gpu.json 18 --run full-gpu 100

## Full CPU checks, useful when most output blocks change every iteration.
tmp/native-validation/run tmp/native-validation/inputs/100000000 \
  tmp/native-validation/full-cpu.json 18 --run full-cpu 100

## Explicit reduced coverage: 65,536 sampled markers, including boundary anchors.
tmp/native-validation/run tmp/native-validation/inputs/100000000 \
  tmp/native-validation/sparse.json 18 --run sparse 100

## Rotate the reproducible sample each iteration; planning cost is included.
tmp/native-validation/run tmp/native-validation/inputs/100000000 \
  tmp/native-validation/sparse-rotating.json 18 --run sparse-rotating 100
```

Existing result files are never overwritten. These runtime commands perform
initial and final full audits, recorded separately. They do **not** repeat an
extra serial audit after every iteration. The placement input is fixed during
each invocation, so repeated placement normally produces identical output.

Omit `--run MODE ITERATIONS` for the controlled benchmark: ten randomized rounds
of full CPU, full GPU comparison and sparse checking, followed by ten trials
each with three changed words and with every block changed. Every benchmark
output receives an additional original serial audit outside the reported timer.
The benchmark does more work than the normal runtime path intentionally.

### What full GPU mode guarantees

Metal does not support the `double` type. This implementation retains the
original CPU FP64 subtraction, three-argument `std::hypot`, finite checks,
coordinate tolerance of `5e-4`, normalized band predicate, counts and maxima.
It does not replace them with FP32 approximations on the GPU.

1. A new context owns copies of the FP64 reference, layers and band edges. Its
   first output receives all original checks.
2. A Metal kernel compares **every** raw output word against the previous
   certified snapshot. This is collision-free bit equality, not a hash or sample.
   Signed zeros, NaN payloads and final partial blocks are included.
3. Each changed 4096-marker block receives all original CPU checks. Only a
   bit-identical block with the same immutable context reuses a cached summary.
4. Counts and maxima are reduced across all block summaries. Failing results are
   cached as failures too; repairing an error replaces that block's summary.

The result therefore covers every coordinate on every full-mode iteration.
It reuses previously established facts rather than proving the mathematics for
all possible inputs. With mostly changed output, full CPU checking can be
faster because GPU comparison adds overhead. Initial certification and context
setup are separate costs, not steady-state speedups.

`GPUCachedValidator` owns the snapshot and `CachedBlockValidator` owns reference
context. Complete the producer GPU command before calling `validate`: the
validator uses a separate queue. Keep the output buffer unchanged for the whole
synchronous call. GPU completion/status are checked before CPU access. Replace
the context with `resetContext` when reference/layers/edges change. Count errors,
command failures and interrupted updates invalidate the certificate. Do not
persist certificates across a checker-policy or executable change.

At 100M markers, placement buffers plus snapshot and flags require about
3.6 GB of GPU-visible shared memory. Reference copies and host inputs also count
toward process memory; the harness checks the GPU working-set budget and reports
allocation, while resource logs record process footprint and swaps.

### Sparse mode is a separate contract

The same per-point FP64 checks and tolerance apply to the sampled points only.
`check.expected` and `check.actual` describe buffer lengths, not checked coverage.
Always retain `effective_validation_mode`, `checked_markers`,
`checked_coordinates`, `coverage` and `full_coverage` with the result. Inputs up
to 65,536 markers fall back to a full check. Boundary anchors are mandatory and
may exceed a caller's unusually small requested sample budget.

Sampling is deterministic without replacement. It includes first/last,
quartile, tile and CPU chunk boundaries. The default runtime reuses its sample;
`sparse-rotating` varies the iteration key and includes sample planning in the
measured clock. Unsampled corruption can pass. A final full audit only certifies
the final output, not unchecked intermediate outputs.

Use full checking after code/compiler/device/context changes, with new input
families, and for correctness evidence. Sparse checking is an explicit runtime
cost/coverage choice after that validation. Finite testing cannot prove that the
math works for every possible input.

### Controls

```sh
xcrun clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -pthread \
  bench/native-metal-xl/validation_cached_blocks_test.cpp -o tmp/native-validation/cache-test
tmp/native-validation/cache-test
xcrun clang++ -std=c++17 -O3 -fno-fast-math -ffp-contract=off -pthread \
  bench/native-metal-xl/validation_sparse_test.cpp -o tmp/native-validation/sparse-test
tmp/native-validation/sparse-test
node bench/native-metal-xl/generate_inputs.mjs 8195 tmp/native-validation/controls
tmp/native-validation/run tmp/native-validation/controls/inputs/8195 \
  tmp/native-validation/gpu-controls.json 18 --controls
```

CPU controls also run under AddressSanitizer and UndefinedBehaviorSanitizer.
GPU controls compare all summaries to the unchanged serial checker, including
single-word/tail changes, restore/repeat, NaN/Inf, zero radius, subnormals, band
failures, all-block changes, count rejection and context replacement.
