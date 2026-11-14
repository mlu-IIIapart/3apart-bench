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
