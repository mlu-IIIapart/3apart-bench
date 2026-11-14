# 100-million-marker placement

The original XL browser run requested a single 1,600,126,976-byte RGBA32F input texture. On the tested Chrome/ANGLE Metal path, both allocation APIs accepted 1,342,177,280 bytes and rejected one additional 262,144-byte row. The test failed before placement despite sufficient total unified memory. Old failure exports remain evidence of that allocation failure; their zero-duration physics samples and derived estimates are not measurements.

The [both-Mac report and plots](reports/2026-11-14-xl/README.md) include raw trial points, full-output validation and the separate CPU/Rust/raw detector baselines. [Execution evidence](reports/2026-11-14-xl/provenance/validation.json) records source hashes and provenance limits.

The replacement keeps four 25-million-marker input/output pairs resident. Each texture occupies 400,031,744 bytes including padding; all eight occupy 3,200,253,952 bytes. This is a per-texture allocation repair, not a change to the sphere mapping. The original shader, global mulberry32 stream, FP64 reference, 24 bands, cap, margin, three warmups and ten resident placements are retained.

## Timing boundaries

These clocks answer different questions and must be shown separately:

- **Resident placement:** draw all four tiles and synchronize every output framebuffer. Inputs and outputs are already allocated on the GPU. Save every wall sample and every available device-query sample.
- **Streaming full readback:** one placement followed by reading every output tile to the host, using a continuous clock. Reference computation and validation occur outside that clock. This series does not retain all output simultaneously.
- **Cold application pipeline:** fresh input generation, allocation, packing/upload, one full placement and complete retained host output under one continuous clock. Reference validation follows. “Cold” describes new data/resources, not restarting the process or evicting OS caches.

Stage clocks locate costs; their sums or summed medians are not substitutes for a measured pipeline clock. Device-query timings are not end-to-end latency. The browser returns padded RGBA32F; NumPy returns packed FP64 XYZ, and native Metal returns packed FP32 XYZ. The byte layouts and precision differ.

## Correctness

Every reported pass requires complete contiguous coverage of 100 million markers and 300 million XYZ coordinates, finite outputs and references, the unchanged tolerance, and zero markers outside their assigned band. Browser readbacks additionally require alpha=1 and sentinel checks. All WebGL errors fail the run. Global input offsets continue across tile boundaries; padding is excluded from marker counts.

Resident and streaming series validate their final deterministic output. Each cold pipeline validates its retained output. The result records state which outputs were checked; final-output verification does not mean every resident trial was independently validated. Negative controls cover nonfinite coordinates, wrong finite values, incorrect tile offsets, missing or duplicate coverage and unwritten buffers.

Coordinate agreement does not establish preservation of temporal ordering at FP32 precision. Keep original time/value data for downstream analysis. This experiment does not validate clinical alerts, financial signals or a complete detector.

## CPU and earlier baselines

`bench/cpu_xl_e2e.py` is an additive FP64 NumPy baseline. It uses unchanged `src/py/engine.py` inputs and placement in the same four-batch plan, retaining full output. It saves ten resident samples and three independent cold pipelines, with exhaustive independent algebraic verification of the final resident output and each cold output. Numerical-library thread requests are one; no exclusive-machine or affinity claim is made.

```sh
python3 -m unittest discover -s bench -p 'test_cpu_xl_e2e.py'
python3 bench/cpu_xl_e2e.py --machine lorium --who joel \
  --output results/joel/lorium/NEW-UTC-TIMESTAMP-numpy-xl-e2e.json
```

Earlier native single-core, persistent multicore and Metal comparisons remain separate baselines. Aquila's raw and Rust detector results concern stream inference, not sphere placement; they must not be relabeled as 100M placement timings. No Rust sphere-placement engine is supplied by this change.

The dedicated geometry action avoids an unrelated preceding large physics workload. The unresolved 10M browser physics failure is not repaired or claimed as a successful measurement here. Geometry and force-directed layouts solve different layout objectives.

## Evidence and interpretation

Keep original result files immutable, including failures and preliminary variants. Record source hashes, implementation variant, runtime/device identity, raw samples and host conditions. Compare the same implementation across Macs. Normal desktop load, existing swap, browser transfers, memory layout and runtime differences prevent interpreting these measurements as an isolated chip-speed comparison.

Local tests and CI check planning, validation and result handling. They do not replace the real-M4 and real-M5 GPU runs and their exhaustive output checks.
