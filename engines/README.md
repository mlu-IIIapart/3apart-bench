# Engines

An engine is any implementation of the 3apart placement: Rust, C++, Swift, CUDA, Metal, WebGPU, anything. Put each in its own folder here (`engines/<name>/`) with a short README saying how to build it.

## The contract

Your engine is a command the harness can run like this:

```
<your command> --inputs <in.bin> --runs <R> --warmup <W> --xyz <out.bin> --json <out.json>
```

It must:

1. Read the inputs file (format below).
2. Place all markers `W` times untimed, then `R` times timed. Time only the placement: not reading the file, not writing the output.
3. Write the final positions to `--xyz`: x, y, z per marker, little-endian, float64 (or float32, see `precision`).
4. Write `--json`:

```json
{
  "engine": "rust-simd",
  "language": "Rust 1.90, AVX2",
  "precision": "f64",
  "device": "cpu",
  "threads": 1,
  "ms": [0.41, 0.40, 0.42]
}
```

`precision` is `f64` or `f32`; `device` is `cpu` or `gpu`; `ms` has one entry per timed run.

## Inputs file

Little-endian, no padding:

| Offset | Type | Field |
|---|---|---|
| 0 | 4 bytes | magic `3AIN` |
| 4 | uint32 | version (1) |
| 8 | uint32 | n, number of markers |
| 12 | uint32 | seed |
| 16 | int32[n] | band (0–23) |
| 16 + 4n | float64[n] | time (0–1) |
| 16 + 12n | float64[n] | value (0–1) |

## Running and verifying

```bash
node tools/engine_bench.mjs --who joel --name rust-simd --cmd "./engines/rust-simd/target/release/place"
```

The harness makes the inputs for 1,000 to 1,000,000 markers (`--sizes` to change), runs your engine, and checks the output against the reference: max abs difference ≤ 1e-9 for f64 or ≤ 5e-4 for f32, and 0 markers outside their band. Results are saved either way, but **unverified results are marked and left out of comparisons.**

Two engines are included as examples and as a check on the harness:

```bash
node tools/engine_bench.mjs --who you --name js-reference --cmd "node engines/js-reference/place.mjs"
node tools/engine_bench.mjs --who you --name py-numpy --cmd "python3 engines/py-numpy/place.py"
```

## Physics engines

A faster physics layout (a GPU force layout with a grid or tree, say) is a fair and useful thing to add: it makes the comparison stronger. Put it under `engines/`, keep the forces in PROTOCOL.md, and report time to a settled layout plus the two quality checks. Open an issue first so we agree how it's measured.
