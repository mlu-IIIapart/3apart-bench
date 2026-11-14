"""Additive FP64 NumPy placement benchmark with full-output verification.

The original disclosure runner, engine and protocol are unchanged. Default run:
100M markers in four 25M batches; three resident warmups, ten resident samples,
then three independent cold pipelines. Validation is outside every timing clock.
"""
from __future__ import annotations

import os

THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
)
for _name in THREAD_VARIABLES:
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "py"))
import engine as E  # noqa: E402

VARIANT = "numpy-fp64-batched-full-output-e2e-v1"
TOLERANCE = 1e-9
MAX_REFERENCE_CHUNK = 1_000_000


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_identity():
    return {str(p.relative_to(ROOT)): digest(p) for p in (
        Path(__file__).resolve(), Path(E.__file__).resolve(),
    )}


def command(argv):
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def peak_rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def slices(n, batch):
    if n <= 0 or batch <= 0:
        raise ValueError("Marker and batch counts must be positive")
    return [(first, min(batch, n - first)) for first in range(0, n, batch)]


def require_coverage(n, ranges):
    """Require the actual write intervals once, in global order, without gaps."""
    cursor = 0
    for first, count in ranges:
        if not isinstance(first, int) or not isinstance(count, int):
            raise ValueError("Coverage offsets and counts must be integers")
        if first != cursor or count <= 0 or first + count > n:
            raise ValueError(f"Invalid coverage at {first}+{count}; expected offset {cursor}")
        cursor += count
    if cursor != n:
        raise ValueError(f"Incomplete coverage: {cursor} of {n} markers")
    return cursor


def place_batches(layer, times, values, output, batch):
    """Use unchanged E.place; retain the complete output in global marker order."""
    n = len(layer)
    if times.shape != (n,) or values.shape != (n,) or output.shape != (n, 3):
        raise ValueError("Input/output shape mismatch")
    ranges = []
    for first, count in slices(n, batch):
        end = first + count
        output[first:end] = E.place(layer[first:end], times[first:end], values[first:end])
        ranges.append((first, count))
    require_coverage(n, ranges)
    return ranges


def validate_output(layer, times, values, output, ranges, chunk=MAX_REFERENCE_CHUNK):
    """Independent algebraic oracle on original FP64 inputs, in bounded chunks.

    E.place uses acos then sin/cos for height/radius. This oracle uses z directly
    and sqrt(1-z*z), with explicit finite, coverage and normalized-band gates.
    It checks every XYZ coordinate; it does not claim an angular-order audit.
    """
    n = len(layer)
    require_coverage(n, ranges)
    if times.shape != (n,) or values.shape != (n,) or output.shape != (n, 3):
        raise ValueError("Input/output shape mismatch")
    if not 1 <= chunk <= MAX_REFERENCE_CHUNK:
        raise ValueError("Reference chunks must contain between1 and1M markers")
    counters = dict(markers_checked=0, xyz_coordinates_checked=0,
                    nonfinite_inputs=0, invalid_input_markers=0,
                    nonfinite_output_coordinates=0, nonfinite_reference_coordinates=0,
                    coordinates_over_tolerance=0, outside_band=0, invalid_radius=0)
    maximum = 0.0
    # Independently reconstruct protocol constants rather than call E.place.
    edges = 0.9 * (1 - 2 * np.arange(25, dtype=np.float64) / 24)
    for first, count in slices(n, chunk):
        end = first + count
        band, t, v, got = layer[first:end], times[first:end], values[first:end], output[first:end]
        finite_input = np.isfinite(band) & np.isfinite(t) & np.isfinite(v)
        valid = finite_input & (band >= 0) & (band < 24) & (band == np.floor(band))
        valid &= (t >= 0) & (t < 1) & (v >= 0) & (v < 1)
        counters["nonfinite_inputs"] += int(np.count_nonzero(~finite_input))
        counters["invalid_input_markers"] += int(np.count_nonzero(~valid))
        safe_band = np.where(valid, band, 0).astype(np.intp)
        z_top, z_bottom = edges[safe_band], edges[safe_band + 1]
        z = z_bottom + (z_top - z_bottom) * (0.12 + 0.76 * v)
        phi = np.pi * (2 * t - 1)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            radius_xy = np.sqrt(1 - z * z)
            reference = np.column_stack((radius_xy * np.sin(phi), -radius_xy * np.cos(phi), z))
            finite_out, finite_ref = np.isfinite(got), np.isfinite(reference)
            counters["nonfinite_output_coordinates"] += int(np.count_nonzero(~finite_out))
            counters["nonfinite_reference_coordinates"] += int(np.count_nonzero(~finite_ref))
            error = np.abs(got - reference)
            finite_error = np.isfinite(error)
            if np.any(finite_error):
                maximum = max(maximum, float(np.max(error[finite_error])))
            counters["coordinates_over_tolerance"] += int(np.count_nonzero(finite_error & (error > TOLERANCE)))
            norm = np.sqrt(np.sum(got * got, axis=1))
            good_radius = np.isfinite(norm) & (norm > 0)
            counters["invalid_radius"] += int(np.count_nonzero(~good_radius))
            height = np.divide(got[:, 2], norm, out=np.full(count, np.nan), where=good_radius)
            counters["outside_band"] += int(np.count_nonzero(~((height <= z_top) & (height >= z_bottom))))
        counters["markers_checked"] += count
        counters["xyz_coordinates_checked"] += 3 * count
    failures = [k for k, v in counters.items() if k not in ("markers_checked", "xyz_coordinates_checked") and v]
    passed = not failures and counters["markers_checked"] == n and counters["xyz_coordinates_checked"] == 3 * n
    return dict(counters, passed=passed, max_abs_error=maximum, tolerance=TOLERANCE,
                reference="Independent FP64 sqrt radius/direct z formula on original E.inputs stream",
                reference_chunk_markers=chunk, coverage_ranges=[list(x) for x in ranges],
                angular_order_tested=False, failing_gates=failures)


def timed_validation(layer, times, values, output, ranges, chunk):
    start = time.perf_counter_ns()
    result = validate_output(layer, times, values, output, ranges, chunk)
    result["wall_ms"] = (time.perf_counter_ns() - start) / 1e6
    if not result["passed"]:
        raise ValueError("Full-output validation failed: " + json.dumps(result))
    return result


def statistics(values):
    if not values or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("Timing samples must be finite and positive")
    return E.stats(values)


def benchmark(args, record, checkpoint):
    n, batch = args.markers, args.batch_markers
    plan = slices(n, batch)
    require_coverage(n, plan)
    record["resident"] = {"warmup_ms": [], "samples": [], "validation": None}
    record["cold_pipelines"] = []
    before = time.perf_counter_ns()
    layer, times, values = E.inputs(n)
    generated = time.perf_counter_ns()
    output = np.full((n, 3), np.nan, dtype=np.float64)
    ready = time.perf_counter_ns()
    record["resident"]["setup"] = {
        "input_generation_ms": (generated - before) / 1e6,
        "output_allocation_and_initialization_ms": (ready - generated) / 1e6,
        "continuous_setup_ms": (ready - before) / 1e6,
    }
    for trial in range(args.warmups + args.runs):
        # Clearing is a coverage sentinel, outside resident placement timing.
        output.fill(np.nan)
        begin = time.perf_counter_ns()
        coverage = place_batches(layer, times, values, output, batch)
        elapsed = (time.perf_counter_ns() - begin) / 1e6
        if trial < args.warmups:
            record["resident"]["warmup_ms"].append(elapsed)
        else:
            record["resident"]["samples"].append({"trial": trial - args.warmups,
                "placement_and_complete_output_materialization_ms": elapsed,
                "coverage_ranges": [list(x) for x in coverage]})
        checkpoint(f"resident {trial + 1}/{args.warmups + args.runs}")
    record["resident"]["validation"] = timed_validation(layer, times, values, output, coverage, args.reference_chunk)
    resident_ms = [r["placement_and_complete_output_materialization_ms"] for r in record["resident"]["samples"]]
    record["resident"]["statistics_ms"] = statistics(resident_ms)
    del layer, times, values, output
    gc.collect()  # Release resident data before independent cold pipelines.
    for trial in range(args.cold_runs):
        start = time.perf_counter_ns()
        layer, times, values = E.inputs(n)
        generated = time.perf_counter_ns()
        output = np.full((n, 3), np.nan, dtype=np.float64)
        allocated = time.perf_counter_ns()
        coverage = place_batches(layer, times, values, output, batch)
        completed = time.perf_counter_ns()
        item = {"trial": trial,
            "end_to_end_ms": (completed - start) / 1e6,
            "input_generation_ms": (generated - start) / 1e6,
            "output_allocation_and_initialization_ms": (allocated - generated) / 1e6,
            "placement_and_complete_output_materialization_ms": (completed - allocated) / 1e6,
            "coverage_ranges": [list(x) for x in coverage]}
        # The recorded E2E clock ends only after all100M output rows exist in RAM.
        item["validation"] = timed_validation(layer, times, values, output, coverage, args.reference_chunk)
        record["cold_pipelines"].append(item)
        checkpoint(f"cold pipeline {trial + 1}/{args.cold_runs}")
        del layer, times, values, output
        gc.collect()
    record["cold_statistics_ms"] = statistics([x["end_to_end_ms"] for x in record["cold_pipelines"]])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--who", default="joel")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markers", type=int, default=100_000_000)
    parser.add_argument("--batch-markers", type=int, default=25_000_000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cold-runs", type=int, default=3)
    parser.add_argument("--reference-chunk", type=int, default=MAX_REFERENCE_CHUNK)
    args = parser.parse_args(argv)
    if min(args.markers, args.batch_markers, args.runs, args.cold_runs) < 1 or args.warmups < 0:
        parser.error("Counts must be positive (warmups may be zero)")
    if not 1 <= args.reference_chunk <= MAX_REFERENCE_CHUNK:
        parser.error("Reference chunk must be1..1000000")
    if args.output.exists():
        parser.error("Output already exists; preserve it and choose a new path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = source_identity()
    record = {
        "schema": 1, "kind": "numpy-xl-e2e", "variant": VARIANT,
        "status": "running", "passed": False, "at": now(),
        "who": args.who, "machine": {"name": args.machine, "hostname": platform.node(),
            "cpu": command(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
            "os": platform.platform(), "logical_cpus": os.cpu_count()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "thread_environment": {k: os.environ[k] for k in THREAD_VARIABLES},
        "precision": "FP64 NumPy; WebGL comparisons use FP32 and are separate implementations",
        "source_hashes": sources, "git_head": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "protocol": {"seed": E.SEED, "bands": E.NL, "cap": E.CAP, "margin": E.MARGIN},
        "plan": {"markers": args.markers, "batch_markers": args.batch_markers,
            "batches": [list(x) for x in slices(args.markers, args.batch_markers)],
            "warmups": args.warmups, "resident_runs": args.runs, "cold_runs": args.cold_runs},
        "bytes": {"input_band_i32": args.markers * 4, "input_time_value_f64": args.markers * 16,
            "complete_output_xyz_f64": args.markers * 24,
            "largest_batch_output_temporary_f64": min(args.markers, args.batch_markers) * 24},
        "scope": {
            "resident": "Original inputs and complete output allocation persist. Timed unchanged E.place calls plus batch output copies and coverage accounting. Sentinel clearing, setup and validation excluded.",
            "cold": "Three independent application pipelines by default: continuous perf_counter_ns from original E.inputs generation through output allocation/NaN initialization and complete batch placement/materialization. No stage-median summation. Cold means new arrays, not OS/filesystem cache eviction or process restart.",
            "validation": "Every XYZ coordinate of final resident output and each cold output, outside timing. Independent FP64 algebraic formula, bounded1M chunks, finite/band/coverage gates. Every timed resident output is not independently validated.",
            "excluded": "Imports, process startup, input/output files, network, reference validation and result serialization. No GPU transfer, detector scoring or alerts are involved.",
            "peak_rss": "Process lifetime high-water RSS including Python/NumPy and validation; not incremental allocation or instantaneous memory.",
            "parallelism": "Requested numerical-library threads1; no affinity or exclusive-machine claim.",
        },
        "peak_rss_bytes": peak_rss_bytes(),
    }
    # Reserve a new result before work. Interrupted runs remain explicitly running.
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")

    def checkpoint(phase):
        record["phase"] = phase
        record["peak_rss_bytes"] = peak_rss_bytes()
        record["updated_at"] = now()
        temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)
        print(json.dumps({"phase": phase, "status": record["status"], "peak_rss_bytes": record["peak_rss_bytes"]}), file=sys.stderr, flush=True)

    try:
        benchmark(args, record, checkpoint)
        record["source_unchanged"] = source_identity() == sources
        if not record["source_unchanged"]:
            raise ValueError("Benchmark source changed during run")
        record["status"] = "passed"
        record["passed"] = True
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["completed_at"] = now()
    checkpoint("complete")
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
