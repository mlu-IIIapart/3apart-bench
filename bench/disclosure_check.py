"""Re-check the disclosure's speed claim on this machine: 100,000 markers placed in 9 to 12 ms (NumPy, one core).

Protocol (fixed, see PROTOCOL.md): 100 runs; each run is the mean of 20 placements of 100,000 markers.
Then a size sweep from 1,000 to 10,000,000 markers, 100 runs per size (fewer with --quick).
With --xl the sweep adds 100,000,000 markers (10 runs; needs about 12 GB of free memory).

Usage: python bench/disclosure_check.py --who <name> [--machine <name>] [--quick] [--xl]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"  # one core, as the claim states

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "py"))
import engine as E  # noqa: E402

CLAIM = (9.0, 12.0)


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def machine(name):
    cpu = platform.processor() or "unknown"
    if sys.platform == "darwin":
        cpu = sh("sysctl -n machdep.cpu.brand_string") or cpu
    elif Path("/proc/cpuinfo").exists():
        m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
        cpu = m.group(1).strip() if m else cpu
    return {"name": name or os.environ.get("BENCH_MACHINE") or platform.node().replace(".local", ""),
            "cpu": cpu, "model": sh("sysctl -n hw.model") if sys.platform == "darwin" else None,
            "cores": os.cpu_count(), "os": f"{platform.system()} {platform.release()} {platform.machine()}"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who", default=os.environ.get("BENCH_WHO"), help="your name; labels your results")
    ap.add_argument("--machine", default=None)
    ap.add_argument("--quick", action="store_true", help="fewer runs, no 10M size (CI / first look)")
    ap.add_argument("--xl", action="store_true", help="also place 100,000,000 markers, 10 runs (needs about 12 GB of free memory)")
    a = ap.parse_args()
    if not a.who:
        ap.error("say who is running this: --who <your-name> (or set BENCH_WHO)")
    who = slug(a.who)
    mach = machine(a.machine)
    log = lambda s: print(s, file=sys.stderr, flush=True)
    log(f"3apart disclosure re-check · {mach['name']} · {mach['cpu']}")

    n = 100_000
    layer, t, v = E.inputs(n)
    for _ in range(20):
        E.place(layer, t, v)
    runs = 20 if a.quick else 100
    means = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(20):
            xyz = E.place(layer, t, v)
        means.append((time.perf_counter() - t0) * 1000 / 20)
    s = E.stats(means)
    outside = int(round(E.outside_share(layer, xyz) * n))
    record = dict(s, means_ms=means, claim=list(CLAIM), below_9=sum(m < CLAIM[0] for m in means),
                  within_9_12=sum(CLAIM[0] <= m <= CLAIM[1] for m in means), above_12=sum(m > CLAIM[1] for m in means),
                  nodes_outside_own_band=outside)
    verdict = "HOLDS" if record["above_12"] == 0 and outside == 0 else "DOES NOT HOLD"
    log(f"  100,000 markers: median {s['median']:.2f} ms, worst {s['max']:.2f} ms; "
        f"{record['below_9']} under 9 ms, {record['within_9_12']} in 9–12, {record['above_12']} over 12; "
        f"{outside} outside band  ->  claim {verdict} on this machine")

    sweep = {}
    sizes = [1_000, 10_000, 100_000, 1_000_000] + ([] if a.quick else [10_000_000]) + ([100_000_000] if a.xl else [])
    for size in sizes:
        layer, t, v = E.inputs(size)
        for _ in range(1 if size >= 100_000_000 else 3 if size >= 1_000_000 else 20):
            E.place(layer, t, v)
        ms = []
        for _ in range(10 if size >= 100_000_000 else 20 if a.quick else 100):
            t0 = time.perf_counter()
            xyz = E.place(layer, t, v)
            ms.append((time.perf_counter() - t0) * 1000)
        st = E.stats(ms)
        st["ns_per_marker"] = st["median"] * 1e6 / size
        st["nodes_outside_own_band"] = int(round(E.outside_share(layer, xyz) * size))
        sweep[str(size)] = st
        log(f"  {size:>10,} markers: median {st['median']:.3f} ms ({st['ns_per_marker']:.0f} ns/marker)")

    out = {"schema": 1, "kind": "numpy", "who": who, "machine": mach,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "commit": sh(f'git -C "{ROOT}" rev-parse --short HEAD'), "plan": ("quick" if a.quick else "standard") + ("+xl" if a.xl else ""),
           "versions": {"python": platform.python_version(), "numpy": np.__version__},
           "protocol": {"seed": E.SEED, "bands": E.NL, "cap": E.CAP, "margin": E.MARGIN},
           "claim_check": record, "sweep": sweep}
    d = ROOT / "results" / who / slug(mach["name"])
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S')}-numpy.json"
    f.write_text(json.dumps(out, indent=1) + "\n")
    log(f"  saved {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
