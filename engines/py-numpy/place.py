#!/usr/bin/env python3
"""NumPy engine implementing the engine contract in engines/README.md."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "py"))
import engine as E  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("inputs", "xyz", "json"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--runs", type=int, required=True)
ap.add_argument("--warmup", type=int, default=5)
a = ap.parse_args()

layer, t, v, _ = E.read_inputs(a.inputs)
for _ in range(a.warmup):
    xyz = E.place(layer, t, v)
ms = []
for _ in range(a.runs):
    t0 = time.perf_counter()
    xyz = E.place(layer, t, v)
    ms.append((time.perf_counter() - t0) * 1000)
np.ascontiguousarray(xyz, dtype="<f8").tofile(a.xyz)
Path(a.json).write_text(json.dumps({"engine": "py-numpy", "language": f"Python {sys.version.split()[0]} / NumPy {np.__version__}",
                                    "precision": "f64", "device": "cpu", "threads": 1, "ms": ms}))
