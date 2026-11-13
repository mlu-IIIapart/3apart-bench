"""3apart shared engine (Python / NumPy).

Same protocol constants, the same input stream as src/js/engine.mjs (bit-identical),
and a vectorised NumPy placement equal to the disclosure's place() + sphere_xyz().

If the disclosure's own reference code differs in any way from `place` below, put it in
reference/ unchanged and make bench/disclosure_check.py call it: the re-check must run
the disclosure's code, not a port of it. See PROTOCOL.md.
"""
import struct
import numpy as np

NL = 24
CAP = 0.9
MARGIN = 0.12
SEED = 20261112
EDGES = CAP * (1 - 2 * np.arange(NL + 1) / NL)

_GOLD = np.uint32(0x6D2B79F5)


def _mulberry32_stream(seed, count):
    """The k-th draw of mulberry32, for k = 1..count, computed in parallel."""
    k = np.arange(1, count + 1, dtype=np.uint64)
    s = ((np.uint64(seed & 0xFFFFFFFF) + k * np.uint64(0x6D2B79F5)) & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    with np.errstate(over="ignore"):
        t = s
        t = (t ^ (t >> np.uint32(15))) * (t | np.uint32(1))
        t = t ^ (t + (t ^ (t >> np.uint32(7))) * (t | np.uint32(61)))
        t = t ^ (t >> np.uint32(14))
    return t.astype(np.float64) / 4294967296.0


def inputs(n, seed=SEED):
    """Band, time and value per marker, drawn in that order (same stream as the JS engine)."""
    r = _mulberry32_stream(seed, 3 * n).reshape(n, 3)
    layer = np.floor(r[:, 0] * NL).astype(np.int32)
    return layer, r[:, 1].copy(), r[:, 2].copy()


def place(layer, t, value):
    """The disclosure's place() + sphere_xyz(): one pass, returns an (n, 3) array."""
    phi = np.pi * (2 * t - 1)
    zt, zb = EDGES[layer], EDGES[layer + 1]
    z = zb + (zt - zb) * (MARGIN + (1 - 2 * MARGIN) * value)
    th = np.arccos(np.clip(z, -1.0, 1.0))
    st = np.sin(th)
    return np.stack((st * np.sin(phi), -st * np.cos(phi), np.cos(th)), axis=1)


def outside_share(layer, xyz):
    """Share of markers whose height on the sphere is outside their own band."""
    r = np.linalg.norm(xyz, axis=1)
    r[r == 0] = 1
    h = xyz[:, 2] / r
    return float(np.mean(~((h <= EDGES[layer]) & (h >= EDGES[layer + 1]))))


def time_order(layer, t, xyz):
    """Share of time-consecutive pairs in a band that run counter-clockwise around the globe."""
    ord_ = np.lexsort((t, layer))
    lp, lq = layer[ord_[:-1]], layer[ord_[1:]]
    same = lp == lq
    a = np.arctan2(xyz[ord_[:-1], 1], xyz[ord_[:-1], 0])
    b = np.arctan2(xyz[ord_[1:], 1], xyz[ord_[1:], 0])
    d = b - a
    ok = (np.sin(d) > 0) | (np.abs(d) < 1e-12)
    tot = int(same.sum())
    return float((ok & same).sum() / tot) if tot else 1.0


def write_inputs(path, layer, t, value, seed=SEED):
    n = len(layer)
    with open(path, "wb") as f:
        f.write(b"3AIN" + struct.pack("<III", 1, n, seed & 0xFFFFFFFF))
        f.write(layer.astype("<i4").tobytes())
        f.write(t.astype("<f8").tobytes())
        f.write(value.astype("<f8").tobytes())


def read_inputs(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:4] != b"3AIN":
        raise ValueError("not a 3apart inputs file")
    _, n, seed = struct.unpack("<III", raw[4:16])
    o = 16
    layer = np.frombuffer(raw, "<i4", n, o)
    t = np.frombuffer(raw, "<f8", n, o + 4 * n)
    v = np.frombuffer(raw, "<f8", n, o + 12 * n)
    return layer.astype(np.int64), t.copy(), v.copy(), seed


def stats(ms):
    a = np.sort(np.asarray(ms, dtype=np.float64))
    n = len(a)
    return {"runs": n, "min": float(a[0]), "median": float(np.median(a)), "mean": float(a.mean()),
            "p95": float(a[min(n - 1, int(np.ceil(0.95 * n)) - 1)]), "max": float(a[-1])}
