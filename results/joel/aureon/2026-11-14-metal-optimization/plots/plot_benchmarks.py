#!/usr/bin/env python3
"""Deterministic, evidence-preserving Aureon benchmark figures.

Run in the adjacent .venv. Sources are never modified. Each dot is an observed
trial, with a diamond marking its median; no inferential intervals are implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.text import Text
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
INK = "#171717"
SECONDARY = "#404040"
GRAY = "#7F8582"
ACCENT = "#1F6F5F"
LIGHT = "#A1A6A3"
GRID = "#E4E7E5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 14,
    "text.color": INK, "axes.labelcolor": SECONDARY,
    "xtick.color": SECONDARY, "ytick.color": INK,
    "svg.fonttype": "none", "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def validated(trials, expected):
    for t in trials:
        check = t["validation"]
        assert check["pass"] is True
        assert check["actual_coordinates"] == expected
        assert check["expected_coordinates"] == expected
        assert check["nonfinite"] == 0
        assert check["outside_band"] == 0
        assert check["coordinates_over_5e_4"] == 0


def figure_base(title, subtitle):
    fig = plt.figure(figsize=(16, 10), dpi=100)
    fig.text(.055, .945, title, fontsize=27, fontweight="semibold", va="top")
    fig.text(.055, .888, subtitle, fontsize=17, color=SECONDARY, va="top")
    return fig


def draw_rows(fig, rows, xlabel, bounds=(.34, .25, .56, .55), xlim=None,
              separators=()):
    ax = fig.add_axes(bounds)
    count = len(rows)
    for index, row in enumerate(rows):
        y = count - index - 1
        values = np.array(row["values"], dtype=float)
        assert np.all(np.isfinite(values)) and np.all(values >= 0)
        med = statistics.median(values)
        color = ACCENT if row.get("focal") else GRAY
        # The deterministic offsets separate repeated values, not uncertainty.
        offsets = np.array([0, .13, -.13, .065, -.065, .19, -.19, .10, -.10, .025])
        offsets = np.resize(offsets, len(values))
        ax.scatter(values, y + offsets, s=45, facecolors="white", edgecolors=color,
                   linewidths=1.25, zorder=3)
        ax.scatter([med], [y], marker="D", s=83, facecolors=color,
                   edgecolors="white", linewidths=.8, zorder=4)
        ax.text(-.025, y, row["label"], transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=14.5, linespacing=1.30,
                color=INK, fontweight="semibold" if row.get("focal") else "normal")
        ax.text(1.045, y, f"{med:.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=16,
                color=ACCENT if row.get("focal") else INK,
                fontweight="semibold" if row.get("focal") else "normal")
        row["median_ms"] = float(med)
        row["sample_count"] = len(values)
    for y in separators:
        ax.axhline(y, color=GRID, lw=.8)
    ax.set_ylim(-.6, count - .4)
    if xlim is None:
        upper = max(max(r["values"]) for r in rows)
        xlim = (0, upper * 1.09)
    ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6, min_n_ticks=4))
    ax.tick_params(axis="x", labelsize=14, length=4, pad=8, color=GRAY)
    ax.set_xlabel(xlabel, labelpad=17, fontsize=16)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["bottom"].set_linewidth(1)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=.7)
    fig.text(bounds[0], bounds[1]+bounds[3]+.04,
             "○  Individual trial     ◆  Median", fontsize=14, color=SECONDARY)
    fig.text(bounds[0]+bounds[2]*1.045, bounds[1]+bounds[3]+.04,
             "Median (ms)", fontsize=14, color=SECONDARY, ha="right")
    return ax


def export(fig, stem, metadata):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    clipping = []
    for obj in fig.findobj(match=Text):
        if not obj.get_visible() or not obj.get_text():
            continue
        box = obj.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width+1 or box.y1 > height+1:
            clipping.append({"text": obj.get_text(), "bbox": list(box.bounds)})
    assert not clipping, clipping
    fig.savefig(HERE / f"{stem}.png", dpi=100, facecolor="white")
    fig.savefig(HERE / f"{stem}@2x.png", dpi=200, facecolor="white")
    fig.savefig(HERE / f"{stem}.svg", facecolor="white")
    plt.close(fig)
    original = Image.open(HERE / f"{stem}.png").convert("RGB")
    assert original.size == (1600, 1000)
    ImageOps.grayscale(original).save(HERE / f"{stem}-grayscale.png")
    # Machado et al. (2009), complete deuteranomaly/deuteranopia approximation.
    # Redundant position/shape/direct labels carry the information regardless.
    rgb = np.asarray(original, dtype=float) / 255
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    matrix = np.array([[.367322,.860646,-.227968],
                       [.280085,.672501,.047413],
                       [-.011820,.042940,.968881]])
    simulated = np.clip(linear @ matrix.T, 0, 1)
    srgb = np.where(simulated <= .0031308, 12.92*simulated,
                    1.055*np.power(simulated, 1/2.4)-.055)
    Image.fromarray(np.uint8(np.clip(srgb*255+.5, 0, 255))).save(
        HERE / f"{stem}-deuteranopia.png")
    metadata["qa"] = {"dimensions_px": [1600, 1000], "text_bounds_check": "pass",
                      "minimum_font_pt": 13.5, "inferential_intervals": "none",
                      "all_raw_trials_shown": True, "visual_review": "pending"}
    dump(HERE / f"{stem}-evidence.json", metadata)


def plot_compute(path):
    source = json.loads(Path(path).read_text())
    assert source["n"] == 100_000_000
    variants = {v["variant"]:v for v in source["variants"]}
    order = [
        ("safe_precise_trig_256", "Safe math · precise functions\n256 threads"),
        ("safe_fast_trig_256", "Safe math · fast functions\n256 threads"),
        ("fast_fast_trig_256", "Fast math · fast functions\n256 threads"),
        ("fast_fast_trig_64", "Fast math · fast functions\n64 threads"),
        ("fast_fast_trig_512", "Fast math · fast functions\n512 threads"),
        ("safe_precise_algebra_256", "Safe math · precise functions\n256 threads · algebraic rewrite"),
        ("safe_fast_algebra_256", "Safe math · fast functions\n256 threads · algebraic rewrite"),
        ("fast_fast_algebra_256", "Fast math · fast functions\n256 threads · algebraic rewrite"),
    ]
    assert set(variants) == {v[0] for v in order}
    rows = []
    for name, label in order:
        v = variants[name]
        assert len(v["trials"]) == source["trials_per_variant"] == 10
        validated(v["trials"], source["n"]*3)
        rows.append({"id": name, "label": label,
                     "values": [t["gpu_device_ms"] for t in v["trials"]],
                     "focal": name == "safe_fast_trig_256"})
    baseline = statistics.median(rows[0]["values"])
    focal = statistics.median(rows[1]["values"])
    fig = figure_base("100 million markers: native Metal compute",
        "Original trig formula in the first five rows; algebraic rewrites shown separately")
    draw_rows(fig, rows, "Resident GPU time (ms)  ·  Lower is better",
              bounds=(.36,.255,.55,.55), xlim=(0,11), separators=(2.5,))
    fig.text(.055, .140,
             f"Changing only the math functions to fast: {baseline/focal:.2f}× baseline throughput in this run.",
             fontsize=16.5, color=ACCENT, fontweight="semibold")
    fig.text(.055, .098,
             "Apple M5 Max · 10 trials per variant, randomized within each round · 3 warmups excluded.",
             fontsize=13.5, color=SECONDARY)
    fig.text(.055, .066,
             "Every output passed all 300 million coordinate checks against the original FP64 reference (tolerance 0.0005).",
             fontsize=13.5, color=SECONDARY)
    fig.text(.055, .034,
             "Resident GPU timer excludes preparation, output copy and validation. Trial spread is shown; diamonds are medians.",
             fontsize=13.5, color=SECONDARY)
    export(fig, "01-metal-compute", {"source":str(Path(path).resolve()),
        "sha256":sha256(path), "metric":"gpu_device_ms", "unit":"ms",
        "aggregation":"median", "direction":"lower", "rows": rows,
        "focal_speedup_ratio_of_medians":baseline/focal,
        "hardware":source["device"], "n":source["n"],
        "required_baseline":"safe_precise_trig_256"})


def plot_validation(path):
    source = json.loads(Path(path).read_text())
    assert source["coordinates"] == 300_000_000
    assert source["all_summaries_bit_identical"] is True
    confirmation = source["paired_confirmation"]
    assert confirmation["pairs"] == 10
    assert confirmation["workers_selected_from_sweep"] == 18
    serial = confirmation["serial_raw_ms"]
    parallel = confirmation["parallel_raw_ms"]
    assert math.isclose(statistics.median(serial), confirmation["serial_median_ms"], abs_tol=1e-10)
    assert math.isclose(statistics.median(parallel), confirmation["parallel_median_ms"], abs_tol=1e-10)
    ratio = statistics.median(serial) / statistics.median(parallel)
    fig = figure_base(f"Full accuracy checks: {ratio:.1f}× faster in this run",
        "Same 300 million coordinates, same FP64 reference, same acceptance tests")
    ax = fig.add_axes((.36,.29,.55,.49))
    ys = np.arange(10, 0, -1)
    for y, before, after in zip(ys, serial, parallel):
        ax.plot([after,before], [y,y], color=GRID, lw=1.5, zorder=1)
    ax.scatter(serial,ys, s=64, facecolors="white", edgecolors=GRAY,
               linewidths=1.6, zorder=3)
    ax.scatter(parallel,ys, s=64, facecolors=ACCENT, edgecolors=ACCENT,
               linewidths=1.6, zorder=3)
    ax.set_yticks(ys, [f"Pair {i}" for i in range(1,11)])
    ax.tick_params(axis="y", length=0, pad=16, labelsize=14)
    ax.tick_params(axis="x", labelsize=14, pad=8, color=GRAY)
    ax.set_xlim(0,400)
    ax.set_ylim(.35,10.65)
    ax.set_xticks([0,100,200,300,400])
    ax.set_xlabel("Full validation time (ms)  ·  Lower is better", fontsize=16, labelpad=17)
    for name in ("top","left","right"):
        ax.spines[name].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID,lw=.7)
    ax.text(0,1.045,"●  Parallel · 18 workers",transform=ax.transAxes,
            fontsize=15,color=ACCENT,ha="left")
    ax.text(1,1.045,"○  Original · serial",transform=ax.transAxes,
            fontsize=15,color=SECONDARY,ha="right")
    fig.text(.055,.70,"Median validation",fontsize=16,color=SECONDARY)
    fig.text(.055,.625,f"{statistics.median(serial):.2f} ms",fontsize=29,
             color=INK,fontweight="semibold")
    fig.text(.055,.582,"Original serial validator",fontsize=14,color=SECONDARY)
    fig.text(.055,.48,f"{statistics.median(parallel):.2f} ms",fontsize=29,
             color=ACCENT,fontweight="semibold")
    fig.text(.055,.437,"Exact parallel validator",fontsize=14,color=SECONDARY)
    fig.text(.055,.14,"Every reported validation summary was bit-for-bit identical.",
             fontsize=16.5,color=ACCENT,fontweight="semibold")
    fig.text(.055,.098,"Aureon · 10 confirmation pairs; serial/parallel execution order alternated AB, BA.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.066,"Each point is one complete validation. Finiteness, max error, tolerance, band and unit-radius checks retained.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.034,"Resident CPU output; file loading, thread-pool creation and GPU computation excluded. No sampling of coordinates.",
             fontsize=13.5,color=SECONDARY)
    export(fig,"02-full-validation",{"source":str(Path(path).resolve()),"sha256":sha256(path),
        "metric":"paired_confirmation serial_raw_ms / parallel_raw_ms","unit":"ms",
        "aggregation":"median","direction":"lower","pair_count":10,
        "serial_values":serial,"parallel_values":parallel,"speedup_ratio_of_medians":ratio,
        "all_summaries_bit_identical":True,"source_reports_no_gpu_work":True,
        "median_serial_ms":statistics.median(serial),"median_parallel_ms":statistics.median(parallel)})


def plot_integrated(path):
    source = json.loads(Path(path).read_text())
    assert source["n"] == 100_000_000
    variants = {v["variant"]:v for v in source["variants"]}
    order = [
        ("baseline_copy_serial", "Precise functions\nCopy + serial checks"),
        ("fast_intrinsics_copy_serial", "Fast functions\nCopy + serial checks"),
        ("baseline_direct_parallel", "Precise functions\nShared output + 18-worker checks"),
        ("fast_intrinsics_direct_parallel", "Fast functions\nShared output + 18-worker checks"),
    ]
    assert set(variants) == {v[0] for v in order}
    rows = []
    for name, label in order:
        v = variants[name]
        assert v["formula"] == "original acos/sin/cos"
        assert v["math_mode"] == "safe"
        assert v["threads"] == 256
        assert len(v["trials"]) == source["trials_per_variant"] == 10
        validated(v["trials"], source["n"]*3)
        assert all(t["all_checker_fields_equal_serial_oracle"] for t in v["trials"])
        rows.append({"id": name, "label": label,
            "values": [t["compute_output_and_validation_ms"] for t in v["trials"]],
            "focal": name == "fast_intrinsics_direct_parallel"})
    ratio = statistics.median(rows[0]["values"])/statistics.median(rows[-1]["values"])
    fig = figure_base(f"Compute + complete checking: {ratio:.1f}× faster",
        "Inputs resident · original trig formula · safe math mode · 256 threads")
    draw_rows(fig, rows, "Compute + output access + full validation (ms)  ·  Lower is better",
              bounds=(.36,.295,.55,.50), xlim=(0,460), separators=(1.5,))
    fig.text(.055,.168,"Parallel checking and direct output access account for most of the gain.",
             fontsize=16.5,color=ACCENT,fontweight="semibold")
    fig.text(.055,.122,"Apple M5 Max · 10 randomized trials per row · Every output passed all 300 million coordinate checks.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.090,"Shared output is CPU-visible after GPU completion; it avoids an extra 1.2 GB copy, with no separate owned copy.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.058,"Timing excludes input preparation, compilation and an additional independent serial-oracle verification.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.026,"All check summaries matched that oracle exactly. Circles show individual trials; diamonds show medians.",
             fontsize=13.5,color=SECONDARY)
    export(fig, "03-compute-and-checking",{"source":str(Path(path).resolve()),
        "sha256":sha256(path),"metric":"compute_output_and_validation_ms",
        "unit":"ms","aggregation":"median","direction":"lower","rows":rows,
        "speedup_ratio_of_medians":ratio,"hardware":source["device"],"n":source["n"],
        "output_endpoint_caveat":"Direct shared output avoids a separate owned copy; copy variants create that copy.",
        "all_checker_fields_equal_serial_oracle":True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--integrated", type=Path)
    args = parser.parse_args()
    if args.compute:
        plot_compute(args.compute)
    if args.validation:
        plot_validation(args.validation)
    if args.integrated:
        plot_integrated(args.integrated)


if __name__ == "__main__":
    main()
