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
        if len(values)>1:
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
    assert all(xlim[0]<=min(r["values"]) and max(r["values"])<=xlim[1] for r in rows), "a raw observation would be clipped"
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
             "Median / one-off (ms)" if any(len(r["values"])==1 for r in rows) else "Median (ms)",
             fontsize=14, color=SECONDARY, ha="right")
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


def full_trials(trials, n):
    validated(trials,n*3)
    for t in trials:
        assert t["full_coverage"] is True
        assert t["all_fields_equal_serial_oracle"] is True
        if "reused_markers" in t:
            assert t["rechecked_markers"]+t["reused_markers"] == n
            assert t["cache_valid"] is True


def load_groups(path):
    d=json.loads(Path(path).read_text())
    assert d["n"] == 100_000_000 and d["workers"] == 18
    groups={}
    for t in d["trials"]+d["changed_output_trials"]:
        name=t.get("scenario",t["mode"])
        groups.setdefault(name,[]).append(t)
    assert set(groups)=={"full_cpu","sparse_cpu","full_gpu_comparison_exact_cpu_revalidation",
                         "three_changed_words","every_block_changed"}
    for name,trials in groups.items():
        assert len(trials)==10
        if name=="sparse_cpu":
            for t in trials:
                assert t["full_coverage"] is False
                assert t["covers_all_markers"] is False
                assert t["checked_markers"]==65536 and t["checked_coordinates"]==196608
                assert t["sample_plan_reused"] is True
                assert t["validation"]["pass"] is True
                assert t["independent_full_serial_audit_pass"] is True
        else:
            full_trials(trials,d["n"])
    full_trials([d["initial_certification"]],d["n"])
    return d,groups


def plot_full(path,d,groups):
    order=[("full_cpu","CPU full check · 18 workers\nAll 100M markers rechecked"),
           ("full_gpu_comparison_exact_cpu_revalidation","GPU cache · unchanged output\n100M markers reuse exact checks"),
           ("three_changed_words","GPU cache · 3 coordinates changed\n8,448 markers rechecked"),
           ("every_block_changed","GPU cache · every block changed\nAll 100M markers rechecked"),
           ("initial_certification","GPU cache · first certification\nAll 100M checked · one observation")]
    rows=[]
    for name,label in order:
        trials=[d[name]] if name=="initial_certification" else groups[name]
        rows.append({"id":name,"label":label,"values":[t["validation_ms"] for t in trials],
                     "focal":name=="full_gpu_comparison_exact_cpu_revalidation"})
    fig=figure_base("Full validation: reuse helps stable outputs",
                   "GPU cache compares every output bit; the CPU rechecks changed 4096-marker blocks")
    ax=draw_rows(fig,rows,"Validation time (ms)  ·  Lower is better",
                 bounds=(.36,.29,.55,.50),xlim=(0,75),separators=(.5,))
    ax.set_xticks([0,15,30,45,60,75])
    fig.text(.055,.16,"All full-coverage results matched the original serial checker bit-for-bit.",
             fontsize=16.5,color=ACCENT,fontweight="semibold")
    fig.text(.055,.115,"Apple M5 Max · 18 CPU workers · 10 trials per repeated case; first certification observed once.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.083,f"Initial context setup was another {d['initial_certification']['context_copy_allocation_compile_ms']:.2f} ms (one observation), outside the certification clock.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.051,"GPU cache timing includes compare/copy + completion, CPU rechecks and summary reduction; no marker sampling.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.019,"Placement, pre-check mutations and independent serial audit excluded. Circles: trials; diamonds: repeated-case medians.",
             fontsize=13.5,color=SECONDARY)
    export(fig,"01-exact-validation-scenarios",{"source":str(Path(path).resolve()),"sha256":sha256(path),
        "metric":"validation_ms","unit":"ms","aggregation":"median; initial observed once",
        "direction":"lower","rows":rows,"hardware":d["device"],"n":d["n"],
        "context_setup_one_observation_ms":d["initial_certification"]["context_copy_allocation_compile_ms"],
        "timing_scope":d["notes"],"all_full_coverage":True})


def plot_coverage(path,d,groups):
    order=[("full_cpu","CPU full check · 18 workers\n100% marker coverage"),
           ("full_gpu_comparison_exact_cpu_revalidation","GPU cache · unchanged output\n100% marker coverage"),
           ("sparse_cpu","Sparse CPU diagnostic\n0.065536% marker coverage")]
    rows=[]
    for name,label in order:
        rows.append({"id":name,"label":label,
            "values":[t["compute_and_validation_ms"] for t in groups[name]],
            "focal":name=="full_gpu_comparison_exact_cpu_revalidation"})
    fig=figure_base("Compute + checks: full coverage or sparse diagnostics",
                   "100 million markers · original formula · fast math functions with safe arithmetic")
    draw_rows(fig,rows,"Placement + validation time (ms)  ·  Lower is better",
              bounds=(.36,.30,.55,.48),xlim=(0,40),separators=(.5,))
    fig.text(.055,.175,"Sparse checks cannot certify the 99.934464% of markers outside the sample.",
             fontsize=16.5,color=INK,fontweight="semibold")
    fig.text(.055,.126,"Apple M5 Max · 10 trials per mode · All modes independently audited in full after their timed path.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.094,"Sparse mode reuses one fixed sample plan: 65,536 markers / 196,608 XYZ coordinates per trial.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.062,"GPU cache shown after initial certification, with identical output. Both full modes match the full serial summary.",
             fontsize=13.5,color=SECONDARY)
    fig.text(.055,.030,"Input loading, context setup, sample planning and independent serial audit excluded. Circles: trials; diamonds: medians.",
             fontsize=13.5,color=SECONDARY)
    export(fig,"02-full-versus-sparse-coverage",{"source":str(Path(path).resolve()),"sha256":sha256(path),
        "metric":"compute_and_validation_ms","unit":"ms","aggregation":"median of10trials",
        "direction":"lower","rows":rows,"hardware":d["device"],"n":d["n"],
        "sparse_checked_markers":65536,"sparse_checked_fraction":65536/d["n"],
        "coverage_caveat":"Sparse does not certify unchecked markers; its full serial audit is outside the timer.",
        "timing_scope":d["notes"]})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    args=parser.parse_args()
    d,groups=load_groups(args.source)
    plot_full(args.source,d,groups)
    plot_coverage(args.source,d,groups)


if __name__=="__main__":
    main()
