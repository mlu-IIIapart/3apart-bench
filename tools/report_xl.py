#!/usr/bin/env python3
"""Render source-hashed benchmark tables and raw-sample dot plots.

Input is an explicit JSON manifest: sources (path, sha256, metadata, assertions)
and figures (title, groups, rows, notes). Each row names a source, raw_path and
optional raw_field; no summary value is accepted instead of the raw samples.
Use separate figures for different workloads. This tool never runs benchmarks.
An absolute local source path requires a relative public_path for the exports.
Relative source paths are resolved beside the manifest, or from this repository
when the source declares relative_to="repository".

Example row:
{"source":"m4_cpu", "group":"resident", "host":"lorium", "engine":"cpu",
 "label":"Lorium M4 · NumPy FP64", "raw_path":["resident","samples"],
 "raw_field":"placement_and_complete_output_materialization_ms"}
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import statistics
import textwrap


COLORS = {"ink": "#171717", "secondary": "#5E6462", "cpu": "#555B59",
          "gpu": "#1F6F5F", "rust": "#555B59", "rule": "#D8D8D4", "grid": "#E8E8E5"}
SCOPES = {
    "resident": "Resident placement",
    "readback": "GPU placement + full host readback",
    "cold": "Cold generation through complete output",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_svg(path):
    """Remove exporter line-end padding without changing SVG content."""
    path = Path(path)
    path.write_text("\n".join(line.rstrip(" \t") for line in path.read_text().splitlines()) + "\n")


def pick(data, path):
    for key in path:
        data = data[key]
    return data


def load_manifest(path):
    manifest = json.loads(path.read_text())
    if manifest.get("draft", False):
        raise ValueError("Draft manifest: final source evidence is still missing")
    sources, records = {}, {}
    for name, item in manifest["sources"].items():
        source = Path(item["path"])
        if not source.is_absolute():
            relative_to = item.get("relative_to", "manifest")
            if relative_to not in ("manifest", "repository"):
                raise ValueError(f"Unknown source root: {name}")
            root = Path(__file__).resolve().parent.parent if relative_to == "repository" else path.parent
            source = root / source
        actual = sha(source)
        if actual != item["sha256"]:
            raise ValueError(f"Source changed: {name}")
        data = json.loads(source.read_text())
        for assertion in item.get("assertions", []):
            got = pick(data, assertion["path"])
            expected = assertion["equals"]
            if got != expected or isinstance(expected, bool) and type(got) is not bool:
                raise ValueError(f"Failed source assertion: {name} {assertion['path']}")
        public_path = item.get("public_path", item["path"])
        if Path(public_path).is_absolute() or ".." in Path(public_path).parts:
            raise ValueError(f"Provide a relative public_path for {name}")
        sources[name] = data
        records[name] = dict(path=public_path, sha256=actual,
                             metadata=item.get("metadata", {}), assertions=item.get("assertions", []))
    figures = []
    for figure in manifest["figures"]:
        rows = []
        group_order = [x["id"] for x in figure["groups"]]
        for row in figure["rows"]:
            if row["group"] not in group_order:
                raise ValueError("Row group is not declared")
            raw = pick(sources[row["source"]], row["raw_path"])
            if "raw_field" in row:
                raw = [pick(item, row["raw_field"] if isinstance(row["raw_field"], list) else [row["raw_field"]]) for item in raw]
            factor = row.get("unit_factor", 1)
            if any(isinstance(x, bool) for x in raw):
                raise ValueError(f"Boolean is not a timing sample: {row['label']}")
            samples = [float(x) * factor for x in raw]
            if not samples or not all(math.isfinite(x) and x > 0 for x in samples):
                raise ValueError(f"Invalid samples: {row['label']}")
            if row.get("expected_runs") is not None and len(samples) != row["expected_runs"]:
                raise ValueError(f"Wrong run count: {row['label']}")
            rows.append(dict(row, samples=samples, count=len(samples), median=statistics.median(samples),
                             minimum=min(samples), maximum=max(samples), source_sha256=records[row["source"]]["sha256"]))
        for host in figure.get("required_hosts", []):
            if host not in {x.get("host") for x in rows}:
                raise ValueError(f"Missing required host: {host}")
        if "expected_rows" in figure and len(rows) != figure["expected_rows"]:
            raise ValueError(f"Wrong row count: {figure['id']}")
        for group in figure["groups"]:
            grouped = [row for row in rows if row["group"] == group["id"]]
            if "expected_rows" in group and len(grouped) != group["expected_rows"]:
                raise ValueError(f"Wrong row count: {figure['id']} / {group['id']}")
            if not set(group.get("required_hosts", [])).issubset({x.get("host") for x in grouped}):
                raise ValueError(f"Missing host in group: {group['id']}")
        figures.append(dict(figure, rows=rows))
    # Public artifacts contain values and source hashes, never private host paths.
    public_text = json.dumps({"sources": records, "figures": figures})
    if "/Users/" in public_text or "/home/" in public_text:
        raise ValueError("Private absolute path in exported metadata or labels")
    return manifest, records, figures


def value_text(row, decimals=2):
    return (f"{row['median']:,.{decimals}f}  "
            f"[{row['minimum']:,.{decimals}f}–{row['maximum']:,.{decimals}f}]")


def visual_spec(figure):
    return {
        "claim": figure["claim"], "audience": "technical-generalist", "visual_form": "table_plus_chart",
        "metric": {"name": figure["metric"], "unit": figure.get("unit", "ms"), "direction": "lower",
                   "aggregation": figure.get("aggregation", "Median, observed minimum–maximum, all individual samples shown; no confidence interval"),
                   "denominator": figure["denominator"], "precision_decimals": figure.get("decimals", 2),
                   "scale": figure.get("scale", "log10")},
        "focal_series": "GPU", "required_baselines": figure.get("required_baselines", []),
        "series_order": list(dict.fromkeys(x["label"] for x in figure["rows"])),
        "category_order": [x["title"] for x in figure["groups"]], "ordering_rationale": "semantic_group",
        "uncertainty": {"available": True, "type": "raw_points", "sample_or_seed_count": None,
                        "note": figure.get("sample_description", "Each row states n. Intervals are the observed sample range, not a confidence interval.")},
        "target": {"medium": "web", "width": 1200, "height": figure.get("height_px", 900),
                   "unit": "px", "minimum_text_size_pt": 10, "background": "white"},
        "table_rules": {"show_actual_values": True, "show_deltas": False, "numeric_alignment": "right",
                        "best_value_emphasis": "none", "missing_symbol": "—"},
        "selected_reference_ids": ["OAI-VIS-001", "OAI-VIS-004", "OAI-VIS-002", "OAI-VIS-166", "OAI-VIS-182"],
        "output_formats": ["svg", "png", "html"],
        "caption_requirements": ["task", "sample", "estimator", "uncertainty", "exclusions", "metric_direction", "latency_basis"],
    }


def render(figure, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager, ticker
    from PIL import Image

    installed = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in ("Arial", "Helvetica", "Inter", "Liberation Sans", "DejaVu Sans") if name in installed), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "font.size": 11, "text.color": COLORS["ink"],
        "axes.labelcolor": COLORS["secondary"], "xtick.color": COLORS["secondary"],
        "axes.edgecolor": COLORS["rule"], "svg.fonttype": "none", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.linewidth": .7,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    height_px = figure.get("height_px", 900)
    fig = plt.figure(figsize=(12, height_px / 100), dpi=100)
    texts = []

    def text(x, y, content, **kwargs):
        artist = fig.text(x, y, content, **kwargs)
        texts.append(artist)
        return artist

    text(.055, .95, figure["title"], fontsize=18, weight="semibold", va="top")
    text(.055, .905, figure["subtitle"], fontsize=11, color=COLORS["secondary"], va="top")
    top, bottom = .815, figure.get("plot_bottom", .265)
    slots, labels = [], []
    for group in figure["groups"]:
        labels.append((len(slots), group))
        slots.append(None)
        slots.extend(row for row in figure["rows"] if row["group"] == group["id"])
        slots.append(None)
    slot_height = (top - bottom) / max(len(slots), 1)
    for index, group in labels:
        y = top - (index + .35) * slot_height
        text(.055, y, group["title"], fontsize=11.5, weight="semibold", va="center")
    plot = fig.add_axes([.655, bottom, .29, top - bottom], zorder=1)
    if figure.get("scale", "log10") == "log10":
        plot.set_xscale("log")
    all_samples = [s for row in figure["rows"] for s in row["samples"]]
    lo, hi = min(all_samples), max(all_samples)
    if figure.get("scale", "log10") == "log10":
        low = 10 ** math.floor(math.log10(lo))
        high = 10 ** math.ceil(math.log10(hi))
        if low == high:
            high *= 10
        plot.set_xlim(low, high)
        plot.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
        plot.xaxis.set_minor_locator(ticker.NullLocator())
        plot.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    else:
        plot.set_xlim(0, hi * 1.12)
        plot.xaxis.set_major_locator(ticker.MaxNLocator(5))
    plot.set_ylim(len(slots), 0)
    plot.set_yticks([])
    plot.grid(axis="x", which="major", color=COLORS["grid"], linewidth=.6)
    plot.set_axisbelow(True)
    plot.tick_params(axis="x", length=3, labelsize=10)
    unit = figure.get("unit", "ms")
    axis_label = f"{unit} · {'log scale · ' if figure.get('scale', 'log10') == 'log10' else ''}lower is better"
    plot.set_xlabel(axis_label, fontsize=10, labelpad=7)
    text(.055, .85, figure.get("row_heading", "Host and implementation"), fontsize=10, color=COLORS["secondary"])
    text(.59, .85, f"Median [min–max], {unit}", fontsize=10, color=COLORS["secondary"], ha="right")
    text(.625, .85, "n", fontsize=10, color=COLORS["secondary"], ha="right")
    for index, row in enumerate(slots):
        if row is None:
            continue
        y = top - (index + .35) * slot_height
        kind = row.get("engine", "cpu")
        color = COLORS.get(kind, COLORS["cpu"])
        marker = "o" if kind == "gpu" else "s"
        text(.07, y, row["label"], fontsize=11, va="center", color=color)
        text(.59, y, value_text(row, figure.get("decimals", 2)), fontsize=11, va="center", ha="right")
        text(.625, y, str(row["count"]), fontsize=10, va="center", ha="right", color=COLORS["secondary"])
        center = index + .35
        offsets = [0] if row["count"] == 1 else [(i / (row["count"] - 1) - .5) * .22 for i in range(row["count"])]
        plot.hlines(center, row["minimum"], row["maximum"], color=color, linewidth=1)
        plot.scatter(row["samples"], [center + z for z in offsets], s=15, marker=marker,
                     facecolors="white", edgecolors=color, linewidths=.65, zorder=3)
        plot.scatter([row["median"]], [center], s=34, marker="D", color=color, zorder=4)
    # Structural rules span labels and numeric columns, never boxed cells.
    for yy in [.835, bottom + .006]:
        fig.add_artist(plt.Line2D([.055, .945], [yy, yy], transform=fig.transFigure, color=COLORS["rule"], linewidth=.65))
    notes = figure["notes"]
    lines = []
    for note in notes:
        lines.extend(textwrap.wrap(note, width=145, break_long_words=False, break_on_hyphens=False))
    note_top = bottom - .065
    for i, line in enumerate(lines):
        text(.055, note_top - i * 18 / height_px, line, fontsize=10, color=COLORS["secondary"], va="top")
    # Actual bounds at delivery dimensions; visual QA still requires inspection.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    bounds = []
    for artist in texts:
        bb = artist.get_window_extent(renderer)
        if bb.x0 < 0 or bb.y0 < 0 or bb.x1 > w or bb.y1 > h:
            bounds.append(artist.get_text())
    if bounds:
        plt.close(fig)
        raise ValueError("Clipped text: " + repr(bounds))
    name = figure["id"]
    fig.savefig(out / f"{name}.svg", format="svg")
    normalize_svg(out / f"{name}.svg")
    fig.savefig(out / f"{name}.png", dpi=200)
    fig.savefig(out / f"{name}-final-size.png", dpi=100)
    with Image.open(out / f"{name}-final-size.png") as im:
        im.convert("L").save(out / f"{name}-grayscale.png")
    plt.close(fig)
    return {"font": font, "target_pixels": [w, h], "export_pixels": [w * 2, h * 2],
            "minimum_ordinary_font_pt": 10, "clipped_text": bounds,
            "manual_final_size_and_grayscale_review_required": True}


def html_table(figure):
    import html
    rows = []
    for group in figure["groups"]:
        rows.append(f'<tr class="group"><th colspan="4">{html.escape(group["title"])}</th></tr>')
        for row in figure["rows"]:
            if row["group"] != group["id"]:
                continue
            rows.append('<tr><td>' + html.escape(row["label"]) + '</td><td class="number">' +
                        value_text(row, figure.get("decimals", 2)) + '</td><td class="number">' + str(row["count"]) +
                        '</td><td>' + html.escape(row["source_sha256"][:12]) + '</td></tr>')
    return '<h2>' + html.escape(figure["title"]) + '</h2><p>' + html.escape(figure["subtitle"]) + '</p>' + \
        '<table><thead><tr><th>Host / implementation</th><th>Median [min–max], ' + html.escape(figure.get("unit", "ms")) + \
        '</th><th>n</th><th>Source SHA-256</th></tr></thead><tbody>' + ''.join(rows) + \
        '</tbody></table><ul>' + ''.join('<li>' + html.escape(n) + '</li>' for n in figure["notes"]) + '</ul>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Verify source hashes/assertions and raw sample paths without rendering")
    args = parser.parse_args()
    manifest, sources, figures = load_manifest(args.manifest)
    if args.check_only:
        print(json.dumps({"passed": True, "sources": len(sources), "figures": len(figures),
                          "rows": sum(len(f["rows"]) for f in figures)}, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    specs = {f["id"]: visual_spec(f) for f in figures}
    (args.output / "visual-specs.json").write_text(json.dumps(specs, indent=2) + "\n")
    # Evidence lock precedes any rendering. All raw timing values remain available.
    (args.output / "plot-data.json").write_text(json.dumps({"sources": sources, "figures": figures}, indent=2, allow_nan=False) + "\n")
    with (args.output / "exact-values.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["figure", "scope", "label", "unit", "median", "min", "max", "n", "source_sha256"])
        for figure in figures:
            for row in figure["rows"]:
                writer.writerow([figure["id"], row["group"], row["label"], figure.get("unit", "ms"),
                                 row["median"], row["minimum"], row["maximum"], row["count"], row["source_sha256"]])
    qa = {f["id"]: render(f, args.output) for f in figures}
    (args.output / "render-preflight.json").write_text(json.dumps(qa, indent=2) + "\n")
    style = 'body{font:15px Arial,sans-serif;color:#171717;max-width:1100px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;width:100%;margin:24px 0}th,td{padding:9px 10px;text-align:left}thead{border-top:1px solid #555;border-bottom:1px solid #aaa}.group th{padding-top:18px;border-top:1px solid #ddd}.number{text-align:right;font-variant-numeric:tabular-nums}li{margin:8px 0}h2{font-size:23px;margin-top:45px}'
    body = ''.join(html_table(f) for f in figures)
    (args.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Validated benchmark timings</title><style>' + style + '</style>' + body)
    if manifest.get("caption"):
        (args.output / "caption.txt").write_text(manifest["caption"].rstrip() + "\n")
    receipt = {"rendered_at_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha(args.manifest),
               "script_sha256": sha(__file__), "sources": sources,
               "export_environment": {"python": platform.python_version(), "matplotlib": version("matplotlib"), "pillow": version("Pillow")},
               "outputs": {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file() and p.name != "render-receipt.json"},
               "status": "rendered_pending_final_size_and_grayscale_inspection",
               "governing_grammar": "Scientific Visuals & Tables2.2.0; GPT-6 Astra north-star tokens",
               "uncertainty": "Raw trial points and observed ranges; no confidence intervals"}
    (args.output / "render-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "figures": [f["id"] for f in figures], "inspection_required": True}))


if __name__ == "__main__":
    main()
