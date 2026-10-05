#!/usr/bin/env python3
"""Full draft context vs a sliding window, on image and video benchmarks.

Reads the results files scripts/sweep_draft_window.sh writes (one per window,
same GPU, same job) and draws grouped bars: one group per benchmark, image
benchmarks left of the divider and video right, one bar per context setting.
The expectation it tests: an image prompt (a few hundred tokens) fits inside
the window, so cutting the draft's context changes nothing; a 48-frame video
prompt (~42k tokens) is almost all far-away frames, and cutting them raises the
acceptance length.

    python scripts/plot_window_ablation.py                       # W=512 vs full
    python scripts/plot_window_ablation.py --windows 512 2048    # three bars
    python scripts/plot_window_ablation.py --metric decode       # decode tok/s
    python scripts/plot_window_ablation.py --summary-only        # table, no figure

Result names: a window W is read from <run-name>_win<W>; full context from
<run-name>_winfull when a sweep re-measured it, else from <run-name> itself, the
main benchmark run (for the 4B image benchmarks that is the 2026-09-23 run of the
same prompted-final draft, block 16, concurrency 1, temperature 0).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_video_results import read_result  # noqa: E402

LABELS = {
    "chartqa": "ChartQA",
    "dynamath": "DynaMath",
    "longvideobench": "LongVideo-\nBench",
    "mvbench": "MVBench",
    "moviechat": "MovieChat",
    "videomme": "Video-MME",
    "vdc": "VDC",
}
# softer fills, full context first, then each window (validated pair: CVD and
# normal-vision separation pass; below 3:1 contrast, so every window bar is
# labelled with its change)
SERIES = ["#6fa3dc", "#ee9970", "#9b8fd6", "#e3b25f"]
TEXT = "#0b0b0b"
FONT = 7  # in-panel text, matching the attention panels of the motivation figure
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--image-run-name", default="mmflash_qwen35-4B_concurrency1_temp0_4096")
    parser.add_argument("--video-run-name", default="video_mmflash_qwen35-4B_concurrency1_temp0_4096_f16-48-48")
    parser.add_argument("--image-benches", nargs="*", default=["chartqa", "dynamath"])
    parser.add_argument("--video-benches", nargs="*", default=["longvideobench", "mvbench"])
    parser.add_argument("--windows", nargs="+", default=["512"], help="windows drawn next to full context")
    parser.add_argument("--metric", choices=["accept", "decode"], default="accept")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "window_ablation"),
        help="path without extension; .pdf and .png are written (default: <repo>/figures/window_ablation)",
    )
    return parser.parse_args()


def load_entry(results_dir: str, name: str, bench: str):
    path = os.path.join(results_dir, f"{name}_results.jsonl")
    if not os.path.isfile(path):
        return None
    try:
        entries = read_result(path).get(bench)
    except (OSError, ValueError):
        return None
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries[0]
    accept = entry.get("accept_length_summary") or {}
    throughput = entry.get("throughput_summary") or {}
    prompts = [s.get("prompt_tokens") for s in entry.get("per_sample_stats") or [] if s.get("prompt_tokens")]
    return {
        "accept": accept.get("mean"),
        "decode": throughput.get("decode_output_throughput"),
        "nq": entry.get("num_questions"),
        "prompt_tokens": sorted(prompts)[len(prompts) // 2] if prompts else None,
    }


def collect(args, windows):
    """{bench: {window: entry or None}} for every bench and window ("full" included)."""
    table = {}
    for kind, benches in (("image", args.image_benches), ("video", args.video_benches)):
        run = args.image_run_name if kind == "image" else args.video_run_name
        for bench in benches:
            table[bench] = {}
            for w in windows:
                names = [f"{run}_winfull", run] if w == "full" else [f"{run}_win{w}"]
                found = None
                for name in names:
                    found = load_entry(args.results_dir, name, bench)
                    if found:
                        found["file"] = name
                        break
                table[bench][w] = found
            table[bench]["kind"] = kind
    return table


def print_sources(table, windows):
    for bench, row in table.items():
        files = [f"{w}: {row[w]['file']}" for w in windows if row.get(w)]
        print(f"  {LABELS.get(bench, bench).replace(chr(10), '').replace('-', '')}: " + "; ".join(files))


def print_summary(table, windows, metric):
    unit = "accept length" if metric == "accept" else "decode tok/s"
    print(f"{unit} by draft context, change relative to full; prompt = median prompt tokens")
    header = f"{'benchmark':<16}{'prompt':>8}" + "".join(f"{('W=' + w) if w != 'full' else 'full':>16}" for w in windows)
    print(header)
    for bench, row in table.items():
        full = (row.get("full") or {}).get(metric)
        prompt = next((e["prompt_tokens"] for w, e in row.items() if w != "kind" and e and e.get("prompt_tokens")), None)
        cells = []
        for w in windows:
            entry = row.get(w)
            value = entry.get(metric) if entry else None
            if value is None:
                cells.append(f"{'not run':>16}")
            elif w == "full" or not full:
                cells.append(f"{value:>16.3f}")
            else:
                cells.append(f"{value:>9.3f} ({(value / full - 1) * 100:+.1f}%)")
        print(f"{LABELS.get(bench, bench).replace(chr(10), '').replace('-', ''):<16}{(str(prompt) if prompt else '-'):>8}" + "".join(cells))


def plot(args, table, windows):
    sys.path.insert(0, "/scratch/fengsicheng/pylibs/mpl")  # matplotlib lives outside the shared env
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    draw_bars_on(ax, table, windows, args.metric)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.output}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {args.output}.pdf / .png")


def rounded_bar(ax, x, height, width, color, radius, **kw):
    """A column with rounded top corners and a square baseline (figures/empirical_study style)."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    r = min(radius, width / 2, height / 2) if height > 0 else 0
    x0, x1, y1 = x - width / 2, x + width / 2, height
    verts = [(x0, 0), (x1, 0), (x1, y1 - r), (x1, y1), (x1 - r, y1),
             (x0 + r, y1), (x0, y1), (x0, y1 - r), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, linewidth=0, **kw))


def draw_bars_on(ax, table, windows, metric: str = "accept", legend_ncol: int = 0):
    """The grouped-bar panel, drawn into an existing axes (also used by plot_sparse_motivation.py);
    legend_ncol 0 = one legend row."""
    from matplotlib.patches import Patch

    benches = list(table)
    n_series = len(windows)
    gap = 0.06
    width = (0.8 - gap * (n_series - 1)) / n_series
    values = {
        (bench, w): (table[bench].get(w) or {}).get(metric) for bench in benches for w in windows
    }
    top = max([v for v in values.values() if v is not None] or [1.0])
    radius = top * 0.012
    for s, w in enumerate(windows):
        for i, bench in enumerate(benches):
            x = i - 0.4 + (width + gap) * s + width / 2
            value = values[(bench, w)]
            if value is None:  # a missing run says so instead of leaving a silent gap
                ax.text(x, top * 0.03, "n/a", ha="center", va="bottom", fontsize=FONT, color=TEXT_SECONDARY, rotation=90)
                continue
            rounded_bar(ax, x, value, width, SERIES[s], radius, zorder=3)
            full = values[(bench, "full")]
            if w != "full" and full:
                change = f"{(value / full - 1) * 100:+.0f}%".replace("-", "\u2212")
                ax.text(x, value + top * 0.015, change, ha="center", va="bottom", fontsize=FONT, color=TEXT_SECONDARY)
    n_image = sum(1 for b in benches if table[b]["kind"] == "image")
    if 0 < n_image < len(benches):
        ax.axvline(n_image - 0.5, color=MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=1)
    ax.set_xticks(range(len(benches)))
    ax.set_xticklabels([LABELS.get(b, b) for b in benches])
    ax.set_ylabel("Accept length $\\tau$" if metric == "accept" else "Decode throughput (tok/s)")
    # the top gridline is the top edge, so the plot area lines up with neighbouring panels
    from matplotlib.ticker import MaxNLocator

    yticks = MaxNLocator(nbins=4).tick_values(0, top * 1.22)
    ax.set_yticks(yticks)
    ax.set_ylim(0, yticks[-1])
    ax.set_xlim(-0.6, len(benches) - 0.4)
    for kind, lo, hi in (("Image", 0, n_image), ("Video", n_image, len(benches))):
        if hi > lo:
            ax.text((lo + hi - 1) / 2, top * 1.16, kind, ha="center", va="center", fontsize=FONT, color=TEXT)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=TEXT_SECONDARY)
    handles = [
        Patch(facecolor=SERIES[s], linewidth=0, label="Full context" if w == "full" else f"Window W={int(w):,}")
        for s, w in enumerate(windows)
    ]
    # legend on top of the panel, as in figures/empirical_study
    ax.legend(handles=handles, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=legend_ncol or n_series,
              handlelength=1.6, columnspacing=2.0, borderaxespad=0.3)


def main() -> int:
    args = parse_args()
    windows = ["full"] + [w for w in args.windows if w != "full"]
    table = collect(args, windows if not args.summary_only else ["full", "512", "2048", "8192"])
    shown = windows if not args.summary_only else ["full", "512", "2048", "8192"]
    print_summary(table, shown, args.metric)
    print("read from results/:")
    print_sources(table, shown)
    if args.summary_only:
        return 0
    plot(args, table, windows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
