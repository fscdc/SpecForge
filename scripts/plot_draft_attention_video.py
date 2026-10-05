#!/usr/bin/env python3
"""Plot where the MMFlash draft attends in a 48-frame video context (CPU only).

Reads what scripts/analyze_draft_attention_video.py wrote and draws:

Figures go to <repo>/figures/ (next to the other paper figures):

  video_attn_map_<id>  one question as a heatmap: a row per decode step (anchor) of
                  the answer, a column per frame, then the question text and
                  the answer so far; colour = attention per token relative to
                  a uniform spread over the visible context (1x = uniform), on
                  a log scale centred at 1, with the colour key on top.
                  --map-window W adds a dashed line where a draft window of W
                  positions starts.
  video_attn_cdf       every question pooled per benchmark: share of the draft's
                  context attention within distance d of the anchor, against
                  the share of context tokens within d (dashed).

and prints the numbers a caption or the text can quote.

    python scripts/plot_draft_attention_video.py
    python scripts/plot_draft_attention_video.py --sample longvideobench-0003 --map-window 2048
    python scripts/plot_draft_attention_video.py --layer 4      # one draft layer instead of the mean
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/scratch/fengsicheng/pylibs/mpl")  # matplotlib lives outside the shared env
import numpy as np  # noqa: E402

DEFAULT_INPUT = "/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/attention_analysis/video-draft-attn"
FIGURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
LABELS = {"longvideobench": "LongVideoBench", "mvbench": "MVBench", "moviechat": "MovieChat",
          "videomme": "Video-MME", "vdc": "VDC"}
# reference palette (dataviz skill)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, TEXT_SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
FONT = 7  # every label, tick, legend and key in these panels (panel letters excepted)
XLABEL_PT = 15  # x-axis label top, points below the axes: shared so side-by-side panels line up
# diverging, in the hues of the soft series colours below: blue arm (under-attended)
# <-> gray midpoint (uniform) <-> orange arm (over-attended); the arms match in
# lightness step for step (CIELAB L* 50/62/75/88, midpoint 95)
DIVERGING = ["#3c7cb0", "#5b9bd5", "#93bcea", "#cedef4", "#f2f1ee", "#fbd5c7", "#f1a78b", "#d97e5b", "#b36141"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--sample", default=None, help="question id for the heatmap (default: first with a full map)")
    parser.add_argument("--layer", default="mean", help="'mean' over draft layers, or a layer index")
    parser.add_argument("--window", type=int, default=2048, help="window quoted in the printed summary")
    parser.add_argument("--map-window", type=int, default=0, help="also mark this window on the map (0 = no line)")
    parser.add_argument("--cdf-window", type=int, default=0, help="also mark this window on the CDF (0 = no line)")
    parser.add_argument("--vrange", type=float, default=100.0, help="colour range is [1/v, v] x uniform")
    parser.add_argument("--question-cols", type=int, default=10)
    parser.add_argument("--answer-cols", type=int, default=16)
    parser.add_argument("--output-dir", default=FIGURES, help="default: <repo>/figures")
    return parser.parse_args()


def load_all(input_dir: str):
    summary = {}
    path = os.path.join(input_dir, "summary.jsonl")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    summary[row["id"]] = row
    samples = {}
    for npz in sorted(glob.glob(os.path.join(input_dir, "*.npz"))):
        sample_id = os.path.basename(npz)[: -len(".npz")]
        samples[sample_id] = npz
    return summary, samples


def bench_of(sample_id: str) -> str:
    return sample_id.rsplit("-", 1)[0]


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------


def sample_numbers(data, window: int):
    """Per-question means over anchors, layers and heads."""
    anchors = data["anchors"].astype(np.float64)
    num_frames = int(data["num_frames"])
    seg = data["seg_mass"].mean(axis=(1, 2))  # (A, F+3)
    counts = data["seg_count"].astype(np.float64)
    spans = data["frame_spans"]
    # frames lying entirely before the window start
    outside = spans[None, :, 1] < (anchors[:, None] - window)  # (A, F)
    frame_mass = seg[:, 1 : 1 + num_frames]
    frame_tokens = counts[:, 1 : 1 + num_frames]
    windows = list(data["windows"])
    out = {
        "context": anchors.mean(),
        "frames_mass": frame_mass.sum(1).mean(),
        "frames_tokens": (frame_tokens.sum(1) / anchors).mean(),
        "far_frames_mass": (frame_mass * outside).sum(1).mean(),
        "far_frames_tokens": ((frame_tokens * outside).sum(1) / anchors).mean(),
        "question_mass": seg[:, num_frames + 1].mean(),
        "answer_mass": seg[:, num_frames + 2].mean(),
        "sink_mass": seg[:, 0].mean(),
        "self_mass": data["self_mass"].mean(),
    }
    if window in windows:
        index = windows.index(window)
        out["window_mass"] = data["win_mass"][..., index].mean()
        out["window_tokens"] = (data["win_count"][:, index] / anchors).mean()
        out["window_mass_by_layer"] = data["win_mass"][..., index].mean(axis=(0, 2))
    return out


def print_numbers(samples, window: int) -> None:
    by_bench = {}
    for sample_id, path in samples.items():
        with np.load(path) as data:
            by_bench.setdefault(bench_of(sample_id), []).append(sample_numbers(data, window))
    print(f"Draft context attention, mean over questions x anchors x layers x heads (W = {window}):")
    head = (f"{'benchmark':<16}{'n':>3}{'ctx':>8}{'frames':>15}{'far frames':>15}"
            f"{'last W':>15}{'question':>10}{'answer':>9}{'sink':>7}{'in-block':>9}")
    print(head)
    for bench, rows in by_bench.items():
        def m(key):
            values = [r[key] for r in rows if key in r]
            return float(np.mean(values)) if values else float("nan")
        print(
            f"{LABELS.get(bench, bench):<16}{len(rows):>3}{m('context') / 1000:>7.1f}k"
            f"{m('frames_mass'):>7.1%} ({m('frames_tokens'):>4.0%})"
            f"{m('far_frames_mass'):>7.1%} ({m('far_frames_tokens'):>4.0%})"
            f"{m('window_mass'):>7.1%} ({m('window_tokens'):>4.1%})"
            f"{m('question_mass'):>10.1%}{m('answer_mass'):>9.1%}{m('sink_mass'):>7.1%}{m('self_mass'):>9.1%}"
        )
        layers = [r["window_mass_by_layer"] for r in rows if "window_mass_by_layer" in r]
        if layers:
            print(f"{'':<19}last-W mass by draft layer: " + "  ".join(f"L{i} {v:.1%}" for i, v in enumerate(np.mean(layers, 0))))
    print("  (x%) = that region's share of the context tokens; 'far frames' = frames wholly outside the last W;")
    print("  'in-block' = attention the 15 predicting slots keep inside their own block (excluded above).")


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": FONT, "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT,
        "legend.fontsize": FONT, "axes.labelcolor": TEXT, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": TEXT_SECONDARY, "ytick.labelcolor": TEXT_SECONDARY,
    })
    return matplotlib, plt


def map_columns(data, last_anchor: int, question_cols: int, answer_cols: int):
    """[(start, end, group)] half-open; a frame's column also takes the
    vision start/end tokens that follow it, so no position is left out."""
    spans = data["frame_spans"]
    prompt_len = int(data["prompt_len"])
    columns = [(0, int(spans[0][0]), "system")]
    for k in range(len(spans)):
        end = int(spans[k + 1][0]) if k + 1 < len(spans) else int(spans[k][1]) + 1
        columns.append((int(spans[k][0]), end, "frame"))
    q_start = int(spans[-1][1]) + 1
    edges = np.unique(np.linspace(q_start, prompt_len, question_cols + 1).round().astype(int))
    columns += [(int(a), int(b), "question") for a, b in zip(edges[:-1], edges[1:]) if b > a]
    step = max(1, int(np.ceil((last_anchor - prompt_len) / answer_cols)))
    for start in range(prompt_len, last_anchor, step):
        columns.append((start, min(start + step, last_anchor), "answer"))
    return columns


def map_data(path, layer="mean", question_cols=10, answer_cols=16, vrange=100.0):
    """One question's heatmap cells: attention per token relative to a uniform
    spread over the visible context, clipped to [1/vrange, vrange]; None when the
    question has no full map."""
    with np.load(path) as data:
        if "full" not in data:
            return None
        full = data["full"]  # (A, L, S)
        attn = full.mean(1) if layer == "mean" else full[:, int(layer)]
        anchors = data["anchors"]
        out = {
            "anchors": anchors,
            "columns": map_columns(data, int(anchors[-1]), question_cols, answer_cols),
            "prompt_len": int(data["prompt_len"]),
            "num_frames": int(data["num_frames"]),
            "seq_len": int(data["seq_len"]),
            "vrange": vrange,
        }
    cumulative = np.concatenate([np.zeros((attn.shape[0], 1)), np.cumsum(attn, axis=1, dtype=np.float64)], axis=1)
    grid = np.full((len(anchors), len(out["columns"])), np.nan)
    for r, anchor in enumerate(anchors):
        for c, (start, end, _group) in enumerate(out["columns"]):
            visible_end = min(end, int(anchor))
            if visible_end <= start:
                continue
            mass = cumulative[r, visible_end] - cumulative[r, start]
            grid[r, c] = mass / (visible_end - start) * int(anchor)  # x uniform
    out["grid"] = np.clip(grid, 1 / vrange, vrange)
    return out


def column_x(columns, position: int) -> float:
    """x of a context position on the map (columns are uneven: one per frame, then text)."""
    for c, (a, b, _g) in enumerate(columns):
        if a <= position < b:
            return c + (position - a) / (b - a)
    return 0.0 if position < columns[0][0] else float(len(columns))


def draw_map_on(ax, m, window: int = 0):
    """The heatmap panel, drawn into an existing axes (also used by
    plot_sparse_motivation.py). The colour key sits on top of the panel, where
    the other panels carry their legends; window > 0 also marks where a draft
    window of that many positions starts."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.patches import Rectangle

    grid, columns, anchors, v = m["grid"], m["columns"], m["anchors"], m["vrange"]
    cmap = LinearSegmentedColormap.from_list("soft_diverging", DIVERGING)
    cmap.set_bad((0, 0, 0, 0))
    # cells past the current position (answer not generated yet) show the hatched
    # background, so they cannot be mistaken for the gray "uniform" midpoint
    ax.set_facecolor("white")
    ax.add_patch(Rectangle((0, 0), len(columns), len(anchors), facecolor="white", edgecolor=GRID,
                           hatch="////", linewidth=0, zorder=0))
    mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap=cmap, norm=LogNorm(1 / v, v), rasterized=True, zorder=1)
    ax.invert_yaxis()

    if window:  # where a window of W positions starts, per row
        xs = [column_x(columns, int(anchor) - window) for anchor in anchors]
        ax.step(xs, np.arange(len(anchors)) + 0.5, where="mid", color=TEXT, linewidth=0.8,
                linestyle=(0, (2.5, 1.5)), zorder=2)
        ax.annotate(f"W={window:,}", xy=(xs[0], 0), xytext=(-2, -2), textcoords="offset points",
                    ha="right", va="top", fontsize=FONT, color=TEXT)

    # x axis: positions in the context under the visual tokens, then one label per
    # region on the x-label line; white gaps split visual tokens | Q | Answer
    groups = [g for _a, _b, g in columns]
    visual_end = max(b for _a, b, g in columns if g == "frame")
    ticks = [p for p in range(0, visual_end, 10000)]
    ax.set_xticks([column_x(columns, p) for p in ticks])
    ax.set_xticklabels(["0" if p == 0 else f"{p // 1000}k" for p in ticks])
    for name, label in (("frame", "Visual tokens"), ("question", "Q"), ("answer", "Answer")):
        cols = [c for c, g in enumerate(groups) if g == name]
        if not cols:
            continue
        ax.annotate(label, xy=((cols[0] + cols[-1] + 1) / 2, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -XLABEL_PT), textcoords="offset points", ha="center", va="top",
                    fontsize=FONT, color=TEXT)
        if name != "frame":
            ax.axvline(cols[0], color="white", linewidth=2.0, zorder=2)

    # y axis: answer tokens decoded so far
    rows = np.linspace(0, len(anchors) - 1, 4).round().astype(int)
    ax.set_yticks(rows + 0.5)
    ax.set_yticklabels([str(int(anchors[r]) - m["prompt_len"]) for r in rows])
    ax.set_ylabel("Answer tokens decoded")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(axis="both", length=0, colors=TEXT_SECONDARY)

    # colour key on top of the panel, like the other panels' legends
    cax = ax.inset_axes([0.04, 1.04, 0.92, 0.045])
    bar = ax.figure.colorbar(mesh, cax=cax, orientation="horizontal")
    decades = [10.0 ** k for k in range(int(round(np.log10(1 / v))), int(round(np.log10(v))) + 1)]
    bar.set_ticks(decades)
    bar.set_ticklabels([f"{d:g}\u00d7" for d in decades])
    bar.minorticks_off()
    bar.outline.set_visible(False)
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")
    cax.tick_params(length=0, pad=1.5, labelsize=FONT, colors=TEXT_SECONDARY)
    bar.set_label("Attention per token vs. uniform", fontsize=FONT, color=TEXT, labelpad=2.5)
    return mesh


def default_map_sample(summary, samples):
    """The first LongVideoBench question with a full map, else the first question with one."""
    with_map = [s for s in samples if summary.get(s, {}).get("full_map")]
    if not with_map:  # summary missing: look inside
        with_map = [s for s, p in samples.items() if "full" in np.load(p).files]
    return sorted(with_map, key=lambda s: (bench_of(s) != "longvideobench", s))[0] if with_map else None


def draw_map(args, sample_id, path, out_dir):
    matplotlib, plt = setup_matplotlib()
    m = map_data(path, args.layer, args.question_cols, args.answer_cols, args.vrange)
    if m is None:
        raise SystemExit(f"{sample_id} has no full map; pick one of the first --dump-full questions")
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    draw_map_on(ax, m, args.map_window)
    layer = "mean of the draft layers" if args.layer == "mean" else f"draft layer {args.layer}"
    print(f"[map] {sample_id}: {LABELS.get(bench_of(sample_id), bench_of(sample_id))}, {m['seq_len'] / 1000:.1f}k-token "
          f"context, {len(m['anchors'])} decode steps, {layer} (the caption goes in the paper, not the figure)")
    out = os.path.join(out_dir, f"video_attn_map_{sample_id}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


def cdf_curves(samples):
    """{benchmark: (n_questions, cumulative attention, cumulative token share)} by log2 distance bin."""
    per_bench = {}
    for sample_id, path in samples.items():
        with np.load(path) as data:
            mass = data["dist_mass"].mean(axis=(1, 2))  # (A, bins)
            tokens = data["dist_count"] / data["anchors"][:, None]
        per_bench.setdefault(bench_of(sample_id), []).append((np.cumsum(mass, 1).mean(0), np.cumsum(tokens, 1).mean(0)))
    return {
        bench: (len(rows), np.mean([r[0] for r in rows], 0), np.mean([r[1] for r in rows], 0))
        for bench, rows in per_bench.items()
    }


# softer line colours (validated: adjacent-pair CVD and normal-vision checks
# pass); blue and lavender are close under protanopia, so every line also
# carries its own marker
SOFT_LINES = ["#5b9bd5", "#ed8f6b", "#9b8fd6"]
LINE_MARKERS = ["o", "s", "^"]


def style_axes(ax) -> None:
    """The empirical_study figure's axes: y grid, a hairline baseline, no ticks."""
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=TEXT_SECONDARY)


def draw_cdf_on(ax, curves, cdf_window: int = 0, columnspacing: float = 2.0):
    """The CDF panel, drawn into an existing axes (also used by plot_sparse_motivation.py)."""
    import matplotlib.ticker
    from matplotlib.transforms import ScaledTranslation

    upper = None
    for i, (bench, (count, mass, tokens)) in enumerate(curves.items()):
        color = SOFT_LINES[i % len(SOFT_LINES)]
        upper = 2.0 ** (np.arange(len(mass)) + 1) - 1  # largest distance in each log2 bin
        keep = tokens < 0.999
        keep[np.argmax(~keep)] = True  # include the first bin that reaches the whole context
        ax.plot(upper[keep], mass[keep], color=color, linewidth=1.8, marker=LINE_MARKERS[i % 3], markersize=3.2,
                markeredgecolor="white", markeredgewidth=0.5, label=LABELS.get(bench, bench), zorder=3, clip_on=False)
        ax.plot(upper[keep], tokens[keep], color=color, linewidth=1.0, linestyle=(0, (2, 2)), zorder=2, clip_on=False)
        print(f"[cdf] {LABELS.get(bench, bench)}: {count} questions")
    if upper is not None:
        if cdf_window:
            ax.axvline(cdf_window, color=TEXT_SECONDARY, linewidth=0.6)
            ax.text(cdf_window * 1.1, 0.03, f"W={cdf_window:,}", fontsize=FONT, color=TEXT_SECONDARY)
        ax.plot([], [], color=MUTED, linewidth=1, linestyle=(0, (2, 2)), label="Share of context tokens")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Distance from the current position (tokens)")
    ax.xaxis.set_label_coords(0.5, 0, transform=ax.transAxes + ScaledTranslation(0, -XLABEL_PT / 72, ax.figure.dpi_scale_trans))
    ax.set_ylabel("Share of attention\nfor draft model")
    ax.set_ylim(0, 1)  # the 100% gridline is the top edge, level with the other panels
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style_axes(ax)
    # legend on top of the panel, as in figures/empirical_study
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.6,
              columnspacing=columnspacing, borderaxespad=0.3)


def draw_cdf(args, samples, out_dir):
    matplotlib, plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    draw_cdf_on(ax, cdf_curves(samples), args.cdf_window)
    fig.tight_layout()
    out = os.path.join(out_dir, "video_attn_cdf")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")


def main() -> int:
    args = parse_args()
    summary, samples = load_all(args.input_dir)
    if not samples:
        raise SystemExit(f"no *.npz under {args.input_dir}; run scripts/analyze_draft_attention_video.py first")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    print_numbers(samples, args.window)
    sample = args.sample or default_map_sample(summary, samples)
    if sample:
        draw_map(args, sample, samples[sample], out_dir)
    draw_cdf(args, samples, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
