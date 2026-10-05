#!/usr/bin/env python3
"""The sparse-draft motivation figure: three panels side by side (CPU only).

  (a) one video question's draft attention as a heatmap: a row per decode step,
      a column per frame, then the question and the answer so far; colour =
      attention per token vs. a uniform spread (scripts/plot_draft_attention_video.py);
  (b) where the MMFlash draft's attention goes in a 48-frame video context:
      share of attention within distance d of the current position, against the
      share of context tokens within d (same script);
  (c) acceptance length with full draft context vs a window of W target
      tokens, image vs video benchmarks (scripts/plot_window_ablation.py).

    python scripts/plot_sparse_motivation.py                    # -> figures/sparse_motivation.pdf/.png
    python scripts/plot_sparse_motivation.py --sample mvbench-0000 --window 2048
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plot_draft_attention_video as attn  # noqa: E402  (puts matplotlib's dir on sys.path)
import plot_window_ablation as bars  # noqa: E402

FIG_W, FIG_H = 7.0, 2.55
BOTTOM, HEIGHT = 0.62, 1.45
PANELS = [(0.36, 1.62), (2.64, 1.68), (4.74, 2.18)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--attn-dir", default=attn.DEFAULT_INPUT, help="analyze_draft_attention_video.py output")
    parser.add_argument("--sample", default=None, help="question id for the heatmap (default: first LongVideoBench one)")
    parser.add_argument("--layer", default="mean", help="heatmap: 'mean' over draft layers, or a layer index")
    parser.add_argument("--vrange", type=float, default=100.0, help="heatmap colour range is [1/v, v] x uniform")
    parser.add_argument("--map-window", type=int, default=0, help="also mark this window on the heatmap (0 = no line)")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--window", default="512", help="window compared with full context in (c)")
    parser.add_argument("--image-benches", nargs="*", default=["chartqa", "dynamath"])
    parser.add_argument("--video-benches", nargs="*", default=["longvideobench", "mvbench"])
    parser.add_argument("--no-panel-labels", action="store_true")
    parser.add_argument("--output", default=os.path.join(os.path.dirname(HERE), "figures", "sparse_motivation"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matplotlib, plt = attn.setup_matplotlib()
    summary, samples = attn.load_all(args.attn_dir)
    if not samples:
        raise SystemExit(f"no *.npz under {args.attn_dir}")
    sample = args.sample or attn.default_map_sample(summary, samples)
    heat = attn.map_data(samples[sample], args.layer, vrange=args.vrange) if sample in samples else None
    if heat is None:
        raise SystemExit(f"no full attention map for {sample!r}; pick one of the first --dump-full questions")
    print(f"[map] {sample}: {heat['seq_len'] / 1000:.1f}k-token context, {len(heat['anchors'])} decode steps")
    table_args = argparse.Namespace(
        results_dir=args.results_dir,
        image_run_name="mmflash_qwen35-4B_concurrency1_temp0_4096",
        video_run_name="video_mmflash_qwen35-4B_concurrency1_temp0_4096_f16-48-48",
        image_benches=args.image_benches,
        video_benches=args.video_benches,
    )
    windows = ["full", str(args.window)]
    table = bars.collect(table_args, windows)
    bars.print_summary(table, windows, "accept")

    # 7 in wide, every panel's plot area spans the same height; (left, width) in
    # inches, each gap sized for the next panel's y label
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    axes = [fig.add_axes([x / FIG_W, BOTTOM / FIG_H, w / FIG_W, HEIGHT / FIG_H]) for x, w in PANELS]
    attn.draw_map_on(axes[0], heat, args.map_window)
    attn.draw_cdf_on(axes[1], attn.cdf_curves(samples), columnspacing=1.2)
    # one legend entry per row, the same height as (b)'s two-row legend
    bars.draw_bars_on(axes[2], table, windows, "accept", legend_ncol=1)
    axes[2].set_xlim(-0.45, len(table) - 0.55)
    if not args.no_panel_labels:
        for ax, label in zip(axes, ("(a)", "(b)", "(c)")):
            ax.annotate(label, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -33), textcoords="offset points",
                        ha="center", va="top", fontsize=attn.FONT + 1, color=attn.TEXT)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.output}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}.pdf / .png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
