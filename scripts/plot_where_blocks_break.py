#!/usr/bin/env python3
"""Per-benchmark "where blocks break" figures from the accept probe.

For every benchmark under ``accept_analysis/`` that has both drafters, three
token- and block-level views of the same probe, all bucketed by visual-KL
quartile (Q1 least, Q4 most image-dependent):

* token accuracy -- the drafter's top-1 accuracy at each drafted position
                    (parallel prediction, so defined for every slot), bucketed
                    by the SLOT's own visual KL;
* block acceptance -- accepted slots / valid slots of the drafted blocks,
                    grouped by the visual KL of the block's FIRST predicted token;
* truncation share -- share of all truncation points whose token falls in each
                    slot-KL quartile (an uninformative score would put 25% in each).

The per-benchmark figure stacks all three; the combined paper figure shows the
two token-level rows (accuracy, truncation share).

Both drafters score the same token sequences, so the KL is identical for a
matched (id, anchor) and the quartile cuts are shared. Blocks are matched by
(id, anchor_pos) so the two bars of a quartile describe the same blocks.

    python scripts/plot_where_blocks_break.py                       # one figure per benchmark
    python scripts/plot_where_blocks_break.py chartqa mmstar dynamath
    python scripts/plot_where_blocks_break.py --combine textvqa mmstar mathvista
                                                                    # one 2x3 figure for the paper

    python scripts/plot_where_blocks_break.py --supp textvqa mmstar mathverse
                                                                    # one 1x3 block-level figure

Writes figures/where_blocks_break_<bench>.pdf (+ .png preview),
figures/empirical_study.pdf for --combine, or figures/empirical_study_supp.pdf
for --supp (block acceptance rate by the first predicted token's quartile).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/scratch/fengsicheng/pylibs/mpl")  # matplotlib lives outside the shared env
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch
from matplotlib.path import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "accept_analysis")
OUT = os.path.join(ROOT, "figures")
DRAFTS = {"zlab": "text-only data", "llava-ov-1M": "multimodal data"}
LABEL = {"chartqa": "ChartQA", "textvqa": "TextVQA", "mmstar": "MMStar",
         "dynamath": "DynaMath", "mathvista": "MathVista", "mathverse": "MathVerse"}

# sequential blue ramp (reference palette steps 200/350/500/650) for Q1..Q4,
# and the darker step of the same ramp for the hatch ink
Q_FILL = ["#9ec5f4", "#5598e7", "#256abf", "#104281"]
Q_INK = ["#5598e7", "#256abf", "#104281", "#0d366b"]
INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#ffffff"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "hatch.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load(bench: str, draft: str):
    dirs = glob.glob(os.path.join(RUNS, f"{bench}-Qwen3.5-4B-{draft}-*"))
    if not dirs:
        return None
    rows = {}
    with open(os.path.join(dirs[0], "per_anchor_accept.jsonl")) as handle:
        for line in handle:
            r = json.loads(line)
            if r.get("visual_kl_first") is None or r["n_valid"] <= 0:
                continue
            c = r["correct"]
            first = next((k for k in range(1, min(16, len(c))) if not c[k]), None)
            kl_break = None if first is None else r["kl_per_slot"][first]
            slots = [(k, float(v), bool(c[k])) for k, v in enumerate(r["kl_per_slot"])
                     if 1 <= k < len(c) and v is not None]
            rows[(r["id"], r["anchor_pos"])] = (
                float(r["visual_kl_first"]), float(r["accept_len"]), int(r["n_valid"]), kl_break, slots)
    return rows


def stats(bench: str):
    Z, O = load(bench, "zlab"), load(bench, "llava-ov-1M")
    if not Z or not O:
        return None
    keys = sorted(set(Z) & set(O))
    kl_first = np.array([O[k][0] for k in keys])
    block_edges = np.quantile(kl_first, [.25, .5, .75])
    q_block = np.searchsorted(block_edges, kl_first)
    slot_kl = np.array([v for k in keys for (_, v, _) in O[k][4]])
    slot_edges = np.quantile(slot_kl, [.25, .5, .75])
    out = {"label": LABEL.get(bench, bench), "n_blocks": len(keys), "n_slots": int(slot_kl.size)}
    for name, D in (("zlab", Z), ("llava-ov-1M", O)):
        al = np.array([D[k][1] for k in keys]); nv = np.array([D[k][2] for k in keys])
        breaks = np.array([D[k][3] for k in keys if D[k][3] is not None], dtype=float)
        q_break = np.searchsorted(slot_edges, breaks)
        # per-slot correctness, paired with the (shared) slot KL by position
        kl_ok = []
        for k in keys:
            own = {pos: ok for pos, _, ok in D[k][4]}
            kl_ok.extend((v, own[pos]) for pos, v, _ in O[k][4] if pos in own)
        kl_ok = np.array(kl_ok, dtype=float)
        q_slot = np.searchsorted(slot_edges, kl_ok[:, 0])
        out[name] = {
            "accept_len": float(al.mean()),
            "token_acc": [float(kl_ok[q_slot == i, 1].mean()) for i in range(4)],
            "accept_rate": [float(al[q_block == i].sum() / nv[q_block == i].sum()) for i in range(4)],
            "trunc_share": [float((q_break == i).mean()) for i in range(4)],
            "n_breaks": int(len(breaks)),
        }
    return out


def rounded_bar(ax, x, height, width, color, r, **kw):
    """A column with 4px-ish rounded top corners and a square baseline."""
    r = min(r, width / 2, height / 2) if height > 0 else 0
    x0, x1, y1 = x - width / 2, x + width / 2, height
    verts = [(x0, 0), (x1, 0), (x1, y1 - r), (x1, y1), (x1 - r, y1),
             (x0 + r, y1), (x0, y1), (x0, y1 - r), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, linewidth=0, **kw))


def panel(ax, s, key, ylabel, fmt, ymax, ref=None, ref_label=None):
    width, gap, xs = 0.34, 0.06, np.arange(4)
    for i in range(4):
        z, o = s["zlab"][key][i] * 100, s["llava-ov-1M"][key][i] * 100
        rounded_bar(ax, xs[i] - (width + gap) / 2, z, width, Q_FILL[i], 0.9,
                    hatch="////", edgecolor=Q_INK[i])
        rounded_bar(ax, xs[i] + (width + gap) / 2, o, width, Q_FILL[i], 0.9)
        if i in (0, 3):  # label the two extremes only; nudge the pair apart so equal values never touch
            for xx, v, dx in ((xs[i] - (width + gap) / 2, z, -0.03), (xs[i] + (width + gap) / 2, o, 0.03)):
                ax.text(xx + dx, v + ymax * 0.015, fmt(v), ha="center", va="bottom", fontsize=6.6, color=INK2)
    if ref is not None:
        ax.axhline(ref, color=INK3, linewidth=0.7, zorder=0)
        if ref_label:
            # over Q1 on the right panel, whose bars stay far below the line
            ax.text(-0.55, ref + ymax * 0.012, ref_label, ha="left", va="bottom", fontsize=6.5, color=INK3)
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(0, ymax)
    ax.set_xticks(xs); ax.set_xticklabels(["Q1\nleast", "Q2", "Q3", "Q4\nmost"])
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=INK2)


def _ymax(s, key):
    return max(max(s["zlab"][key]), max(s["llava-ov-1M"][key])) * 100 * 1.22


def draw(bench: str, s: dict):
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(3.3, 6.3))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.13, top=0.95, hspace=0.45)
    pct = lambda v: f"{v:.0f}%"  # noqa: E731
    panel(a1, s, "token_acc", "Acceptance rate", pct, ymax=_ymax(s, "token_acc"))
    panel(a2, s, "accept_rate", "Block acceptance rate", pct, ymax=_ymax(s, "accept_rate"))
    panel(a3, s, "trunc_share", "Share of truncation points", pct, ymax=_ymax(s, "trunc_share"))
    fig.text(0.17, 0.982, s["label"], fontsize=10, fontweight="bold", color=INK, va="center")
    # the legend tells the two drafters apart by texture only; colour belongs to the quartiles
    handles = [Patch(facecolor="#dedcd6", edgecolor=INK2, hatch="////", linewidth=0,
                     label=f"DFlash, {DRAFTS['zlab']}"),
               Patch(facecolor="#b9b7b0", linewidth=0, label=f"DFlash, {DRAFTS['llava-ov-1M']}")]
    fig.legend(handles=handles, loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.56, 0.0),
               handlelength=1.6, labelspacing=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"where_blocks_break_{bench}.{ext}"), dpi=200 if ext == "png" else None,
                    facecolor=SURFACE)
    plt.close(fig)


def draw_combined(benches, all_stats):
    """One row per panel, one column per benchmark; legend on top, names below."""
    n = len(benches)
    fig, axes = plt.subplots(2, n, figsize=(2.25 * n + 0.6, 4.1))
    fig.subplots_adjust(left=0.25 / n, right=0.995, bottom=0.165, top=0.915, hspace=0.42, wspace=0.28)
    for j, b in enumerate(benches):
        s = all_stats[b]; z, o = s["zlab"], s["llava-ov-1M"]
        panel(axes[0, j], s, "token_acc", "Acceptance rate" if j == 0 else "", lambda v: f"{v:.0f}%", ymax=_ymax(s, "token_acc"))
        panel(axes[1, j], s, "trunc_share", "Share of truncation points" if j == 0 else "", lambda v: f"{v:.0f}%", ymax=_ymax(s, "trunc_share"))
        # benchmark name centred under its column, below the x labels
        box = axes[1, j].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.055, s["label"], ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=INK)
    handles = [Patch(facecolor="#dedcd6", edgecolor=INK2, hatch="////", linewidth=0, label=f"DFlash, {DRAFTS['zlab']}"),
               Patch(facecolor="#b9b7b0", linewidth=0, label=f"DFlash, {DRAFTS['llava-ov-1M']}")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"empirical_study.{ext}"), dpi=200 if ext == "png" else None,
                    facecolor=SURFACE)
    plt.close(fig)


def draw_supp(benches, all_stats):
    """One row: block acceptance rate (blocks grouped by their first predicted token)."""
    n = len(benches)
    fig, axes = plt.subplots(1, n, figsize=(2.25 * n + 0.6, 2.35))
    fig.subplots_adjust(left=0.25 / n, right=0.995, bottom=0.27, top=0.86, wspace=0.28)
    for j, b in enumerate(benches):
        s = all_stats[b]
        panel(axes[j], s, "accept_rate", "Block acceptance rate" if j == 0 else "",
              lambda v: f"{v:.0f}%", ymax=_ymax(s, "accept_rate"))
        box = axes[j].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.075, s["label"], ha="center", va="center",
                 fontsize=9.5, fontweight="bold", color=INK)
    handles = [Patch(facecolor="#dedcd6", edgecolor=INK2, hatch="////", linewidth=0, label=f"DFlash, {DRAFTS['zlab']}"),
               Patch(facecolor="#b9b7b0", linewidth=0, label=f"DFlash, {DRAFTS['llava-ov-1M']}")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"empirical_study_supp.{ext}"), dpi=200 if ext == "png" else None,
                    facecolor=SURFACE)
    plt.close(fig)


def main(argv):
    os.makedirs(OUT, exist_ok=True)
    combine = "--combine" in argv
    supp = "--supp" in argv
    argv = [a for a in argv if a not in ("--combine", "--supp")]
    benches = argv or ["chartqa", "textvqa", "mmstar", "dynamath", "mathvista", "mathverse"]
    summary = {}
    for b in benches:
        s = stats(b)
        if s is None:
            print(f"[skip] {b}: need both drafters under {RUNS}")
            continue
        if not (combine or supp):
            draw(b, s)
        summary[b] = s
        z, o = s["zlab"], s["llava-ov-1M"]
        print(f"{s['label']:9s} blocks={s['n_blocks']:5d} accept {z['accept_len']:.2f}->{o['accept_len']:.2f} | "
              f"token-acc Q1/Q4 {z['token_acc'][0]/z['token_acc'][3]:.2f}->{o['token_acc'][0]/o['token_acc'][3]:.2f} | "
              f"block-acc Q1/Q4 {z['accept_rate'][0]/z['accept_rate'][3]:.2f}->{o['accept_rate'][0]/o['accept_rate'][3]:.2f} | "
              f"break share Q1 {100*z['trunc_share'][0]:.1f}->{100*o['trunc_share'][0]:.1f}%  Q4 {100*z['trunc_share'][3]:.1f}->{100*o['trunc_share'][3]:.1f}%  "
              f"Q3+Q4 {100*(z['trunc_share'][2]+z['trunc_share'][3]):.0f}->{100*(o['trunc_share'][2]+o['trunc_share'][3]):.0f}%")
    kept = [b for b in benches if b in summary]
    if combine:
        draw_combined(kept, summary)
        print(f"wrote {OUT}/empirical_study.pdf")
    if supp:
        draw_supp(kept, summary)
        print(f"wrote {OUT}/empirical_study_supp.pdf")
    if not (combine or supp):
        # the json is the full six-benchmark summary; a partial run must not overwrite it
        if not argv:
            with open(os.path.join(OUT, "where_blocks_break.json"), "w") as handle:
                json.dump(summary, handle, indent=1)
        print(f"wrote {len(summary)} figure(s) to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
