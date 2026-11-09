#!/usr/bin/env python3
"""The concurrency table from the vLLM runs of scripts/benchmark_vllm_concurrency.sh.

    python benchmarks/vllm/summarize_concurrency.py [--model-tag qwen35-4B] [--run-tag _smoke]

Read like the SGLang concurrency tables: per benchmark and in flight, accept length /
decode throughput (tok/s) / speedup, with the mean decode batch after it. Throughput is
the server-side decode-step figure (decode tokens over the time the server spent on
steps that only decoded), speedup is MMFlash over the target alone at the same number
in flight, and the average is the mean over the benchmarks of each quantity. The
compact table after it adds the wall-clock view (all generated tokens over the whole
run, prefill included) and, where they exist, the SGLang speedups from the
origin_/mmflash_ results of the same target.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Any, Dict, Optional

NAMES = {
    "chartqa": "ChartQA", "charxiv": "CharXiv", "mmstar": "MMStar", "mmbench-origin": "MMBench",
    "mmbench": "MMBench", "dynamath": "DynaMath", "mathvista": "MathVista", "mathverse": "MathVerse",
}


def load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def entry(results: Optional[Dict[str, Any]], bench: str) -> Optional[Dict[str, Any]]:
    runs = (results or {}).get(bench) or []
    return runs[-1] if runs else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model-tag", default="qwen35-4B")
    parser.add_argument("--temp-tag", default="0")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument(
        "--benches", nargs="+", default=None,
        help="default: the seven image benchmarks, MMBench as the SGLang table of this target has it",
    )
    args = parser.parse_args()
    mmbench = "mmbench" if args.model_tag.endswith("9B") else "mmbench-origin"
    benches = args.benches or ["chartqa", "charxiv", "mmstar", mmbench, "dynamath", "mathvista", "mathverse"]

    def path(prefix: str, method: str, c: int, tag: str = "") -> str:
        return os.path.join(
            args.results_dir,
            f"{prefix}{method}_{args.model_tag}_concurrency{c}_temp{args.temp_tag}_4096{tag}_results.jsonl",
        )

    rows = {}
    for c in args.concurrencies:
        for method in ("origin", "mmflash"):
            results = load(path("vllm_", method, c, args.run_tag))
            cells = {}
            for bench in benches:
                e = entry(results, bench)
                sa = (e or {}).get("server_accounting") or {}
                if not e or "decode_step_throughput" not in sa:
                    continue
                accept = (e.get("accept_length_summary") or {}).get("mean") if method == "mmflash" else None
                ts = e.get("throughput_summary") or {}
                cells[bench] = dict(
                    al=accept, tp=sa["decode_step_throughput"], batch=sa.get("mean_decode_batch", 0.0),
                    pre=sa.get("prefill_share", 0.0), wall=ts.get("wall_output_throughput"),
                    acc=((e.get("metrics") or [{}])[0]).get("accuracy"),
                )
            rows[(method, c)] = cells

    def fmt(cell: Dict[str, Any], sp: float) -> str:
        al = "—" if cell["al"] is None else f"{cell['al']:.2f}"
        return f"{al} / {cell['tp']:,.0f} / {sp:.2f}× ({cell['batch']:.1f})"

    header = "| Method · in flight | " + " | ".join(NAMES[b] for b in benches) + " | Average |"
    print(f"\nvLLM, {args.model_tag}, temperature {args.temp_tag}: accept / decode tok/s / speedup (mean decode batch)\n")
    print(header)
    print("|---|" + "---|" * (len(benches) + 1))
    compact = []
    for c in args.concurrencies:
        origin, mm = rows.get(("origin", c), {}), rows.get(("mmflash", c), {})
        for method, cells, label in (("origin", origin, "Baseline"), ("mmflash", mm, "MMFlash")):
            if not cells:
                print(f"| {label} · {c} | " + " | ".join("pending" for _ in benches) + " | pending |")
                continue
            parts, sps = [], []
            for bench in benches:
                cell = cells.get(bench)
                base = origin.get(bench)
                if cell is None or base is None:
                    parts.append("pending")
                    continue
                sp = cell["tp"] / base["tp"]
                sps.append(sp)
                parts.append(fmt(cell, sp))
            complete = len(sps) == len(benches)
            if complete:
                avg = {
                    "al": None if method == "origin" else statistics.fmean(cells[b]["al"] for b in benches),
                    "tp": statistics.fmean(cells[b]["tp"] for b in benches),
                    "batch": statistics.fmean(cells[b]["batch"] for b in benches),
                }
                parts.append(fmt(avg, statistics.fmean(sps)))
            else:
                parts.append(f"({len(sps)}/{len(benches)})")
            print(f"| {label} · {c} | " + " | ".join(parts) + " |")
        if origin and mm and all(b in origin and b in mm for b in benches):
            sglang_sp = None
            so = [entry(load(path("", "origin", c)), b) for b in benches]
            sm = [entry(load(path("", "mmflash", c)), b) for b in benches]
            if all(so) and all(sm):
                def tp(e):
                    sa = e.get("server_accounting") or {}
                    # the single-request SGLang files predate the server accounting
                    return sa.get("decode_step_throughput") or e["throughput_summary"]["decode_output_throughput"]
                sglang_sp = statistics.fmean(tp(m) / tp(o) for o, m in zip(so, sm))
            compact.append(dict(
                c=c,
                tp_o=statistics.fmean(origin[b]["tp"] for b in benches),
                tp_m=statistics.fmean(mm[b]["tp"] for b in benches),
                sp=statistics.fmean(mm[b]["tp"] / origin[b]["tp"] for b in benches),
                al=statistics.fmean(mm[b]["al"] for b in benches),
                batch_o=statistics.fmean(origin[b]["batch"] for b in benches),
                batch_m=statistics.fmean(mm[b]["batch"] for b in benches),
                pre_m=statistics.fmean(mm[b]["pre"] for b in benches),
                wall_sp=statistics.fmean(mm[b]["wall"] / origin[b]["wall"] for b in benches),
                acc_o=[origin[b]["acc"] for b in benches],
                acc_m=[mm[b]["acc"] for b in benches],
                sglang_sp=sglang_sp,
            ))

    if compact:
        print("\n| in flight | baseline tok/s | MMFlash tok/s | speedup | accept | batch (base / MMFlash) "
              "| MMFlash prefill share | wall-clock speedup | SGLang speedup |")
        print("|---|---|---|---|---|---|---|---|---|")
        for r in compact:
            sg = "—" if r["sglang_sp"] is None else f"{r['sglang_sp']:.2f}×"
            print(f"| {r['c']} | {r['tp_o']:,.0f} | {r['tp_m']:,.0f} | {r['sp']:.2f}× | {r['al']:.2f} | "
                  f"{r['batch_o']:.1f} / {r['batch_m']:.1f} | {r['pre_m']:.1%} | {r['wall_sp']:.2f}× | {sg} |")
        print("\naccuracy (%, benchmarks with reference answers), baseline vs MMFlash:")
        for r in compact:
            pairs = [(o, m) for o, m in zip(r["acc_o"], r["acc_m"]) if isinstance(o, (int, float)) and isinstance(m, (int, float))]
            if pairs:
                print(f"  {r['c']:>2} in flight: {100 * statistics.fmean(o for o, _ in pairs):.1f} vs "
                      f"{100 * statistics.fmean(m for _, m in pairs):.1f} over {len(pairs)} benchmarks")


if __name__ == "__main__":
    main()
