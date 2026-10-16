#!/usr/bin/env python3
"""Summarize scripts/profile_dflash_steps_hpc.sh: where a DFlash decode step goes,
and what share of the end-to-end time drafting and verification take.

    python scripts/summarize_dflash_step_timing.py results/dflash_step_timing/<run>

Per configuration and benchmark:

  decode step (GPU ms per step, from the CUDA events)
    draft   = the draft model's forward + choosing its tokens through the target
              lm_head + the sparse context gather (sparse runs)
    verify  = the target forward over the drafted block
    kv      = writing the committed tokens' target hidden states into the draft KV
    other   = the rest of the step: building the draft inputs, the acceptance check
  end to end (share of the server's busy time, from the results file's
  server_accounting): prefill, then the decode time split by the same fractions;
  "decode other" also holds the host time between steps that no GPU region covers.
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from summarize_video_results import read_result  # noqa: E402

REGIONS = ("draft_forward", "draft_sample", "sparse_gather", "verify_forward", "kv_append")
NAMES = {"longvideobench": "LongVideoBench", "mvbench": "MVBench", "videomme": "Video-MME"}


def load_totals(cfg_dir):
    merged = {}
    for path in glob.glob(os.path.join(cfg_dir, "timing", "*.json")):
        with open(path, encoding="utf-8") as handle:
            totals = json.load(handle).get("totals") or {}
        for label, entry in totals.items():
            into = merged.setdefault(label, {})
            for key, value in entry.items():
                into[key] = into.get(key, 0) + value
    return merged


def read_config(cfg_dir):
    out = {}
    with open(os.path.join(cfg_dir, "config.txt"), encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.strip().partition("=")
            out[key] = value
    return out


def main() -> int:
    run = sys.argv[1]
    repo = os.path.dirname(HERE)
    rows = []
    for cfg_dir in sorted(glob.glob(os.path.join(run, "*", ""))):
        name = os.path.basename(os.path.normpath(cfg_dir))
        if not os.path.isfile(os.path.join(cfg_dir, "config.txt")):
            continue
        config = read_config(cfg_dir)
        totals = load_totals(cfg_dir)
        results_path = os.path.join(repo, "results", config.get("results", "") + "_results.jsonl")
        results = read_result(results_path) if os.path.isfile(results_path) else {}
        for bench, entry in totals.items():
            if bench not in NAMES or not entry.get("steps"):
                continue
            steps = entry["steps"]
            step = entry["step_ms"] / steps
            draft = sum(entry.get(r, 0.0) for r in ("draft_forward", "draft_sample", "sparse_gather")) / steps
            verify = entry.get("verify_forward", 0.0) / steps
            kv = entry.get("kv_append", 0.0) / steps
            other = step - draft - verify - kv
            res = (results.get(bench) or [{}])[0] if isinstance(results.get(bench), list) else {}
            acc = res.get("server_accounting") or {}
            tau = (res.get("accept_length_summary") or {}).get("mean")
            busy = acc.get("busy_time") or 0.0
            prefill = acc.get("prefill_time") or 0.0
            decode = acc.get("decode_time") or 0.0
            server_step_ms = acc.get("decode_step_ms")
            rows.append(dict(config=name, bench=bench, steps=steps, step=step, draft=draft, verify=verify, kv=kv,
                             other=other, draft_fwd=entry.get("draft_forward", 0.0) / steps,
                             draft_sample=entry.get("draft_sample", 0.0) / steps,
                             gather=entry.get("sparse_gather", 0.0) / steps, tau=tau, busy=busy, prefill=prefill,
                             decode=decode, server_step_ms=server_step_ms,
                             tokens_per_step=acc.get("tokens_per_decode_step")))
    if not rows:
        print(f"no timing data under {run}")
        return 1

    print("Decode step, GPU ms per step (share of the step)")
    print(f"{'config':24s}{'benchmark':16s}{'steps':>7s}{'tau':>6s}{'step':>8s}{'draft':>15s}{'verify':>15s}{'kv write':>14s}{'other':>13s}"
          f"{'  draft = fwd + sample + gather':>36s}")
    for r in rows:
        def cell(x):
            return f"{x:6.2f} ({x / r['step']:4.0%})"
        tau = f"{r['tau']:.2f}" if r["tau"] is not None else "-"
        print(f"{r['config']:24s}{NAMES[r['bench']]:16s}{r['steps']:7d}{tau:>6s}{r['step']:8.2f}{cell(r['draft']):>15s}{cell(r['verify']):>15s}"
              f"{cell(r['kv']):>14s}{cell(r['other']):>13s}   {r['draft_fwd']:.2f} + {r['draft_sample']:.2f} + {r['gather']:.2f}")
    print()
    print("End to end, share of the server's busy time (prefill + decode); decode split by the step fractions above")
    print(f"{'config':24s}{'benchmark':16s}{'busy s':>8s}{'prefill':>9s}{'draft':>8s}{'verify':>8s}{'kv+other':>10s}{'server step ms':>16s}")
    agg = {}
    for r in rows:
        if not r["busy"]:
            continue
        gpu_sum = r["step"] * r["steps"] / 1000.0
        scale = r["decode"] / gpu_sum if gpu_sum else 0.0  # wall decode time per GPU-measured second
        d = r["draft"] * r["steps"] / 1000.0 * scale
        v = r["verify"] * r["steps"] / 1000.0 * scale
        rest = r["decode"] - d - v
        shares = (r["prefill"] / r["busy"], d / r["busy"], v / r["busy"], rest / r["busy"])
        agg.setdefault(r["config"], []).append(shares)
        print(f"{r['config']:24s}{NAMES[r['bench']]:16s}{r['busy']:8.0f}" + "".join(f"{x:9.1%}" if i == 0 else (f"{x:8.1%}" if i < 3 else f"{x:10.1%}") for i, x in enumerate(shares))
              + f"{r['server_step_ms'] or float('nan'):16.2f}")
    print()
    print("Mean over benchmarks")
    for config in dict.fromkeys(r["config"] for r in rows):
        sub = [r for r in rows if r["config"] == config]
        mean = lambda k: statistics.mean(r[k] for r in sub)  # noqa: E731
        step = mean("step")
        line = (f"{config:24s} step {step:5.2f} ms | draft {mean('draft'):5.2f} ({mean('draft') / step:.0%}) | "
                f"verify {mean('verify'):5.2f} ({mean('verify') / step:.0%}) | kv {mean('kv'):4.2f} | other {mean('other'):4.2f}")
        if config in agg:
            p, d, v, rest = (statistics.mean(s[i] for s in agg[config]) for i in range(4))
            line += f" || e2e: prefill {p:.1%}, draft {d:.1%}, verify {v:.1%}, other decode {rest:.1%}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
