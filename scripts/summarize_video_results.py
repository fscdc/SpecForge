#!/usr/bin/env python
"""Tabulate the video (VDC) runs under results/.

One row per results file that carries the benchmark, with the numbers that
matter for a draft on video prompts: decode-only throughput (the prefill is
~98% of the wall time on VDC, so wall/e2e throughput cannot separate drafts),
the acceptance length, and enough context (TTFT, decode tokens and seconds)
to judge how noisy the throughput figure is.

    python scripts/summarize_video_results.py
    python scripts/summarize_video_results.py --benchmark chartqa --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
import time

_DECODER = json.JSONDecoder()


def read_result(path: str) -> dict:
    """The one pretty-printed JSON object a bench_mm results file holds."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    obj, _ = _DECODER.raw_decode(text, index)
    return obj


def window_from_name(name: str) -> str:
    match = re.search(r"_win(\d+|full)", name)
    return match.group(1) if match else "full"


def collect(results_dir: str, benchmark: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*_results.jsonl")), key=os.path.getmtime):
        try:
            result = read_result(path)
        except (OSError, ValueError) as exc:
            print(f"skip {path}: {exc}", file=sys.stderr)
            continue
        entries = result.get(benchmark)
        if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
            continue
        entry = entries[0]
        throughput = entry.get("throughput_summary") or {}
        accept = entry.get("accept_length_summary") or {}
        name = os.path.basename(path).replace("_results.jsonl", "")
        rows.append(
            {
                "name": name,
                "model": str(result.get("model", "")).replace("Qwen/Qwen3.5-", ""),
                "window": window_from_name(name),
                "block": result.get("block_size"),
                "conc": result.get("concurrency"),
                "temp": (result.get("sampling_params") or {}).get("temperature"),
                "nq": entry.get("num_questions"),
                "decode_tok_s": throughput.get("decode_output_throughput"),
                "wall_tok_s": throughput.get("wall_output_throughput"),
                "accept_mean": accept.get("mean"),
                "accept_tw": accept.get("token_weighted_mean"),
                "ttft_s": throughput.get("ttft_mean"),
                "prefill_share": throughput.get("prefill_share"),
                "decode_tokens": throughput.get("decode_tokens"),
                "decode_s": throughput.get("decode_latency_sum"),
                "mtime": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(path))),
            }
        )
    return rows


def fmt(value, width: int, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        return f"{value:{width}.{digits}f}"
    return f"{'-' if value is None else str(value):>{width}s}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--benchmark", default="vdc")
    parser.add_argument("--csv", default=None, help="also write the rows to this CSV")
    args = parser.parse_args()

    rows = collect(args.results_dir, args.benchmark)
    if not rows:
        print(f"no results file under {args.results_dir} carries a {args.benchmark!r} entry")
        return 1

    # the origin run of each model anchors the speedup column
    origin = {}
    for row in rows:
        if row["block"] in (0, None) and row["accept_mean"] is None:
            origin.setdefault(row["model"], row["decode_tok_s"])

    header = (
        f"{'name':70s} {'model':5s} {'win':>5s} {'blk':>3s} {'conc':>4s} {'nq':>3s} "
        f"{'decode':>8s} {'x':>5s} {'acc':>5s} {'acc_tw':>6s} {'ttft':>6s} {'pref':>5s} {'dtok':>5s} {'dsec':>6s} {'mtime':>11s}"
    )
    print(header)
    for row in rows:
        base = origin.get(row["model"])
        speedup = (row["decode_tok_s"] / base) if base and isinstance(row["decode_tok_s"], (int, float)) else None
        print(
            f"{row['name']:70s} {row['model']:5s} {row['window']:>5s} {str(row['block']):>3s} {str(row['conc']):>4s} {str(row['nq']):>3s} "
            f"{fmt(row['decode_tok_s'], 8)} {fmt(speedup, 5)} {fmt(row['accept_mean'], 5)} {fmt(row['accept_tw'], 6)} "
            f"{fmt(row['ttft_s'], 6, 1)} {fmt(row['prefill_share'], 5)} {fmt(row['decode_tokens'], 5, 0)} {fmt(row['decode_s'], 6, 1)} {row['mtime']:>11s}"
        )

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
