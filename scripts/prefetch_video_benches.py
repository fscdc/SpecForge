#!/usr/bin/env python3
"""Fetch the videos and decode the frames a video bench_mm run will read (CPU only).

The video benchmarks download their videos and decode frames inside
``load_data()``. Two eval jobs doing that at once can tear each other's frame
cache (``materialize_frames`` writes one JPEG at a time and a reader takes
whatever frames it finds), and LongVideoBench asks the first N questions *whose
video is on disk*, so jobs that download while they run can end up on different
question sets. Running this once fills the caches; eval jobs then run with
downloads off (``NO_DOWNLOAD=1`` in scripts/eval_mmflash_video_hpc.sh) and all
read the same questions.

Frame settings are the eval script's (48 frames, VDC 16, MVBench at 720p area)
and the run happens from the repo root, where the frame caches' relative paths
resolve for bench_mm too. Each benchmark is loaded at N + warmup questions, as
bench_mm loads it.

    python scripts/prefetch_video_benches.py longvideobench:200 mvbench:200 videomme:200 \
        --done-file /path/to/prefetch.done

The done file gets, per benchmark, the number of questions and a fingerprint of
their frames and text; it is written only when every benchmark loaded its full
count, so an eval job waiting on it never starts on a partial cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_SETTINGS = {
    "VDC_NUM_FRAMES": "16",
    "LVB_NUM_FRAMES": "48",
    "MOVIECHAT_NUM_FRAMES": "48",
    "VIDEOMME_NUM_FRAMES": "48",
    "MVBENCH_NUM_FRAMES": "48",
    "MVBENCH_FRAME_PIXELS": "1280x720",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("benches", nargs="+", help="<benchmark>:<questions>, as in bench_mm --benchmark-list")
    parser.add_argument("--warmup", type=int, default=1, help="bench_mm's warmup count (concurrency x servers)")
    parser.add_argument("--done-file", default=None, help="written once every benchmark loaded its full count")
    return parser.parse_args()


def fingerprint(questions) -> str:
    digest = hashlib.sha1()
    for question in questions:
        for kind, value in question["parts"]:
            digest.update(f"{kind}:{value}\n".encode())
    return digest.hexdigest()[:16]


def main() -> int:
    args = parse_args()
    os.chdir(REPO)
    sys.path.insert(0, os.path.join(REPO, "benchmarks"))
    for key, value in FRAME_SETTINGS.items():
        os.environ.setdefault(key, value)
    from mm_benchmarker import MM_BENCHMARKS

    summary, complete = {}, True
    for item in args.benches:
        name, _, count = item.partition(":")
        wanted = int(count) + args.warmup
        start = time.time()
        print(f"[prefetch] {name}: loading {wanted} questions ({count} + {args.warmup} warmup)", flush=True)
        questions, _labels = MM_BENCHMARKS.get(name)(num_samples=wanted).load_data()
        frames = [sum(1 for kind, _ in q["parts"] if kind == "image") for q in questions]
        summary[name] = {
            "questions": len(questions),
            "wanted": wanted,
            "frames_min": min(frames, default=0),
            "frames_max": max(frames, default=0),
            "fingerprint": fingerprint(questions),
            "seconds": round(time.time() - start, 1),
        }
        print(f"[prefetch] {name}: {summary[name]}", flush=True)
        complete &= len(questions) >= wanted
    if not complete:
        print("[prefetch] ERROR: some benchmark loaded fewer questions than asked; no done file", flush=True)
        return 1
    if args.done_file:
        with open(args.done_file + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)
        os.replace(args.done_file + ".tmp", args.done_file)
        print(f"[prefetch] wrote {args.done_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
