#!/usr/bin/env python3
"""Pre-training completeness check of a regen jsonl against its visual-KL sidecar.

Run it once the regen has finished and the last scoring pass
(scripts/score_visual_kl_hpc.sh) has ended, and before training starts:

    cd <repo> && HF_HUB_OFFLINE=1 PYTHONPATH=$PWD nice -n 19 python scripts/check_visual_kl_complete.py \
        --config scripts/mmtraining_configs/<run>_hpc.yaml     # regen, sidecar, max_length from it
    [--regen <jsonl>] [--sidecar <dir>] [--max-length N] [--max-lines N]   # N: test on a head only

Read-only, CPU-only, streaming (one pass over the regen, one over the sidecar;
memory ~ the id sets). It does NOT need the processor: alignment of every row
(n_tokens / fingerprint) is what the producer re-checks at start-up
(data.visual_score_max_unscored_frac turns that into a hard stop); this script
checks everything that can be checked from the files alone and cross-checks the
scorer's own counters.

Exit code 0 = PASS, 1 = FAIL (each failed check is printed with FAIL).
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--regen", default=None, help="default: the config's data.train_data_path")
    p.add_argument("--sidecar", default=None, help="the directory (or single file) data.visual_score_path names; "
                   "default: the config's")
    p.add_argument("--max-length", type=int, default=None, help="default: the config's data.max_length, else 4096")
    p.add_argument("--max-kept-unscored", type=float, default=0.001,
                   help="FAIL above this share of image rows the last pass could not score although "
                   "training keeps them (image_error, length_mismatch, unaligned, oom: they train at g=0)")
    p.add_argument("--max-lines", type=int, default=None, help="testing only: read the first N lines of every file")
    p.add_argument("--stat-images", type=int, default=2000, help="stat every k-th image path (0: none)")
    p.add_argument("--config", default=None, help="training yaml (run from the repo with PYTHONPATH=repo): "
                   "also compares the scorer's header/end ids and library versions with this env")
    return p.parse_args()


def check_env_against_meta(config_path, metas, fail):
    """The scorer's tokenisation inputs vs what the producer will use now."""
    import transformers

    from specforge.config.schema import load_config
    from specforge.data.mm_preprocessing import _assistant_header_ids, _end_token_ids, load_mm_processor

    cfg = load_config(config_path)
    processor = load_mm_processor(cfg)
    header, end = list(_assistant_header_ids(processor)), sorted(int(i) for i in _end_token_ids(processor))
    for m in metas:
        meta = json.load(open(m))
        if meta.get("header_ids") != header or meta.get("end_ids") != end:
            fail(f"{m}: scorer header/end ids {meta.get('header_ids')}/{meta.get('end_ids')} != now {header}/{end}")
        if meta.get("versions", {}).get("transformers") != transformers.__version__:
            fail(f"{m}: scored with transformers {meta.get('versions', {}).get('transformers')}, env has {transformers.__version__}")
        if meta.get("target_model_path") != cfg.model.target_model_path:
            fail(f"{m}: target {meta.get('target_model_path')} != {cfg.model.target_model_path}")
    print(f"[env] header_ids={header} end_ids={end} transformers={transformers.__version__} "
          f"image_max_tokens={cfg.data.image_max_tokens} max_length={cfg.data.max_length}", flush=True)


def sidecar_files(path):
    """Same rule as specforge.data.visual_score._sidecar_files."""
    if os.path.isdir(path):
        return sorted(f for f in glob.glob(os.path.join(path, "*.jsonl")) if not os.path.basename(f).startswith("."))
    return [path]


def last_byte_is_newline(path):
    with open(path, "rb") as h:
        h.seek(0, os.SEEK_END)
        if h.tell() == 0:
            return True
        h.seek(-1, os.SEEK_END)
        return h.read(1) == b"\n"


def main():
    a = parse_args()
    if a.config:
        from specforge.config.schema import load_config

        cfg = load_config(a.config)
        a.regen = a.regen or cfg.data.train_data_path
        a.sidecar = a.sidecar or cfg.data.visual_score_path
        if a.max_length is not None and a.max_length != cfg.data.max_length:
            sys.exit(f"--max-length {a.max_length} != the config's data.max_length {cfg.data.max_length}")
        a.max_length = cfg.data.max_length
    if not a.regen or not a.sidecar:
        sys.exit("need --config, or --regen and --sidecar")
    a.max_length = a.max_length or 4096
    print(f"[args] regen {a.regen}\n[args] sidecar {a.sidecar}\n[args] max_length {a.max_length}", flush=True)
    failures = []

    def fail(msg):
        failures.append(msg)
        print(f"FAIL {msg}", flush=True)

    t0 = time.time()
    # ------------------------------------------------------------ regen
    if a.max_lines is None and not last_byte_is_newline(a.regen):
        fail(f"{a.regen} ends mid-line: the regen is still being written")
    image_ids, text_ids = set(), set()
    dup_ids = Counter()
    no_id = rel_paths = missing_images = stat_checked = 0
    with open(a.regen, encoding="utf-8") as h:
        for n, line in enumerate(h, start=1):
            if a.max_lines and n > a.max_lines:
                break
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("id")
            if rid is None:
                no_id += 1
                continue
            rid = str(rid)
            if rid in image_ids or rid in text_ids:
                dup_ids[rid] += 1
            img = rec.get("image")
            if img is None:
                text_ids.add(rid)
                continue
            image_ids.add(rid)
            first = img[0] if isinstance(img, list) else img
            if not os.path.isabs(first):
                rel_paths += 1
            elif a.stat_images and len(image_ids) % a.stat_images == 0:
                stat_checked += 1
                missing_images += not os.path.isfile(first)
    print(f"[regen] {len(image_ids)} image rows, {len(text_ids)} text rows, rows without id {no_id}, "
          f"duplicate ids {sum(dup_ids.values())}, relative image paths {rel_paths}, "
          f"image files stat-checked {stat_checked} missing {missing_images} ({time.time() - t0:.0f}s)", flush=True)
    if no_id:
        fail(f"{no_id} regen rows carry no id (the trainer joins on record['id'], the scorer falls back to the line number)")
    if dup_ids:
        fail(f"duplicate regen ids, e.g. {list(dup_ids)[:5]}")
    if missing_images:
        fail(f"{missing_images} sampled image paths do not exist")

    # ------------------------------------------------------------ sidecar
    files = sidecar_files(a.sidecar)
    print(f"[sidecar] files read by the trainer: {[os.path.basename(f) for f in files]}", flush=True)
    stray = [f for f in files if not os.path.basename(f).startswith("visual_kl.")]
    if stray:
        fail(f"non-scorer *.jsonl in the sidecar dir (the trainer reads them, the scorer's resume ignores them): {stray}")
    scored = {}  # id -> file index
    dups = Counter()
    bad_rows = nonfinite = unaligned_rows = too_long = 0
    tokens = 0
    for fi, f in enumerate(files):
        if not last_byte_is_newline(f):
            fail(f"{f} ends mid-line (a scorer is writing it, or was killed mid-write): the trainer raises 'invalid JSON'")
        with open(f, encoding="utf-8") as h:
            for n, line in enumerate(h, start=1):
                if a.max_lines and n > a.max_lines:
                    break
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    fail(f"{f}:{n} is not valid JSON")
                    continue
                rid = str(r["id"])
                if rid in scored:
                    dups[rid] += 1
                scored[rid] = fi
                kl, ent = r.get("kl", []), r.get("entropy")
                n_loss = int(r.get("n_loss", len(kl)))
                if len(kl) != n_loss or ent is None or len(ent) != n_loss or "n_tokens" not in r or not r.get("fp"):
                    bad_rows += 1
                if not all(map(math.isfinite, kl)) or (ent is not None and not all(map(math.isfinite, ent))):
                    nonfinite += 1
                unaligned_rows += bool(r.get("n_unaligned"))
                too_long += int(r.get("n_tokens", 0)) > a.max_length
                tokens += len(kl)
    print(f"[sidecar] {len(scored)} distinct ids, {sum(dups.values())} duplicate rows, {tokens} scored tokens, "
          f"malformed rows {bad_rows}, rows with NaN/inf {nonfinite}, rows with n_unaligned>0 {unaligned_rows}, "
          f"n_tokens>max_length {too_long} ({time.time() - t0:.0f}s)", flush=True)
    if dups:
        fail(f"{sum(dups.values())} duplicate sidecar ids (VisualScoreTable.load raises 'lists ... twice'), e.g. {list(dups)[:5]}")
    if bad_rows:
        fail(f"{bad_rows} sidecar rows with len(kl)/len(entropy)/n_loss/n_tokens/fp inconsistent")
    if nonfinite:
        fail(f"{nonfinite} sidecar rows hold NaN/inf (they would poison the quantile edges)")
    if too_long:
        fail(f"{too_long} sidecar rows longer than max_length={a.max_length}: scored with another max_length")

    # ------------------------------------------------------------ join
    sc = set(scored)
    unscored = image_ids - sc
    orphans = sc - image_ids - text_ids
    on_text = sc & text_ids
    print(f"[join] image rows with a score {len(image_ids & sc)}; image rows WITHOUT a score {len(unscored)} "
          f"({len(unscored) / max(len(image_ids), 1):.2%}); sidecar ids not in the regen (orphans: filtered or "
          f"another regen) {len(orphans)}; sidecar ids that are text rows in the regen {len(on_text)}", flush=True)
    if on_text:
        fail(f"{len(on_text)} sidecar ids are TEXT rows in this regen: the sidecar belongs to another regen")

    # ------------------------------------------------------------ scorer accounting
    if os.path.isdir(a.sidecar) and not a.max_lines:
        metas = sorted(glob.glob(os.path.join(a.sidecar, "meta.*.json")))
        dropped = Counter()
        regen_mtime = os.path.getmtime(a.regen)
        for m in metas:
            meta = json.load(open(m))
            c = meta["counts"]
            for k in ("encode_none", "encode_error", "image_error", "length_mismatch", "unaligned", "oom"):
                dropped[k] += c.get(k, 0)
            prog = os.path.join(a.sidecar, os.path.basename(m).replace("meta.", "progress.", 1))
            elapsed = json.load(open(prog)).get("elapsed_s", 0) if os.path.isfile(prog) else 0
            started = time.mktime(time.strptime(meta["finished_at"], "%Y-%m-%d %H:%M:%S")) - elapsed
            print(f"[meta] {os.path.basename(m)}: started ~{time.strftime('%F %T', time.localtime(started))}, "
                  f"finished {meta['finished_at']}, max_length={meta['max_length']}, counts={c}", flush=True)
            if meta.get("train_data_path") != a.regen:
                fail(f"{m} scored {meta.get('train_data_path')}, not {a.regen}")
            if int(meta.get("max_length", -1)) != a.max_length:
                fail(f"{m} scored with max_length={meta.get('max_length')}")
            if started < regen_mtime:
                fail(f"{m}: its last pass started before the regen file was last modified "
                     f"({time.strftime('%F %T', time.localtime(regen_mtime))}): run a final pass")
        if a.config:
            check_env_against_meta(a.config, metas, fail)
        total_dropped = sum(dropped.values())
        print(f"[meta] last-pass drop counters summed over shards: {dict(dropped)} = {total_dropped}; "
              f"image rows without a score: {len(unscored)}", flush=True)
        if total_dropped != len(unscored):
            fail(f"unscored image rows ({len(unscored)}) != rows the last pass dropped ({total_dropped}): "
                 "the last pass did not cover the whole regen, or the files changed since")
        kept_unscored = total_dropped - dropped["encode_none"]  # encode_none rows: training drops them too
        if kept_unscored > a.max_kept_unscored * max(len(image_ids), 1):
            fail(f"{kept_unscored} image rows ({kept_unscored / max(len(image_ids), 1):.3%}) failed scoring for a "
                 f"reason training does not share ({ {k: v for k, v in dropped.items() if k != 'encode_none'} }): "
                 "they would train at g=0; re-score them or raise --max-kept-unscored deliberately")
        q = os.path.join(a.sidecar, "quantiles.json")
        if os.path.isfile(q):
            try:
                cached = json.load(open(q))
            except (OSError, ValueError):
                cached = {}
            signature = [[f, os.stat(f).st_size, os.stat(f).st_mtime_ns] for f in files]
            state = "matches" if cached.get("signature") == signature and cached.get("num_tokens") == tokens else "STALE"
            print(f"[quantiles] {q}: num_tokens={cached.get('num_tokens')} vs {tokens} scored tokens now -> {state}"
                  + (" (the producer recomputes a stale cache; deleting it is harmless)" if state == "STALE" else ""),
                  flush=True)
    print(f"[done] {time.time() - t0:.0f}s: {'PASS' if not failures else f'FAIL ({len(failures)} checks)'}", flush=True)
    print("[next] exact per-row alignment = what the producer prints at start-up: "
          "'[visual-score] channel over the N kept prompts: misaligned=0, missing=<~0>'", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
