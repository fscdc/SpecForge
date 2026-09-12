#!/usr/bin/env python3
"""Offline visual-dependency scorer for MMFlash training data.

For every image row of the training jsonl this writes one line to a sidecar::

    {"id": ..., "n_tokens": S, "n_loss": L, "kl": [...], "entropy": [...]}

where ``kl[i]`` is, for the i-th loss-mask position (ascending), how far the
target's next-token distribution moves when the image is removed,

    KL( p(. | prefix, image) || p(. | prefix, no image) )     [nats]

and ``entropy[i]`` is the entropy of the with-image distribution (the control
that separates "visual" from merely "uncertain"). The training producer turns
``kl`` into ``g in [0, 1]`` (``data.visual_score_transform``) and weights the
MMFlash objective with it; see ``specforge/data/visual_score.py``.

Exactness matters more than speed here: the row is tokenised with the SAME
``encode_mm_record`` the trainer uses (same processor, chat template,
``max_length``, image resolution), and ``n_tokens`` is stored so a sidecar
built from a different tokenisation is detected and ignored at training time
rather than silently misaligned.

The with-image pass is fed the processor's full batch (``pixel_values``,
``image_grid_thw``): Qwen3.5 embeds ``<|image_pad|>`` as plain text when
``pixel_values`` is absent, which would make the "with image" side blind.

Sharding / resume
-----------------
Rows are assigned to shards by line index (``line % num_shards == shard``), so
shards are independent and can run on different GPUs or nodes; each appends
to its own file and skips ids it already wrote, so a killed job is resumed by
re-running the same command. ``scripts/score_visual_kl_hpc.sh`` fans one
shard per GPU.

    python scripts/score_visual_kl.py \
        --config scripts/mmtraining_configs/qwen3.5-4b-mmflash_hpc.yaml \
        --output-dir /scratch/.../visual_kl/llava-ov15-1M-prompted \
        --shard-index 0 --num-shards 4

Point ``data.visual_score_path`` at ``--output-dir`` afterwards (the loader
reads every ``*.jsonl`` in the directory).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="training yaml: supplies data.train_data_path, data.max_length, model.*")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-rows", type=int, default=None, help="stop after scoring this many rows (smoke test)")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--no-resume", action="store_true", help="overwrite this shard's file instead of appending")
    parser.add_argument("--expected-rows", type=int, default=None, help="rows in the jsonl, only used for the ETA (wc -l)")
    return parser.parse_args()


def _load_analysis_module():
    """``visual_kl_per_token`` lives in the accept-analysis script (not a package)."""
    spec = importlib.util.spec_from_file_location(
        "analyze_dflash_accept", REPO / "scripts" / "analyze_dflash_accept.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint_input_ids(input_ids) -> str:
    """A short digest of the exact token sequence a score was measured on.

    The join key is the record id, but ids are shared across the regen files
    built from the same source corpus while their RESPONSES differ -- and the
    KL of a token depends on the whole prefix, response included. Without this,
    pointing ``data.visual_score_path`` at the sidecar of a *different* regen
    of the same corpus would be caught only when the token counts happened to
    disagree. With it, every row is checked.
    """
    import hashlib
    from array import array

    return hashlib.blake2b(
        array("i", input_ids).tobytes(), digest_size=8
    ).hexdigest()


def _shard_paths(output_dir: str, shard: int, num_shards: int):
    tag = f"shard{shard:03d}-of-{num_shards:03d}"
    return (
        os.path.join(output_dir, f"visual_kl.{tag}.jsonl"),
        os.path.join(output_dir, f"meta.{tag}.json"),
        os.path.join(output_dir, f"progress.{tag}.json"),
    )


def _already_done(path: str) -> set:
    done = set()
    if not os.path.isfile(path):
        return done
    # a killed job can leave a torn last line; drop it so the append stays valid
    with open(path, "rb+") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size and handle.seek(size - 1) is not None and handle.read(1) != b"\n":
            handle.seek(0)
            data = handle.read()
            cut = data.rfind(b"\n")
            handle.seek(0)
            handle.truncate(cut + 1 if cut >= 0 else 0)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    done.add(str(json.loads(line)["id"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")
    sys.path.insert(0, str(REPO))
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText

    from specforge.config.schema import load_config
    from specforge.data.mm_preprocessing import (
        _assistant_header_ids,
        _end_token_ids,
        encode_mm_record,
        load_mm_processor,
        to_chat_messages,
    )
    from specforge.data.prompt_builder import _iter_records

    analysis = _load_analysis_module()
    visual_kl_per_token = analysis.visual_kl_per_token

    config = load_config(args.config)
    source = config.data.train_data_path
    max_length = int(config.data.max_length)
    print(f"[scorer] config={args.config}", flush=True)
    print(f"[scorer] data={source} max_length={max_length} shard={args.shard_index}/{args.num_shards}", flush=True)

    processor = load_mm_processor(config)
    header_ids = _assistant_header_ids(processor)
    end_ids = _end_token_ids(processor)
    dtype = getattr(torch, args.dtype)
    started_load = time.monotonic()
    target = AutoModelForImageTextToText.from_pretrained(
        config.model.target_model_path,
        torch_dtype=dtype,
        trust_remote_code=config.model.trust_remote_code,
        cache_dir=config.model.cache_dir,
    )
    target = target.to(args.device).eval()
    torch.set_grad_enabled(False)
    print(
        f"[scorer] target {config.model.target_model_path} loaded in {time.monotonic() - started_load:.0f}s "
        f"on {args.device} ({dtype}); header_ids={header_ids} end_ids={sorted(end_ids)}",
        flush=True,
    )

    out_path, meta_path, progress_path = _shard_paths(args.output_dir, args.shard_index, args.num_shards)
    done = set() if args.no_resume else _already_done(out_path)
    if done:
        print(f"[scorer] resuming: {len(done)} rows already in {out_path}", flush=True)
    mode = "w" if args.no_resume else "a"

    counts: Dict[str, int] = {
        "seen": 0, "text_only": 0, "resumed": 0, "encode_error": 0, "encode_none": 0,
        "image_error": 0, "length_mismatch": 0, "unaligned": 0, "oom": 0, "scored": 0,
    }
    kl_sum = entropy_sum = 0.0
    kl_tokens = 0
    unaligned_positions = 0
    started = time.monotonic()
    last_log = started

    def _log(final: bool = False) -> None:
        elapsed = max(time.monotonic() - started, 1e-6)
        rate = counts["scored"] / elapsed
        eta = ""
        if args.expected_rows and rate > 0:
            share = 0.655  # image rows are ~65% of the corpus; only a hint for the ETA
            remaining = max(args.expected_rows / args.num_shards * share - counts["scored"] - counts["resumed"], 0)
            eta = f", ~{remaining / rate / 3600:.1f} h left"
        mean_kl = kl_sum / max(kl_tokens, 1)
        mean_h = entropy_sum / max(kl_tokens, 1)
        print(
            f"[scorer] {'done' if final else 'progress'} shard={args.shard_index}: "
            + ", ".join(f"{k}={v}" for k, v in counts.items())
            + f" | {rate:.2f} rows/s{eta} | mean KL={mean_kl:.3f} nats, mean H={mean_h:.3f}, "
            f"scored tokens={kl_tokens}, unaligned positions={unaligned_positions}",
            flush=True,
        )
        with open(progress_path, "w", encoding="utf-8") as handle:
            json.dump({**counts, "elapsed_s": elapsed, "mean_kl": mean_kl, "mean_entropy": mean_h}, handle)

    with open(out_path, mode, encoding="utf-8") as sink:
        for line_number, record in _iter_records(source):
            if (line_number - 1) % args.num_shards != args.shard_index:
                continue
            counts["seen"] += 1
            record_id = str(record.get("id", line_number))
            if record.get("image") is None:
                counts["text_only"] += 1
                continue
            if record_id in done:
                counts["resumed"] += 1
                continue
            try:
                payload = encode_mm_record(
                    record,
                    processor,
                    image_root=config.data.image_root,
                    max_length=max_length,
                    train_only_last_turn=config.data.train_only_last_turn,
                    header_ids=header_ids,
                    end_ids=end_ids,
                )
            except (OSError, ValueError):
                counts["encode_error"] += 1
                continue
            if payload is None:
                counts["encode_none"] += 1
                continue

            try:
                with Image.open(payload["image"]) as handle:
                    image = handle.convert("RGB")
                    text = processor.apply_chat_template(
                        to_chat_messages(record["conversations"]),
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                    inputs = processor(text=[text], images=[image], return_tensors="pt")
            except (OSError, ValueError):
                counts["image_error"] += 1
                continue
            if inputs["input_ids"].shape[1] != len(payload["input_ids"]):
                counts["length_mismatch"] += 1
                continue
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            input_ids = inputs["input_ids"]

            try:
                measured = visual_kl_per_token(
                    target,
                    processor,
                    record,
                    input_ids,
                    payload["loss_mask"],
                    max_length,
                    model_inputs=inputs,
                )
            except torch.cuda.OutOfMemoryError:
                counts["oom"] += 1
                torch.cuda.empty_cache()
                continue
            if measured is None:
                counts["unaligned"] += 1
                continue
            kl, entropy = measured
            positions = [index for index, flag in enumerate(payload["loss_mask"]) if flag]
            kl_row = [float(kl.get(p, 0.0)) for p in positions]
            entropy_row = [float(entropy.get(p, 0.0)) for p in positions]
            missing = sum(1 for p in positions if p not in kl)
            unaligned_positions += missing
            sink.write(
                json.dumps(
                    {
                        "id": record_id,
                        "n_tokens": len(payload["input_ids"]),
                        "n_loss": len(positions),
                        "fp": fingerprint_input_ids(payload["input_ids"]),
                        "kl": [round(v, 6) for v in kl_row],
                        "entropy": [round(v, 6) for v in entropy_row],
                        **({"n_unaligned": missing} if missing else {}),
                    }
                )
                + "\n"
            )
            counts["scored"] += 1
            kl_sum += sum(kl_row)
            entropy_sum += sum(entropy_row)
            kl_tokens += len(kl_row)
            if counts["scored"] % args.flush_every == 0:
                sink.flush()
            if counts["scored"] % args.log_every == 0 or time.monotonic() - last_log > 300:
                _log()
                last_log = time.monotonic()
            if args.max_rows is not None and counts["scored"] >= args.max_rows:
                print(f"[scorer] --max-rows {args.max_rows} reached", flush=True)
                break
        sink.flush()

    _log(final=True)
    import transformers

    meta: Dict[str, Any] = {
        "config": os.path.abspath(args.config),
        "train_data_path": source,
        "max_length": max_length,
        "target_model_path": config.model.target_model_path,
        "processor": getattr(processor, "name_or_path", None)
        or getattr(getattr(processor, "tokenizer", None), "name_or_path", None),
        "header_ids": list(header_ids),
        "end_ids": sorted(int(i) for i in end_ids),
        "dtype": args.dtype,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "counts": counts,
        "unaligned_positions": unaligned_positions,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[scorer] wrote {out_path} and {meta_path}", flush=True)


if __name__ == "__main__":
    main()
