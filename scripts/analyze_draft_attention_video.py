"""Where does the MMFlash draft attend when the context is a 48-frame video?

The question behind the paper's sparse-draft motivation: at a decode step deep
into a ~43k-token video prompt, how much of the draft's attention still goes to
the early frames, and how much to the last few thousand positions (the recent
frames, the question, the answer so far)?

Per benchmark question -- the same questions, frames and prompt text that
``benchmarks/bench_mm.py`` sends, taken from the benchmark classes themselves:

1. **Answer.** Preferably the answer SGLang actually generated in a
   full-context benchmark run (``--bench-generations``: the prefix of the
   ``<name>_<benchmark>_generations.jsonl`` files ``bench_mm.py
   --save-generations`` writes; scripts/analyze_draft_attention_video.sh makes
   them). Without it the HF target answers greedily -- slow (2-5 min a question
   with the torch fallback of the linear-attention layers) and not the
   benchmark's answer: HF and SGLang greedy decoding diverge at 42k tokens, and
   one LongVideoBench question looped to the token cap under HF but not SGLang.
2. **Re-encode.** (frames, prompt, answer) goes through ``encode_mm_record``,
   the training path, and the prompt part is checked token-for-token against the
   generation prompt, so the context analysed is the one the draft sees when
   serving.
3. **Probe.** One target forward gives the draft's five target layers. At
   anchors spread over the answer the draft runs one block exactly as at a
   decode step (context = every position before the anchor, plus the block),
   under eager attention so the weights exist (see
   ``analyze_dflash_attention.py`` for why this cannot go through SGLang).

Attention is the mean over the 15 predicting rows of the block (row 0, the
anchor slot, predicts nothing), renormalised over the CONTEXT columns: the
block attending to itself is reported separately as ``self_mass``.

Per question, ``<id>.npz`` holds, for every anchor x layer x head:
  seg_mass   mass on [sink, frame 1..F, other prompt text, answer so far]
  dist_mass  mass by distance anchor - p in log2 bins [1,2), [2,4), ...
  win_mass   mass on the last W positions, for each --windows W
  self_mass  share of the raw attention kept inside the block
plus the token counts of each segment/bin, and for the first ``--dump-full``
questions of each benchmark the full-resolution head-averaged map
(anchors x layers x positions) the heatmap is drawn from.

    python scripts/analyze_draft_attention_video.py            # 3 benches x 20
    python scripts/analyze_draft_attention_video.py --benchmarks longvideobench:4

Then ``python scripts/plot_draft_attention_video.py`` (CPU) draws the figure.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# scripts/ is sys.path[0] when this file is run
from analyze_dflash_accept import (  # noqa: E402
    _import_mm_benchmarks,
    _language_model,
    load_target,
    report_specforge_source,
    strip_thinking,
)
from analyze_dflash_attention import build_block_inputs, load_draft, record_attention  # noqa: E402

DEFAULT_DRAFT = (
    "/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/draft_models/"
    "qwen3.5-4b-mmflash-llava-ov15-1M-prompted-final"
)
DEFAULT_OUTPUT = "/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/attention_analysis/video-draft-attn"
DEFAULT_BENCHMARKS = ["longvideobench:20", "mvbench:20", "moviechat:20"]
NUM_DIST_BINS = 17  # distances 1 .. 2^17 > any prompt here


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--draft-model-path", default=DEFAULT_DRAFT)
    parser.add_argument("--target-model-path", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        help="name:count, as bench_mm's --benchmark-list (same questions for the same count)",
    )
    parser.add_argument("--num-frames", type=int, default=48, help="frames per video, as the benchmark runs")
    parser.add_argument(
        "--bench-generations",
        default=None,
        help="prefix of bench_mm's <name>_<benchmark>_generations.jsonl files (full-context run); "
        "without it the HF target generates the answers",
    )
    parser.add_argument("--keep-cut", action="store_true", help="also probe answers that hit the token cap")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="HF generation only")
    parser.add_argument("--max-length", type=int, default=65536)
    parser.add_argument("--max-anchors", type=int, default=48, help="anchors per answer, evenly spread")
    parser.add_argument("--dump-full", type=int, default=2, help="questions per benchmark that keep the full map")
    parser.add_argument("--windows", type=int, nargs="+", default=[512, 2048, 8192])
    parser.add_argument("--sink", type=int, default=4, help="leading positions reported as the sink")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    return parser.parse_args()


# --------------------------------------------------------------------------
# questions
# --------------------------------------------------------------------------


def load_questions(spec: str, num_frames: int) -> List[Dict[str, Any]]:
    """``name:count`` -> [{"id", "benchmark", "frames", "text"}] from the benchmark class."""
    name, _, count = spec.partition(":")
    registry = _import_mm_benchmarks()
    cls = registry.get(name)
    if cls is None:
        raise SystemExit(f"unknown benchmark {name!r}")
    kwargs: Dict[str, Any] = {"num_samples": int(count) if count else None}
    if "num_frames" in inspect.signature(cls.__init__).parameters:
        kwargs["num_frames"] = num_frames
    questions, _labels = cls(**kwargs).load_data()
    rows = []
    for index, question in enumerate(questions):
        if "parts" in question:
            frames = [value for kind, value in question["parts"] if kind == "image"]
            text = "".join(value for kind, value in question["parts"] if kind == "text")
        else:
            frames = [question["image_path"]] if question.get("image_path") else []
            text = question["question"]
        if frames:
            rows.append({"id": f"{name}-{index:04d}", "benchmark": name, "frames": frames, "text": text})
    print(f"[data] {spec}: {len(rows)} questions, {len(rows[0]['frames']) if rows else 0} frames each", flush=True)
    return rows


def user_turn(row: Dict[str, Any]) -> Dict[str, str]:
    """Frames first, then the benchmark text: the layout bench_mm sends and
    scripts/prepare_data_mm.py::build_video_record stores (verified to render
    identically to the server's prompt)."""
    from specforge.data.mm_preprocessing import IMAGE_PLACEHOLDER

    return {"role": "user", "content": IMAGE_PLACEHOLDER * len(row["frames"]) + "\n" + row["text"]}


def load_images(paths):
    from PIL import Image

    images = []
    for path in paths:
        with Image.open(path) as handle:
            images.append(handle.convert("RGB"))
    return images


# --------------------------------------------------------------------------
# stage 1: the target's answer
# --------------------------------------------------------------------------


def generate(args, processor, target, row) -> Dict[str, Any]:
    from specforge.data.mm_preprocessing import to_chat_messages
    from specforge.data.visual_score import fingerprint_input_ids

    prompt = processor.apply_chat_template(
        to_chat_messages([user_turn(row)]),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[prompt], images=load_images(row["frames"]), return_tensors="pt")
    inputs = {key: value.to(target.device) for key, value in inputs.items()}
    prompt_ids = inputs["input_ids"][0].tolist()
    started = time.time()
    with torch.no_grad():
        output = target.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    new_ids = output[0, len(prompt_ids):].tolist()
    eos = target.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    finish = "stop" if new_ids and new_ids[-1] in eos else "length"
    answer = strip_thinking(processor.tokenizer.decode(new_ids, skip_special_tokens=True))
    return {
        "id": row["id"],
        "benchmark": row["benchmark"],
        "frames": row["frames"],
        "text": row["text"],
        "answer": answer,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(new_ids),
        "finish_reason": finish,
        "generate_s": round(time.time() - started, 1),
        "prompt_fp": fingerprint_input_ids(prompt_ids),
    }


def bench_generations(prefix: str, benchmark: str) -> List[Dict[str, Any]]:
    path = f"{prefix}_{benchmark}_generations.jsonl"
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing; run bench_mm.py --save-generations first (see the .sh runner)")
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return sorted(records, key=lambda record: record["index"])


def from_bench(processor, row: Dict[str, Any], record: Dict[str, Any], source: str) -> Dict[str, Any]:
    """The answer SGLang generated for this question, checked to be about the
    same frames and text and to have been prompted with the same token count."""
    from specforge.data.mm_preprocessing import to_chat_messages
    from specforge.data.visual_score import fingerprint_input_ids

    parts = record.get("parts") or []
    frames = [value for kind, value in parts if kind == "image"]
    text = "".join(value for kind, value in parts if kind == "text")
    if frames != row["frames"] or text != row["text"]:
        raise SystemExit(
            f"{row['id']}: the bench generation at index {record.get('index')} is for a different question; "
            "regenerate it with the same benchmark list and counts"
        )
    prompt = processor.apply_chat_template(
        to_chat_messages([user_turn(row)]), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = processor(text=[prompt], images=load_images(row["frames"]), return_tensors="pt")["input_ids"][0].tolist()
    return {
        "id": row["id"],
        "benchmark": row["benchmark"],
        "frames": row["frames"],
        "text": row["text"],
        "answer": strip_thinking(record.get("generation") or ""),
        # SGLang's count; probe() rejects the question when ours differs
        "prompt_tokens": int(record.get("prompt_tokens") or -1),
        "local_prompt_tokens": len(prompt_ids),
        "completion_tokens": record.get("completion_tokens"),
        "finish_reason": record.get("finish_reason"),
        "accept_length": record.get("accept_length"),
        "prompt_fp": fingerprint_input_ids(prompt_ids),
        "source": source,
    }


# --------------------------------------------------------------------------
# stage 2-3: probe
# --------------------------------------------------------------------------


def segment_layout(input_ids: torch.Tensor, prompt_len: int, sink: int, visual_ids) -> Dict[str, Any]:
    """Segment id per position: 0 sink, 1..F frame k, F+1 other prompt text, F+2 answer."""
    ids = input_ids[0]
    seq_len = ids.shape[0]
    visual = torch.zeros_like(ids, dtype=torch.bool)
    for token_id in visual_ids:
        visual |= ids == token_id
    starts = (visual & ~torch.cat([visual.new_zeros(1), visual[:-1]])).nonzero().flatten().tolist()
    ends = (visual & ~torch.cat([visual[1:], visual.new_zeros(1)])).nonzero().flatten().tolist()
    spans = list(zip(starts, ends))  # inclusive, one per frame
    num_frames = len(spans)
    segment = torch.full((seq_len,), num_frames + 1, dtype=torch.long, device=ids.device)
    for index, (start, end) in enumerate(spans):
        segment[start : end + 1] = index + 1
    segment[prompt_len:] = num_frames + 2
    segment[: min(sink, prompt_len)] = 0
    return {"segment": segment, "spans": spans, "num_frames": num_frames, "visual": visual}


def choose_anchors(prompt_len: int, seq_len: int, block_size: int, count: int) -> List[int]:
    eligible = list(range(prompt_len, seq_len - block_size + 1))
    if len(eligible) <= count:
        return eligible
    step = (len(eligible) - 1) / max(count - 1, 1)
    return [eligible[round(i * step)] for i in range(count)]


def anchor_stats(weights, seq_len: int, anchor: int, segment, num_segments: int, windows) -> Dict[str, Any]:
    """Where one layer's attention went at one anchor.

    ``weights`` is that layer's (heads, block, S + block) attention for the
    block at ``anchor``. Rows 1.. are the predicting slots; their mean over the
    context columns, renormalised to sum to one, is split by segment, by log2
    distance to the anchor, and by window.
    """
    device = weights.device
    weights = weights[:, 1:, :].float()  # (heads, block-1, S + block)
    context = weights[..., :seq_len].mean(1)  # (heads, S)
    self_mass = weights[..., seq_len:].sum(-1).mean(1)
    positions = torch.arange(seq_len, device=device)
    visible = positions < anchor
    # zero past the anchor already (the mask); multiplying keeps the sums exact
    context = context * visible
    context = context / context.sum(-1, keepdim=True).clamp_min(1e-12)
    heads = context.shape[0]
    distance = (anchor - positions).clamp(min=1)
    dist_bin = torch.floor(torch.log2(distance.float())).long().clamp(max=NUM_DIST_BINS - 1)
    in_window = (distance.unsqueeze(0) <= windows.unsqueeze(1)) & visible.unsqueeze(0)  # (W, S)
    return {
        "context": context,
        "self": self_mass,
        "seg": torch.zeros(heads, num_segments, device=device).index_add_(1, segment, context),
        "dist": torch.zeros(heads, NUM_DIST_BINS, device=device).index_add_(1, dist_bin[visible], context[:, visible]),
        "win": context @ in_window.float().T,
        "counts": {
            "seg": torch.bincount(segment[visible], minlength=num_segments),
            "dist": torch.bincount(dist_bin[visible], minlength=NUM_DIST_BINS),
            "win": in_window.sum(-1),
        },
    }


def probe(args, processor, target, draft, embed_tokens, generation, header_ids, end_ids, dump_full: bool):
    from specforge.data.mm_preprocessing import encode_mm_record, to_chat_messages
    from specforge.data.visual_score import fingerprint_input_ids
    from specforge.modeling.draft.dflash import extract_context_feature

    record = {
        "id": generation["id"],
        "image": list(generation["frames"]),
        "conversations": [user_turn(generation), {"role": "assistant", "content": generation["answer"]}],
    }
    payload = encode_mm_record(
        record,
        processor,
        image_root="",
        max_length=args.max_length,
        train_only_last_turn=False,
        header_ids=header_ids,
        end_ids=end_ids,
    )
    if payload is None:
        return None, "encode_none"
    loss_mask = payload["loss_mask"]
    if 1 not in loss_mask:
        return None, "empty_answer"
    prompt_len = loss_mask.index(1)
    if prompt_len != generation["prompt_tokens"]:
        # e.g. SGLang counted a different prompt than this processor renders
        return None, "prompt_length_mismatch"
    # the context analysed must be the one the draft saw when serving
    if fingerprint_input_ids(list(payload["input_ids"])[:prompt_len]) != generation["prompt_fp"]:
        return None, "prompt_mismatch"

    text = processor.apply_chat_template(to_chat_messages(record["conversations"]), tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[text], images=load_images(record["image"]), return_tensors="pt")
    if inputs["input_ids"].shape[1] != len(payload["input_ids"]):
        return None, "length_mismatch"
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = target(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
    target_hidden = extract_context_feature(list(outputs.hidden_states), draft.target_layer_ids)
    del outputs

    visual_ids = [getattr(target.config, n) for n in ("image_token_id", "video_token_id") if isinstance(getattr(target.config, n, None), int)]
    layout = segment_layout(input_ids, prompt_len, args.sink, visual_ids)
    num_segments = layout["num_frames"] + 3
    anchors = choose_anchors(prompt_len, seq_len, draft.block_size, args.max_anchors)
    if not anchors:
        return None, "answer_too_short"

    windows = torch.tensor(args.windows, device=args.device)
    num_layers = len(draft.layers)
    seg_mass, dist_mass, win_mass, self_mass = [], [], [], []
    seg_count, dist_count, win_count, full = [], [], [], []
    for anchor in anchors:
        noise_embedding, position_ids, mask = build_block_inputs(draft, embed_tokens, input_ids, anchor)
        with torch.no_grad(), record_attention(draft) as captured:
            draft(position_ids=position_ids, noise_embedding=noise_embedding, target_hidden=target_hidden, attention_mask=mask)
        layers = [anchor_stats(captured[layer][0], seq_len, anchor, layout["segment"], num_segments, windows) for layer in range(num_layers)]
        del captured
        for key, sink_list in (("seg", seg_mass), ("dist", dist_mass), ("win", win_mass), ("self", self_mass)):
            sink_list.append(torch.stack([stats[key] for stats in layers]).cpu())
        counts = layers[0]["counts"]
        seg_count.append(counts["seg"].cpu())
        dist_count.append(counts["dist"].cpu())
        win_count.append(counts["win"].cpu())
        if dump_full:
            # float32: frame tokens sit far below the 1/S uniform level, where
            # float16 is subnormal
            full.append(torch.stack([stats["context"].mean(0) for stats in layers]).cpu())

    result = {
        "anchors": np.asarray(anchors, dtype=np.int64),
        "seq_len": seq_len,
        "prompt_len": prompt_len,
        "num_frames": layout["num_frames"],
        "frame_spans": np.asarray(layout["spans"], dtype=np.int64).reshape(-1, 2),
        "sink": args.sink,
        "windows": np.asarray(args.windows, dtype=np.int64),
        "seg_mass": torch.stack(seg_mass).numpy().astype(np.float32),     # (A, L, H, F+3)
        "dist_mass": torch.stack(dist_mass).numpy().astype(np.float32),   # (A, L, H, 17)
        "win_mass": torch.stack(win_mass).numpy().astype(np.float32),     # (A, L, H, W)
        "self_mass": torch.stack(self_mass).numpy().astype(np.float32),   # (A, L, H)
        "seg_count": torch.stack(seg_count).numpy().astype(np.int64),     # (A, F+3)
        "dist_count": torch.stack(dist_count).numpy().astype(np.int64),   # (A, 17)
        "win_count": torch.stack(win_count).numpy().astype(np.int64),     # (A, W)
    }
    if dump_full:
        result["full"] = torch.stack(full).numpy().astype(np.float32)  # (A, L, S)
    return result, "ok"


def describe(result: Dict[str, Any], windows: List[int]) -> str:
    """One line: where the context attention of this answer went (mean over anchors, layers, heads)."""
    seg = result["seg_mass"].mean(axis=(1, 2)).mean(0)
    frames = float(seg[1 : 1 + result["num_frames"]].sum())
    ctx = result["anchors"]  # positions 0..anchor-1 are the context at each anchor
    frame_tokens = result["seg_count"][:, 1 : 1 + result["num_frames"]].sum(1) / ctx
    parts = [f"frames {frames:.1%} of mass (tokens {frame_tokens.mean():.1%})"]
    win = result["win_mass"].mean(axis=(1, 2)).mean(0)
    share = (result["win_count"] / ctx[:, None]).mean(0)
    for index, w in enumerate(windows):
        parts.append(f"last {w}: {win[index]:.1%} (tokens {share[index]:.1%})")
    parts.append(f"answer-so-far {float(seg[-1]):.1%}, question {float(seg[-2]):.1%}, sink {float(seg[0]):.1%}")
    return "; ".join(parts)


def main() -> None:
    args = parse_args()
    report_specforge_source()
    os.makedirs(args.output_dir, exist_ok=True)

    from specforge.data.mm_preprocessing import _assistant_header_ids, _end_token_ids

    processor, target = load_target(args)
    draft = load_draft(args, args.device)
    print(
        f"[draft] {args.draft_model_path}: block_size={draft.block_size}, layers={len(draft.layers)}, "
        f"target_layer_ids={draft.target_layer_ids}, attn=eager",
        flush=True,
    )
    embed_tokens = _language_model(target).get_input_embeddings()
    header_ids, end_ids = _assistant_header_ids(processor), _end_token_ids(processor)

    # where the answers come from; a result is reused only for the same source
    source = f"sglang:{os.path.basename(args.bench_generations)}" if args.bench_generations else "hf"
    gen_path = os.path.join(args.output_dir, "generations.jsonl")
    cached: Dict[str, Dict[str, Any]] = {}
    if os.path.isfile(gen_path):
        with open(gen_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("source", "hf") == source:
                        cached[row["id"]] = row
    summary_path = os.path.join(args.output_dir, "summary.jsonl")
    done = set()
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("source", "hf") == source:
                        done.add(row["id"])
    print(f"[answers] {source}; {len(done)} questions already probed with these answers", flush=True)

    for spec in args.benchmarks:
        rows = load_questions(spec, args.num_frames)
        bench_records = bench_generations(args.bench_generations, rows[0]["benchmark"]) if args.bench_generations and rows else None
        if bench_records is not None and len(bench_records) != len(rows):
            raise SystemExit(f"{spec}: {len(bench_records)} bench generations for {len(rows)} questions")
        for position, row in enumerate(rows):
            npz_path = os.path.join(args.output_dir, f"{row['id']}.npz")
            if row["id"] in done and os.path.isfile(npz_path):
                continue
            generation = cached.get(row["id"])
            if generation is None or generation.get("frames") != row["frames"] or generation.get("text") != row["text"]:
                if bench_records is not None:
                    generation = from_bench(processor, row, bench_records[position], source)
                else:
                    generation = {**generate(args, processor, target, row), "source": source}
                with open(gen_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(generation, ensure_ascii=False) + "\n")
                cached[row["id"]] = generation
            started = time.time()
            if generation.get("finish_reason") == "length" and not args.keep_cut:
                # a looping answer cut at the cap would dominate the anchors with the loop
                result, status = None, "answer_cut_at_cap"
            else:
                result, status = None, None
            try:
                if status is None:
                    result, status = probe(
                        args, processor, target, draft, embed_tokens, generation, header_ids, end_ids,
                        dump_full=position < args.dump_full,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                result, status = None, "oom"
            line = {
                "id": row["id"],
                "benchmark": row["benchmark"],
                "source": source,
                "status": status,
                "prompt_tokens": generation["prompt_tokens"],
                "completion_tokens": generation["completion_tokens"],
                "finish_reason": generation["finish_reason"],
            }
            if result is not None:
                np.savez_compressed(npz_path, **result)
                line.update(
                    {
                        "anchors": len(result["anchors"]),
                        "num_frames": result["num_frames"],
                        "full_map": "full" in result,
                        "probe_s": round(time.time() - started, 1),
                        "frames_mass": float(result["seg_mass"][..., 1 : 1 + result["num_frames"]].sum(-1).mean()),
                        "win_mass": {str(w): float(result["win_mass"][..., i].mean()) for i, w in enumerate(args.windows)},
                        "win_tokens": {str(w): float((result["win_count"][:, i] / result["anchors"]).mean()) for i, w in enumerate(args.windows)},
                        "self_mass": float(result["self_mass"].mean()),
                    }
                )
                print(f"[probe] {row['id']} S={result['seq_len']} answer={generation['completion_tokens']} "
                      f"anchors={len(result['anchors'])}: {describe(result, args.windows)}", flush=True)
            else:
                if os.path.isfile(npz_path):
                    os.remove(npz_path)  # a map from earlier answers must not reach the plots
                print(f"[probe] {row['id']} skipped: {status}", flush=True)
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
            done.add(row["id"])
            torch.cuda.empty_cache()

    print(f"[done] outputs in {args.output_dir}; plot with scripts/plot_draft_attention_video.py --input-dir {args.output_dir}")


if __name__ == "__main__":
    sys.exit(main())
