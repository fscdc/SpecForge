"""Do the 16 draft queries of a DFlash block look at the same context?

If every position in a draft block attends to roughly the same context tokens,
then one shared summary of that context would serve all of them -- which is the
premise the whole "compress the draft's visual context" idea rests on. This
script measures that directly.

**Why not through the server.** SGLang cannot return attention weights: the LM
path has no field for them, and FA3/FlashInfer never materialize the q x kv
matrix in the first place -- not writing it back is the point of those kernels.
The repository's SGLang patches hook tensors that already exist (hidden states,
verify entropy); attention weights are not among them. So this runs the draft
offline in PyTorch instead.

**What is reproduced.** One draft block at a chosen anchor, built exactly as
`OnlineDFlashModel._forward_draft_blocks` builds it with a single anchor:

    Q_LEN  = block_size                      (16 mask/anchor slots)
    KV_LEN = S + block_size                  (context, then the block itself)
    mask   = (kv < anchor) | (same block)

which is the same shape and the same visibility rule a decode step has, since
the draft's KV cache at that moment holds exactly the positions before the
anchor. Running the training-side path rather than `spec_generate` keeps the
anchor under our control and needs no generation loop.

**Three traps this handles.**

1. `create_dflash_sdpa_mask` returns a BOOLEAN mask. SDPA accepts that; eager
   attention does `attn_weights + attention_mask`, so a bool mask would add 1.0
   to the visible entries and 0.0 to the hidden ones -- no error, pure garbage.
   It is converted to an additive -inf mask here.
2. `eager_attention_forward` calls `repeat_kv` first, so the weights come back
   with 32 query heads, not the 8 KV heads the projections produce.
3. The last `block_size` columns are the block attending to itself, not to the
   context. They are reported separately, and the context columns are
   renormalized before any similarity is computed -- otherwise "how much mass
   stayed inside the block" would leak into every similarity number.

Row 0 of the block is the anchor slot, whose output is never used for a
prediction (`_sample_draft_tokens` reads the last `block_size - 1` positions),
so the metrics run over rows 1..15.

**No target-model control.** Comparing against the target's own attention at the
same positions would need `output_attentions=True` on a 32-layer model, i.e. a
(16, S, S) matrix per layer -- terabytes at S ~ 9k. It needs a different
instrument than this one.

Usage -- the generations file is the one `analyze_dflash_accept.py` writes:

    python scripts/analyze_dflash_attention.py \\
        --draft-model-path /path/to/qwen3.5-4b-mmflash-sharegpt4v-pt-120000 \\
        --generations .../accept_analysis/chartqa-120000/generations.jsonl \\
        --output-dir .../attention_analysis/chartqa-120000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

# both scripts live in scripts/, which is sys.path[0] when either is run
from analyze_dflash_accept import (
    _language_model,
    load_target,
    read_jsonl,
    report_specforge_source,
    target_features,
    write_jsonl,
)

DEFAULT_TOPK = (32, 128, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    model = parser.add_argument_group("models")
    model.add_argument("--draft-model-path", required=True)
    model.add_argument("--target-model-path", default="Qwen/Qwen3.5-4B")
    model.add_argument("--device", default="cuda")
    model.add_argument("--dtype", default="bfloat16")
    model.add_argument("--trust-remote-code", action="store_true", default=True)

    data = parser.add_argument_group("data")
    data.add_argument(
        "--generations",
        required=True,
        help="generations.jsonl from analyze_dflash_accept.py",
    )
    data.add_argument("--num-samples", type=int, default=50)
    data.add_argument("--max-length", type=int, default=4096)

    probe = parser.add_argument_group("probe")
    probe.add_argument(
        "--anchors-per-sample",
        type=int,
        default=4,
        help="Anchors per generation, spread evenly over the response",
    )
    probe.add_argument(
        "--topk",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOPK),
        help="Context sizes at which overlap and union are measured",
    )
    probe.add_argument(
        "--mass-threshold",
        type=float,
        default=0.9,
        help="Fraction of attention mass the effective support must cover",
    )

    parser.add_argument("--output-dir", default="./cache/attention_analysis")
    return parser.parse_args()


# --------------------------------------------------------------------------
# capturing the weights
# --------------------------------------------------------------------------


@contextmanager
def record_attention(draft) -> Iterator[Dict[int, torch.Tensor]]:
    """Capture every draft layer's attention weights for one forward pass.

    `Qwen3DFlashAttention.forward` already returns `(attn_output, attn_weights)`;
    the decoder layer drops the second element. A forward hook sees the tuple
    before that happens.
    """
    if draft.config._attn_implementation != "eager":
        raise RuntimeError(
            "attention weights only exist under the eager implementation; set "
            "draft.config._attn_implementation = 'eager' before recording"
        )

    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            weights = output[1] if isinstance(output, tuple) else None
            if weights is not None:
                captured[layer_index] = weights.detach().float()

        return hook

    for index, layer in enumerate(draft.layers):
        handles.append(layer.self_attn.register_forward_hook(make_hook(index)))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def build_block_inputs(
    draft,
    embed_tokens,
    input_ids: torch.Tensor,
    anchor: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One draft block at `anchor`, exactly as the training path builds it.

    Returns (noise_embedding, position_ids, additive_mask).
    """
    from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask

    device = input_ids.device
    seq_len = input_ids.shape[1]
    block_size = draft.block_size

    # [anchor token, <mask> x (block_size - 1)] -- _create_noise_embed's layout
    noise_ids = torch.full(
        (1, block_size), draft.mask_token_id, dtype=torch.long, device=device
    )
    noise_ids[0, 0] = input_ids[0, anchor]
    noise_embedding = embed_tokens(noise_ids)

    position_ids = torch.cat(
        [
            torch.arange(seq_len, device=device),
            anchor + torch.arange(block_size, device=device),
        ]
    ).unsqueeze(0)

    bool_mask = create_dflash_sdpa_mask(
        anchor_positions=torch.tensor([[anchor]], device=device),
        block_keep_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
        S=seq_len,
        block_size=block_size,
        device=device,
    )
    # eager attention ADDS the mask, so a bool one would be silently wrong
    dtype = noise_embedding.dtype
    additive = torch.zeros(bool_mask.shape, dtype=dtype, device=device)
    additive.masked_fill_(~bool_mask, torch.finfo(dtype).min)

    return noise_embedding, position_ids, additive


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

EPS = 1e-12


def _upper_pairs(size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.triu_indices(size, size, offset=1, device=device).unbind(0)


def pairwise_cosine(rows: torch.Tensor) -> float:
    """Mean cosine similarity between every pair of query distributions."""
    unit = rows / rows.norm(dim=-1, keepdim=True).clamp_min(EPS)
    similarity = unit @ unit.T
    i, j = _upper_pairs(rows.shape[0], rows.device)
    return float(similarity[i, j].mean())


def pairwise_js(rows: torch.Tensor) -> Tuple[float, float]:
    """Mean and first-vs-last Jensen-Shannon divergence, in nats.

    0 means the two queries attend identically; ln(2) ~ 0.693 means disjoint.
    """
    p = rows.unsqueeze(1)
    q = rows.unsqueeze(0)
    mixture = (0.5 * (p + q)).clamp_min(EPS).log()
    kl_p = (p * (p.clamp_min(EPS).log() - mixture)).sum(-1)
    kl_q = (q * (q.clamp_min(EPS).log() - mixture)).sum(-1)
    divergence = 0.5 * kl_p + 0.5 * kl_q
    i, j = _upper_pairs(rows.shape[0], rows.device)
    return float(divergence[i, j].mean()), float(divergence[0, -1])


def topk_stats(rows: torch.Tensor, k: int) -> Tuple[float, float]:
    """Mean pairwise top-k overlap, and |union of top-k| / k.

    The union ratio is the decisive one: 1.0 means all queries rank the same k
    context positions highest, so a single k-sized summary would serve the whole
    block; the row count (15) means they look at disjoint places.
    """
    k_eff = min(k, rows.shape[-1])
    indices = rows.topk(k_eff, dim=-1).indices
    membership = torch.zeros(
        rows.shape[0], rows.shape[-1], dtype=torch.float32, device=rows.device
    )
    membership.scatter_(1, indices, 1.0)
    overlap = (membership @ membership.T) / k_eff
    i, j = _upper_pairs(rows.shape[0], rows.device)
    union = int(torch.unique(indices).numel())
    return float(overlap[i, j].mean()), union / k_eff


def effective_support(row: torch.Tensor, threshold: float) -> int:
    """How many context positions carry `threshold` of one query's mass."""
    ordered, _ = row.sort(descending=True)
    return int((ordered.cumsum(-1) < threshold).sum()) + 1


def analyze_head(
    weights: torch.Tensor,
    seq_len: int,
    visual_columns: Optional[torch.Tensor],
    topks: Sequence[int],
    threshold: float,
) -> Dict[str, Any]:
    """Every metric for one (layer, head), from its (block_size, KV_LEN) slice."""
    rows = weights[1:, :]  # the anchor slot is never used for a prediction
    context = rows[:, :seq_len]
    block_self = float(rows[:, seq_len:].sum(-1).mean())

    # renormalize over the context, so "mass kept inside the block" does not
    # leak into the similarity numbers
    normalized = context / context.sum(-1, keepdim=True).clamp_min(EPS)

    mean_js, first_last_js = pairwise_js(normalized)
    record: Dict[str, Any] = {
        "block_self_mass": block_self,
        "cosine_mean": pairwise_cosine(normalized),
        "js_mean": mean_js,
        "js_first_vs_last": first_last_js,
        "support_per_query": float(
            statistics.fmean(
                effective_support(row, threshold) for row in normalized
            )
        ),
        "support_of_mean": effective_support(normalized.mean(0), threshold),
    }
    for k in topks:
        overlap, union_ratio = topk_stats(normalized, k)
        record[f"overlap@{k}"] = overlap
        record[f"union_ratio@{k}"] = union_ratio
    if visual_columns is not None and int(visual_columns.sum()) > 0:
        visual_mass = normalized[:, visual_columns].sum(-1)
        record["visual_mass_mean"] = float(visual_mass.mean())
        record["visual_mass_std"] = float(visual_mass.std(unbiased=False))
    return record


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def load_draft(args, device):
    """The draft checkpoint, forced onto the eager attention path."""
    from specforge.modeling.draft.dflash import DFlashDraftModel

    draft = DFlashDraftModel.from_pretrained(
        args.draft_model_path,
        torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    # Qwen3DFlashAttention reads self.config, which is the same object
    draft.config._attn_implementation = "eager"
    return draft.to(device).eval()


def visual_column_mask(
    input_ids: torch.Tensor, target_config
) -> Optional[torch.Tensor]:
    """Which context columns are image or video placeholder tokens."""
    ids = [
        getattr(target_config, name, None)
        for name in ("image_token_id", "video_token_id")
    ]
    ids = [value for value in ids if isinstance(value, int)]
    if not ids:
        return None
    mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    for token_id in ids:
        mask |= input_ids[0] == token_id
    return mask


def choose_anchors(loss_mask: Sequence[int], seq_len: int, block_size: int, count: int):
    """Evenly spread anchors over the response positions with a full block ahead."""
    eligible = [
        index
        for index, flag in enumerate(loss_mask)
        if flag and index + block_size <= seq_len
    ]
    if not eligible:
        return []
    if count >= len(eligible):
        return eligible
    step = (len(eligible) - 1) / max(count - 1, 1)
    return [eligible[round(position * step)] for position in range(count)]


def summarize(rows: List[Dict[str, Any]], topks: Sequence[int]) -> None:
    if not rows:
        print("No attention was recorded.")
        return

    pairs = len({(row["id"], row["anchor"]) for row in rows})
    print(f"\nHeads measured: {len(rows)} over {pairs} (sample, anchor) pairs")

    by_layer: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)

    primary = topks[0]
    header = (
        f"{'layer':>5}  {'cos':>6}  {'JS':>6}  {'JS 1v15':>8}  "
        f"{'ovl@' + str(primary):>9}  {'union@' + str(primary):>11}  "
        f"{'supp/q':>8}  {'supp(avg)':>10}  {'blk-self':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for layer in sorted(by_layer):
        group = by_layer[layer]

        def mean(key: str) -> float:
            return statistics.fmean(row[key] for row in group)

        print(
            f"{layer:>5}  {mean('cosine_mean'):>6.3f}  {mean('js_mean'):>6.3f}  "
            f"{mean('js_first_vs_last'):>8.3f}  "
            f"{mean(f'overlap@{primary}'):>9.3f}  "
            f"{mean(f'union_ratio@{primary}'):>11.2f}  "
            f"{mean('support_per_query'):>8.1f}  {mean('support_of_mean'):>10.1f}  "
            f"{mean('block_self_mass'):>9.3f}"
        )

    print("\nUnion ratio by k (1.0 = one shared summary serves the whole block, "
          "15.0 = every query looks elsewhere):")
    for k in topks:
        values = [row[f"union_ratio@{k}"] for row in rows]
        print(
            f"  k={k:<5} mean {statistics.fmean(values):.2f}   "
            f"median {statistics.median(values):.2f}   max {max(values):.2f}"
        )

    if any("visual_mass_mean" in row for row in rows):
        visual = [row["visual_mass_mean"] for row in rows if "visual_mass_mean" in row]
        spread = [row["visual_mass_std"] for row in rows if "visual_mass_std" in row]
        share = statistics.fmean(row["visual_share"] for row in rows)
        print(
            f"\nVisual tokens are {share:.1%} of the context and take "
            f"{statistics.fmean(visual):.1%} of the attention mass "
            f"(std across the 15 queries: {statistics.fmean(spread):.4f})"
        )


def main() -> None:
    args = parse_args()
    report_specforge_source()
    os.makedirs(args.output_dir, exist_ok=True)

    processor, target = load_target(args)
    draft = load_draft(args, args.device)
    print(
        f"Draft: block_size={draft.block_size}, layers={len(draft.layers)}, "
        f"target_layer_ids={draft.target_layer_ids}, attn=eager"
    )

    import types

    from PIL import Image

    from specforge.data.mm_preprocessing import (
        _assistant_header_ids,
        _end_token_ids,
        encode_mm_record,
        to_chat_messages,
    )

    # target_features() reads probe.draft_model; this probe has no wrapper
    shim = types.SimpleNamespace(draft_model=draft)
    header_ids = _assistant_header_ids(processor)
    end_ids = _end_token_ids(processor)
    embed_tokens = _language_model(target).get_input_embeddings()

    records = read_jsonl(args.generations)[: args.num_samples]
    print(f"Read {len(records)} generations from {args.generations}")

    rows: List[Dict[str, Any]] = []
    skipped = 0
    for position, record in enumerate(records):
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
            skipped += 1
            continue

        input_ids = torch.tensor([payload["input_ids"]], device=args.device)
        seq_len = input_ids.shape[1]
        anchors = choose_anchors(
            payload["loss_mask"], seq_len, draft.block_size, args.anchors_per_sample
        )
        if not anchors:
            skipped += 1
            continue

        with Image.open(payload["image"]) as handle:
            image = handle.convert("RGB")
            text = processor.apply_chat_template(
                to_chat_messages(record["conversations"]),
                tokenize=False,
                add_generation_prompt=False,
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(args.device) for key, value in inputs.items()}
        if inputs["input_ids"].shape[1] != seq_len:
            skipped += 1
            continue

        target_hidden = target_features(target, shim, inputs)
        visual = visual_column_mask(input_ids, target.config)
        visual_share = float(visual.float().mean()) if visual is not None else 0.0

        for anchor in anchors:
            noise_embedding, position_ids, mask = build_block_inputs(
                draft, embed_tokens, input_ids, anchor
            )
            with torch.no_grad(), record_attention(draft) as captured:
                draft(
                    position_ids=position_ids,
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    attention_mask=mask,
                )
            for layer, weights in sorted(captured.items()):
                # (1, n_heads, block_size, seq_len + block_size)
                for head in range(weights.shape[1]):
                    row = analyze_head(
                        weights[0, head],
                        seq_len,
                        visual,
                        args.topk,
                        args.mass_threshold,
                    )
                    row.update(
                        {
                            "id": record["id"],
                            "anchor": anchor,
                            "seq_len": seq_len,
                            "layer": layer,
                            "head": head,
                            "visual_share": visual_share,
                        }
                    )
                    rows.append(row)
            del captured

        if (position + 1) % 10 == 0:
            print(f"  probed {position + 1}/{len(records)} generations", flush=True)

    out_path = os.path.join(args.output_dir, "attention_similarity.jsonl")
    written = write_jsonl(out_path, iter(rows))
    print(f"\nWrote {written} per-head records to {out_path} ({skipped} skipped)")
    summarize(rows, args.topk)


if __name__ == "__main__":
    main()
