"""Per-anchor accept length of a DFlash draft, resolved by token.

The serving-side benchmark reports one accept length per request
(``completion_tokens / spec_verify_ct``), which is too coarse to ask whether a
draft stumbles on *visually grounded* tokens specifically. This script measures
the same quantity at the resolution the training objective already works at: one
accept length per anchor position, aligned to an absolute token index.

Pipeline, for a multimodal target and a DFlash-family draft checkpoint:

1. **Generate.** The target answers ``--num-samples`` benchmark questions under
   the benchmark's own instruction. The measurement has to run over what the
   target actually emits, not over the dataset's reference answer: ChartQA's
   references are a phrase, far short of the ``2 * block_size`` trainable tokens
   a DFlash block needs, while its instruction ("describe the image in detail,
   then present your reasoning") yields a few hundred tokens that split into a
   visually grounded half and a text-driven one.
2. **Re-encode.** Each (image, prompt, generation) triple goes back through
   ``encode_mm_record``, i.e. the exact path online training uses, so the
   ``input_ids`` carry the processor's image-token expansion and the
   ``loss_mask`` covers the assistant span alone.
3. **Score.** The target runs once with ``output_hidden_states=True``; the five
   layers named by the draft's ``target_layer_ids`` are concatenated by
   ``extract_context_feature`` into what the draft was trained on. The draft's
   block forward then runs, and ``compute_accept_len`` turns its argmax into the
   accepted prefix of every block.
4. **Attribute.** Optionally the target runs a second time with the image
   removed. The per-token KL between the two next-token distributions is how
   much the image changed the target's mind at that token, which is the
   modality-specific grounding signal a positional heuristic cannot provide.

Nothing in the training path is modified: the draft is loaded as-is and only
read from.

The per-anchor JSONL is the artifact; the printed summary is a first look at it.

Usage (one GPU is enough for a 4B target):

    python scripts/analyze_dflash_accept.py \\
        --draft-model-path /path/to/qwen3.5-4b-mmflash-sharegpt4v-pt-120000 \\
        --target-model-path Qwen/Qwen3.5-4B \\
        --num-samples 200 \\
        --output-dir ./cache/accept_analysis

Re-run the scoring without regenerating by passing ``--generations`` the
``generations.jsonl`` of an earlier run.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

BENCHMARK_INSTRUCTIONS = {
    # benchmarks/mm_benchmarker/chartqa.py::CHARTQA_INSTRUCTION, whose
    # "describe the image, then reason" shape is what makes one generation
    # contain both a grounded and an ungrounded stretch.
    "chartqa": (
        "Analyze the image and question carefully, using step-by-step reasoning.\n"
        "First, describe any image provided in detail. Then, present your "
        "reasoning. And finally your final answer in this format:\n"
        "Final Answer: <answer>\n"
        "where <answer> follows the following instructions:\n"
        "- <answer> should should be a single phrase or number.\n"
        "- <answer> should not paraphrase or reformat the text in the image.\n"
        "- If <answer> is a ratio, it should be a decimal value like 0.25 instead "
        "of 1:4.\n"
        "If the question is a Yes/No question, <answer> should be Yes or No.\n"
        "- If <answer> is a number, it should not contain any units.\n"
        "- If <answer> is a percentage, it should include a % sign.\n"
        "- If <answer> is an entity, it should include the full label from the "
        "graph.\n"
        "IMPORTANT: Remember, to end your answer, start your final answer with "
        '"Final Answer:".'
    ),
}


def report_specforge_source() -> None:
    """Say which ``specforge`` this run imported, and warn on a mismatch.

    `specforge train` runs from the INSTALLED package, so an analysis that
    quietly imported a different copy would be explaining a model with code
    that never trained it. Reporting is deliberate rather than forcing this
    repo's copy onto sys.path: forcing would hide the same mismatch in the
    other, more harmful direction.

    The two agree only when the package was installed with `pip install -e .`.
    """
    import specforge

    imported = Path(specforge.__file__).resolve().parent
    repo = Path(__file__).resolve().parent.parent / "specforge"
    print(f"specforge imported from: {imported}")
    if imported == repo.resolve():
        return
    print(
        f"  WARNING: this is NOT the repo's copy at {repo}\n"
        "  Local edits under specforge/ are not in effect, and this analysis "
        "may not match the code that trained the checkpoint.\n"
        "  Fix with: pip install -e . --no-deps --no-build-isolation"
    )


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
        "--benchmark", default="chartqa", choices=sorted(BENCHMARK_INSTRUCTIONS)
    )
    data.add_argument("--split", default="val")
    data.add_argument("--num-samples", type=int, default=200)
    data.add_argument("--max-new-tokens", type=int, default=512)
    data.add_argument("--max-length", type=int, default=4096)
    data.add_argument(
        "--generations",
        default=None,
        help="Reuse this generations.jsonl instead of running the target again",
    )

    probe = parser.add_argument_group("probe")
    probe.add_argument(
        "--num-anchors",
        type=int,
        default=512,
        help=(
            "Anchors sampled per generation. The default exceeds a 512-token "
            "response, so effectively every response position is measured"
        ),
    )
    probe.add_argument(
        "--attention-backend",
        default="sdpa",
        choices=("sdpa", "eager", "flex_attention"),
    )
    probe.add_argument(
        "--logit-chunk-blocks",
        type=int,
        default=16,
        help="Blocks per lm_head chunk; the vocabulary is what makes this matter",
    )
    probe.add_argument(
        "--no-visual-kl",
        action="store_true",
        help="Skip the image-ablated second target pass",
    )
    probe.add_argument(
        "--visual-top-n",
        type=int,
        default=10,
        help=(
            "LVSpec's N: how many of the most similar visual tokens the "
            "relevance score averages over (paper default 10; 1 is noisy and "
            "100 dilutes the salient cues with background)"
        ),
    )
    probe.add_argument(
        "--no-visual-relevance",
        action="store_true",
        help="Skip the LVSpec cosine-similarity relevance score",
    )
    probe.add_argument("--seed", type=int, default=42)
    probe.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Let the target reason before answering. Off by default: the "
            "mmflash draft is trained on regenerations captured with "
            "--reasoning disable, so a thinking response is off-distribution, "
            "and the template renders it inside <think>, which no assistant "
            "header can then match"
        ),
    )

    parser.add_argument("--output-dir", default="./cache/accept_analysis")
    return parser.parse_args()


# --------------------------------------------------------------------------
# stage 1: what the target actually generates
# --------------------------------------------------------------------------


def load_benchmark_rows(args) -> List[Dict[str, Any]]:
    """Materialize the images and pair each with the benchmark's instruction."""
    from datasets import load_dataset

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    dataset = load_dataset("HuggingFaceM4/ChartQA")[args.split]
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    instruction = BENCHMARK_INSTRUCTIONS[args.benchmark]
    rows = []
    for index, row in enumerate(dataset):
        image_path = os.path.join(image_dir, f"{index:06d}.png")
        if not os.path.exists(image_path):
            row["image"].convert("RGB").save(image_path, "PNG")
        label = row.get("label")
        rows.append(
            {
                "id": f"{args.benchmark}-{index:06d}",
                "image": image_path,
                "question": str(row["query"]),
                "prompt": f"{row['query']}\n{instruction}",
                "reference": label[0] if isinstance(label, list) and label else label,
            }
        )
    return rows


def strip_thinking(text: str) -> str:
    """Drop a reasoning block, marker or not, from a decoded generation.

    ``skip_special_tokens=True`` removes the ``</think>`` token itself, so a
    thinking response comes back as reasoning text welded to the answer. Storing
    that would make the chat template re-render the whole thing inside
    ``<think>``, and the assistant header the loss mask looks for would never
    appear.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return text.replace("<think>", "").strip()


def generate_responses(args, rows, processor, target) -> List[Dict[str, Any]]:
    """Answer every row with the target, as ShareGPT records for stage 2."""
    from PIL import Image

    from specforge.data.mm_preprocessing import IMAGE_PLACEHOLDER

    records = []
    for position, row in enumerate(rows):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": row["prompt"]},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        with Image.open(row["image"]) as handle:
            image = handle.convert("RGB")
            inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = {key: value.to(target.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated = target.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        prompt_len = inputs["input_ids"].shape[1]
        answer = strip_thinking(
            processor.tokenizer.decode(
                generated[0][prompt_len:], skip_special_tokens=True
            )
        )

        records.append(
            {
                "id": row["id"],
                "image": row["image"],
                "reference": row["reference"],
                "conversations": [
                    {
                        "role": "user",
                        "content": f"{IMAGE_PLACEHOLDER}\n{row['prompt']}",
                    },
                    {"role": "assistant", "content": answer},
                ],
            }
        )
        if (position + 1) % 20 == 0:
            print(f"  generated {position + 1}/{len(rows)}", flush=True)
    return records


# --------------------------------------------------------------------------
# stage 2: what the draft would have got accepted
# --------------------------------------------------------------------------


def build_probe(args, target):
    """Load the draft checkpoint into the training wrapper, read-only."""
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    from specforge.modeling.draft.dflash import DFlashDraftModel

    draft = DFlashDraftModel.from_pretrained(
        args.draft_model_path,
        torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    # The mask kind OnlineDFlashModel builds has to match the attention the
    # draft actually runs. create_dflash_sdpa_mask returns a BOOLEAN mask: SDPA
    # accepts one, but eager does `attn_weights + attention_mask`, so a bool
    # would be ADDED as 1.0/0.0 -- no error, silently wrong numbers -- and
    # flex_attention needs a BlockMask instead. Pin the draft to the same
    # choice rather than trusting whatever the checkpoint's config says.
    if args.attention_backend == "eager":
        raise ValueError(
            "eager attention adds its mask, so the boolean mask this path "
            "builds would be added instead of masking; use sdpa or "
            "flex_attention"
        )
    draft.config._attn_implementation = args.attention_backend
    draft = draft.to(target.device).eval()

    language_model = _language_model(target)
    probe = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=target.get_output_embeddings(),
        target_embed_tokens=language_model.get_input_embeddings(),
        mask_token_id=draft.mask_token_id,
        block_size=draft.block_size,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        # the loss is never evaluated here, only the block forward and the argmax
        loss_type="dflash",
    )
    return probe.to(target.device).eval()


def _language_model(target):
    """The text stack of a VL checkpoint, or the model itself for a text one."""
    for attribute in ("language_model", "model"):
        candidate = getattr(target, attribute, None)
        if candidate is not None and hasattr(candidate, "get_input_embeddings"):
            nested = getattr(candidate, "language_model", None)
            return nested if nested is not None else candidate
    return target


def target_features(
    target,
    probe,
    inputs: Dict[str, torch.Tensor],
    return_last_hidden: bool = False,
):
    """The concatenated target layers the draft was trained to read.

    With ``return_last_hidden`` the final layer's states come back too, which is
    what LVSpec's relevance score is computed from. The default keeps the
    single-tensor signature the attention probe imports.
    """
    from specforge.modeling.draft.dflash import extract_context_feature

    with torch.no_grad():
        outputs = target(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states
    target_hidden = extract_context_feature(
        list(hidden_states), probe.draft_model.target_layer_ids
    )
    expected = (
        len(probe.draft_model.target_layer_ids)
        * probe.draft_model.config.hidden_size
    )
    if target_hidden.shape[-1] != expected:
        raise RuntimeError(
            f"target features are {target_hidden.shape[-1]}-wide, the draft's fc "
            f"expects {expected}; check target_layer_ids against the target"
        )
    if return_last_hidden:
        return target_hidden, hidden_states[-1]
    return target_hidden


def per_anchor_accept(
    probe,
    input_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    loss_mask: torch.Tensor,
    logit_chunk_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accepted prefix length of every sampled block.

    Rebuilds the labels and the weight mask exactly as
    ``OnlineDFlashModel.forward`` does -- same offsets, same exclusion of the
    anchor slot, same gather of the original loss mask -- and replaces the loss
    with ``compute_accept_len``. The logits are chunked over blocks because a
    250k vocabulary times 512 blocks does not fit anywhere.

    Returns (anchor_positions, accepted, correct_flags, weight_mask), all on the
    probe's device and squeezed to the single sequence in the batch.
    """
    from specforge.algorithms.common.dflash_family_model import compute_accept_len

    device = input_ids.device
    seq_len = input_ids.shape[1]
    block_size = probe.block_size

    with torch.no_grad():
        anchor_positions, block_keep_mask, output_hidden = probe._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=target_hidden,
            loss_mask=loss_mask,
        )

        label_offsets = torch.arange(0, block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        weight_mask = weight_mask * torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        hidden_4d = output_hidden.reshape(
            input_ids.shape[0], anchor_positions.shape[1], block_size, -1
        )
        predicted = torch.empty_like(target_ids)
        for start in range(0, hidden_4d.shape[1], logit_chunk_blocks):
            stop = start + logit_chunk_blocks
            chunk = hidden_4d[:, start:stop]
            logits = probe.lm_head(
                chunk.reshape(chunk.shape[0], -1, chunk.shape[-1])
            ).reshape(chunk.shape[0], chunk.shape[1], block_size, -1)
            predicted[:, start:stop] = logits.argmax(dim=-1)

        valid = weight_mask > 0
        accepted = compute_accept_len(predicted, target_ids, valid)
        correct = (predicted == target_ids) & valid

    return anchor_positions[0], accepted[0], correct[0], valid[0]


def visual_relevance_per_token(
    last_hidden: torch.Tensor,
    visual_mask: torch.Tensor,
    positions: Sequence[int],
    top_n: int,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """LVSpec's visual relevance, per generated token.

    From "See the Forest for the Trees" (arXiv 2604.05650), Eq. 8: cosine
    similarity between a text token's last-layer hidden state and every visual
    token's, averaged over the Top-N largest. The Top-N is the point of the
    metric -- averaging over all visual tokens lets abundant background cues
    dilute the sharp similarity a grounded token has with the salient ones, so
    grounded and ungrounded tokens stop being separable.

    Unlike the paper this runs over the whole response from one teacher-forced
    pass rather than per decoding step. The paper computes it step-locally
    because it ranks scores *within* a draft block, where absolute drift over a
    growing context would not matter; here the raw score is wanted, so the two
    differ by exactly that drift.

    Returns (top-N score, all-visual mean) per position, the second only so the
    dilution the paper reports can be checked on this data.
    """
    if int(visual_mask.sum()) == 0 or not positions:
        return {}, {}

    visual = last_hidden[0][visual_mask].float()
    visual = visual / visual.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    index = torch.tensor(positions, device=last_hidden.device)
    text = last_hidden[0][index].float()
    text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    top_scores: Dict[int, float] = {}
    all_scores: Dict[int, float] = {}
    n_eff = min(top_n, visual.shape[0])
    # chunked so a video prompt's (response x 40k visual) matrix stays bounded
    for start in range(0, len(positions), 256):
        rows = text[start : start + 256]
        similarity = rows @ visual.T
        top = similarity.topk(n_eff, dim=-1).values.mean(dim=-1)
        mean = similarity.mean(dim=-1)
        for offset, position in enumerate(positions[start : start + 256]):
            top_scores[position] = float(top[offset])
            all_scores[position] = float(mean[offset])
    return top_scores, all_scores


def visual_token_mask(input_ids: torch.Tensor, processor) -> torch.Tensor:
    """Which positions hold an image or video placeholder token."""
    tokenizer = processor.tokenizer
    mask = torch.zeros_like(input_ids[0], dtype=torch.bool)
    for token in ("<|image_pad|>", "<|video_pad|>"):
        try:
            value = tokenizer.convert_tokens_to_ids(token)
        except Exception:  # pragma: no cover - tokenizer without the token
            continue
        if isinstance(value, int) and value >= 0:
            mask |= input_ids[0] == value
    return mask


def visual_kl_per_token(
    target,
    processor,
    record: Dict[str, Any],
    input_ids: torch.Tensor,
    loss_mask_list: Sequence[int],
    max_length: int,
) -> Optional[Tuple[Dict[int, float], Dict[int, float]]]:
    """Per token: how much the image moves the target, and how unsure it is.

    Both come from the same pair of forward passes, and the second is what
    separates a real finding from a rediscovery of "uncertain tokens are hard".
    A position where the target has many valid continuations has a distribution
    that moves a lot when the image is dropped AND is hard for any draft, so a
    raw KL-versus-acceptance correlation cannot tell the two apart. The entropy
    of the WITH-image distribution -- the one generation actually samples from
    -- is the control.

    The same conversation is encoded a second time with the image dropped. The
    assistant span is identical text in both, and sits at the end, so the two
    sequences align by a constant offset -- which is asserted token by token
    rather than assumed. ``None`` when they do not align, e.g. because the
    template rendered the image-free turn differently.
    """
    from specforge.data.mm_preprocessing import IMAGE_PLACEHOLDER, to_chat_messages

    stripped = [
        {**turn, "content": turn["content"].replace(IMAGE_PLACEHOLDER, "").lstrip()}
        for turn in record["conversations"]
    ]
    text = processor.apply_chat_template(
        to_chat_messages(stripped), tokenize=False, add_generation_prompt=False
    )
    blind = processor(text=[text], return_tensors="pt")
    blind_ids = blind["input_ids"].to(input_ids.device)
    if blind_ids.shape[1] > max_length or blind_ids.shape[1] >= input_ids.shape[1]:
        return None

    offset = input_ids.shape[1] - blind_ids.shape[1]
    positions = [index for index, flag in enumerate(loss_mask_list) if flag]
    if not positions:
        return None
    aligned = [index for index in positions if index - offset >= 1]
    if not aligned:
        return None
    if not torch.equal(
        input_ids[0, aligned],
        blind_ids[0, [index - offset for index in aligned]],
    ):
        return None

    # Only the assistant span is scored, and it is contiguous at the end of both
    # sequences, so the trailing window that covers it is all the logits needed.
    # Asking for the whole sequence would materialize (S, 248320) twice -- 700 MB
    # per pass at chartqa's length, and tens of GB on a video prompt.
    seen_len = input_ids.shape[1]
    blind_len = blind_ids.shape[1]
    seen_keep = seen_len - (min(aligned) - 1)
    blind_keep = blind_len - (min(aligned) - offset - 1)
    with torch.no_grad():
        seen = target(
            input_ids=input_ids, use_cache=False, logits_to_keep=seen_keep
        ).logits
        unseen = target(
            **{k: v.to(input_ids.device) for k, v in blind.items()},
            use_cache=False,
            logits_to_keep=blind_keep,
        ).logits
    # a kept window holds the LAST k positions, so absolute p sits at p - base
    seen_base = seen_len - seen.shape[1]
    blind_base = blind_len - unseen.shape[1]

    kl: Dict[int, float] = {}
    entropy: Dict[int, float] = {}
    # the distribution that produced token t lives at position t - 1
    for start in range(0, len(aligned), 64):
        chunk = aligned[start : start + 64]
        seen_rows = [index - 1 - seen_base for index in chunk]
        blind_rows = [index - offset - 1 - blind_base for index in chunk]
        if min(seen_rows) < 0 or min(blind_rows) < 0:
            return None
        rows_seen = seen[0, seen_rows].float()
        rows_unseen = unseen[0, blind_rows].float()
        log_p = torch.log_softmax(rows_seen, dim=-1)
        log_q = torch.log_softmax(rows_unseen, dim=-1)
        probability = log_p.exp()
        values = (probability * (log_p - log_q)).sum(dim=-1)
        # entropy of the with-image distribution, in nats
        surprise = -(probability * log_p).sum(dim=-1)
        for index, value, bits in zip(chunk, values.tolist(), surprise.tolist()):
            kl[index] = value
            entropy[index] = bits
    return kl, entropy


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    if den_x <= 0 or den_y <= 0:
        return None
    return num / (den_x * den_y) ** 0.5


def _ranks(values: Sequence[float]) -> List[float]:
    """Average ranks, so ties do not bias the rank correlation."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        stop = position
        while (
            stop + 1 < len(order)
            and values[order[stop + 1]] == values[order[position]]
        ):
            stop += 1
        shared = (position + stop) / 2.0
        for index in order[position : stop + 1]:
            ranks[index] = shared
        position = stop + 1
    return ranks


def _bucket_table(
    rows: List[Dict[str, Any]], key: str, buckets: int = 4
) -> List[Tuple[str, int, float]]:
    """Mean accept length per quantile of `key`, smallest bucket first."""
    usable = [row for row in rows if row.get(key) is not None]
    if len(usable) < buckets:
        return []
    usable.sort(key=lambda row: row[key])
    size = len(usable) / buckets
    table = []
    for index in range(buckets):
        start, stop = int(index * size), int((index + 1) * size)
        group = usable[start:stop] if index < buckets - 1 else usable[start:]
        if not group:
            continue
        label = f"{group[0][key]:.4g}..{group[-1][key]:.4g}"
        table.append(
            (label, len(group), statistics.fmean(row["accept_len"] for row in group))
        )
    return table


def _quartiles(values: Sequence[float]) -> List[float]:
    """The three cut points that split `values` into four equal groups."""
    ordered = sorted(values)
    return [ordered[int(len(ordered) * q)] for q in (0.25, 0.5, 0.75)]


def _bucket_of(value: float, cuts: Sequence[float]) -> int:
    return sum(1 for cut in cuts if value >= cut)


def partial_correlation(
    rows: List[Dict[str, Any]], x_key: str, y_key: str, control_key: str
) -> Optional[float]:
    """corr(x, y) with `control` held fixed, on the rows that carry all three.

    r_xy.z = (r_xy - r_xz r_yz) / sqrt((1 - r_xz^2)(1 - r_yz^2)).
    """
    usable = [
        row
        for row in rows
        if all(row.get(key) is not None for key in (x_key, y_key, control_key))
    ]
    if len(usable) < 4:
        return None
    xs = [row[x_key] for row in usable]
    ys = [row[y_key] for row in usable]
    zs = [row[control_key] for row in usable]
    r_xy, r_xz, r_yz = (
        _pearson(xs, ys),
        _pearson(xs, zs),
        _pearson(ys, zs),
    )
    if None in (r_xy, r_xz, r_yz):
        return None
    denominator = ((1 - r_xz**2) * (1 - r_yz**2)) ** 0.5
    if denominator <= 0:
        return None
    return (r_xy - r_xz * r_yz) / denominator


def print_cross_table(
    rows: List[Dict[str, Any]], row_key: str, col_key: str
) -> None:
    """Mean accepted length over a quartile x quartile grid.

    Reading a row left to right holds `row_key` roughly fixed, which is what
    tells a genuine second factor apart from one that only rode along with it.
    """
    usable = [
        row
        for row in rows
        if row.get(row_key) is not None and row.get(col_key) is not None
    ]
    if len(usable) < 16:
        return
    row_cuts = _quartiles([row[row_key] for row in usable])
    col_cuts = _quartiles([row[col_key] for row in usable])
    grid: Dict[Tuple[int, int], List[float]] = {}
    for row in usable:
        key = (_bucket_of(row[row_key], row_cuts), _bucket_of(row[col_key], col_cuts))
        grid.setdefault(key, []).append(row["accept_len"])

    print(f"\nMean accept length, {row_key} (rows) x {col_key} (columns):")
    header = f"  {row_key[:12]:<14}" + "".join(
        f"{f'{col_key[:8]} Q{c + 1}':>14}" for c in range(4)
    )
    print(header)
    for r in range(4):
        cells = []
        for c in range(4):
            group = grid.get((r, c), [])
            cells.append(
                f"{statistics.fmean(group):>9.2f} ({len(group):>3})" if group else
                f"{'-':>14}"
            )
        print(f"  Q{r + 1:<13}" + "".join(cells))


def _print_correlation(rows: List[Dict[str, Any]], key: str, label: str) -> None:
    """Pearson and Spearman of one score against the accepted length."""
    pairs = [
        (row[key], row["accept_len"]) for row in rows if row.get(key) is not None
    ]
    if len(pairs) < 3:
        return
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    pearson = _pearson(xs, ys)
    spearman = _pearson(_ranks(xs), _ranks(ys))
    print(
        f"  corr({label}, accept length): "
        f"pearson={pearson if pearson is None else f'{pearson:.4f}'}, "
        f"spearman={spearman if spearman is None else f'{spearman:.4f}'}"
    )


def print_token_level(rows: List[Dict[str, Any]]) -> None:
    """Draft accuracy per PREDICTED TOKEN, against that token's own visual KL.

    The block-level tables pair one accepted-prefix length with the mean KL of
    the 15 tokens the block covers, which blends tokens the draft never had to
    get right. Here every predicted slot is its own observation: the draft's hit
    at slot k paired with the KL of the token at slot k.

    DFlash decodes a block in one parallel pass, so slot k does not condition on
    the draft's own guess at slot k-1 -- every slot is an independent prediction
    from the same context, which is what makes this pairing clean. Accuracy
    still decays with k because slot k sits k tokens past the anchor, so the
    table is also split by slot to keep that out of the visual effect.
    """
    observations = []
    for row in rows:
        kls = row.get("kl_per_slot") or []
        correct = row.get("correct") or []
        for slot, (value, hit) in enumerate(zip(kls, correct)):
            if slot == 0 or value is None:
                continue
            observations.append((value, 1.0 if hit else 0.0, slot))
    if len(observations) < 32:
        return

    values = sorted(observation[0] for observation in observations)
    cuts = [values[int(len(values) * q)] for q in (0.25, 0.5, 0.75)]
    bucket_of = lambda v: sum(1 for cut in cuts if v >= cut)  # noqa: E731

    print(
        f"\nTOKEN LEVEL: draft accuracy by the token's own visual KL "
        f"({len(observations):,} predicted tokens)"
    )
    groups: Dict[int, List[float]] = {}
    for value, hit, _slot in observations:
        groups.setdefault(bucket_of(value), []).append(hit)
    for index in range(4):
        hits = groups.get(index, [])
        if hits:
            edge = "" if index == 3 else f"< {cuts[index]:.4g}"
            low = "" if index == 0 else f"{cuts[index - 1]:.4g} <= "
            print(
                f"  KL Q{index + 1}  {low}{edge or 'max':<22}"
                f"n={len(hits):>7,}  accuracy={statistics.fmean(hits):6.1%}"
            )

    bands = ((1, 3), (4, 7), (8, 11), (12, 15))
    print("\n  split by slot (accuracy decays with distance from the anchor):")
    header = "  " + " " * 8 + "".join(f"{f'slot {a}-{b}':>14}" for a, b in bands)
    print(header)
    for index in range(4):
        cells = []
        for low, high in bands:
            hits = [
                hit
                for value, hit, slot in observations
                if bucket_of(value) == index and low <= slot <= high
            ]
            cells.append(
                f"{statistics.fmean(hits):>9.1%} ({len(hits) // 1000}k)"
                if len(hits) >= 1000
                else (f"{statistics.fmean(hits):>9.1%} ({len(hits):>3})" if hits
                      else f"{'-':>14}")
            )
        print(f"  KL Q{index + 1:<4}" + "".join(cells))


def summarize(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No anchors were measured.")
        return

    accept = [row["accept_len"] for row in rows]
    print(
        f"\nAnchors measured: {len(rows)} over "
        f"{len({row['id'] for row in rows})} generations"
    )
    print(f"Mean accepted prefix: {statistics.fmean(accept):.4f} tokens")
    print(f"  (+1 for the anchor itself: {statistics.fmean(accept) + 1:.4f})")

    print("\nAccept length by position within the response:")
    for label, count, mean in _bucket_table(rows, "rel_pos"):
        print(f"  rel_pos {label:>16}  n={count:>6}  accept={mean:.4f}")

    for key, label in (
        ("visual_rel", "LVSpec relevance, block mean"),
        ("visual_rel_first", "LVSpec relevance of the first predicted token"),
    ):
        if not any(row.get(key) is not None for row in rows):
            continue
        print(f"\nAccept length by {label}:")
        for bucket, count, mean in _bucket_table(rows, key):
            print(f"  {bucket:>22}  n={count:>6}  accept={mean:.4f}")
        _print_correlation(rows, key, label)

    if any(row.get("visual_rel_allmean") is not None for row in rows):
        _print_correlation(
            rows, "visual_rel_allmean", "relevance without Top-N (all visual)"
        )

    if any(row.get("visual_kl") is not None for row in rows):
        print("\nAccept length by how much the image moves the target (KL):")
        for label, count, mean in _bucket_table(rows, "visual_kl"):
            print(f"  KL {label:>20}  n={count:>6}  accept={mean:.4f}")
        pairs = [
            (row["visual_kl"], row["accept_len"])
            for row in rows
            if row.get("visual_kl") is not None
        ]
        xs = [pair[0] for pair in pairs]
        ys = [pair[1] for pair in pairs]
        pearson = _pearson(xs, ys)
        spearman = _pearson(_ranks(xs), _ranks(ys))
        print(
            f"\n  corr(visual KL, accept length): "
            f"pearson={pearson if pearson is None else f'{pearson:.4f}'}, "
            f"spearman={spearman if spearman is None else f'{spearman:.4f}'}"
        )

    if any(row.get("entropy") is not None for row in rows):
        print("\nAccept length by the target's own uncertainty (entropy, nats):")
        for bucket, count, mean in _bucket_table(rows, "entropy"):
            print(f"  H {bucket:>20}  n={count:>6}  accept={mean:.4f}")
        _print_correlation(rows, "entropy", "entropy")

        # the decisive pair: does visual dependence survive holding entropy fixed?
        print("\n--- is the KL effect just entropy in disguise? ---")
        for x_key, label in (
            ("visual_kl", "visual KL"),
            ("visual_rel_first", "LVSpec relevance"),
        ):
            if not any(row.get(x_key) is not None for row in rows):
                continue
            plain = _pearson(
                *zip(
                    *[
                        (row[x_key], row["accept_len"])
                        for row in rows
                        if row.get(x_key) is not None
                    ]
                )
            )
            partial = partial_correlation(rows, x_key, "accept_len", "entropy")
            print(
                f"  {label:<18} raw r={plain if plain is None else f'{plain:+.4f}'}"
                f"   controlling for entropy r="
                f"{partial if partial is None else f'{partial:+.4f}'}"
            )
        print_cross_table(rows, "entropy", "visual_kl")
        print_token_level(rows)
    else:
        print(
            "\nNo visual-KL signal was computed "
            "(--no-visual-kl, or alignment failed)."
        )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def load_target(args):
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    # A draft checkpoint handed to --target-model-path fails deep inside the
    # auto-factory with a wall of every vision config it knows, which says
    # nothing about the actual mistake.
    config = AutoConfig.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    architectures = getattr(config, "architectures", None) or []
    if any("DFlash" in name or "Draft" in name for name in architectures):
        raise ValueError(
            f"--target-model-path points at a DRAFT checkpoint "
            f"({args.target_model_path}, architectures={architectures}).\n"
            "The target is the vision-language model the draft speculates for, "
            "e.g. Qwen/Qwen3.5-4B; pass the checkpoint to --draft-model-path."
        )

    processor = AutoProcessor.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    target = AutoModelForImageTextToText.from_pretrained(
        args.target_model_path,
        torch_dtype=getattr(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    return processor, target.to(args.device).eval()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str, rows: Iterator[Dict[str, Any]]) -> int:
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            written += 1
    return written


def check_header_alignment(
    records, processor, header_ids, end_ids, args, generations_path: str = ""
) -> None:
    """Stop early if no generation can ever be loss-masked.

    A template whose assistant header does not appear in the stored replies
    masks nothing, and every generation is then dropped one by one with no hint
    as to why. The usual cause is a reasoning response: the template wraps it in
    ``<think>`` instead of emitting the empty block the header carries.
    """
    from specforge.data.mm_preprocessing import encode_mm_record, to_chat_messages

    for record in records[:8]:
        payload = encode_mm_record(
            record,
            processor,
            image_root="",
            max_length=args.max_length,
            train_only_last_turn=False,
            header_ids=header_ids,
            end_ids=end_ids,
        )
        if payload is not None and any(payload["loss_mask"]):
            return

    tokenizer = processor.tokenizer
    rendered = processor.apply_chat_template(
        to_chat_messages(records[0]["conversations"]),
        tokenize=False,
        add_generation_prompt=False,
    )
    marker = rendered.find("assistant")
    raise RuntimeError(
        "no assistant token could be loss-masked in any of the first "
        "generations, so nothing would be measured.\n"
        f"  header the template implies: {tokenizer.decode(header_ids)!r}\n"
        f"  what the stored reply renders as: "
        f"{rendered[max(marker - 12, 0): marker + 90]!r}\n"
        "  If the reply is wrapped in <think>, it was generated with thinking "
        "on, which this run no longer does. Delete "
        f"{generations_path or 'the generations file'} and re-run to "
        "regenerate without it."
    )


def main() -> None:
    args = parse_args()
    report_specforge_source()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    processor, target = load_target(args)

    generations_path = args.generations or os.path.join(
        args.output_dir, "generations.jsonl"
    )
    if args.generations or os.path.exists(generations_path):
        records = read_jsonl(generations_path)
        print(f"Reusing {len(records)} generations from {generations_path}")
    else:
        print(
            f"Generating {args.num_samples} {args.benchmark} answers "
            "with the target..."
        )
        records = generate_responses(args, load_benchmark_rows(args), processor, target)
        write_jsonl(generations_path, iter(records))
        print(f"Wrote {len(records)} generations to {generations_path}")

    probe = build_probe(args, target)
    print(
        f"Draft: block_size={probe.block_size}, "
        f"target_layer_ids={probe.draft_model.target_layer_ids}, "
        f"num_anchors={probe.num_anchors}"
    )

    from PIL import Image

    from specforge.data.mm_preprocessing import (
        _assistant_header_ids,
        _end_token_ids,
        encode_mm_record,
        to_chat_messages,
    )

    header_ids = _assistant_header_ids(processor)
    end_ids = _end_token_ids(processor)

    check_header_alignment(
        records, processor, header_ids, end_ids, args, generations_path
    )

    rows: List[Dict[str, Any]] = []
    # why a generation never reached the draft; an empty run is otherwise
    # indistinguishable from a broken one
    skipped: Dict[str, int] = {}
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
            skipped["encode returned None (too long, or no trainable token)"] = (
                skipped.get("encode returned None (too long, or no trainable token)", 0)
                + 1
            )
            continue

        input_ids = torch.tensor([payload["input_ids"]], device=args.device)
        loss_mask = torch.tensor(
            [payload["loss_mask"]], device=args.device, dtype=torch.float32
        )
        response_positions = [i for i, flag in enumerate(payload["loss_mask"]) if flag]
        if len(response_positions) < 2 * probe.block_size:
            reason = f"fewer than {2 * probe.block_size} trainable tokens"
            skipped[reason] = skipped.get(reason, 0) + 1
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
        if inputs["input_ids"].shape[1] != input_ids.shape[1]:
            # encode_mm_record and this call must produce the same sequence
            reason = "encode/processor length mismatch"
            skipped[reason] = skipped.get(reason, 0) + 1
            continue

        relevance: Dict[int, float] = {}
        relevance_all: Dict[int, float] = {}
        if args.no_visual_relevance:
            target_hidden = target_features(target, probe, inputs)
        else:
            target_hidden, last_hidden = target_features(
                target, probe, inputs, return_last_hidden=True
            )
            relevance, relevance_all = visual_relevance_per_token(
                last_hidden,
                visual_token_mask(input_ids, processor),
                response_positions,
                args.visual_top_n,
            )
            del last_hidden

        kl = entropy = None
        if not args.no_visual_kl:
            measured = visual_kl_per_token(
                target,
                processor,
                record,
                input_ids,
                payload["loss_mask"],
                args.max_length,
            )
            if measured is not None:
                kl, entropy = measured

        anchors, accepted, correct, valid = per_anchor_accept(
            probe, input_ids, target_hidden, loss_mask, args.logit_chunk_blocks
        )

        first_response = response_positions[0]
        span = max(response_positions[-1] - first_response, 1)
        tokenizer = processor.tokenizer
        for block in range(anchors.shape[0]):
            if not bool(valid[block].any()):
                continue
            anchor_pos = int(anchors[block].item())
            # aligned with `correct`: slot k holds the value for token
            # anchor+k, or None where the slot is invalid. Slot 0 is the anchor
            # itself and is never predicted.
            slots = range(probe.block_size)
            per_slot = lambda src: [  # noqa: E731 - a local alias, not an API
                (src.get(anchor_pos + k) if src and bool(valid[block, k]) else None)
                for k in slots
            ]
            block_positions = [
                anchor_pos + offset
                for offset in range(1, probe.block_size)
                if bool(valid[block, offset])
            ]
            block_kl = (
                [kl[p] for p in block_positions if p in kl] if kl else []
            )
            block_entropy = (
                [entropy[p] for p in block_positions if p in entropy]
                if entropy
                else []
            )
            block_rel = [relevance[p] for p in block_positions if p in relevance]
            block_rel_all = [
                relevance_all[p] for p in block_positions if p in relevance_all
            ]
            rows.append(
                {
                    "id": record["id"],
                    "anchor_pos": anchor_pos,
                    "rel_pos": (anchor_pos - first_response) / span,
                    "accept_len": float(accepted[block].item()),
                    "anchor_token": tokenizer.decode(
                        [payload["input_ids"][anchor_pos]]
                    ),
                    "next_token": (
                        tokenizer.decode([payload["input_ids"][anchor_pos + 1]])
                        if anchor_pos + 1 < len(payload["input_ids"])
                        else None
                    ),
                    "correct": [bool(x) for x in correct[block].tolist()],
                    "n_valid": int(valid[block].sum().item()),
                    # token level: one value per predicted slot, so a slot's
                    # own visual dependence can be paired with its own hit
                    "kl_per_slot": per_slot(kl),
                    "entropy_per_slot": per_slot(entropy),
                    "visual_kl": statistics.fmean(block_kl) if block_kl else None,
                    "visual_kl_first": (
                        kl.get(anchor_pos + 1) if kl else None
                    ),
                    # the control for visual_kl: is the target simply unsure here?
                    "entropy": (
                        statistics.fmean(block_entropy) if block_entropy else None
                    ),
                    "entropy_first": (
                        entropy.get(anchor_pos + 1) if entropy else None
                    ),
                    # LVSpec's cosine relevance: one forward pass, no ablation
                    "visual_rel": (
                        statistics.fmean(block_rel) if block_rel else None
                    ),
                    "visual_rel_first": relevance.get(anchor_pos + 1),
                    "visual_rel_allmean": (
                        statistics.fmean(block_rel_all) if block_rel_all else None
                    ),
                }
            )
        if (position + 1) % 20 == 0:
            print(f"  scored {position + 1}/{len(records)} generations", flush=True)

    anchors_path = os.path.join(args.output_dir, "per_anchor_accept.jsonl")
    written = write_jsonl(anchors_path, iter(rows))
    total_skipped = sum(skipped.values())
    print(
        f"\nWrote {written} anchor records to {anchors_path} "
        f"({total_skipped} of {len(records)} generations skipped)"
    )
    for reason, count in sorted(skipped.items(), key=lambda item: -item[1]):
        print(f"  {count:>5}  {reason}")
    summarize(rows)


if __name__ == "__main__":
    main()
