# coding=utf-8
"""Sparse draft context for MMFlash: the training twin of SGLang's sparse mode.

At inference the patched SGLang worker (``patches/sglang/v0.5.14/
dflash-draft-sparse-context.patch``, enabled by ``SGLANG_DFLASH_DRAFT_SPARSE``
together with ``--speculative-draft-window-size``) lets every draft block see
only part of the committed target context. This module trains the draft under
the SAME visibility, so a checkpoint trained here is served with the identical
pattern instead of meeting it for the first time at inference.

Visibility rule. A draft block whose anchor (first token) sits at absolute
position ``a`` sees context position ``p`` iff ``p < a`` and one of

  * ``p < sink``                                   -- attention sinks;
  * ``p`` is not a visual token and ``text``       -- prompt text, the chat
    template's tokens (``<|vision_start|>``/``<|vision_end|>`` included) and
    the answer tokens before the anchor, which SGLang treats as "generated
    tokens are text";
  * ``p`` lies in a visual span ``[s, e]`` with ``p - s >= stride // 2`` and
    ``(p - s - stride // 2) % stride == 0``       -- a strided overview of every
    image/frame (``stride = 0`` keeps none, ``1`` keeps all);
  * ``window > 0`` and ``p >= a - window``         -- the most recent positions.

A visual span is a maximal run of the target's image (or video) pad token.
That is exactly what SGLang's multimodal item offsets are for these prompts
(one ``(start, end)`` per image, end inclusive, the ``<|vision_start|>`` /
``<|vision_end|>`` delimiters outside), so the spans are rebuilt from
``input_ids`` here and nothing new travels through capture.

Everything else is unchanged from the dense MMFlash mask: blocks are
bidirectional internally, blind to each other, and RoPE positions stay
absolute (the K/V the draft reads are the same entries either way). With no
sparse configuration nothing in this module runs: ``build_mmflash_model``
keeps constructing the plain ``OnlineMMFlashModel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from specforge.algorithms.common.mmflash_model import (
    OnlineMMFlashModel,
    create_block_mask,
)

#: The environment variable the patched SGLang worker reads.
SGLANG_SPARSE_ENV = "SGLANG_DFLASH_DRAFT_SPARSE"
#: Key under which the setting is persisted (resume contract, training state)
#: and the ``dflash_config`` key an exported draft config carries it under.
CONTRACT_KEY = "mmflash_draft_sparse"
EXPORT_CONFIG_KEY = "draft_sparse"

_SPEC_KEYS = ("sink", "text", "stride", "window")


@dataclass(frozen=True)
class DraftSparseContext:
    """Which committed context positions a draft block may attend to.

    Same four knobs, meanings and string grammar as SGLang's
    ``DFlashDraftSparseContext``. ``window`` is always explicit here: SGLang
    falls back to ``--speculative-draft-window-size`` when the env string
    omits it, which would let training and serving silently disagree.
    """

    sink: int
    text: bool
    stride: int
    window: int

    def __post_init__(self) -> None:
        for name in ("sink", "stride", "window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"draft sparse {name} must be an integer >= 0, got {value!r}")
        if not isinstance(self.text, bool):
            raise ValueError(f"draft sparse text must be a bool, got {self.text!r}")

    @classmethod
    def from_config(cls, config: Any) -> Optional["DraftSparseContext"]:
        """From ``training.draft_sparse`` (a schema model, a mapping or None)."""
        if config is None:
            return None
        if isinstance(config, DraftSparseContext):
            return config
        get = config.get if isinstance(config, dict) else lambda key: getattr(config, key)
        return cls(
            sink=int(get("sink")),
            text=bool(get("text")),
            stride=int(get("stride")),
            window=int(get("window")),
        )

    @classmethod
    def parse(cls, spec: str) -> "DraftSparseContext":
        """Parse ``"sink=4,text=1,stride=32,window=2048"`` (all four keys)."""
        values: Dict[str, int] = {}
        for part in str(spec).split(","):
            part = part.strip()
            if not part:
                continue
            key, sep, value = part.partition("=")
            key = key.strip()
            if not sep or key not in _SPEC_KEYS or key in values:
                raise ValueError(
                    f"bad draft sparse spec {spec!r}; expected "
                    "sink=<n>,text=<0|1>,stride=<n>,window=<n>"
                )
            values[key] = int(value.strip())
        missing = [key for key in _SPEC_KEYS if key not in values]
        if missing:
            raise ValueError(f"draft sparse spec {spec!r} is missing {missing}")
        if values["text"] not in (0, 1):
            raise ValueError(f"draft sparse text must be 0 or 1 in {spec!r}")
        return cls(
            sink=values["sink"],
            text=bool(values["text"]),
            stride=values["stride"],
            window=values["window"],
        )

    def to_sglang_env(self) -> str:
        """The exact ``SGLANG_DFLASH_DRAFT_SPARSE`` value that serves this pattern."""
        return (
            f"sink={self.sink},text={int(self.text)},stride={self.stride},"
            f"window={self.window}"
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "sink": self.sink,
            "text": int(self.text),
            "stride": self.stride,
            "window": self.window,
        }

    def serving_window_size(self, block_size: int) -> int:
        """A ``--speculative-draft-window-size`` that enables the sparse mode.

        SGLang needs the compact draft cache switched on and requires that
        window to cover a block; the env string's own ``window`` then decides
        the recent positions, so any value >= the block size serves the same
        pattern. ``window`` itself is used when it is large enough.
        """
        return max(int(self.window), int(block_size))

    def __str__(self) -> str:
        return self.to_sglang_env()


def visual_token_ids_of(target_config: Any) -> Tuple[int, ...]:
    """The pad token ids whose runs are visual spans, from the target config.

    Qwen-VL style configs carry ``image_token_id`` and ``video_token_id`` at the
    top level (Qwen3.5-4B: 248056 and 248057), the same fields SGLang builds
    its multimodal offsets from.
    """
    ids = []
    for name in ("image_token_id", "video_token_id"):
        value = getattr(target_config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            ids.append(value)
    return tuple(dict.fromkeys(ids))


def prompt_lengths_from_loss_mask(loss_mask: torch.Tensor) -> torch.Tensor:
    """(B,) the first trained position of every row, i.e. its prompt length.

    The first loss position of a row is where the target's answer starts, which
    is exactly the length of the prompt the serving request carries (both end
    with the same ``<|im_start|>assistant`` header, ``<think></think>``
    included). A row without any loss position counts as all prompt.
    """
    seq_len = loss_mask.shape[1]
    positions = torch.arange(seq_len, device=loss_mask.device).unsqueeze(0)
    return torch.where(loss_mask > 0.5, positions, seq_len).min(dim=1).values


def context_keep_table(
    input_ids: torch.Tensor,
    visual_token_ids: Sequence[int],
    sparse: DraftSparseContext,
    prompt_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """(B, S) bool: the anchor-independent part of the visibility rule.

    ``keep[b, p] = p < sink | (visual[b, p] ? stride_hit[b, p] : text)``, the
    vectorised form of SGLang's ``prompt_keep_mask`` extended over the answer
    (non-visual, so it follows ``text`` as generated tokens do there). Only the
    window term depends on the anchor and is applied inside the mask.
    Right padding (token 0, never visual) is marked as text, which is harmless:
    a row's anchors all lie inside the row, and only ``p < anchor`` is visible.

    SGLang applies the sink to prompt positions only (``keep[:min(sink,
    prompt_len)]``); with ``prompt_lengths`` given the sink is clipped the same
    way, which only matters for a sink longer than the prompt. That is exact
    for single-turn rows, the only kind the video data has: in a multi-turn
    row, SGLang's prompt for a later turn would also cover earlier answers.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be (B, S), got {tuple(input_ids.shape)}")
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    visual = torch.zeros((bsz, seq_len), dtype=torch.bool, device=device)
    for token_id in visual_token_ids:
        visual |= input_ids == int(token_id)
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
    if sparse.stride > 0:
        previous = torch.zeros_like(visual)
        previous[:, 1:] = visual[:, :-1]
        # start of the visual run each position belongs to (no wrap-around:
        # unlike SGLang's torch.roll based finder, a run touching both ends of
        # the row is not merged)
        run_start = torch.where(visual & ~previous, positions, torch.zeros_like(positions))
        run_start = run_start.cummax(dim=1).values
        offset = positions - run_start
        phase = sparse.stride // 2
        stride_hit = visual & (offset >= phase) & ((offset - phase) % sparse.stride == 0)
    else:
        stride_hit = torch.zeros_like(visual)
    keep = torch.where(visual, stride_hit, torch.full_like(visual, sparse.text))
    if sparse.sink > 0:
        if prompt_lengths is None:
            sink_end = torch.full((bsz,), min(sparse.sink, seq_len), device=device)
        else:
            sink_end = prompt_lengths.to(device).clamp(max=sparse.sink)
        keep = keep | (positions < sink_end.unsqueeze(1))
    return keep


def create_mmflash_sparse_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    context_keep: torch.Tensor,
    window: int,
):
    """Flex BlockMask: ``create_mmflash_block_mask`` with the sparse context rule.

    Identical to the dense builder except for the context term, which also
    requires ``context_keep[b, kv] | kv >= anchor - window``.
    """
    window = int(window)

    def mmflash_sparse_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)
        # kv_idx runs over the draft blocks too; clamp before indexing the
        # (B, S) table -- those positions are masked out by is_context anyway
        visible = context_keep[b, kv_idx.clamp(max=S - 1)]
        if window > 0:
            visible = visible | (kv_idx >= anchor_pos - window)
        mask_context = mask_context & visible

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    return create_block_mask(
        mmflash_sparse_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


def create_mmflash_sparse_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    context_keep: torch.Tensor,
    window: int,
) -> torch.Tensor:
    """Dense (B, 1, Q, KV) bool twin of ``create_mmflash_sparse_block_mask``.

    For small sequences and tests: at video lengths the dense mask alone is
    close to a gigabyte, so video training uses ``flex_attention``.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)
    q_block_ids = q_indices // block_size
    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    keep_kv = torch.zeros((B, KV_LEN), dtype=torch.bool, device=device)
    keep_kv[:, :S] = context_keep
    visible = keep_kv.view(B, 1, 1, KV_LEN)
    if int(window) > 0:
        visible = visible | (kv_indices >= anchor_expanded - int(window))
    mask_context = mask_context & visible

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    return (mask_context | mask_draft) & valid_block


def visible_context_counts(
    context_keep: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(visible, available) context positions summed over the valid blocks.

    ``available`` is ``sum(anchor)`` (a block may see ``[0, anchor)``) and
    ``visible`` how many of those the sparse rule keeps, so their ratio is the
    fraction of the context the draft actually attends to.
    """
    bsz, seq_len = context_keep.shape
    zeros = torch.zeros((bsz, 1), dtype=torch.long, device=context_keep.device)
    kept_before = torch.cat([zeros, context_keep.long().cumsum(dim=1)], dim=1)
    dropped_before = torch.cat([zeros, (~context_keep).long().cumsum(dim=1)], dim=1)
    anchors = anchor_positions.clamp(min=0, max=seq_len)
    visible = torch.gather(kept_before, 1, anchors)
    if int(window) > 0:
        low = (anchors - int(window)).clamp(min=0)
        visible = visible + (
            torch.gather(dropped_before, 1, anchors) - torch.gather(dropped_before, 1, low)
        )
    valid = block_keep_mask.long()
    return (visible * valid).sum(), (anchors * valid).sum()


class OnlineSparseMMFlashModel(OnlineMMFlashModel):
    """``OnlineMMFlashModel`` whose draft blocks see the sparse context.

    Only the mask changes: anchors, block inputs, positions, the objective and
    every metric are inherited. One extra ratio metric, ``ctx_visible_frac``,
    reports the share of ``[0, anchor)`` the blocks actually attend to.
    """

    #: dense bool masks are applied by the flex and SDPA paths; transformers'
    #: eager path adds a bool mask to the scores instead of masking with it
    _SUPPORTED_BACKENDS = ("flex_attention", "sdpa")

    def __init__(
        self,
        *args: Any,
        draft_sparse: DraftSparseContext,
        visual_token_ids: Sequence[int],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(draft_sparse, DraftSparseContext):
            raise TypeError(f"draft_sparse must be a DraftSparseContext, got {draft_sparse!r}")
        ids = tuple(int(token_id) for token_id in visual_token_ids)
        if not ids:
            raise ValueError(
                "sparse draft context needs the target's visual pad token id(s) "
                "to find image spans; the target config exposes none"
            )
        if self.attention_backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                f"training.draft_sparse needs attention_backend in "
                f"{self._SUPPORTED_BACKENDS}, got {self.attention_backend!r}"
            )
        self.draft_sparse = draft_sparse
        self.visual_token_ids = ids
        self._sparse_counts: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _forward_draft_blocks(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        context_keep = context_keep_table(
            input_ids,
            self.visual_token_ids,
            self.draft_sparse,
            prompt_lengths=prompt_lengths_from_loss_mask(loss_mask),
        )
        window = self.draft_sparse.window
        builder = (
            create_mmflash_sparse_block_mask
            if self.attention_backend == "flex_attention"
            else create_mmflash_sparse_sdpa_mask
        )
        mmflash_attn_mask = builder(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=seq_len,
            block_size=self.block_size,
            device=device,
            context_keep=context_keep,
            window=window,
        )
        with torch.no_grad():
            self._sparse_counts = visible_context_counts(
                context_keep, anchor_positions, block_keep_mask, window
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=mmflash_attn_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        visual_score: Optional[torch.Tensor] = None,
    ):
        self._sparse_counts = None
        loss, accuracy, metrics = super().forward(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            visual_score=visual_score,
        )
        if self._sparse_counts is not None:
            visible, available = self._sparse_counts
            # float32, not the loss dtype: the sums run into the tens of
            # millions, which bf16 cannot hold exactly
            metrics.setdefault("ratio_metrics", {})["ctx_visible_frac"] = (
                visible.detach().float(),
                available.detach().float(),
            )
            self._sparse_counts = None
        return loss, accuracy, metrics


__all__ = [
    "CONTRACT_KEY",
    "DraftSparseContext",
    "EXPORT_CONFIG_KEY",
    "OnlineSparseMMFlashModel",
    "SGLANG_SPARSE_ENV",
    "context_keep_table",
    "create_mmflash_sparse_block_mask",
    "create_mmflash_sparse_sdpa_mask",
    "visible_context_counts",
    "visual_token_ids_of",
]
