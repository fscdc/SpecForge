# coding=utf-8
"""MMFlash online training model and its masking helpers.

Forked from ``OnlineDFlashModel`` (and the mask builders it uses) in
``dflash_family_model.py`` so the MMFlash objective can diverge without
touching DFlash, Domino or DSpark. The block forward, anchor sampling and masks
are still DFlash's; the objective is not.

Objective. ONE formula for every token of every row:

    w_k  = base_k * (1 + visual_alpha * g_k * (1 - p_k))
    loss = sum_k w_k * ce_k / bsz            (loss_type = D-PACE family)
         = sum_k w_k * ce_k / sum_k w_k      (loss_type = "dflash")

* ``base_k`` is the verification-aware weight ``loss_type`` names. It answers
  "is training this position useful right now":

  - ``"mmflash"``: the *prefix confidence* ``prod_{i<=k} q~_i`` with
    ``q~ = (1 - lambda) q + lambda`` (``mmflash_smoothing``) -- the smoothed
    probability that sequential verification reaches position k at all. A
    position after a likely miss gets almost no gradient; a block the draft is
    sure of keeps its later positions in play. Reduced as ``sum(w*ce)/bsz``.
  - ``"dflash"``: DFlash's fixed positional decay, reduced as a weighted mean.
  - the ``"dpace*"`` values: D-PACE weights, exactly as ``dflash_family_model``
    computes them (``"mmflash"`` coincides with its cumulative-only variant).
* ``g_k in [0, 1]`` is the token's visual grounding from the per-token
  ``visual_score`` channel (``specforge.data.visual_score``): how much the
  target's next-token distribution moves when the image is taken away, gated
  by the target's confidence WITH the image, so a token scores high only when
  the image both changes and settles the prediction. Static, from the
  offline sidecar. It answers "is this a multimodal token worth extra effort".
* ``p_k = exp(-ce_k)`` is the draft's current probability on the target token.
  ``(1 - p_k)`` makes the boost fade once the draft has learned the token, so
  the extra effort moves on. Dynamic, per step, detached.

The multiplier is bounded in ``[1, 1 + visual_alpha]`` and never touches the
base, so the verification structure (positional decay, chain survival) is
preserved for every token. A text-only row carries ``g = 0`` everywhere (the
channel's sentinel), so its multiplier is exactly 1 and it trains on the plain
``loss_type`` objective; the same holds for an image row without a sidecar
entry and for the whole batch when ``visual_alpha = 0``. In all those cases
the loss is bit-identical to ``OnlineDFlashModel`` with the same ``loss_type``
-- there is no separate text path and no per-row rescaling.

The ``loss_type`` values (``"dflash"``, ``"dpace"``, ...) are NOT renamed:
they name objectives, are shared with ``training.loss_type`` in the config
schema, and are persisted into resume contracts.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.mmflash import MMFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None

# NPU workaround: flex_attention is not available on Ascend NPU.
if hasattr(torch, "npu") and torch.npu.is_available():
    FLEX_ATTENTION_AVAILABLE = False


_VALID_LOSS_TYPES = {
    "dflash",
    "mmflash",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DPACE_LOSS_TYPES = _VALID_LOSS_TYPES - {"dflash", "mmflash"}


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute per-block acceptance length."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def create_mmflash_sdpa_mask(anchor_positions, block_keep_mask, S, block_size, device):
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_mmflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct Flex Attention BlockMask for MMFlash training.

    KV: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Rules:
      1. Each block sees context strictly before its anchor (kv_idx < anchor_pos).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (block_keep_mask=False) see nothing.
    """

    def mmflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

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
        mmflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device
    )


class OnlineMMFlashModel(nn.Module):
    """MMFlash online training wrapper with MMFlash and D-PACE losses."""

    def __init__(
        self,
        draft_model: MMFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
        visual_alpha: float = 1.0,
        mmflash_smoothing: float = 0.5,
    ):
        super().__init__()
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; must be one of {sorted(_VALID_LOSS_TYPES)}"
            )
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")
        if not visual_alpha >= 0.0:
            raise ValueError(f"visual_alpha must be >= 0, got {visual_alpha}")
        if not 0.0 <= mmflash_smoothing <= 1.0:
            raise ValueError(f"mmflash_smoothing must be in [0, 1], got {mmflash_smoothing}")
        if objective_chunk_blocks < 0:
            raise ValueError("objective_chunk_blocks must be >= 0")

        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.objective_chunk_blocks = int(objective_chunk_blocks)
        # the base (verification-aware) weight of every token, see the module doc
        self.loss_type = loss_type
        self.dpace_alpha = dpace_alpha
        # floor of the smoothed confidence inside the "mmflash" prefix weight
        self.mmflash_smoothing = float(mmflash_smoothing)
        # upper bound of the visual multiplier 1 + visual_alpha * g * (1 - p)
        self.visual_alpha = float(visual_alpha)
        self._objective_printed = False

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Create absolute position IDs for parallel draft blocks."""
        bsz, n_blocks = anchor_positions.shape
        device = anchor_positions.device
        offsets = torch.arange(self.block_size, device=device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        block_starts = torch.arange(n, device=device) * bs
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        return self.embed_tokens(noise_ids)

    def _dpace_weight(
        self,
        prob: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_b: torch.Tensor,
        loss_type: str,
    ) -> torch.Tensor:
        """Compute detached D-PACE position weights.

        ``prob`` is the draft probability on the target token at each draft
        position. Invalid positions are treated as multiplicative no-ops inside
        prefix products and excluded from suffix sums; the caller still
        multiplies the returned weights by ``binary_mask`` before reduction.
        """
        smooth = (1.0 - self.dpace_alpha) * prob + self.dpace_alpha
        smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        if loss_type == "dpace-cumulative-confidence-only":
            return prefix

        suffix = torch.flip(
            torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
            dims=[-1],
        )

        if loss_type == "dpace":
            return suffix
        if loss_type == "dpace-continuation-value-only":
            return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        raise ValueError(f"unknown D-PACE loss_type {loss_type!r}")

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

        if self.attention_backend == "flex_attention":
            mmflash_attn_mask = create_mmflash_block_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )
        else:
            mmflash_attn_mask = create_mmflash_sdpa_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                S=seq_len,
                block_size=self.block_size,
                device=device,
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=mmflash_attn_mask,
        )
        return anchor_positions, block_keep_mask, output_hidden

    def _base_weights(
        self,
        neg_log_q: torch.Tensor,
        weight_mask: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """The verification-aware base weight of every position, per ``loss_type``.

        Applied to every row alike, image or text, and never rescaled: a block
        the draft is confident about carries more total weight than one it is
        unsure about, which is part of the objective.
        """
        if self.loss_type == "dflash":
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                decay = torch.exp(
                    -(positions - 1).clamp(min=0).float() / self.loss_decay_gamma
                )
                return decay.expand_as(weight_mask)
            return torch.ones_like(weight_mask)
        # half-precision CE would make the cumulative products noisy; wider
        # dtypes (the tests run in float64) are kept as they are
        prob_dtype = torch.promote_types(neg_log_q.dtype, torch.float32)
        if self.loss_type == "mmflash":
            # prefix confidence: the smoothed probability that verification
            # reaches position k. Positions the loss does not cover (and the
            # anchor slot) are multiplicative no-ops so they never break a chain.
            confidence = torch.exp(-neg_log_q.detach().to(prob_dtype))
            smooth = (1.0 - self.mmflash_smoothing) * confidence + self.mmflash_smoothing
            smooth = torch.where(weight_mask > 0, smooth, torch.ones_like(smooth))
            return torch.cumprod(smooth, dim=-1)
        if self.loss_type not in _DPACE_LOSS_TYPES:  # defensive, validated in __init__
            raise ValueError(f"unknown loss_type {self.loss_type!r}")
        return self._dpace_weight(
            torch.exp(-neg_log_q.detach().to(prob_dtype)),
            weight_mask.to(prob_dtype),
            weight_mask > 0,
            self.loss_type,
        )

    def _mmflash_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        visual_score: torch.Tensor,
        multimodal_rows: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Additive loss and telemetry terms for one slice of anchor blocks.

        Shapes: ``hidden`` (B, n, K, H); ``target_ids`` / ``weight_mask`` /
        ``visual_score`` (B, n, K) with ``visual_score`` already clamped to
        [0, 1]; ``multimodal_rows`` (B, n, 1), 1.0 on rows that carry an image.
        Every returned tensor is a plain sum over the slice, so the caller can
        add slices and normalise once (see the module docstring).
        """
        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        neg_log_q = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)

        valid = weight_mask > 0.5
        positions = torch.arange(block_size, device=hidden.device).view(1, 1, -1)
        with torch.no_grad():
            predicted_ids = logits.argmax(dim=-1)
            correct = predicted_ids == target_ids

            base = self._base_weights(neg_log_q, weight_mask, positions)
            # the draft's current probability on the target token
            target_probability = torch.exp(-neg_log_q.detach().to(base.dtype))
            g = visual_score.to(base.dtype)
            boost = 1.0 + self.visual_alpha * g * (1.0 - target_probability)
            loss_weights = base * boost * weight_mask.to(base.dtype)

        loss_num = (neg_log_q * loss_weights).sum()
        if self.loss_type == "dflash":
            loss_den = loss_weights.sum()
        else:
            # D-PACE convention: the caller divides by the batch size instead
            loss_den = loss_num.new_zeros(())

        with torch.no_grad():
            correct_f = correct.float()
            mm_mask = weight_mask * multimodal_rows
            text_mask = weight_mask - mm_mask
            ce = neg_log_q.detach().float()
            weighted_ce = neg_log_q.detach() * loss_weights  # same dtype rule as loss_num
            correct_num = (correct_f * weight_mask).sum()
            accuracy_den = weight_mask.sum()

            mm_row_flag = multimodal_rows.squeeze(-1)  # (B, n)
            block_has_loss = valid.any(dim=-1).float()  # (B, n)
            mm_blocks = block_has_loss * mm_row_flag
            text_blocks = block_has_loss - mm_blocks
            accepted = compute_accept_len(predicted_ids, target_ids, valid)  # (B, n)

            telemetry = (
                (ce * mm_mask).sum(),
                mm_mask.sum(),
                (ce * text_mask).sum(),
                text_mask.sum(),
                (correct_f * mm_mask).sum(),
                (correct_f * text_mask).sum(),
                (g * mm_mask).sum(),
                (boost * mm_mask).sum(),
                (loss_weights * multimodal_rows).sum(),
                (loss_weights * (1.0 - multimodal_rows)).sum(),
                (weighted_ce * multimodal_rows).sum(),
                weighted_ce.sum(),
                (accepted * mm_blocks).sum(),
                mm_blocks.sum(),
                (accepted * text_blocks).sum(),
                text_blocks.sum(),
            )
        return (loss_num, loss_den, correct_num, accuracy_den) + telemetry

    def _describe_objective_once(self, has_visual_channel: bool) -> None:
        if self._objective_printed:
            return
        self._objective_printed = True
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        if rank != 0:
            return
        if self.loss_type == "dflash":
            gamma = self.loss_decay_gamma
            base = f"dflash decay exp(-(k-1)/{gamma})" if gamma and gamma > 0 else "uniform"
            reduction = "sum(w*ce)/sum(w)"
        elif self.loss_type == "mmflash":
            base = (
                "prefix confidence prod_(i<=k) ((1-lambda)*p_i + lambda), "
                f"lambda={self.mmflash_smoothing}"
            )
            reduction = "sum(w*ce)/bsz"
        else:
            base = f"{self.loss_type} (dpace_alpha={self.dpace_alpha})"
            reduction = "sum(w*ce)/bsz"
        print(
            f"[mmflash-objective] every row: w = base * (1 + alpha*g*(1-p)), "
            f"base={base}, alpha={self.visual_alpha}, reduction={reduction}; "
            f"visual_score channel present={has_visual_channel}"
            + ("" if has_visual_channel else " (g=0 everywhere: plain base objective)"),
            flush=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        visual_score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel block-wise training forward pass; returns
        (loss, accuracy, metrics) — same shape as Domino's forward.

        ``visual_score`` is the (B, S) float channel of
        ``specforge.data.visual_score``: ``g in [0, 1]`` on image rows, ``-1``
        on every position of a text-only row. ``None`` treats the whole batch
        as text-only.
        """
        if self.attention_backend == "flex_attention" and not FLEX_ATTENTION_AVAILABLE:
            raise ValueError(
                "flex_attention is not available on this device; use sdpa/eager."
            )
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask, output_hidden = self._forward_draft_blocks(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
        )
        num_blocks = anchor_positions.size(1)

        # --- Labels: same-position prediction (position k predicts token anchor+k) ---
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )

        # --- Weight mask: block validity * bounds * exclude anchor (pos 0) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()

        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # --- Visual-dependency channel -> block layout + per-row image flag ---
        has_visual_channel = visual_score is not None
        if visual_score is None:
            visual_score = torch.full(
                (bsz, seq_len), -1.0, device=device, dtype=torch.float32
            )
        visual_score = visual_score.to(device=device, dtype=torch.float32)
        if visual_score.shape != (bsz, seq_len):
            raise ValueError(
                f"visual_score must be {(bsz, seq_len)}, got {tuple(visual_score.shape)}"
            )
        multimodal_row = (visual_score >= 0).any(dim=1)  # (B,)
        g_full = visual_score.clamp(min=0.0, max=1.0)
        # gathered exactly like loss_mask, and kept OUT of weight_mask: the
        # objective needs the two separately (mask decides validity, g decides
        # weight) and the accuracy terms assume a binary mask
        visual_gathered = torch.gather(
            g_full.unsqueeze(1).expand(-1, num_blocks, -1), 2, safe_label_indices
        )
        multimodal_rows = (
            multimodal_row.float().view(bsz, 1, 1).expand(-1, num_blocks, 1).contiguous()
        )
        loss_mask_f = loss_mask.float()
        # Image rows whose channel is all zero, which happens when the row has
        # no sidecar entry (or none is configured) and also -- legitimately --
        # when the confidence gate or the "binary" transform zeroes every token
        # of the row. Either way those rows train on the plain base objective,
        # which is what the metric is for; it is NOT a count of join failures,
        # the producer's own [visual-score] lines are.
        zero_score_row = multimodal_row & ((g_full * loss_mask_f).sum(dim=1) <= 0)

        hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)
        terms = checkpointed_chunk_reduce(
            self._mmflash_objective_chunk_terms,
            hidden_4d,
            target_ids,
            weight_mask,
            visual_gathered,
            multimodal_rows,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        (
            loss_num,
            loss_den,
            correct_num,
            accuracy_denom,
            mm_ce,
            mm_tokens,
            text_ce,
            text_tokens,
            mm_correct,
            text_correct,
            g_sum,
            boost_sum_mm,
            w_sum_mm,
            w_sum_text,
            wce_mm,
            wce_total,
            mm_accepted,
            mm_blocks,
            text_accepted,
            text_blocks,
        ) = terms
        if self.loss_type == "dflash":
            loss = loss_num / (loss_den + 1e-6)
        else:
            loss = loss_num / float(bsz)
        accuracy = correct_num / (accuracy_denom + 1e-6)

        self._describe_objective_once(has_visual_channel)

        rows = loss.new_tensor(float(bsz))
        detach = lambda value: value.detach()  # noqa: E731
        ratio_metrics = {
            # what the batch is made of
            "mm_row_frac": (detach(multimodal_row.float().sum()), rows),
            "mm_zero_score_row_frac": (detach(zero_score_row.float().sum()), rows),
            "mm_token_frac": (detach(mm_tokens), detach(accuracy_denom)),
            # per-modality loss / accuracy (unweighted CE, so comparable to `loss`)
            "mm_ce": (detach(mm_ce), detach(mm_tokens)),
            "text_ce": (detach(text_ce), detach(text_tokens)),
            "mm_acc": (detach(mm_correct), detach(mm_tokens)),
            "text_acc": (detach(text_correct), detach(text_tokens)),
            # the visual channel and what the objective did with it. The
            # multiplier averages 1 on text rows by construction, so only its
            # image-row mean is reported; the two w_mean values share a scale
            # (same base objective), and the shares say how the batch's weight
            # and loss split between the modalities.
            "g_mean_mm": (detach(g_sum), detach(mm_tokens)),
            "boost_mean_mm": (detach(boost_sum_mm), detach(mm_tokens)),
            "w_mean_mm": (detach(w_sum_mm), detach(mm_tokens)),
            "w_mean_text": (detach(w_sum_text), detach(text_tokens)),
            "w_share_mm": (detach(w_sum_mm), detach(w_sum_mm + w_sum_text)),
            "loss_share_mm": (detach(wce_mm), detach(wce_total)),
            # simulated verification from the block's own argmax
            "sim_accept_len_mm": (detach(mm_accepted), detach(mm_blocks)),
            "sim_accept_len_text": (detach(text_accepted), detach(text_blocks)),
        }
        return (
            loss,
            accuracy,
            {"accuracy_denom": accuracy_denom.detach(), "ratio_metrics": ratio_metrics},
        )
