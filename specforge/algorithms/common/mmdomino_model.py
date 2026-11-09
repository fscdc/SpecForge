# coding=utf-8
"""MMDomino: Domino's drafter trained with the MMFlash objective.

Domino is the DFlash block-parallel backbone plus a GRU logits correction
(``DominoDraftModel.apply_logits_head``). ``OnlineDominoModel`` trains it on

    loss = (1 - lambda_base) * L_final + lambda_base * L_base,

the CE of the corrected ("final") logits that draft at inference and of the
backbone's ("base") logits, with ``lambda_base`` decaying to 0 over training
(``DominoTrainStrategy``). MMDomino keeps that blend and Domino's training
forward (anchor sampling, the ``shift_label`` label layout, the GRU head
inputs) and gives each term MMFlash's per-token weight (``mmflash_model``):

    w_k = base_k * (1 + visual_alpha * g_k * (1 - p_k))

computed per head from that head's own probability ``p`` on the target token,
so ``L_final`` is MMFlash's objective on the drafting distribution and
``L_base`` the same objective on the backbone: lambda_base = 1 trains the
backbone exactly as MMFlash trains its draft, lambda_base = 0 trains the
logits Domino actually drafts with.

``loss_type`` names ``base_k`` and the reduction, as in MMFlash:

* ``"mmflash"``: the prefix confidence ``prod_{i<=k} ((1 - s) p_i + s)``
  (``s = mmflash_smoothing``), reduced as ``sum(w * ce) / bsz``;
* ``"dflash"``: Domino's fixed positional decay (``loss_decay_gamma``, offset
  by ``shift_label`` exactly as ``OnlineDominoModel`` offsets it), reduced as a
  weighted mean. With ``visual_alpha = 0`` (or no visual channel) this is
  ``OnlineDominoModel``'s loss bit for bit: the Domino baseline on the same
  multimodal data.

``g_k`` is the target token's visual grounding from the ``visual_score``
channel (``specforge.data.visual_score``), ``-1`` on every position of a
text-only row, so text rows train on the plain ``loss_type`` objective.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from specforge.algorithms.common.dflash_family_model import (
    FLEX_ATTENTION_AVAILABLE,
    OnlineDominoModel,
    compute_accept_len,
)
from specforge.core.chunking import checkpointed_chunk_reduce

_VALID_LOSS_TYPES = {"dflash", "mmflash"}


class OnlineMMDominoModel(OnlineDominoModel):
    """Domino online training wrapper with the MMFlash objective on both heads."""

    def __init__(
        self,
        draft_model,
        target_lm_head,
        target_embed_tokens,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        objective_chunk_blocks: int = 128,
        shift_label: bool = False,
        loss_type: str = "mmflash",
        visual_alpha: float = 1.0,
        mmflash_smoothing: float = 0.5,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
            objective_chunk_blocks=objective_chunk_blocks,
            shift_label=shift_label,
        )
        if loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type={loss_type!r}; MMDomino supports {sorted(_VALID_LOSS_TYPES)}"
            )
        if not visual_alpha >= 0.0:
            raise ValueError(f"visual_alpha must be >= 0, got {visual_alpha}")
        if not 0.0 <= mmflash_smoothing <= 1.0:
            raise ValueError(f"mmflash_smoothing must be in [0, 1], got {mmflash_smoothing}")
        # the parent pins "dflash" for its own objective; this one is configurable
        self.loss_type = loss_type
        self.visual_alpha = float(visual_alpha)
        self.mmflash_smoothing = float(mmflash_smoothing)
        self._objective_printed = False

    def _loss_weights(
        self,
        neg_log_q: torch.Tensor,
        weight_mask: torch.Tensor,
        visual_score: torch.Tensor,
    ) -> torch.Tensor:
        """Detached MMFlash weight ``base * boost * mask`` of one head's positions."""
        if self.loss_type == "dflash":
            base = torch.ones_like(weight_mask)
            if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
                k = torch.arange(self.block_size, device=weight_mask.device).view(1, 1, -1)
                offset = 0 if self.shift_label else 1
                base = torch.exp(
                    -(k - offset).clamp(min=0).float() / self.loss_decay_gamma
                ).expand_as(weight_mask)
        else:
            # half-precision CE would make the cumulative products noisy; wider
            # dtypes (the tests run in float64) are kept as they are
            prob_dtype = torch.promote_types(neg_log_q.dtype, torch.float32)
            confidence = torch.exp(-neg_log_q.detach().to(prob_dtype))
            smooth = (1.0 - self.mmflash_smoothing) * confidence + self.mmflash_smoothing
            # positions outside the loss are multiplicative no-ops in the chain
            smooth = torch.where(weight_mask > 0, smooth, torch.ones_like(smooth))
            base = torch.cumprod(smooth, dim=-1)
        target_probability = torch.exp(-neg_log_q.detach().to(base.dtype))
        boost = 1.0 + self.visual_alpha * visual_score.to(base.dtype) * (
            1.0 - target_probability
        )
        return base * boost * weight_mask.to(base.dtype)

    def _mmdomino_objective_chunk_terms(
        self,
        hidden: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        visual_score: torch.Tensor,
        multimodal_rows: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """Additive loss and telemetry terms for one slice of anchor blocks.

        Shapes: ``hidden`` (B, n, K, H); ``prev_ids`` / ``target_ids`` /
        ``weight_mask`` (binary) / ``visual_score`` (B, n, K) with the score
        clamped to [0, 1]; ``multimodal_rows`` (B, n, 1), 1.0 on image rows.
        """
        batch_size, num_blocks, block_size, hidden_size = hidden.shape
        base_logits = self.lm_head(
            hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
        ).reshape(batch_size, num_blocks, block_size, -1)
        final_logits = self._apply_domino_head(
            base_logits4d=base_logits,
            hidden4d=hidden,
            prev_ids=prev_ids,
            target_ids=target_ids,
        )
        final_ce = F.cross_entropy(
            final_logits.reshape(-1, final_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)
        base_ce = F.cross_entropy(
            base_logits.reshape(-1, base_logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape_as(target_ids)

        with torch.no_grad():
            final_weights = self._loss_weights(final_ce, weight_mask, visual_score)
            base_weights = self._loss_weights(base_ce, weight_mask, visual_score)
        final_num = (final_ce * final_weights).sum()
        base_num = (base_ce * base_weights).sum()
        if self.loss_type == "dflash":
            final_den = final_weights.sum()
            base_den = base_weights.sum()
        else:
            # MMFlash convention: the caller divides by the batch size instead
            final_den = final_num.new_zeros(())
            base_den = base_num.new_zeros(())

        with torch.no_grad():
            valid = weight_mask > 0.5
            predicted_ids = final_logits.argmax(dim=-1)
            base_predicted_ids = base_logits.argmax(dim=-1)
            correct = (predicted_ids == target_ids) & valid
            correct_num = correct.sum().float()
            base_correct_num = ((base_predicted_ids == target_ids) & valid).sum().float()
            accuracy_den = weight_mask.sum()

            accepted = compute_accept_len(predicted_ids, target_ids, valid)  # (B, n)
            base_accepted = compute_accept_len(base_predicted_ids, target_ids, valid)
            valid_blocks = valid.any(dim=-1).float()
            # Domino's accept_len counts the bonus token: accepted drafts + 1
            accept_num = ((accepted + 1.0) * valid_blocks).sum()
            base_accept_num = ((base_accepted + 1.0) * valid_blocks).sum()
            accept_den = valid_blocks.sum()

            # per-modality telemetry of the drafting (final) head, named and
            # counted as OnlineMMFlashModel logs them
            correct_f = correct.float()
            mm_mask = weight_mask * multimodal_rows
            text_mask = weight_mask - mm_mask
            ce = final_ce.detach().float()
            mm_row_flag = multimodal_rows.squeeze(-1)  # (B, n)
            mm_blocks = valid_blocks * mm_row_flag
            text_blocks = valid_blocks - mm_blocks
            target_probability = torch.exp(-ce)
            boost = 1.0 + self.visual_alpha * visual_score.float() * (
                1.0 - target_probability
            )
            weighted_ce = final_ce.detach() * final_weights
            telemetry = (
                (ce * mm_mask).sum(),
                mm_mask.sum(),
                (ce * text_mask).sum(),
                text_mask.sum(),
                (correct_f * mm_mask).sum(),
                (correct_f * text_mask).sum(),
                (visual_score.float() * mm_mask).sum(),
                (boost * mm_mask).sum(),
                (final_weights * multimodal_rows).sum(),
                (final_weights * (1.0 - multimodal_rows)).sum(),
                (weighted_ce * multimodal_rows).sum(),
                weighted_ce.sum(),
                (accepted * mm_blocks).sum(),
                mm_blocks.sum(),
                (accepted * text_blocks).sum(),
                text_blocks.sum(),
            )
        return (
            final_num,
            base_num,
            final_den,
            base_den,
            correct_num,
            base_correct_num,
            accuracy_den,
            accept_num,
            base_accept_num,
            accept_den,
        ) + telemetry

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
            base = f"dflash positional decay (gamma={gamma}, shift_label={self.shift_label})"
            reduction = "sum(w*ce)/sum(w)"
        else:
            base = (
                "prefix confidence prod_(i<=k) ((1-lambda)*p_i + lambda), "
                f"lambda={self.mmflash_smoothing}"
            )
            reduction = "sum(w*ce)/bsz"
        print(
            "[mmdomino] objective per head (final = Domino-corrected logits, base = "
            "backbone logits, p = that head's probability): "
            f"w = base * (1 + alpha * g * (1 - p)), base={base}, alpha={self.visual_alpha}, "
            f"reduction={reduction}; loss = (1 - lambda_base) * L_final + "
            f"lambda_base * L_base; visual_score channel present={has_visual_channel}"
            + ("" if has_visual_channel else " (g=0 everywhere: plain base objective)"),
            flush=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        visual_score: Optional[torch.Tensor] = None,
        lambda_base: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Parallel MMDomino training forward pass; returns (loss, accuracy, metrics).

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

        # --- Labels: Domino's layout (shifted by one with shift_label) ---
        label_start = 1 if self.shift_label else 0
        label_offsets = torch.arange(
            label_start, label_start + self.block_size, device=device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_target_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_target_indices,
        )
        hidden4d, prev_ids = self._build_domino_head_inputs(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            target_ids=target_ids,
            output_hidden=output_hidden,
        )

        # --- Binary weight mask: block validity * bounds * (anchor slot) * loss_mask ---
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()
        if not self.shift_label:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_target_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        # --- Visual-dependency channel at the label positions + per-row image flag ---
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
        visual_gathered = torch.gather(
            g_full.unsqueeze(1).expand(-1, num_blocks, -1), 2, safe_target_indices
        )
        multimodal_rows = (
            multimodal_row.float().view(bsz, 1, 1).expand(-1, num_blocks, 1).contiguous()
        )
        # image rows whose channel is all zero over the loss: no sidecar entry,
        # or the confidence gate zeroed them; they train on the plain base
        zero_score_row = multimodal_row & (
            (g_full * loss_mask.float()).sum(dim=1) <= 0
        )

        (
            final_num,
            base_num,
            final_den,
            base_den,
            correct_num,
            base_correct_num,
            accuracy_denom,
            accept_num,
            base_accept_num,
            accept_den,
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
        ) = checkpointed_chunk_reduce(
            self._mmdomino_objective_chunk_terms,
            hidden4d,
            prev_ids,
            target_ids,
            weight_mask,
            visual_gathered,
            multimodal_rows,
            chunk_size=self.objective_chunk_blocks,
            dim=1,
        )
        if self.loss_type == "dflash":
            final_loss = final_num / (final_den + 1e-6)
            base_loss = base_num / (base_den + 1e-6)
        else:
            final_loss = final_num / float(bsz)
            base_loss = base_num / float(bsz)
        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
        accuracy = correct_num / (accuracy_denom + 1e-6)

        self._describe_objective_once(has_visual_channel)

        rows = loss.new_tensor(float(bsz))
        detach = lambda value: value.detach()  # noqa: E731
        ratio_metrics = {
            "mm_row_frac": (detach(multimodal_row.float().sum()), rows),
            "mm_zero_score_row_frac": (detach(zero_score_row.float().sum()), rows),
            "mm_token_frac": (detach(mm_tokens), detach(accuracy_denom)),
            "mm_ce": (detach(mm_ce), detach(mm_tokens)),
            "text_ce": (detach(text_ce), detach(text_tokens)),
            "mm_acc": (detach(mm_correct), detach(mm_tokens)),
            "text_acc": (detach(text_correct), detach(text_tokens)),
            "g_mean_mm": (detach(g_sum), detach(mm_tokens)),
            "boost_mean_mm": (detach(boost_sum_mm), detach(mm_tokens)),
            "w_mean_mm": (detach(w_sum_mm), detach(mm_tokens)),
            "w_mean_text": (detach(w_sum_text), detach(text_tokens)),
            "w_share_mm": (detach(w_sum_mm), detach(w_sum_mm + w_sum_text)),
            "loss_share_mm": (detach(wce_mm), detach(wce_total)),
            # accepted drafts per block (no bonus token), as MMFlash logs them
            "sim_accept_len_mm": (detach(mm_accepted), detach(mm_blocks)),
            "sim_accept_len_text": (detach(text_accepted), detach(text_blocks)),
        }
        metrics = {
            "final_loss": final_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_accuracy": (base_correct_num / (accuracy_denom + 1e-6)).detach(),
            "accept_len": (accept_num / (accept_den + 1e-6)).detach(),
            "base_accept_len": (base_accept_num / (accept_den + 1e-6)).detach(),
            "lambda_base": torch.tensor(lambda_base, device=loss.device),
            "accuracy_denom": accuracy_denom.detach(),
            "ratio_metrics": ratio_metrics,
        }
        return loss, accuracy, metrics


__all__ = ["OnlineMMDominoModel"]
