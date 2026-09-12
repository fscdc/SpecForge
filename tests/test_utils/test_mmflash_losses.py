# coding=utf-8
"""Formula-level tests for the MMFlash objective.

Mirrors ``test_dflash_losses.py``: ``mmflash_model.py`` is loaded with a stub
draft model so the wrapper runs on CPU, anchors / draft output / LM-head
output are made deterministic, and every case is checked against a naive
re-implementation of the weighting written independently below.

What the objective must satisfy (see the module docstring of
``specforge/algorithms/common/mmflash_model.py``):

* no ``visual_score`` -> every row is text-only -> ``loss_type`` as before
  (bit-identical to DFlash for ``loss_type="dflash"``);
* text-only rows under D-PACE -> D-PACE weights rescaled to mean 1 per block;
* image rows -> ``g * (1 + alpha) + (1 - g) * w_vat`` with the VAT decay
  re-anchored at the block's first top-1 miss;
* one weighted mean over the whole batch;
* the result does not depend on ``objective_chunk_blocks``.
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]

_stub_mmflash_draft = types.ModuleType("specforge.modeling.draft.mmflash")


class _MMFlashDraftStub(nn.Module):
    pass


_stub_mmflash_draft.MMFlashDraftModel = _MMFlashDraftStub

_spec = importlib.util.spec_from_file_location(
    "specforge.algorithms.common.mmflash_model",
    REPO / "specforge" / "algorithms" / "common" / "mmflash_model.py",
)
_mmflash_module = importlib.util.module_from_spec(_spec)

_pkg_specforge = types.ModuleType("specforge")
_pkg_specforge.__path__ = [str(REPO / "specforge")]
_pkg_algorithms = types.ModuleType("specforge.algorithms")
_pkg_algorithms.__path__ = [str(REPO / "specforge" / "algorithms")]
_pkg_common = types.ModuleType("specforge.algorithms.common")
_pkg_common.__path__ = [str(REPO / "specforge" / "algorithms" / "common")]
_pkg_modeling = types.ModuleType("specforge.modeling")
_pkg_modeling.__path__ = [str(REPO / "specforge" / "modeling")]
_pkg_draft = types.ModuleType("specforge.modeling.draft")
_pkg_draft.__path__ = [str(REPO / "specforge" / "modeling" / "draft")]

with patch.dict(
    sys.modules,
    {
        "specforge": _pkg_specforge,
        "specforge.algorithms": _pkg_algorithms,
        "specforge.algorithms.common": _pkg_common,
        "specforge.algorithms.common.mmflash_model": _mmflash_module,
        "specforge.modeling": _pkg_modeling,
        "specforge.modeling.draft": _pkg_draft,
        "specforge.modeling.draft.mmflash": _stub_mmflash_draft,
    },
):
    _spec.loader.exec_module(_mmflash_module)
OnlineMMFlashModel = _mmflash_module.OnlineMMFlashModel


# ----------------------------------------------------------------------------
# deterministic wrapper
# ----------------------------------------------------------------------------


class _FixedDraft(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        bsz, draft_len = noise_embedding.shape[:2]
        return torch.zeros(
            bsz, draft_len, self.hidden_size, dtype=noise_embedding.dtype
        )


class _FixedHead(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, hidden_states):
        # hidden_states is (B, n_chunk * K, H): return the matching logits slice
        # so the chunked objective sees exactly the blocks it asked about
        bsz, n_times_k, _ = hidden_states.shape
        full = self.fixed_logits  # (B, N, K, V)
        block_size = full.shape[2]
        n_chunk = n_times_k // block_size
        start = getattr(self, "_cursor", 0)
        if start + n_chunk > full.shape[1]:
            start = 0
        out = full[:, start : start + n_chunk].reshape(bsz, n_times_k, -1)
        self._cursor = (start + n_chunk) % full.shape[1]
        return out


def _fixed_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
    bsz, n_blocks = anchor_positions.shape
    return torch.zeros(
        bsz,
        n_blocks * self.block_size,
        self.embed_tokens.embedding_dim,
        dtype=torch.double,
    )


def _fixed_anchor_sampler(anchors, keep_mask):
    def _sample(self, seq_len, loss_mask, device):
        return anchors.to(device), keep_mask.to(device)

    return _sample


def _make_model(logits, anchors, keep_mask, **kwargs):
    bsz, n_blocks, block_size, vocab_size = logits.shape
    model = OnlineMMFlashModel(
        draft_model=_FixedDraft(hidden_size=4),
        target_lm_head=_FixedHead(logits),
        target_embed_tokens=nn.Embedding(vocab_size, 4).double(),
        mask_token_id=0,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=n_blocks,
        **kwargs,
    ).double()
    model._sample_anchor_positions = types.MethodType(
        _fixed_anchor_sampler(anchors, keep_mask), model
    )
    model._create_noise_embed = types.MethodType(_fixed_noise_embed, model)
    return model


def _sample_tensors():
    torch.manual_seed(123)
    bsz, n_blocks, block_size, vocab_size = 2, 3, 5, 13
    seq_len = 12
    logits = torch.randn(bsz, n_blocks, block_size, vocab_size, dtype=torch.double)
    input_ids = torch.tensor(
        [
            [1, 4, 2, 8, 3, 7, 5, 6, 9, 2, 1, 4],
            [2, 5, 1, 4, 7, 3, 8, 10, 11, 6, 5, 3],
        ],
        dtype=torch.long,
    )
    # make some positions hits so the simulated verification has structure
    for b in range(bsz):
        for n in range(n_blocks):
            for k in (1, 2):
                anchor = [[0, 3, 6], [1, 4, 7]][b][n]
                logits[b, n, k, input_ids[b, anchor + k]] += 6.0
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.double)
    loss_mask[0, 7] = 0.0
    loss_mask[1, 6] = 0.0
    anchors = torch.tensor([[0, 3, 6], [1, 4, 7]], dtype=torch.long)
    keep_mask = torch.tensor([[True, True, True], [True, False, True]])
    hidden_states = torch.zeros(bsz, seq_len, 4, dtype=torch.double)
    return logits, input_ids, loss_mask, hidden_states, anchors, keep_mask


def _targets_and_mask(input_ids, loss_mask, anchors, keep_mask, block_size):
    bsz, seq_len = input_ids.shape
    n_blocks = anchors.shape[1]
    offsets = torch.arange(block_size).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    targets = torch.gather(input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices)
    mask = keep_mask.unsqueeze(-1).expand(-1, -1, block_size).double()
    mask = mask * (label_indices < seq_len).double()
    mask = mask * (offsets > 0).double()
    mask = mask * torch.gather(loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_indices)
    return targets, mask, safe_indices


def _neg_log_q(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view_as(targets)


# ----------------------------------------------------------------------------
# naive references
# ----------------------------------------------------------------------------


def _naive_dpace_weight(prob, binary_mask, alpha, loss_type):
    smooth = (1.0 - alpha) * prob + alpha
    smooth = torch.where(binary_mask > 0, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(smooth, dim=-1)
    if loss_type == "dpace-cumulative-confidence-only":
        return prefix
    suffix = torch.flip(torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1), dims=[-1])
    if loss_type == "dpace":
        return suffix
    if loss_type == "dpace-continuation-value-only":
        return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    raise ValueError(loss_type)


def _naive_text_weights(neg_log_q, mask, loss_type, dpace_alpha, gamma):
    block_size = mask.shape[-1]
    positions = torch.arange(block_size, dtype=torch.double).view(1, 1, -1)
    if loss_type == "dflash":
        if gamma:
            return torch.exp(-(positions - 1).clamp(min=0) / gamma).expand_as(mask)
        return torch.ones_like(mask)
    raw = _naive_dpace_weight(torch.exp(-neg_log_q), mask, dpace_alpha, loss_type)
    block_mean = (raw * mask).sum(-1, keepdim=True) / mask.sum(-1, keepdim=True).clamp_min(1.0)
    return raw / block_mean.clamp_min(torch.finfo(raw.dtype).tiny)


def _naive_vat_weights(predicted, targets, mask, gamma):
    block_size = mask.shape[-1]
    positions = torch.arange(block_size).view(1, 1, -1)
    hit = (predicted == targets) | (mask <= 0.5) | (positions == 0)
    reached = torch.cumprod(torch.cat([torch.ones_like(hit[..., :1]), hit[..., :-1]], -1).long(), -1)
    first_reject = reached.sum(-1, keepdim=True) - 1
    distance = (positions - first_reject).clamp(min=0).double()
    if gamma:
        return torch.exp(-distance / gamma), first_reject
    return torch.ones_like(distance), first_reject


def _naive_loss(logits, targets, mask, g4d, mm_rows, *, loss_type, dpace_alpha, gamma, alpha):
    neg_log_q = _neg_log_q(logits, targets)
    predicted = logits.argmax(-1)
    vat, _ = _naive_vat_weights(predicted, targets, mask, gamma)
    visual = g4d * (1.0 + alpha) + (1.0 - g4d) * vat
    text = _naive_text_weights(neg_log_q, mask, loss_type, dpace_alpha, gamma)
    w = torch.where(mm_rows.view(-1, 1, 1) > 0.5, visual, text) * mask
    return (neg_log_q * w).sum() / (w.sum() + 1e-6)


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------


class TestMMFlashObjective(unittest.TestCase):
    def setUp(self):
        (
            self.logits,
            self.input_ids,
            self.loss_mask,
            self.hidden_states,
            self.anchors,
            self.keep_mask,
        ) = _sample_tensors()
        self.block_size = self.logits.shape[2]
        self.targets, self.mask, self.safe_indices = _targets_and_mask(
            self.input_ids, self.loss_mask, self.anchors, self.keep_mask, self.block_size
        )
        bsz, seq_len = self.input_ids.shape
        torch.manual_seed(7)
        self.g_full = torch.rand(bsz, seq_len, dtype=torch.double)
        self.g4d = torch.gather(
            self.g_full.unsqueeze(1).expand(-1, self.anchors.shape[1], -1), 2, self.safe_indices
        )

    def _run(self, visual_score, **kwargs):
        model = _make_model(self.logits, self.anchors, self.keep_mask, **kwargs)
        loss, accuracy, metrics = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
            visual_score=visual_score,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(accuracy))
        return loss, metrics

    def _reference(self, visual_score, **kwargs):
        loss_type = kwargs.get("loss_type", "dflash")
        gamma = kwargs.get("loss_decay_gamma")
        alpha = kwargs.get("visual_alpha", 1.0)
        dpace_alpha = kwargs.get("dpace_alpha", 0.5)
        if visual_score is None:
            mm_rows = torch.zeros(self.input_ids.shape[0])
            g4d = torch.zeros_like(self.g4d)
        else:
            mm_rows = (visual_score >= 0).any(dim=1).double()
            g4d = torch.gather(
                visual_score.clamp(min=0).unsqueeze(1).expand(-1, self.anchors.shape[1], -1),
                2,
                self.safe_indices,
            )
        return _naive_loss(
            self.logits, self.targets, self.mask, g4d, mm_rows,
            loss_type=loss_type, dpace_alpha=dpace_alpha, gamma=gamma, alpha=alpha,
        )

    # --- legacy paths ------------------------------------------------------
    def test_no_channel_is_the_dflash_weighted_mean(self):
        got, _ = self._run(None)
        neg_log_q = _neg_log_q(self.logits, self.targets)
        want = (neg_log_q * self.mask).sum() / (self.mask.sum() + 1e-6)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-9)

    def test_no_channel_keeps_the_dflash_decay(self):
        got, _ = self._run(None, loss_decay_gamma=7.0)
        positions = torch.arange(self.block_size, dtype=torch.double).view(1, 1, -1)
        weight = self.mask * torch.exp(-(positions - 1).clamp(min=0) / 7.0)
        neg_log_q = _neg_log_q(self.logits, self.targets)
        want = (neg_log_q * weight).sum() / (weight.sum() + 1e-6)
        # the decay itself is evaluated in float32, exactly as DFlash does
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

    def test_all_text_rows_use_per_block_normalised_dpace(self):
        text_only = torch.full_like(self.g_full, -1.0)
        for loss_type in ("dpace", "dpace-cumulative-confidence-only", "dpace-continuation-value-only"):
            with self.subTest(loss_type=loss_type):
                got, _ = self._run(text_only, loss_type=loss_type, loss_decay_gamma=7.0)
                want = self._reference(text_only, loss_type=loss_type, loss_decay_gamma=7.0)
                torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

    def test_text_rows_dpace_weights_average_to_one_per_block(self):
        model = _make_model(self.logits, self.anchors, self.keep_mask, loss_type="dpace")
        neg_log_q = _neg_log_q(self.logits, self.targets)
        positions = torch.arange(self.block_size).view(1, 1, -1)
        weights, block_mean = model._text_row_weights(
            neg_log_q, self.mask.float(), positions
        )
        self.assertIsNotNone(block_mean)
        per_block = (weights * self.mask).sum(-1) / self.mask.sum(-1).clamp_min(1.0)
        has_tokens = self.mask.sum(-1) > 0
        torch.testing.assert_close(
            per_block[has_tokens], torch.ones_like(per_block[has_tokens]), rtol=1e-5, atol=1e-5
        )

    # --- image rows --------------------------------------------------------
    def test_image_rows_without_scores_get_vat_weights(self):
        zeros = torch.zeros_like(self.g_full)  # g=0 everywhere, but still image rows
        got, metrics = self._run(zeros, loss_decay_gamma=7.0)
        want = self._reference(zeros, loss_decay_gamma=7.0)
        torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
        num, den = metrics["ratio_metrics"]["mm_zero_score_row_frac"]
        self.assertEqual(float(num / den), 1.0)

    def test_vat_weights_are_full_up_to_the_first_miss_then_decay(self):
        predicted = self.logits.argmax(-1)
        vat, first_reject = _naive_vat_weights(predicted, self.targets, self.mask, gamma=7.0)
        positions = torch.arange(self.block_size).view(1, 1, -1)
        before = positions <= first_reject
        self.assertTrue(bool((vat[before] == 1.0).all()))
        after = (positions > first_reject) & (self.mask > 0.5)
        if after.any():
            self.assertTrue(bool((vat[after] < 1.0).all()))
        # a block that passes whole has first_reject == K-1 and never decays
        hit = (predicted == self.targets) | (self.mask <= 0.5) | (positions == 0)
        whole = hit.all(-1)
        self.assertTrue(bool((first_reject.squeeze(-1)[whole] == self.block_size - 1).all()))

    def test_fully_visual_rows_reduce_to_plain_mean_ce(self):
        ones = torch.ones_like(self.g_full)
        neg_log_q = _neg_log_q(self.logits, self.targets)
        want = (neg_log_q * self.mask).sum() / (self.mask.sum() + 1e-6)
        for alpha in (0.0, 1.0, 3.0):
            with self.subTest(alpha=alpha):
                got, _ = self._run(ones, loss_decay_gamma=7.0, visual_alpha=alpha)
                torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)

    def test_mixed_batch_matches_reference(self):
        mixed = self.g_full.clone()
        mixed[1] = -1.0  # row 1 text-only
        for alpha in (0.0, 1.0, 2.0):
            with self.subTest(alpha=alpha):
                got, metrics = self._run(mixed, loss_type="dpace", loss_decay_gamma=7.0, visual_alpha=alpha)
                want = self._reference(mixed, loss_type="dpace", loss_decay_gamma=7.0, visual_alpha=alpha)
                torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
                num, den = metrics["ratio_metrics"]["mm_row_frac"]
                self.assertAlmostEqual(float(num / den), 0.5)

    def test_visual_alpha_raises_image_row_weight(self):
        mixed = self.g_full.clone()
        mixed[1] = -1.0
        _, low = self._run(mixed, loss_type="dpace", loss_decay_gamma=7.0, visual_alpha=0.0)
        _, high = self._run(mixed, loss_type="dpace", loss_decay_gamma=7.0, visual_alpha=2.0)
        w_low = low["ratio_metrics"]["w_mean_mm"]
        w_high = high["ratio_metrics"]["w_mean_mm"]
        self.assertGreater(float(w_high[0] / w_high[1]), float(w_low[0] / w_low[1]))
        # text rows are untouched by alpha
        t_low = low["ratio_metrics"]["dpace_block_mean_text"]
        t_high = high["ratio_metrics"]["dpace_block_mean_text"]
        torch.testing.assert_close(t_low[0] / t_low[1], t_high[0] / t_high[1])

    def test_kstar_metric_ignores_blocks_that_pass_whole(self):
        """k* is averaged over rejecting blocks only, and the pass rate is separate."""
        mixed = self.g_full.clone()
        mixed[1] = -1.0
        _, metrics = self._run(mixed, loss_decay_gamma=7.0)
        r = metrics["ratio_metrics"]
        predicted = self.logits.argmax(-1)
        _, first_reject = _naive_vat_weights(predicted, self.targets, self.mask, gamma=7.0)
        positions = torch.arange(self.block_size).view(1, 1, -1)
        hit = (predicted == self.targets) | (self.mask <= 0.5) | (positions == 0)
        has_miss = (~hit).any(-1)
        block_has_loss = (self.mask > 0.5).any(-1)
        for row, tag in ((0, "mm"), (1, "text")):
            rejects = (has_miss[row] & block_has_loss[row])
            want_frac = rejects.float().sum() / block_has_loss[row].float().sum()
            num, den = r[f"sim_reject_frac_{tag}"]
            torch.testing.assert_close(num / den, want_frac.to(num.dtype), rtol=1e-6, atol=1e-6)
            if rejects.any():
                want_k = first_reject.squeeze(-1)[row][rejects].double().mean()
                num, den = r[f"sim_first_reject_{tag}"]
                torch.testing.assert_close(num / den, want_k.to(num.dtype), rtol=1e-6, atol=1e-6)
                # the sentinel would have pulled the mean towards K-1
                self.assertLess(float(num / den), float(self.block_size - 1))

    def test_no_dpace_block_mean_metric_under_dflash_loss(self):
        _, metrics = self._run(None, loss_decay_gamma=7.0)
        self.assertNotIn("dpace_block_mean_text", metrics["ratio_metrics"])
        self.assertNotIn("w_mean_text", metrics["ratio_metrics"])

    # --- telemetry and invariances ------------------------------------------
    def test_g_mean_metric_is_the_masked_mean_of_the_channel(self):
        _, metrics = self._run(self.g_full, loss_decay_gamma=7.0)
        num, den = metrics["ratio_metrics"]["g_mean_mm"]
        want = (self.g4d * self.mask).sum() / self.mask.sum()
        torch.testing.assert_close(num / den, want.to(num.dtype), rtol=1e-5, atol=1e-6)

    def test_chunking_does_not_change_loss_or_metrics(self):
        mixed = self.g_full.clone()
        mixed[1] = -1.0
        whole, m_whole = self._run(mixed, loss_type="dpace", loss_decay_gamma=7.0, objective_chunk_blocks=0)
        chunked, m_chunk = self._run(mixed, loss_type="dpace", loss_decay_gamma=7.0, objective_chunk_blocks=1)
        torch.testing.assert_close(whole, chunked, rtol=1e-9, atol=1e-9)
        for name, (num, den) in m_whole["ratio_metrics"].items():
            num2, den2 = m_chunk["ratio_metrics"][name]
            torch.testing.assert_close(num, num2, rtol=1e-9, atol=1e-9, msg=name)
            torch.testing.assert_close(den, den2, rtol=1e-9, atol=1e-9, msg=name)

    def test_gradient_flows_through_image_and_text_rows(self):
        mixed = self.g_full.clone()
        mixed[1] = -1.0
        model = _make_model(self.logits, self.anchors, self.keep_mask, loss_type="dpace", loss_decay_gamma=7.0)
        # make the head's logits a leaf so we can look at their gradient
        model.lm_head.fixed_logits.requires_grad_(True)
        loss, _, _ = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
            visual_score=mixed,
        )
        loss.backward()
        grad = model.lm_head.fixed_logits.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad[0].abs().sum()), 0.0)  # image row
        self.assertGreater(float(grad[1].abs().sum()), 0.0)  # text row

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            _make_model(self.logits, self.anchors, self.keep_mask, visual_alpha=-0.5)
        model = _make_model(self.logits, self.anchors, self.keep_mask)
        with self.assertRaises(ValueError):
            model(
                input_ids=self.input_ids,
                hidden_states=self.hidden_states,
                loss_mask=self.loss_mask,
                visual_score=torch.zeros(2, 3),
            )


if __name__ == "__main__":
    unittest.main()
