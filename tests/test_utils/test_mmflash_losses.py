# coding=utf-8
"""Formula-level tests for the MMFlash objective.

Mirrors ``test_dflash_losses.py``: ``mmflash_model.py`` is loaded with a stub
draft model so the wrapper runs on CPU, anchors / draft output / LM-head
output are made deterministic, and every case is checked against a naive
re-implementation of the weighting written independently below.

What the objective must satisfy (see the module docstring of
``specforge/algorithms/common/mmflash_model.py``):

* every token: ``w = base * (1 + alpha * g * (1 - p))`` with ``base`` the
  ``loss_type`` weight exactly as DFlash computes it and ``p`` the draft's
  probability on the target token;
* reduction follows the base: weighted mean for ``"dflash"``, ``/ bsz`` for
  the D-PACE family;
* ``g = 0`` (text-only rows, image rows without scores, no channel at all) or
  ``alpha = 0`` gives the plain ``loss_type`` objective, bit for bit;
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


def _naive_prefix_confidence(prob, mask, smoothing):
    """mmflash base: smoothed probability that verification reaches position k."""
    smooth = (1.0 - smoothing) * prob + smoothing
    smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))
    return torch.cumprod(smooth, dim=-1)


def _naive_base(neg_log_q, mask, loss_type, dpace_alpha, gamma, smoothing=0.5):
    block_size = mask.shape[-1]
    positions = torch.arange(block_size, dtype=torch.double).view(1, 1, -1)
    if loss_type == "dflash":
        if gamma:
            return torch.exp(-(positions - 1).clamp(min=0) / gamma).expand_as(mask)
        return torch.ones_like(mask)
    if loss_type == "mmflash":
        return _naive_prefix_confidence(torch.exp(-neg_log_q), mask, smoothing)
    return _naive_dpace_weight(torch.exp(-neg_log_q), mask, dpace_alpha, loss_type)


def _naive_loss(logits, targets, mask, g4d, *, loss_type, dpace_alpha, gamma, alpha, smoothing=0.5):
    """The ``loss_type`` base weight times the visual multiplier, reduced per base."""
    neg_log_q = _neg_log_q(logits, targets)
    base = _naive_base(neg_log_q, mask, loss_type, dpace_alpha, gamma, smoothing)
    p = torch.exp(-neg_log_q)
    w = base * (1.0 + alpha * g4d * (1.0 - p)) * mask
    if loss_type == "dflash":
        return (neg_log_q * w).sum() / (w.sum() + 1e-6)
    return (neg_log_q * w).sum() / float(logits.shape[0])


def _naive_accept_len(predicted, targets, mask):
    block_size = mask.shape[-1]
    positions = torch.arange(block_size).view(1, 1, -1)
    hit = (predicted == targets) | (mask <= 0.5) | (positions == 0)
    reached = torch.cumprod(torch.cat([torch.ones_like(hit[..., :1]), hit[..., :-1]], -1).long(), -1)
    # accepted = leading run of valid hits after the anchor slot
    return ((reached > 0) & hit & (mask > 0.5)).sum(-1).double()


ALL_LOSS_TYPES = ("dflash", "mmflash", "dpace", "dpace-cumulative-confidence-only", "dpace-continuation-value-only")


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
        smoothing = kwargs.get("mmflash_smoothing", 0.5)
        if visual_score is None:
            g4d = torch.zeros_like(self.g4d)
        else:
            g4d = torch.gather(
                visual_score.clamp(min=0).unsqueeze(1).expand(-1, self.anchors.shape[1], -1),
                2,
                self.safe_indices,
            )
        return _naive_loss(
            self.logits, self.targets, self.mask, g4d,
            loss_type=loss_type, dpace_alpha=dpace_alpha, gamma=gamma, alpha=alpha,
            smoothing=smoothing,
        )

    def _plain(self, loss_type, gamma=None):
        """The loss_type objective with no visual term, written out directly."""
        neg_log_q = _neg_log_q(self.logits, self.targets)
        base = _naive_base(neg_log_q, self.mask, loss_type, 0.5, gamma) * self.mask
        if loss_type == "dflash":
            return (neg_log_q * base).sum() / (base.sum() + 1e-6)
        return (neg_log_q * base).sum() / float(self.logits.shape[0])

    @staticmethod
    def _ratio(metrics, name):
        num, den = metrics["ratio_metrics"][name]
        return float(num / den)

    # --- the objective collapses to loss_type whenever the visual term is off --
    def test_no_channel_is_the_dflash_weighted_mean(self):
        got, _ = self._run(None)
        neg_log_q = _neg_log_q(self.logits, self.targets)
        want = (neg_log_q * self.mask).sum() / (self.mask.sum() + 1e-6)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-9)

    def test_no_channel_keeps_the_dflash_decay(self):
        got, _ = self._run(None, loss_decay_gamma=7.0)
        # the decay itself is evaluated in float32, exactly as DFlash does
        torch.testing.assert_close(got, self._plain("dflash", 7.0), rtol=1e-6, atol=1e-6)

    def test_mmflash_base_is_the_prefix_confidence(self):
        """loss_type="mmflash": base_k = prod_{i<=k} ((1-s) p_i + s), reduced / bsz."""
        neg_log_q = _neg_log_q(self.logits, self.targets)
        for smoothing in (0.3, 0.5, 0.7):
            with self.subTest(smoothing=smoothing):
                got, _ = self._run(None, loss_type="mmflash", mmflash_smoothing=smoothing)
                prefix = _naive_prefix_confidence(torch.exp(-neg_log_q), self.mask, smoothing)
                want = (neg_log_q * prefix * self.mask).sum() / float(self.logits.shape[0])
                torch.testing.assert_close(got, want, rtol=0, atol=1e-9)
                # the chain can only lose probability along the block (a masked
                # position is a no-op, not a break), and never falls below s^k
                self.assertTrue(bool((prefix[..., 1:] <= prefix[..., :-1] + 1e-12).all()))
                floor = smoothing ** (self.block_size - 1)
                self.assertTrue(bool((prefix >= floor - 1e-12).all()))
        with self.assertRaises(ValueError):
            _make_model(self.logits, self.anchors, self.keep_mask, loss_type="mmflash", mmflash_smoothing=1.5)

    def test_no_channel_is_dflash_dpace_bit_for_bit(self):
        for loss_type in ALL_LOSS_TYPES[2:]:
            with self.subTest(loss_type=loss_type):
                got, _ = self._run(None, loss_type=loss_type, loss_decay_gamma=7.0)
                torch.testing.assert_close(got, self._plain(loss_type), rtol=0, atol=1e-9)

    def test_text_only_channel_and_alpha_zero_and_zero_scores_all_equal_plain(self):
        text_only = torch.full_like(self.g_full, -1.0)
        zeros = torch.zeros_like(self.g_full)
        for loss_type in ALL_LOSS_TYPES:
            with self.subTest(loss_type=loss_type):
                plain = self._plain(loss_type, 7.0)
                cases = {
                    "sentinel everywhere": self._run(text_only, loss_type=loss_type, loss_decay_gamma=7.0),
                    "alpha=0 with real g": self._run(self.g_full, loss_type=loss_type, loss_decay_gamma=7.0, visual_alpha=0.0),
                    "image rows, g=0": self._run(zeros, loss_type=loss_type, loss_decay_gamma=7.0, visual_alpha=3.0),
                }
                for name, (got, metrics) in cases.items():
                    torch.testing.assert_close(got, plain, rtol=1e-6, atol=1e-6, msg=name)
                # the zero-score rows are still counted as image rows
                self.assertEqual(self._ratio(cases["image rows, g=0"][1], "mm_zero_score_row_frac"), 1.0)
                self.assertEqual(self._ratio(cases["sentinel everywhere"][1], "mm_row_frac"), 0.0)

    # --- the multiplier ------------------------------------------------------
    def test_mixed_batch_matches_reference_for_every_base(self):
        mixed = self.g_full.clone()
        mixed[1] = -1.0  # row 1 text-only
        for loss_type in ALL_LOSS_TYPES:
            for alpha in (0.0, 1.0, 2.0):
                with self.subTest(loss_type=loss_type, alpha=alpha):
                    got, metrics = self._run(mixed, loss_type=loss_type, loss_decay_gamma=7.0, visual_alpha=alpha)
                    want = self._reference(mixed, loss_type=loss_type, loss_decay_gamma=7.0, visual_alpha=alpha)
                    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)
                    self.assertAlmostEqual(self._ratio(metrics, "mm_row_frac"), 0.5)

    def test_multiplier_is_bounded_and_fades_with_draft_confidence(self):
        alpha = 2.0
        _, metrics = self._run(self.g_full, loss_type="dpace", visual_alpha=alpha)
        boost = self._ratio(metrics, "boost_mean_mm")
        self.assertGreaterEqual(boost, 1.0)
        self.assertLessEqual(boost, 1.0 + alpha)
        # the same channel on a draft that is sure of every token -> multiplier 1
        sure = self.logits.clone()
        for b in range(sure.shape[0]):
            for n in range(sure.shape[1]):
                for k in range(sure.shape[2]):
                    sure[b, n, k, self.targets[b, n, k]] += 60.0
        model = _make_model(sure, self.anchors, self.keep_mask, loss_type="dpace", visual_alpha=alpha)
        _, _, m = model(input_ids=self.input_ids, hidden_states=self.hidden_states,
                        loss_mask=self.loss_mask, visual_score=self.g_full)
        self.assertAlmostEqual(self._ratio(m, "boost_mean_mm"), 1.0, places=6)

    def test_alpha_raises_image_rows_only(self):
        mixed = self.g_full.clone()
        mixed[1] = -1.0
        _, low = self._run(mixed, loss_type="dpace", visual_alpha=0.0)
        _, high = self._run(mixed, loss_type="dpace", visual_alpha=2.0)
        self.assertGreater(self._ratio(high, "w_mean_mm"), self._ratio(low, "w_mean_mm"))
        self.assertGreater(self._ratio(high, "w_share_mm"), self._ratio(low, "w_share_mm"))
        self.assertGreater(self._ratio(high, "loss_share_mm"), self._ratio(low, "loss_share_mm"))
        # text rows are untouched by alpha, and at alpha=0 both modalities share a scale
        self.assertAlmostEqual(self._ratio(high, "w_mean_text"), self._ratio(low, "w_mean_text"), places=9)
        neg_log_q = _neg_log_q(self.logits, self.targets)
        base = _naive_dpace_weight(torch.exp(-neg_log_q), self.mask, 0.5, "dpace") * self.mask
        want_text = float(base[1].sum() / self.mask[1].sum())
        self.assertAlmostEqual(self._ratio(low, "w_mean_text"), want_text, places=6)

    # --- telemetry and invariances ------------------------------------------
    def test_g_mean_metric_is_the_masked_mean_of_the_channel(self):
        _, metrics = self._run(self.g_full, loss_decay_gamma=7.0)
        num, den = metrics["ratio_metrics"]["g_mean_mm"]
        want = (self.g4d * self.mask).sum() / self.mask.sum()
        torch.testing.assert_close(num / den, want.to(num.dtype), rtol=1e-5, atol=1e-6)

    def test_sim_accept_len_matches_a_naive_count(self):
        mixed = self.g_full.clone()
        mixed[1] = -1.0
        _, metrics = self._run(mixed, loss_type="dpace")
        accepted = _naive_accept_len(self.logits.argmax(-1), self.targets, self.mask)
        has_loss = (self.mask > 0.5).any(-1)
        for row, tag in ((0, "mm"), (1, "text")):
            want = accepted[row][has_loss[row]].mean()
            num, den = metrics["ratio_metrics"][f"sim_accept_len_{tag}"]
            torch.testing.assert_close(num / den, want.to(num.dtype), rtol=1e-6, atol=1e-6)

    def test_chunking_does_not_change_loss_or_metrics(self):
        mixed = self.g_full.clone()
        mixed[1] = -1.0
        for loss_type in ("dflash", "dpace"):
            with self.subTest(loss_type=loss_type):
                whole, m_whole = self._run(mixed, loss_type=loss_type, loss_decay_gamma=7.0, objective_chunk_blocks=0)
                chunked, m_chunk = self._run(mixed, loss_type=loss_type, loss_decay_gamma=7.0, objective_chunk_blocks=1)
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
