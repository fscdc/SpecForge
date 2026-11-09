# coding=utf-8
"""MMDomino = Domino's drafter on the MMFlash objective; pin both reductions.

* ``loss_type="dflash"`` with no visual term is ``OnlineDominoModel`` exactly
  (the Domino baseline on the same data).
* ``lambda_base = 1`` without ``shift_label`` is ``OnlineMMFlashModel`` exactly:
  the base term is MMFlash's objective on the backbone logits.
* ``lambda_base < 1`` trains the GRU correction, and the visual boost only
  touches image rows.
"""

import unittest
from types import MethodType, SimpleNamespace

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
from specforge.algorithms.common.mmdomino_model import OnlineMMDominoModel
from specforge.algorithms.common.mmflash_model import OnlineMMFlashModel
from specforge.modeling.draft.domino import DominoDraftModel

HIDDEN, VOCAB, BLOCK = 4, 7, 4


def _bare_domino(shift_label: bool) -> DominoDraftModel:
    model = DominoDraftModel.__new__(DominoDraftModel)
    nn.Module.__init__(model)
    model.block_size = BLOCK
    model.shift_label = shift_label
    model.pure_draft_prefix_len = 0
    model.prefix_gru = nn.GRU(
        input_size=HIDDEN, hidden_size=3, num_layers=1, batch_first=True, bias=False
    )
    model.embed_proj = nn.Sequential(
        nn.Linear(HIDDEN + 3, 2, bias=False),
        nn.SiLU(),
        nn.Linear(2, VOCAB, bias=False),
    )
    return model.double()


def _fixture(seed: int = 7):
    torch.manual_seed(seed)
    input_ids = torch.randint(1, VOCAB, (2, 12))
    loss_mask = torch.ones(2, 12)
    loss_mask[:, :2] = 0  # prompt
    loss_mask[1, 10:] = 0  # padding
    anchors = torch.tensor([[2, 6], [3, 5]])
    keep = torch.tensor([[True, True], [True, False]])
    hidden = torch.randn(2, 2 * BLOCK, HIDDEN, dtype=torch.float64)
    # row 0 has an image (g in [0, 1]); row 1 is text-only (sentinel -1)
    visual = torch.rand(2, 12, dtype=torch.float64)
    visual[1] = -1.0
    return SimpleNamespace(
        input_ids=input_ids,
        loss_mask=loss_mask,
        anchors=anchors,
        keep=keep,
        hidden=hidden,
        visual=visual,
    )


def _pin_blocks(model, fx):
    def fixed(self, input_ids, hidden_states, loss_mask):
        del input_ids, hidden_states, loss_mask
        return fx.anchors, fx.keep, fx.hidden

    model._forward_draft_blocks = MethodType(fixed, model)
    return model


def _heads(seed: int = 3):
    torch.manual_seed(seed)
    head = nn.Linear(HIDDEN, VOCAB, bias=False).double()
    embed = nn.Embedding(VOCAB, HIDDEN).double()
    return head, embed


def _common(draft, head, embed, **kwargs):
    return dict(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=0,
        block_size=BLOCK,
        attention_backend="sdpa",
        num_anchors=2,
        **kwargs,
    )


class MMDominoObjectiveTest(unittest.TestCase):
    def test_dflash_loss_without_visual_term_is_domino_exactly(self):
        fx = _fixture()
        for shift_label in (False, True):
            for gamma in (None, 7.0):
                for lambda_base in (0.0, 0.3, 1.0):
                    for alpha, visual in ((0.0, fx.visual), (1.0, None)):
                        with self.subTest(
                            shift_label=shift_label,
                            gamma=gamma,
                            lambda_base=lambda_base,
                            alpha=alpha,
                            visual=visual is not None,
                        ):
                            draft = _bare_domino(shift_label)
                            head, embed = _heads()
                            domino = _pin_blocks(
                                OnlineDominoModel(
                                    **_common(draft, head, embed, loss_decay_gamma=gamma),
                                    shift_label=shift_label,
                                ),
                                fx,
                            )
                            ours = _pin_blocks(
                                OnlineMMDominoModel(
                                    **_common(draft, head, embed, loss_decay_gamma=gamma),
                                    shift_label=shift_label,
                                    loss_type="dflash",
                                    visual_alpha=alpha,
                                ),
                                fx,
                            )
                            args = dict(
                                input_ids=fx.input_ids,
                                hidden_states=torch.zeros(1),
                                loss_mask=fx.loss_mask,
                                lambda_base=lambda_base,
                            )
                            d_loss, d_acc, d_metrics = domino(**args)
                            m_loss, m_acc, m_metrics = ours(**args, visual_score=visual)
                            torch.testing.assert_close(m_loss, d_loss, rtol=0, atol=1e-12)
                            torch.testing.assert_close(m_acc, d_acc, rtol=0, atol=0)
                            for key in (
                                "final_loss",
                                "base_loss",
                                "base_accuracy",
                                "accept_len",
                                "base_accept_len",
                            ):
                                torch.testing.assert_close(
                                    m_metrics[key], d_metrics[key], rtol=0, atol=1e-12
                                )

    def test_lambda_one_is_mmflash_objective_on_the_backbone(self):
        fx = _fixture(11)
        for loss_type in ("mmflash", "dflash"):
            for alpha in (0.0, 1.0, 2.5):
                with self.subTest(loss_type=loss_type, alpha=alpha):
                    draft = _bare_domino(shift_label=False)
                    head, embed = _heads()
                    mmflash = _pin_blocks(
                        OnlineMMFlashModel(
                            **_common(draft, head, embed, loss_decay_gamma=7.0),
                            loss_type=loss_type,
                            visual_alpha=alpha,
                            mmflash_smoothing=0.3,
                        ),
                        fx,
                    )
                    ours = _pin_blocks(
                        OnlineMMDominoModel(
                            **_common(draft, head, embed, loss_decay_gamma=7.0),
                            shift_label=False,
                            loss_type=loss_type,
                            visual_alpha=alpha,
                            mmflash_smoothing=0.3,
                        ),
                        fx,
                    )
                    f_loss, f_acc, _ = mmflash(
                        input_ids=fx.input_ids,
                        hidden_states=torch.zeros(1),
                        loss_mask=fx.loss_mask,
                        visual_score=fx.visual,
                    )
                    m_loss, _, m_metrics = ours(
                        input_ids=fx.input_ids,
                        hidden_states=torch.zeros(1),
                        loss_mask=fx.loss_mask,
                        visual_score=fx.visual,
                        lambda_base=1.0,
                    )
                    torch.testing.assert_close(m_loss, f_loss, rtol=0, atol=1e-12)
                    torch.testing.assert_close(
                        m_metrics["base_accuracy"], f_acc, rtol=0, atol=1e-12
                    )

    def test_visual_boost_touches_only_image_rows_and_final_head_trains(self):
        fx = _fixture(5)
        draft = _bare_domino(shift_label=True)
        head, embed = _heads()

        def run(alpha, visual, lambda_base=0.0):
            model = _pin_blocks(
                OnlineMMDominoModel(
                    **_common(draft, head, embed, loss_decay_gamma=7.0),
                    shift_label=True,
                    loss_type="mmflash",
                    visual_alpha=alpha,
                ),
                fx,
            )
            return model(
                input_ids=fx.input_ids,
                hidden_states=torch.zeros(1),
                loss_mask=fx.loss_mask,
                visual_score=visual,
                lambda_base=lambda_base,
            )

        text_only = torch.full_like(fx.visual, -1.0)
        loss_text_a0, _, _ = run(0.0, text_only)
        loss_text_a1, _, _ = run(1.0, text_only)
        torch.testing.assert_close(loss_text_a1, loss_text_a0, rtol=0, atol=0)

        loss_img_a0, _, _ = run(0.0, fx.visual)
        loss_img_a1, _, metrics = run(1.0, fx.visual)
        self.assertGreater(loss_img_a1.item(), loss_img_a0.item())
        num, den = metrics["ratio_metrics"]["boost_mean_mm"]
        self.assertGreater((num / den).item(), 1.0)
        self.assertLessEqual((num / den).item(), 2.0)

        draft.zero_grad(set_to_none=True)
        loss, _, _ = run(1.0, fx.visual, lambda_base=0.25)
        loss.backward()
        for module in (draft.prefix_gru, draft.embed_proj):
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.0)

    def test_chunked_objective_matches_unchunked(self):
        fx = _fixture(9)
        for shift_label in (False, True):
            with self.subTest(shift_label=shift_label):
                draft = _bare_domino(shift_label)
                head, embed = _heads()
                losses = []
                for chunk in (0, 1):
                    model = _pin_blocks(
                        OnlineMMDominoModel(
                            **_common(
                                draft,
                                head,
                                embed,
                                loss_decay_gamma=7.0,
                                objective_chunk_blocks=chunk,
                            ),
                            shift_label=shift_label,
                            loss_type="mmflash",
                            visual_alpha=1.0,
                        ),
                        fx,
                    )
                    loss, accuracy, metrics = model(
                        input_ids=fx.input_ids,
                        hidden_states=torch.zeros(1),
                        loss_mask=fx.loss_mask,
                        visual_score=fx.visual,
                        lambda_base=0.4,
                    )
                    losses.append((loss, accuracy, metrics["accept_len"]))
                for a, b in zip(*losses):
                    torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_rejects_dpace_loss_types(self):
        draft = _bare_domino(shift_label=True)
        head, embed = _heads()
        with self.assertRaises(ValueError):
            OnlineMMDominoModel(**_common(draft, head, embed), loss_type="dpace")

    def test_strategy_forwards_visual_score_and_ratio_metrics(self):
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.strategies.base import (
            MMDominoTrainStrategy,
            StepContext,
        )

        fx = _fixture(13)
        draft = _bare_domino(shift_label=True)
        head, embed = _heads()
        model = _pin_blocks(
            OnlineMMDominoModel(
                **_common(draft, head, embed, loss_decay_gamma=7.0),
                shift_label=True,
                loss_type="mmflash",
                visual_alpha=1.0,
            ),
            fx,
        )
        strategy = MMDominoTrainStrategy(model, lambda_start=1.0, decay_ratio=1.0)
        batch = TrainBatch(
            sample_ids=["a", "b"],
            strategy="mmdomino",
            tensors={
                "input_ids": fx.input_ids,
                "hidden_states": torch.zeros(2, 12, HIDDEN, dtype=torch.float64),
                "loss_mask": fx.loss_mask,
                "visual_score": fx.visual,
            }
        )
        out = strategy.forward_loss(batch, StepContext(global_step=25, total_steps=100))
        self.assertAlmostEqual(0.75, float(out.metrics["lambda_base"]))
        self.assertIn("boost_mean_mm", out.ratio_metrics)
        self.assertNotIn("ratio_metrics", out.metrics)
        mm_rows, rows = out.ratio_metrics["mm_row_frac"]
        self.assertEqual((1.0, 2.0), (float(mm_rows), float(rows)))


if __name__ == "__main__":
    unittest.main()
