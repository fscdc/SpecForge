# coding=utf-8
"""The MMFlash sparse draft context trains exactly what SGLang serves.

``SGLangReference`` below is a verbatim port of the keep logic in
patches/sglang/v0.5.14/dflash-draft-sparse-context.patch
(``prompt_keep_mask`` + ``_gather_draft_sparse_context``) and of SGLang's
``get_mm_items_offset``. Every test compares the training masks against it, per
row and per anchor, so a drift on either side shows up here.
"""

from __future__ import annotations

import itertools
import random
import unittest
from types import SimpleNamespace

import torch

from specforge.algorithms.common.mmflash_model import (
    OnlineMMFlashModel,
    create_mmflash_block_mask,
    create_mmflash_sdpa_mask,
)
from specforge.algorithms.common.mmflash_sparse import (
    CONTRACT_KEY,
    DraftSparseContext,
    OnlineSparseMMFlashModel,
    context_keep_table,
    create_mmflash_sparse_block_mask,
    create_mmflash_sparse_sdpa_mask,
    prompt_lengths_from_loss_mask,
    visible_context_counts,
    visual_token_ids_of,
)

try:
    from torch.nn.attention.flex_attention import create_mask

    FLEX = True
except ImportError:  # pragma: no cover
    FLEX = False

IMG, VID, VS, VE = 248056, 248057, 248053, 248054
BS = 4


# --------------------------------------------------------------------------
# SGLang reference (ported verbatim from the patch and base_processor)
# --------------------------------------------------------------------------
def get_mm_items_offset(input_ids: torch.Tensor, mm_token_id: int):
    mask = input_ids == mm_token_id
    start_positions = (mask & ~torch.roll(mask, 1)).nonzero(as_tuple=True)[0]
    end_positions = (mask & ~torch.roll(mask, -1)).nonzero(as_tuple=True)[0]
    return list(zip(start_positions.tolist(), end_positions.tolist()))


class SGLangReference:
    def __init__(self, sparse: DraftSparseContext):
        self.sink, self.text = sparse.sink, sparse.text
        self.stride, self.window = sparse.stride, sparse.window

    def prompt_keep_mask(self, prompt_len, spans):
        keep = torch.full((prompt_len,), self.text, dtype=torch.bool)
        for start, end in spans:
            start = max(int(start), 0)
            end = min(int(end), prompt_len - 1)
            if end < start:
                continue
            keep[start : end + 1] = False
            if self.stride > 0:
                keep[start + self.stride // 2 : end + 1 : self.stride] = True
        if self.sink > 0:
            keep[: min(self.sink, prompt_len)] = True
        return keep

    def draft_keep(self, prompt_mask, seq_len):
        keep = torch.zeros(seq_len, dtype=torch.bool)
        n_prompt = min(seq_len, int(prompt_mask.numel()))
        keep[:n_prompt] = prompt_mask[:n_prompt]
        if self.text and seq_len > n_prompt:
            keep[n_prompt:] = True  # generated tokens are text
        if self.window > 0:
            keep[max(seq_len - self.window, 0) :] = True
        return keep


# --------------------------------------------------------------------------
# synthetic rows: text, then frames <|vision_start|> pad* <|vision_end|>, a
# question and an answer -- the layout of the video training rows
# --------------------------------------------------------------------------
def make_row(n_images, pad_tokens, n_pre, n_question, n_answer, rng, pad=IMG):
    ids = [rng.randint(10, 990) for _ in range(n_pre)]
    for _ in range(n_images):
        ids += [VS] + [pad] * pad_tokens + [VE]
    ids += [rng.randint(10, 990) for _ in range(n_question)]
    prompt_len = len(ids)
    ids += [rng.randint(10, 990) for _ in range(n_answer)]
    return ids, [0] * prompt_len + [1] * n_answer, prompt_len


def make_batch(rows):
    seq_len = max(len(row[0]) for row in rows)
    input_ids = torch.zeros((len(rows), seq_len), dtype=torch.long)  # collator pads 0
    loss_mask = torch.zeros((len(rows), seq_len), dtype=torch.long)
    for index, (ids, loss, _) in enumerate(rows):
        input_ids[index, : len(ids)] = torch.tensor(ids)
        loss_mask[index, : len(loss)] = torch.tensor(loss)
    return input_ids, loss_mask, seq_len


def sample_anchors(seq_len, loss_mask, num_anchors=64, seed=0):
    model = OnlineMMFlashModel.__new__(OnlineMMFlashModel)
    torch.nn.Module.__init__(model)
    model.block_size, model.num_anchors = BS, num_anchors
    torch.manual_seed(seed)
    return model._sample_anchor_positions(seq_len, loss_mask, torch.device("cpu"))


SPECS = [
    DraftSparseContext(sink=0, text=True, stride=0, window=0),
    DraftSparseContext(sink=4, text=False, stride=0, window=7),   # "sink4": window + sinks
    DraftSparseContext(sink=4, text=True, stride=4, window=7),
    DraftSparseContext(sink=0, text=False, stride=1, window=0),   # every image token
    DraftSparseContext(sink=3, text=False, stride=5, window=0),
    DraftSparseContext(sink=2, text=True, stride=3, window=10),
    DraftSparseContext(sink=0, text=False, stride=0, window=5),   # plain window
    DraftSparseContext(sink=100, text=False, stride=2, window=3),  # sink past the prompt start region
]


def rows_for(seed):
    rng = random.Random(seed)
    return [
        make_row(3, 37, 9, 7, 60, rng),
        make_row(2, 29, 5, 4, 33, rng),  # shorter -> right padded
        make_row(1, 11, 1, 2, 25, rng, pad=VID),
    ]


class DraftSparseContextTest(unittest.TestCase):
    def test_env_string_round_trip(self):
        sparse = DraftSparseContext(sink=4, text=True, stride=32, window=2048)
        self.assertEqual("sink=4,text=1,stride=32,window=2048", sparse.to_sglang_env())
        self.assertEqual(sparse, DraftSparseContext.parse(sparse.to_sglang_env()))
        self.assertEqual(sparse, DraftSparseContext.parse(" window=2048, stride=32,text=1 ,sink=4"))
        self.assertEqual({"sink": 4, "text": 1, "stride": 32, "window": 2048}, sparse.as_dict())

    def test_from_config(self):
        self.assertIsNone(DraftSparseContext.from_config(None))
        expected = DraftSparseContext(sink=1, text=False, stride=2, window=3)
        self.assertEqual(expected, DraftSparseContext.from_config(SimpleNamespace(sink=1, text=False, stride=2, window=3)))
        self.assertEqual(expected, DraftSparseContext.from_config({"sink": 1, "text": False, "stride": 2, "window": 3}))

    def test_rejects_malformed(self):
        for spec in ("sink=1,text=1,stride=0", "sink=1,text=2,stride=0,window=4",
                     "sink=1,text=1,stride=0,window=4,depth=2", "sink=1,sink=2,text=1,stride=0,window=4",
                     "sink=-1,text=1,stride=0,window=4", "sink"):
            with self.assertRaises(ValueError, msg=spec):
                DraftSparseContext.parse(spec)
        with self.assertRaises(ValueError):
            DraftSparseContext(sink=True, text=True, stride=0, window=0)
        with self.assertRaises(ValueError):
            DraftSparseContext(sink=0, text=1, stride=0, window=0)

    def test_serving_window_covers_a_block(self):
        self.assertEqual(2048, DraftSparseContext(sink=0, text=True, stride=0, window=2048).serving_window_size(16))
        self.assertEqual(16, DraftSparseContext(sink=4, text=True, stride=32, window=0).serving_window_size(16))

    def test_visual_token_ids_from_target_config(self):
        self.assertEqual((IMG, VID), visual_token_ids_of(SimpleNamespace(image_token_id=IMG, video_token_id=VID)))
        self.assertEqual((IMG,), visual_token_ids_of(SimpleNamespace(image_token_id=IMG)))
        self.assertEqual((), visual_token_ids_of(SimpleNamespace()))


class MaskMatchesSGLangTest(unittest.TestCase):
    """Per row and per anchor, the training mask keeps exactly SGLang's positions."""

    def _check(self, mask, input_ids, rows, anchors, blocks, sparse, seq_len):
        reference = SGLangReference(sparse)
        num_blocks = anchors.shape[1]
        for b, (ids, _, prompt_len) in enumerate(rows):
            prompt = torch.tensor(ids[:prompt_len])
            spans = get_mm_items_offset(prompt, IMG) + get_mm_items_offset(prompt, VID)
            prompt_mask = reference.prompt_keep_mask(prompt_len, spans)
            for n in range(num_blocks):
                q_rows = mask[b, 0, n * BS : (n + 1) * BS]
                if not bool(blocks[b, n]):
                    self.assertFalse(bool(q_rows.any()), "an invalid block must see nothing")
                    continue
                anchor = int(anchors[b, n])
                expected = reference.draft_keep(prompt_mask, anchor)
                for q in range(BS):
                    context = q_rows[q, :seq_len]
                    self.assertTrue(torch.equal(context[:anchor], expected), f"row {b} anchor {anchor} {sparse}")
                    self.assertFalse(bool(context[anchor:].any()), "nothing at or after the anchor")
                    draft = q_rows[q, seq_len:]
                    own = torch.zeros_like(draft)
                    own[n * BS : (n + 1) * BS] = True
                    self.assertTrue(torch.equal(draft, own), "block attends to itself only")

    def test_sdpa_mask_matches_sglang(self):
        for seed, sparse in itertools.product(range(3), SPECS):
            rows = rows_for(seed)
            input_ids, loss_mask, seq_len = make_batch(rows)
            anchors, blocks = sample_anchors(seq_len, loss_mask, seed=seed)
            keep = context_keep_table(input_ids, (IMG, VID), sparse, prompt_lengths_from_loss_mask(loss_mask))
            mask = create_mmflash_sparse_sdpa_mask(anchors, blocks, seq_len, BS, "cpu", keep, sparse.window)
            self._check(mask, input_ids, rows, anchors, blocks, sparse, seq_len)

    @unittest.skipUnless(FLEX, "flex_attention unavailable")
    def test_flex_mask_matches_sdpa_mask(self):
        for seed, sparse in itertools.product(range(2), SPECS):
            rows = rows_for(seed)
            input_ids, loss_mask, seq_len = make_batch(rows)
            anchors, blocks = sample_anchors(seq_len, loss_mask, seed=seed)
            keep = context_keep_table(input_ids, (IMG, VID), sparse, prompt_lengths_from_loss_mask(loss_mask))
            block_mask = create_mmflash_sparse_block_mask(anchors, blocks, seq_len, BS, "cpu", keep, sparse.window)
            num_q, num_kv = anchors.shape[1] * BS, seq_len + anchors.shape[1] * BS
            dense = create_mask(block_mask.mask_mod, anchors.shape[0], 1, num_q, num_kv, device="cpu")
            sdpa = create_mmflash_sparse_sdpa_mask(anchors, blocks, seq_len, BS, "cpu", keep, sparse.window)
            self.assertTrue(torch.equal(dense, sdpa), str(sparse))

    def test_keep_everything_is_the_dense_mask(self):
        keep_all = DraftSparseContext(sink=0, text=True, stride=1, window=0)
        rows = rows_for(7)
        input_ids, loss_mask, seq_len = make_batch(rows)
        anchors, blocks = sample_anchors(seq_len, loss_mask, seed=7)
        keep = context_keep_table(input_ids, (IMG, VID), keep_all, prompt_lengths_from_loss_mask(loss_mask))
        self.assertTrue(bool(keep.all()))
        sparse = create_mmflash_sparse_sdpa_mask(anchors, blocks, seq_len, BS, "cpu", keep, 0)
        dense = create_mmflash_sdpa_mask(anchors, blocks, seq_len, BS, "cpu")
        self.assertTrue(torch.equal(sparse, dense))
        if FLEX:
            num_q, num_kv = anchors.shape[1] * BS, seq_len + anchors.shape[1] * BS
            flex_sparse = create_mmflash_sparse_block_mask(anchors, blocks, seq_len, BS, "cpu", keep, 0)
            flex_dense = create_mmflash_block_mask(anchors, blocks, seq_len, BS, "cpu")
            self.assertTrue(torch.equal(
                create_mask(flex_sparse.mask_mod, anchors.shape[0], 1, num_q, num_kv, device="cpu"),
                create_mask(flex_dense.mask_mod, anchors.shape[0], 1, num_q, num_kv, device="cpu"),
            ))

    def test_stride_phase_and_spans(self):
        # one frame of 10 pads at [2, 11]; stride 4 keeps offsets 2 and 6 -> 4, 8
        ids = torch.tensor([[7, VS] + [IMG] * 10 + [VE, 9, 9]])
        keep = context_keep_table(ids, (IMG,), DraftSparseContext(sink=0, text=False, stride=4, window=0))
        self.assertEqual([4, 8], keep[0].nonzero().flatten().tolist())
        # adjacent frames are separate spans: each restarts its phase
        ids = torch.tensor([[VS] + [IMG] * 5 + [VE, VS] + [IMG] * 5 + [VE]])
        keep = context_keep_table(ids, (IMG,), DraftSparseContext(sink=0, text=False, stride=3, window=0))
        self.assertEqual([2, 5, 9, 12], keep[0].nonzero().flatten().tolist())

    def test_sink_is_clipped_to_the_prompt(self):
        ids = torch.tensor([[5, 6, 7, 8, 9, 10]])
        loss = torch.tensor([[0, 0, 0, 1, 1, 1]])
        sparse = DraftSparseContext(sink=5, text=False, stride=0, window=0)
        lengths = prompt_lengths_from_loss_mask(loss)
        self.assertEqual([3], lengths.tolist())
        self.assertEqual([0, 1, 2], context_keep_table(ids, (IMG,), sparse, lengths)[0].nonzero().flatten().tolist())
        self.assertEqual([0, 1, 2, 3, 4], context_keep_table(ids, (IMG,), sparse)[0].nonzero().flatten().tolist())
        self.assertEqual([6], prompt_lengths_from_loss_mask(torch.zeros(1, 6, dtype=torch.long)).tolist())

    def test_visible_counts(self):
        for seed, sparse in itertools.product(range(2), SPECS):
            rows = rows_for(seed)
            input_ids, loss_mask, seq_len = make_batch(rows)
            anchors, blocks = sample_anchors(seq_len, loss_mask, seed=seed)
            keep = context_keep_table(input_ids, (IMG, VID), sparse, prompt_lengths_from_loss_mask(loss_mask))
            visible, available = visible_context_counts(keep, anchors, blocks, sparse.window)
            mask = create_mmflash_sparse_sdpa_mask(anchors, blocks, seq_len, BS, "cpu", keep, sparse.window)
            brute_visible = sum(
                int(mask[b, 0, n * BS, :seq_len].sum())
                for b in range(anchors.shape[0])
                for n in range(anchors.shape[1])
                if bool(blocks[b, n])
            )
            self.assertEqual(brute_visible, int(visible))
            self.assertEqual(int((anchors * blocks).sum()), int(available))


def _tiny_models(attention_backend, sparse=None, visual_token_ids=(IMG, VID)):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

    from specforge.modeling.draft.mmflash import MMFlashDraftModel

    config = Qwen3Config(hidden_size=64, intermediate_size=128, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, num_hidden_layers=2, vocab_size=1000,
                         max_position_embeddings=4096, rope_theta=1e7,
                         layer_types=["full_attention"] * 2)
    config.num_target_layers = 4
    config.block_size = BS
    config.dflash_config = {"target_layer_ids": [0, 1], "mask_token_id": 999}
    config._attn_implementation = attention_backend
    torch.manual_seed(7)
    draft = MMFlashDraftModel(config)
    lm_head = torch.nn.Linear(64, 1000, bias=False)
    embed = torch.nn.Embedding(1000, 64)
    common = dict(mask_token_id=999, block_size=BS, attention_backend=attention_backend,
                  num_anchors=32, loss_type="mmflash", objective_chunk_blocks=8)
    if sparse is None:
        return OnlineMMFlashModel(draft, lm_head, embed, **common)
    return OnlineSparseMMFlashModel(draft, lm_head, embed, draft_sparse=sparse,
                                    visual_token_ids=visual_token_ids, **common)


class OnlineSparseModelTest(unittest.TestCase):
    def setUp(self):
        rng = random.Random(3)
        # row ids stay below the tiny vocab; the pad ids are remapped for the
        # embedding lookup only through the anchor tokens, which are answers
        rows = [make_row(3, 37, 9, 7, 60, rng), make_row(2, 29, 5, 4, 33, rng)]
        self.rows = rows
        self.input_ids, self.loss_mask, self.seq_len = make_batch(rows)
        torch.manual_seed(11)
        self.hidden = torch.randn(2, self.seq_len, 128)
        self.visual_score = torch.where(self.loss_mask > 0, torch.rand(2, self.seq_len), torch.zeros(2, self.seq_len))

    def _run(self, model, seed=5, backward=True):
        torch.manual_seed(seed)  # same anchors for every model
        # anchors are answer tokens (< 1000), so the tiny embedding never sees a pad id
        loss, accuracy, metrics = model(self.input_ids, self.hidden, self.loss_mask, self.visual_score)
        if backward:
            loss.backward()
        return loss, accuracy, metrics

    def test_keep_everything_reproduces_the_dense_loss(self):
        dense = _tiny_models("sdpa")
        keep_all = _tiny_models("sdpa", DraftSparseContext(sink=0, text=True, stride=1, window=0))
        dense_loss, _, _ = self._run(dense)
        sparse_loss, _, metrics = self._run(keep_all)
        self.assertTrue(torch.equal(dense_loss.detach(), sparse_loss.detach()))
        visible, available = metrics["ratio_metrics"]["ctx_visible_frac"]
        self.assertEqual(float(visible), float(available))

    def test_sparse_forward_backward_and_metric(self):
        sparse = DraftSparseContext(sink=4, text=False, stride=4, window=7)
        model = _tiny_models("sdpa", sparse)
        loss, accuracy, metrics = self._run(model)
        self.assertTrue(torch.isfinite(loss))
        grads = [p.grad for p in model.draft_model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        visible, available = metrics["ratio_metrics"]["ctx_visible_frac"]
        self.assertGreater(float(available), 0.0)
        self.assertLess(float(visible), float(available))
        self.assertIn("sim_accept_len_mm", metrics["ratio_metrics"])  # inherited metrics intact

    @unittest.skipUnless(FLEX, "flex_attention unavailable")
    def test_flex_forward_matches_sdpa(self):
        sparse = DraftSparseContext(sink=2, text=True, stride=3, window=10)
        # CPU flex has no backward: keep autograd off for both
        with torch.no_grad():
            flex_loss, _, _ = self._run(_tiny_models("flex_attention", sparse), backward=False)
            sdpa_loss, _, _ = self._run(_tiny_models("sdpa", sparse), backward=False)
        self.assertTrue(torch.allclose(flex_loss, sdpa_loss, atol=1e-5, rtol=1e-5))

    def test_rejects_eager_and_missing_visual_ids(self):
        sparse = DraftSparseContext(sink=0, text=True, stride=0, window=8)
        with self.assertRaises(ValueError):
            _tiny_models("eager", sparse)
        with self.assertRaises(ValueError):
            _tiny_models("sdpa", sparse, visual_token_ids=())


class ResumeContractTest(unittest.TestCase):
    def _contract(self, **extra):
        from specforge.algorithms.builtin import builtin_algorithm_registry

        mmflash = builtin_algorithm_registry().resolve("mmflash")
        draft = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=3), target_layer_ids=[1, 5, 9])
        model = SimpleNamespace(block_size=16, mask_token_id=7, attention_backend="flex_attention",
                                num_anchors=32, loss_decay_gamma=None, loss_type="mmflash",
                                dpace_alpha=0.5, visual_alpha=1.0, mmflash_smoothing=0.5, **extra)
        return mmflash.providers.step.resume_contract(None, draft, model)

    def test_dense_contract_has_no_sparse_key(self):
        self.assertNotIn(CONTRACT_KEY, self._contract())
        self.assertNotIn(CONTRACT_KEY, self._contract(draft_sparse=None))

    def test_sparse_contract_records_the_env_string(self):
        sparse = DraftSparseContext(sink=4, text=True, stride=32, window=2048)
        contract = self._contract(draft_sparse=sparse)
        self.assertEqual("sink=4,text=1,stride=32,window=2048", contract[CONTRACT_KEY])
        self.assertEqual({k: v for k, v in contract.items() if k != CONTRACT_KEY}, self._contract())


if __name__ == "__main__":
    unittest.main()
