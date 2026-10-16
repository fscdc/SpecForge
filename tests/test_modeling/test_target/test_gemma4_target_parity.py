# coding=utf-8
"""Gemma 4 target parity on CPU: the trainer's frozen target pieces compute
what the target itself (HF) and the serving engine (SGLang) compute.

A tiny random ``Gemma4ForConditionalGeneration`` saved to a temp dir has the
real checkpoint's layout -- one ``model.safetensors``, no index, a tied head
stored only as ``model.language_model.embed_tokens.weight``. Its embedding is
scaled x40 before saving so the raw logits exceed the softcap (30) and the cap
actually engages.
"""

import glob
import math
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.core.compact_teacher import build_offline_teacher_inputs
from specforge.modeling.target.target_head import TargetHead
from specforge.modeling.target.target_utils import (
    TargetEmbeddingsAndHead,
    _ScaledEmbedding,
    target_embedding_scale,
    target_logit_softcap,
)

try:
    from transformers import (
        Gemma4Config,
        Gemma4ForConditionalGeneration,
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )
except ImportError:  # transformers without Gemma 4
    Gemma4Config = None

EMBED_KEY = "model.language_model.embed_tokens.weight"
HIDDEN = 80
CAP = 30.0
WEIGHT_FILES = ["*.json", "*.safetensors", "*.bin"]


def _gemma4_config():
    text = Gemma4TextConfig(
        vocab_size=512,
        vocab_size_per_layer_input=512,
        hidden_size=HIDDEN,
        hidden_size_per_layer_input=8,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=32,
        sliding_window=8,
        final_logit_softcapping=CAP,
        layer_types=["sliding_attention", "full_attention"] * 3,
        # each shared layer type needs an earlier unshared layer of that type
        num_kv_shared_layers=2,
        tie_word_embeddings=True,
        pad_token_id=0,
        max_position_embeddings=512,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1e6,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
    )
    vision = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        patch_size=16,
        pooling_kernel_size=3,
        position_embedding_size=64,
    )
    return Gemma4Config(
        text_config=text,
        vision_config=vision,
        audio_config=None,
        image_token_id=500,
        boi_token_id=501,
        eoi_token_id=502,
        video_token_id=503,
        audio_token_id=504,
        tie_word_embeddings=True,
    )


def _qwen3_config(**overrides):
    from transformers import Qwen3Config

    values = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    values.update(overrides)
    return Qwen3Config(**values)


def _dflash_draft_config(**dflash_config):
    """A one-layer DFlash-family draft over the tiny target's taps 1 and 3."""
    config = _qwen3_config(
        vocab_size=512,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=1,
        head_dim=16,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    config.block_size = 4
    config.num_target_layers = 6
    config.dflash_config = {
        "mask_token_id": 4,
        "target_layer_ids": [1, 3],
        **dflash_config,
    }
    config._attn_implementation = "sdpa"
    return config


@unittest.skipIf(Gemma4Config is None, "transformers has no Gemma 4 support")
class Gemma4TargetParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.target_dir = cls._tmp.name
        model = Gemma4ForConditionalGeneration(_gemma4_config()).eval()
        with torch.no_grad():
            model.get_input_embeddings().weight.mul_(40)
        model.save_pretrained(cls.target_dir)
        cls.model = model
        generator = torch.Generator().manual_seed(1)
        cls.input_ids = torch.randint(5, 500, (1, 12), generator=generator)
        with torch.no_grad():
            cls.output = model(input_ids=cls.input_ids, output_hidden_states=True)
        cls.weight = model.get_input_embeddings().weight.detach()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_fixture_has_the_real_checkpoint_layout(self):
        self.assertEqual(glob.glob(os.path.join(self.target_dir, "*.index.json")), [])
        with safe_open(
            os.path.join(self.target_dir, "model.safetensors"), framework="pt"
        ) as handle:
            keys = set(handle.keys())
        self.assertIn(EMBED_KEY, keys)
        self.assertNotIn("lm_head.weight", keys)
        # the per-layer-embedding table must not look like the tied embedding
        self.assertEqual(
            [key for key in keys if key.endswith("embed_tokens.weight")], [EMBED_KEY]
        )

    def test_scale_is_the_python_float_sglang_serves_with(self):
        self.assertEqual(target_embedding_scale(self.model.config), math.sqrt(HIDDEN))
        self.assertEqual(
            target_embedding_scale(self.model.config.text_config), math.sqrt(HIDDEN)
        )
        self.assertIsInstance(target_embedding_scale(self.model.config), float)
        self.assertEqual(target_logit_softcap(self.model.config), CAP)

    def test_scaled_embedding_matches_hf_and_keeps_the_head_raw(self):
        parts = TargetEmbeddingsAndHead.from_pretrained(
            self.target_dir,
            embed_key=EMBED_KEY,
            device="cpu",
            dtype=torch.float32,
        )
        hidden = self.output.hidden_states[-1]

        self.assertIsInstance(parts.embed_tokens, _ScaledEmbedding)
        with torch.no_grad():
            self.assertTrue(
                torch.equal(
                    parts.embed_tokens(self.input_ids),
                    self.model.get_input_embeddings()(self.input_ids),
                )
            )
            self.assertIs(parts.lm_head.weight, parts.embed_tokens.weight)
            self.assertTrue(torch.equal(parts.embed_tokens.weight, self.weight))
            self.assertTrue(
                torch.equal(parts.lm_head(hidden), F.linear(hidden, self.weight))
            )

    def test_bf16_embedding_rounds_like_sglang(self):
        parts = TargetEmbeddingsAndHead.from_pretrained(
            self.target_dir,
            embed_key=EMBED_KEY,
            device="cpu",
            dtype=torch.bfloat16,
        )
        weight = self.weight.to(torch.bfloat16)
        expected = F.embedding(self.input_ids, weight) * float(math.sqrt(HIDDEN))

        with torch.no_grad():
            actual = parts.embed_tokens(self.input_ids)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))
        # The move to bf16 left the scale unrounded (a buffer would be 8.9375).
        self.assertEqual(parts.embed_tokens.embed_scale, math.sqrt(HIDDEN))
        self.assertIs(parts.lm_head.weight, parts.embed_tokens.weight)
        self.assertTrue(torch.equal(parts.lm_head.weight, weight))

    def test_scale_override(self):
        off = TargetEmbeddingsAndHead(self.model.config, embed_scale=1.0)
        self.assertIs(type(off.embed_tokens), nn.Embedding)
        custom = TargetEmbeddingsAndHead(self.model.config, embed_scale=50.5)
        self.assertEqual(custom.embed_tokens.embed_scale, 50.5)

    def test_target_head_loads_the_single_file_checkpoint_and_caps_like_hf(self):
        head = TargetHead(self.target_dir)
        head.load_weights(self.target_dir, embedding_key=EMBED_KEY)
        hidden = self.output.hidden_states[-1]

        self.assertEqual(head.final_logit_softcapping, CAP)
        self.assertTrue(torch.equal(head.fc.weight, self.weight))
        with torch.no_grad():
            raw = head.fc(hidden)
            capped = head(hidden)
        self.assertGreater(raw.abs().max().item(), CAP)
        self.assertFalse(torch.allclose(raw, self.output.logits, atol=1e-2))
        torch.testing.assert_close(capped, self.output.logits, rtol=0, atol=1e-4)

    def test_target_head_finds_the_tied_embedding_without_a_key(self):
        head = TargetHead(self.target_dir)
        head.load_weights(self.target_dir)
        self.assertTrue(torch.equal(head.fc.weight, self.weight))

    def test_target_head_from_pretrained_returns_bf16_capped_logits(self):
        head = TargetHead.from_pretrained(self.target_dir, embedding_key=EMBED_KEY)
        hidden = self.output.hidden_states[-1].to(torch.bfloat16)

        self.assertEqual(head.fc.weight.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(head.fc.weight, self.weight.to(torch.bfloat16)))
        with torch.no_grad():
            raw = head.fc(hidden)
            logits = head(hidden)
        # fp32 cap math stored as bf16: the values the FSDP root's input cast
        # used to hand the loss, without a full fp32 copy of the logits.
        self.assertGreater(raw.abs().max().item(), CAP)
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertLessEqual(logits.abs().max().item(), CAP)
        reference = raw.float().div(CAP).tanh().mul(CAP)
        self.assertTrue(torch.equal(logits, reference.to(torch.bfloat16)))

    def test_target_head_caps_every_row_chunk_in_place(self):
        head = TargetHead.from_pretrained(self.target_dir, embedding_key=EMBED_KEY)
        # 600 rows: two full 256-row chunks and a partial one, each holding
        # every fixture row, so the cap engages in all three
        hidden = self.output.hidden_states[-1].to(torch.bfloat16).repeat(2, 25, 1)
        fc_outputs = []
        with torch.no_grad():
            raw = head.fc(hidden)
            head.fc.register_forward_hook(
                lambda _module, _args, output: fc_outputs.append(output)
            )
            logits = head(hidden)

        # capped in fc's own buffer, not in a second full-size one
        self.assertEqual(logits.data_ptr(), fc_outputs[0].data_ptr())
        self.assertEqual(logits.dtype, torch.bfloat16)
        self.assertGreater(raw.abs().max().item(), CAP)
        self.assertLessEqual(logits.abs().max().item(), CAP)
        # The reference caps all rows in one call; CPU tanh may then differ by
        # an fp32 ulp at thread-range edges, so allow one bf16 step (2**-3 on
        # [16, 32)).
        reference = raw.float().div(CAP).tanh().mul(CAP).to(torch.bfloat16)
        torch.testing.assert_close(logits, reference, rtol=0, atol=2**-3)

    def test_target_head_needs_one_weight_file_without_an_index(self):
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy(os.path.join(self.target_dir, "config.json"), directory)
            for name in ("model-a.safetensors", "model-b.safetensors"):
                save_file({EMBED_KEY: self.weight}, os.path.join(directory, name))
            head = TargetHead(directory)
            with self.assertRaisesRegex(FileNotFoundError, "exactly one"):
                head.load_weights(directory, embedding_key=EMBED_KEY)

    def test_hub_ids_resolve_only_the_weight_files(self):
        from specforge.export.to_hf import _load_embedding_tensor
        from specforge.modeling.draft.base import Eagle3DraftModel

        class _Draft(nn.Module):
            load_embedding = Eagle3DraftModel.load_embedding

            def __init__(self):
                super().__init__()
                self.embed_tokens = nn.Embedding(512, HIDDEN)

        repo = "example-org/gemma-4-tiny"
        with mock.patch(
            "specforge.modeling.target.target_head.snapshot_download",
            return_value=self.target_dir,
        ) as download:
            head = TargetHead(self.target_dir)
            head.load_weights(repo, cache_dir="/hf-cache", embedding_key=EMBED_KEY)
        download.assert_called_once_with(
            repo_id=repo, cache_dir="/hf-cache", allow_patterns=WEIGHT_FILES
        )
        self.assertTrue(torch.equal(head.fc.weight, self.weight))

        with mock.patch(
            "specforge.modeling.draft.base.snapshot_download",
            return_value=self.target_dir,
        ) as download:
            draft = _Draft()
            draft.load_embedding(repo, embedding_key=EMBED_KEY)
        download.assert_called_once_with(repo_id=repo, allow_patterns=WEIGHT_FILES)
        self.assertTrue(torch.equal(draft.embed_tokens.weight, self.weight))

        with mock.patch(
            "specforge.export.to_hf.snapshot_download",
            return_value=self.target_dir,
        ) as download:
            # a copy: the loaded tensor maps the checkpoint, which would keep
            # an NFS temp dir from being removed
            embedding = _load_embedding_tensor(repo, EMBED_KEY).clone()
        download.assert_called_once_with(repo_id=repo, allow_patterns=WEIGHT_FILES)
        self.assertTrue(torch.equal(embedding, self.weight))

    def test_compact_teacher_refuses_a_softcapped_head(self):
        head = TargetHead(self.target_dir)
        head.load_weights(self.target_dir, embedding_key=EMBED_KEY)
        hidden = self.output.hidden_states[-1]

        with self.assertRaisesRegex(ValueError, "final_logit_softcapping=30.0"):
            build_offline_teacher_inputs(
                compact=True,
                target_model=head,
                target_hidden=hidden,
                chunk_size_arg=None,
            )
        with torch.no_grad():
            logits, kwargs = build_offline_teacher_inputs(
                compact=False,
                target_model=head,
                target_hidden=hidden,
                chunk_size_arg=None,
            )
        self.assertEqual(kwargs, {})
        torch.testing.assert_close(logits, self.output.logits, rtol=0, atol=1e-4)

    def test_spec_generate_embeds_through_the_target_accessor(self):
        from specforge.modeling.draft.dflash import DFlashDraftModel
        from specforge.modeling.draft.domino import DominoDraftModel
        from specforge.modeling.draft.mmflash import MMFlashDraftModel

        # HF Gemma 4 keeps its embedding under the language model, so the
        # helpers must not reach for target.model.embed_tokens.
        self.assertFalse(hasattr(self.model.model, "embed_tokens"))
        prompt = self.input_ids[:, :6]
        drafts = (
            (DFlashDraftModel, {}),
            (MMFlashDraftModel, {}),
            (DominoDraftModel, {"emb_dim": 16, "gru_hidden_dim": 8}),
        )
        for draft_class, head_config in drafts:
            with self.subTest(draft=draft_class.__name__):
                torch.manual_seed(0)
                draft = draft_class(_dflash_draft_config(**head_config)).eval()
                output = draft.spec_generate(
                    self.model,
                    prompt,
                    max_new_tokens=8,
                    stop_token_ids=None,
                    temperature=0.0,
                )
                self.assertTrue(torch.equal(output[:, : prompt.shape[1]], prompt))
                self.assertGreater(output.shape[1], prompt.shape[1])


class NonGemmaTargetTest(unittest.TestCase):
    def test_qwen_and_gemma2_targets_get_a_plain_embedding(self):
        configs = {"qwen3": _qwen3_config()}
        try:
            from transformers import Gemma2Config

            configs["gemma2"] = Gemma2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=8,
            )
        except ImportError:
            pass
        for name, config in configs.items():
            with self.subTest(target=name):
                self.assertIsNone(target_embedding_scale(config))
                parts = TargetEmbeddingsAndHead(config)
                self.assertIs(type(parts.embed_tokens), nn.Embedding)

    def test_gemma3_text_target_is_scaled(self):
        try:
            from transformers import Gemma3TextConfig
        except ImportError:
            self.skipTest("transformers has no Gemma 3")
        config = Gemma3TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        self.assertEqual(target_embedding_scale(config), math.sqrt(32))

    def test_qwen_target_head_is_uncapped(self):
        from transformers import Qwen3ForCausalLM

        torch.manual_seed(0)
        model = Qwen3ForCausalLM(_qwen3_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            head = TargetHead(directory)
            head.load_weights(directory)
        hidden = torch.randn(1, 5, 32)

        self.assertIsNone(head.final_logit_softcapping)
        self.assertIsNone(target_logit_softcap(model.config))
        with torch.no_grad():
            self.assertTrue(torch.equal(head(hidden), model.lm_head(hidden)))
            _, kwargs = build_offline_teacher_inputs(
                compact=True,
                target_model=head,
                target_hidden=hidden,
                chunk_size_arg=None,
            )
        self.assertIs(kwargs["target_head_weight"], head.fc.weight)
        # The HF eval helpers' accessor is the same module for a text target.
        self.assertIs(model.get_input_embeddings(), model.model.embed_tokens)

    def test_non_positive_softcap_is_off(self):
        for cap in (None, 0.0, -1.0):
            with self.subTest(cap=cap):
                config = SimpleNamespace(final_logit_softcapping=cap)
                self.assertIsNone(target_logit_softcap(config))


class TargetScaleResumeContractTest(unittest.TestCase):
    registry = builtin_algorithm_registry()

    @staticmethod
    def _family_model(embed_tokens):
        return SimpleNamespace(
            block_size=16,
            mask_token_id=4,
            attention_backend="flex_attention",
            num_anchors=512,
            loss_decay_gamma=7.0,
            loss_type="dflash",
            dpace_alpha=0.5,
            visual_alpha=1.0,
            mmflash_smoothing=0.0,
            embed_tokens=embed_tokens,
        )

    def test_dflash_family_records_the_scale_only_when_one_applies(self):
        draft = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=5),
            target_layer_ids=[1, 10, 20, 30, 39],
        )
        scale = math.sqrt(2560)
        for name in ("dflash", "mmflash"):
            key = f"{name}_target_embed_scale"
            contract = self.registry.resolve(name).providers.step.resume_contract
            with self.subTest(algorithm=name):
                scaled = _ScaledEmbedding(8, 4, embed_scale=scale)
                self.assertEqual(
                    contract(None, draft, self._family_model(scaled))[key], scale
                )
                for unscaled in (
                    nn.Embedding(8, 4),
                    _ScaledEmbedding(8, 4, embed_scale=1.0),
                    None,
                ):
                    self.assertNotIn(
                        key, contract(None, draft, self._family_model(unscaled))
                    )

    def test_dflash_family_records_an_explicit_scale_of_one(self):
        # target_embedding_scale: 1.0 turns the Gemma scale off. Recording it
        # makes resuming a scaled checkpoint fail, since the trainer compares
        # only the current run's keys; an auto (None) run stays keyless.
        draft = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=5),
            target_layer_ids=[1, 10, 20, 30, 39],
        )
        unscaled = self._family_model(nn.Embedding(8, 4))

        def run_config(scale):
            return SimpleNamespace(model=SimpleNamespace(target_embedding_scale=scale))

        for name in ("dflash", "mmflash"):
            key = f"{name}_target_embed_scale"
            contract = self.registry.resolve(name).providers.step.resume_contract
            with self.subTest(algorithm=name):
                self.assertEqual(contract(run_config(1.0), draft, unscaled)[key], 1.0)
                self.assertNotIn(key, contract(run_config(None), draft, unscaled))

    def test_eagle3_records_the_teacher_softcap_only_when_set(self):
        contract = self.registry.resolve("eagle3").providers.step.resume_contract
        config = SimpleNamespace(
            training=SimpleNamespace(
                compact_teacher=False,
                compact_teacher_chunk_size=None,
            )
        )
        draft = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1))
        model = SimpleNamespace(
            length=7,
            attention_backend="flex_attention",
            lk_loss_type=None,
            kl_scale=1.0,
            kl_decay=1.0,
        )
        self.assertNotIn("eagle3_target_logit_softcap", contract(config, draft, model))
        model.target_logit_softcap = None
        self.assertNotIn("eagle3_target_logit_softcap", contract(config, draft, model))
        model.target_logit_softcap = CAP
        self.assertEqual(
            contract(config, draft, model)["eagle3_target_logit_softcap"], CAP
        )


class DFlashFamilyTargetScaleWiringTest(unittest.TestCase):
    def test_the_run_override_reaches_the_target_embedding(self):
        from specforge.algorithms.model_providers import _build_dflash_family_model

        draft_model = SimpleNamespace(
            config=SimpleNamespace(vocab_size=64, dflash_config={"mask_token_id": 4}),
            target_layer_ids=[1],
            block_size=4,
        )
        parts = SimpleNamespace(
            lm_head=nn.Linear(2, 64), embed_tokens=nn.Embedding(64, 2)
        )
        for override in (None, 1.0, 50.5):
            cfg = SimpleNamespace(
                model=SimpleNamespace(
                    mask_token_id=None,
                    target_model_path="target",
                    embedding_key=EMBED_KEY,
                    lm_head_key="lm_head.weight",
                    cache_dir=None,
                    trust_remote_code=False,
                    target_embedding_scale=override,
                    torch_dtype="float32",
                ),
                training=SimpleNamespace(
                    attention_backend="flex_attention",
                    num_anchors=8,
                    loss_decay_gamma=None,
                    objective_chunk_blocks=0,
                ),
            )
            with (
                self.subTest(override=override),
                mock.patch(
                    "specforge.modeling.target.target_utils."
                    "TargetEmbeddingsAndHead.from_pretrained",
                    return_value=parts,
                ) as load,
            ):
                _build_dflash_family_model(
                    cfg, draft_model, None, lambda common: nn.Linear(1, 1)
                )
            self.assertEqual(load.call_args.kwargs["embed_scale"], override)


if __name__ == "__main__":
    unittest.main()
