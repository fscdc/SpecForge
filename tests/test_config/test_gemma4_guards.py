# coding=utf-8
"""Fail-fast guards for a Gemma 4 target, checked on CPU before a 16 GB load.

* the managed capture server must run Gemma4 on an attention backend SGLang
  accepts (triton / trtllm_mha), and an image capture on the regen and serving
  servers' triton attention and sdpa vision attention
  (``--mm-attention-backend``);
* the managed preflight refuses an SGLang without the gemma4-spec patch;
* capture layer ids must lie inside the target's depth;
* an image run's processor resolves the tokenizer's ``<mask>`` id;
* no draft config is derived from a target whose RoPE is keyed by layer type.

Everything is mocked or tiny; the checks against the cached Gemma 4 snapshot
are skipped when it is not in the local Hugging Face cache.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

from pydantic import ValidationError

from specforge import launch_plan
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config, ModelConfig, apply_overrides
from specforge.launch_plan import LaunchPlan, _sglang_argv, build_launch_plan
from specforge.training.capture_contract import (
    ServerCaptureContract,
    resolve_server_capture_contract,
)

REGISTRY = builtin_algorithm_registry()
GEMMA4_SNAPSHOT_ID = "google/gemma-4-E4B-it"
CONTRACT_PATCH = "specforge.training.capture_contract.resolve_server_capture_contract"
CLONE_SENTINEL = "aux_hidden_states.append(hidden_states.clone())"
MASK_SENTINEL = "and vision_bidirectional"
DECODE_SENTINEL = "gpu_image_decode = False"


def _gemma4_snapshot():
    """The cached Gemma 4 E4B snapshot directory, or None."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            GEMMA4_SNAPSHOT_ID,
            allow_patterns=["*.json", "*.jinja"],
            local_files_only=True,
        )
    except Exception:
        return None


def _contract(*architectures, layers=(1, 10, 20, 30, 39)):
    return ServerCaptureContract(
        method="dflash",
        aux_layer_ids=layers,
        target_hidden_size=2560,
        target_vocab_size=262144,
        draft_vocab_size=262144,
        target_architectures=architectures,
    )


GEMMA4 = _contract("Gemma4ForConditionalGeneration")
QWEN35 = _contract("Qwen3_5ForConditionalGeneration", layers=(1, 8, 15, 22, 29))


def _managed_config(control_dir, *, servers=None, **model):
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft",
                **model,
            },
            "data": {"prompts_path": "prompts.jsonl"},
            "training": {"strategy": "dflash", "max_steps": 1},
            "deployment": {
                "mode": "disaggregated",
                "trainer": {"nnodes": 1, "nproc_per_node": 1},
                "disaggregated": {
                    "control_dir": control_dir,
                    "backend": "mooncake",
                    "managed_local": {
                        "trainer_cuda_visible_devices": ["1"],
                        "capture_servers": servers
                        or [{"port": 30000, "cuda_visible_devices": ["0"]}],
                    },
                },
            },
        }
    )


def _plan(cfg, contract):
    with mock.patch(CONTRACT_PATCH, return_value=contract):
        return build_launch_plan(
            cfg,
            algorithm=REGISTRY.resolve("dflash"),
            config_path="run.yaml",
            env={},
        )


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


class Gemma4CaptureLaunchGuardTest(unittest.TestCase):
    def setUp(self):
        self._root = tempfile.TemporaryDirectory()
        self.control_dir = os.path.join(self._root.name, "attempt")

    def tearDown(self):
        self._root.cleanup()

    def test_default_flashinfer_backend_is_refused_for_gemma4(self):
        cfg = _managed_config(self.control_dir)
        self.assertEqual(cfg.model.sglang_attention_backend, "flashinfer")
        with self.assertRaisesRegex(ValueError, "model.sglang_attention_backend"):
            _plan(cfg, GEMMA4)
        servers = [
            {"port": 30000, "cuda_visible_devices": ["0"]},
            {
                "port": 30001,
                "cuda_visible_devices": ["2"],
                "attention_backend": "fa3",
            },
        ]
        cfg = _managed_config(
            self.control_dir, servers=servers, sglang_attention_backend="triton"
        )
        with self.assertRaisesRegex(ValueError, "capture server 1 .* 'fa3'"):
            _plan(cfg, GEMMA4)

    def test_triton_and_trtllm_mha_are_accepted(self):
        cases = {
            "model triton": ({"sglang_attention_backend": "triton"}, None),
            "server trtllm_mha": (
                {},
                [
                    {
                        "port": 30000,
                        "cuda_visible_devices": ["0"],
                        "attention_backend": "trtllm_mha",
                    }
                ],
            ),
        }
        for name, (model, servers) in cases.items():
            with self.subTest(case=name):
                cfg = _managed_config(self.control_dir, servers=servers, **model)
                plan = _plan(cfg, GEMMA4)
                argv = plan.services[1].command.argv
                self.assertIn(
                    _flag(argv, "--attention-backend"), ("triton", "trtllm_mha")
                )
                self.assertIsNone(_flag(argv, "--mm-attention-backend"))

    def test_gemma4_image_capture_needs_the_vision_backend(self):
        # only sdpa applies the vision tower's padding-patch mask
        for mm_backend in (None, "fa3", "triton_attn"):
            with self.subTest(mm_backend=mm_backend):
                cfg = _managed_config(
                    self.control_dir,
                    input_modality="image",
                    sglang_attention_backend="triton",
                    sglang_mm_attention_backend=mm_backend,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    rf"sglang_mm_attention_backend 'sdpa' \(got {mm_backend!r}\)",
                ):
                    _plan(cfg, GEMMA4)

        cfg = _managed_config(
            self.control_dir,
            input_modality="image",
            sglang_attention_backend="triton",
            sglang_mm_attention_backend="sdpa",
        )
        argv = _plan(cfg, GEMMA4).services[1].command.argv
        self.assertEqual(_flag(argv, "--attention-backend"), "triton")
        self.assertEqual(_flag(argv, "--mm-attention-backend"), "sdpa")
        self.assertEqual(
            argv[argv.index("--attention-backend") + 2], "--mm-attention-backend"
        )

    def test_gemma4_image_capture_needs_triton(self):
        # SGLang runs Gemma4 on trtllm_mha too, but the regen and serving run
        # triton, the only backend that applies the per-image bidirectional
        # attention of a use_bidirectional_attention="vision" target
        cases = {
            "model trtllm_mha": ({"sglang_attention_backend": "trtllm_mha"}, None, 0),
            "server trtllm_mha": (
                {"sglang_attention_backend": "triton"},
                [
                    {"port": 30000, "cuda_visible_devices": ["0"]},
                    {
                        "port": 30001,
                        "cuda_visible_devices": ["2"],
                        "attention_backend": "trtllm_mha",
                    },
                ],
                1,
            ),
        }
        for name, (model, servers, index) in cases.items():
            with self.subTest(case=name):
                cfg = _managed_config(
                    self.control_dir,
                    servers=servers,
                    input_modality="image",
                    sglang_mm_attention_backend="sdpa",
                    **model,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    f"capture server {index} .* 'trtllm_mha', but the regen and "
                    "serving servers run triton",
                ):
                    _plan(cfg, GEMMA4)

        cfg = _managed_config(
            self.control_dir,
            input_modality="text",
            sglang_attention_backend="trtllm_mha",
        )
        argv = _plan(cfg, GEMMA4).services[1].command.argv
        self.assertEqual(_flag(argv, "--attention-backend"), "trtllm_mha")

    def test_gemma4_plan_requires_the_gemma4_sglang_patch(self):
        cfg = _managed_config(
            self.control_dir,
            input_modality="image",
            sglang_attention_backend="triton",
            sglang_mm_attention_backend="sdpa",
        )
        plan = _plan(cfg, GEMMA4)
        self.assertEqual(plan.required_sglang_patches, ("gemma4-spec",))
        self.assertEqual(
            json.loads(plan.render())["required_sglang_patches"], ["gemma4-spec"]
        )

    def test_other_targets_keep_their_plan(self):
        cfg = _managed_config(self.control_dir, input_modality="image")
        plan = _plan(cfg, QWEN35)
        argv = plan.services[1].command.argv
        self.assertEqual(_flag(argv, "--attention-backend"), "flashinfer")
        self.assertNotIn("--mm-attention-backend", argv)
        self.assertEqual(plan.required_sglang_patches, ())
        self.assertNotIn("required_sglang_patches", json.loads(plan.render()))


class ModelConfigFieldTest(unittest.TestCase):
    def test_mm_attention_backend_becomes_a_flag_only_when_set(self):
        argv = _sglang_argv(ModelConfig(target_model_path="target"))
        self.assertNotIn("--mm-attention-backend", argv)
        argv = _sglang_argv(
            ModelConfig(target_model_path="target", sglang_mm_attention_backend="sdpa")
        )
        self.assertEqual(_flag(argv, "--mm-attention-backend"), "sdpa")

    def test_target_embedding_scale_is_an_optional_positive_override(self):
        default = ModelConfig(target_model_path="target")
        self.assertIsNone(default.target_embedding_scale)
        for invalid in (0, -1.0):
            with self.subTest(scale=invalid):
                with self.assertRaises(ValidationError):
                    ModelConfig(
                        target_model_path="target", target_embedding_scale=invalid
                    )
        with tempfile.TemporaryDirectory() as root:
            cfg = apply_overrides(
                _managed_config(os.path.join(root, "attempt")),
                [
                    "model.target_embedding_scale=1.0",
                    "model.sglang_mm_attention_backend=sdpa",
                ],
            )
        self.assertEqual(cfg.model.target_embedding_scale, 1.0)
        self.assertEqual(cfg.model.sglang_mm_attention_backend, "sdpa")


def _fake_sglang(root, *, clone=True, mask=True, decode=True):
    package = os.path.join(root, "sglang")
    files = {
        # the spec-capture patch's split-storage fix, checked first
        ("srt", "managers", "mm_utils.py"): "def _own_split_storage(items):\n",
        ("srt", "models", "gemma4_causal.py"): (
            f"        {CLONE_SENTINEL}\n"
            if clone
            else "        aux_hidden_states.append(hidden_states)\n"
        ),
        ("srt", "models", "gemma4_mm.py"): (
            f"            {MASK_SENTINEL}\n"
            if mask
            else "            and forward_batch.contains_image_inputs()\n"
        ),
        ("srt", "multimodal", "processors", "gemma4.py"): (
            f"    {DECODE_SENTINEL}\n" if decode else "    models = []\n"
        ),
    }
    for parts, body in files.items():
        path = os.path.join(package, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(body)
    return package


class Gemma4SGLangPatchPreflightTest(unittest.TestCase):
    def setUp(self):
        self._root = tempfile.TemporaryDirectory()
        self.root = self._root.name

    def tearDown(self):
        self._root.cleanup()

    def _preflight(self, *, package, patches=("gemma4-spec",), env=None):
        def find_spec(name):
            if name == "sglang":
                return SimpleNamespace(
                    submodule_search_locations=[package] if package else []
                )
            return object()

        plan = LaunchPlan(
            "managed_supervisor",
            "both",
            managed_root=os.path.join(self.root, "attempt"),
            required_sglang_patches=patches,
        )
        with (
            mock.patch(
                "specforge.launch_plan.shutil.which",
                return_value="/usr/bin/mooncake_master",
            ),
            mock.patch(
                "specforge.launch_plan.importlib.util.find_spec",
                side_effect=find_spec,
            ),
            mock.patch.dict(os.environ, env or {}, clear=False),
        ):
            if not env:
                os.environ.pop("SPECFORGE_ALLOW_UNPATCHED_GEMMA4", None)
            launch_plan._managed_preflight(plan)

    def test_unpatched_sglang_fails_for_a_gemma4_target(self):
        package = _fake_sglang(self.root, clone=False, mask=False, decode=False)
        with self.assertRaisesRegex(RuntimeError, "gemma4-spec") as caught:
            self._preflight(package=package)
        message = str(caught.exception)
        self.assertIn("srt/models/gemma4_causal.py", message)
        self.assertIn("srt/models/gemma4_mm.py", message)
        self.assertIn("srt/multimodal/processors/gemma4.py", message)
        self.assertIn("SPECFORGE_ALLOW_UNPATCHED_GEMMA4=1", message)

    def test_each_sentinel_is_required(self):
        for clone, mask, decode, missing in (
            (False, True, True, "srt/models/gemma4_causal.py"),
            (True, False, True, "srt/models/gemma4_mm.py"),
            (True, True, False, "srt/multimodal/processors/gemma4.py"),
        ):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as root:
                package = _fake_sglang(root, clone=clone, mask=mask, decode=decode)
                with self.assertRaisesRegex(RuntimeError, missing):
                    self._preflight(package=package)

    def test_patched_sglang_passes(self):
        self._preflight(package=_fake_sglang(self.root))

    def test_the_escape_hatch_waives_the_check(self):
        package = _fake_sglang(self.root, clone=False, mask=False, decode=False)
        self._preflight(
            package=package, env={"SPECFORGE_ALLOW_UNPATCHED_GEMMA4": "1"}
        )

    def test_other_targets_do_not_need_the_patch(self):
        package = _fake_sglang(self.root, clone=False, mask=False, decode=False)
        self._preflight(package=package, patches=())

    def test_an_unlocatable_sglang_counts_as_unpatched(self):
        with self.assertRaisesRegex(RuntimeError, "gemma4-spec"):
            self._preflight(package=None)


def _resolve_contract(target_config, draft_payload, *, strategy="dflash"):
    cfg = Config.model_validate(
        {
            "model": {"target_model_path": "target/gemma4"},
            "data": {"train_data_path": "train.jsonl"},
            "training": {"strategy": strategy, "role": "producer", "total_steps": 2},
            "deployment": {
                "mode": "disaggregated",
                "disaggregated": {
                    "control_dir": "outputs/test/control",
                    "backend": "mooncake",
                    "server_urls": ["http://capture"],
                },
            },
        }
    )
    with (
        mock.patch(
            "transformers.AutoConfig.from_pretrained", return_value=target_config
        ),
        mock.patch(
            "specforge.training.model_loading.draft_config_dict",
            return_value=draft_payload,
        ),
    ):
        return resolve_server_capture_contract(
            cfg, algorithm=REGISTRY.resolve(strategy)
        )


def _gemma4_like_target(num_layers=42):
    return SimpleNamespace(
        architectures=["Gemma4ForConditionalGeneration"],
        text_config=SimpleNamespace(
            hidden_size=2560, vocab_size=262144, num_hidden_layers=num_layers
        ),
    )


def _dflash_payload(layers):
    return {
        "architectures": ["DFlashDraftModel"],
        "vocab_size": 262144,
        "dflash_config": {"mask_token_id": 4, "target_layer_ids": list(layers)},
    }


class CaptureLayerRangeTest(unittest.TestCase):
    def test_in_range_ids_resolve_with_the_target_architecture(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            contract = _resolve_contract(
                _gemma4_like_target(), _dflash_payload([1, 10, 20, 30, 39])
            )
        self.assertEqual(contract.aux_layer_ids, (1, 10, 20, 30, 39))
        self.assertEqual(
            contract.target_architectures, ("Gemma4ForConditionalGeneration",)
        )

    def test_ids_past_the_target_depth_are_refused(self):
        with self.assertRaisesRegex(ValueError, r"\[42\] are out of range"):
            _resolve_contract(
                _gemma4_like_target(), _dflash_payload([1, 10, 20, 30, 42])
            )
        eagle3_payload = {
            "architectures": ["LlamaForCausalLMEagle3"],
            "vocab_size": 262144,
            "draft_vocab_size": 32000,
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 20, 42]},
        }
        with self.assertRaisesRegex(ValueError, "out of range"):
            _resolve_contract(_gemma4_like_target(), eagle3_payload, strategy="eagle3")

    def test_the_last_layer_warns(self):
        # recorded directly: assertWarns walks every loaded module's
        # __warningregistry__, which Transformers' lazy modules log about
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _resolve_contract(_gemma4_like_target(), _dflash_payload([1, 20, 41]))
        self.assertTrue(
            any("last decoder layer" in str(warning.message) for warning in caught)
        )

    def test_a_target_without_a_layer_count_is_not_range_checked(self):
        target = SimpleNamespace(hidden_size=64, vocab_size=128)
        contract = _resolve_contract(target, _dflash_payload([3, 700]))
        self.assertEqual(contract.aux_layer_ids, (3, 700))
        self.assertEqual(contract.target_architectures, ())


class ProcessorMaskTokenTest(unittest.TestCase):
    @staticmethod
    def _resolve(tokenizer, *, draft_mask=None):
        from specforge.algorithms.model_providers import _resolve_mask_token_id

        cfg = SimpleNamespace(model=SimpleNamespace(mask_token_id=None))
        dflash_config = {} if draft_mask is None else {"mask_token_id": draft_mask}
        draft = SimpleNamespace(
            config=SimpleNamespace(vocab_size=262144, dflash_config=dflash_config)
        )
        return _resolve_mask_token_id(cfg, draft, tokenizer)

    def test_a_processor_resolves_through_its_tokenizer(self):
        tokenizer = SimpleNamespace(
            mask_token_id=4, pad_token_id=0, eos_token_id=1, unk_token_id=3
        )
        self.assertEqual(self._resolve(SimpleNamespace(tokenizer=tokenizer)), 4)
        self.assertEqual(self._resolve(tokenizer), 4)
        # the draft config's id still wins
        self.assertEqual(
            self._resolve(SimpleNamespace(tokenizer=tokenizer), draft_mask=7), 7
        )

    def test_the_gemma4_processor_resolves_its_mask_token(self):
        snapshot = _gemma4_snapshot()
        if snapshot is None:
            self.skipTest(f"{GEMMA4_SNAPSHOT_ID} is not in the local HF cache")
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
        self.assertIsNone(getattr(processor, "mask_token_id", None))
        self.assertEqual(processor.tokenizer.mask_token_id, 4)
        self.assertEqual(self._resolve(processor), 4)


class DraftAutoGenerationTest(unittest.TestCase):
    @staticmethod
    def _resolve_draft(target_config, strategy):
        from specforge.training.model_loading import resolve_draft_config

        model = {"target_model_path": "target/gemma4"}
        if strategy == "eagle3":
            model["vocab_mapping_path"] = "/mapping.pt"
        cfg = Config.model_validate(
            {
                "model": model,
                "data": {"hidden_states_path": "/features"},
                "training": {"strategy": strategy},
            }
        )
        with mock.patch(
            "transformers.AutoConfig.from_pretrained", return_value=target_config
        ):
            return resolve_draft_config(
                cfg, provider=REGISTRY.resolve(strategy).providers.model.draft_config
            )

    def test_per_layer_type_rope_refuses_generation(self):
        try:
            from transformers import Gemma4TextConfig
        except ImportError:
            self.skipTest("transformers has no Gemma 4 support")
        target = Gemma4TextConfig(
            vocab_size=512,
            hidden_size=80,
            intermediate_size=128,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=["sliding_attention", "full_attention"] * 3,
        )
        for strategy in ("eagle3", "dflash", "mmflash"):
            with self.subTest(strategy=strategy):
                with self.assertRaisesRegex(
                    ValueError, "by layer type.*model.draft_model_config"
                ):
                    self._resolve_draft(target, strategy)

    def test_the_cached_gemma4_config_refuses_generation(self):
        snapshot = _gemma4_snapshot()
        if snapshot is None:
            self.skipTest(f"{GEMMA4_SNAPSHOT_ID} is not in the local HF cache")
        from transformers import AutoConfig

        target = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        with self.assertRaisesRegex(ValueError, "by layer type"):
            self._resolve_draft(target, "dflash")

    def test_a_single_rope_still_generates(self):
        from transformers import Qwen3Config

        target = Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=12,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
        draft = self._resolve_draft(target, "dflash")
        self.assertEqual(draft.architectures, ["DFlashDraftModel"])

    def test_flat_multimodal_rope_is_not_mistaken_for_per_layer(self):
        from specforge.training.model_loading import _rope_field_keyed_by_layer_type

        qwen35_text = SimpleNamespace(
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
                "rope_type": "default",
                "rope_theta": 10000000,
                "partial_rotary_factor": 0.25,
            },
        )
        self.assertIsNone(_rope_field_keyed_by_layer_type(qwen35_text))


class Gemma4SnapshotPlanTest(unittest.TestCase):
    """The guards against the real cached Gemma 4 config, without mocks."""

    def setUp(self):
        self.snapshot = _gemma4_snapshot()
        if self.snapshot is None:
            self.skipTest(f"{GEMMA4_SNAPSHOT_ID} is not in the local HF cache")
        self._root = tempfile.TemporaryDirectory()
        self.root = self._root.name

    def tearDown(self):
        self._root.cleanup()

    def _draft_json(self, layers):
        path = os.path.join(self.root, f"draft-{len(os.listdir(self.root))}.json")
        payload = {
            "architectures": ["MMFlashDraftModel"],
            "model_type": "qwen3",
            "auto_map": {"AutoModel": "mmflash.MMFlashDraftModel"},
            "block_size": 16,
            "dflash_config": {"mask_token_id": 4, "target_layer_ids": layers},
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 2560,
            "intermediate_size": 9728,
            "max_position_embeddings": 131072,
            "num_attention_heads": 32,
            "num_hidden_layers": 5,
            "num_key_value_heads": 8,
            "num_target_layers": 42,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000,
            "vocab_size": 262144,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def _config(self, layers, **model):
        return Config.model_validate(
            {
                "model": {
                    "target_model_path": self.snapshot,
                    "draft_model_config": self._draft_json(layers),
                    "input_modality": "image",
                    **model,
                },
                "data": {"train_data_path": "train.jsonl", "max_length": 4096},
                "training": {"strategy": "mmflash", "max_steps": 1},
                "deployment": {
                    "mode": "disaggregated",
                    "trainer": {"nnodes": 1, "nproc_per_node": 1},
                    "disaggregated": {
                        "control_dir": os.path.join(self.root, "attempt"),
                        "backend": "mooncake",
                        "managed_local": {
                            "trainer_cuda_visible_devices": ["1"],
                            "capture_servers": [
                                {"port": 30000, "cuda_visible_devices": ["0"]}
                            ],
                        },
                    },
                },
            }
        )

    def _plan(self, cfg):
        return build_launch_plan(
            cfg,
            algorithm=REGISTRY.resolve("mmflash"),
            config_path="run.yaml",
            env={},
        )

    def test_a_configured_gemma4_mmflash_run_plans_its_capture_server(self):
        cfg = self._config(
            [1, 10, 20, 30, 39],
            sglang_attention_backend="triton",
            sglang_mm_attention_backend="sdpa",
        )
        plan = self._plan(cfg)
        argv = plan.services[1].command.argv
        layer_index = argv.index("--spec-capture-aux-layer-ids")
        self.assertEqual(
            argv[layer_index + 1 : layer_index + 6], ("1", "10", "20", "30", "39")
        )
        self.assertEqual(_flag(argv, "--attention-backend"), "triton")
        self.assertEqual(_flag(argv, "--mm-attention-backend"), "sdpa")
        self.assertEqual(_flag(argv, "--context-length"), "4103")
        self.assertEqual(plan.required_sglang_patches, ("gemma4-spec",))

    def test_the_defaults_and_bad_layer_ids_are_refused(self):
        with self.assertRaisesRegex(ValueError, "model.sglang_attention_backend"):
            self._plan(self._config([1, 10, 20, 30, 39]))
        with self.assertRaisesRegex(ValueError, "out of range"):
            self._plan(
                self._config(
                    [1, 10, 20, 30, 42],
                    sglang_attention_backend="triton",
                    sglang_mm_attention_backend="sdpa",
                )
            )


if __name__ == "__main__":
    unittest.main()
