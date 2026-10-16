# coding=utf-8
"""Gemma 4 E4B-it draft configs, HPC recipes and their job scripts.

Weight-free and CPU-only: the draft JSONs are instantiated on the meta device,
the target is read through its cached tokenizer, config and safetensors header,
the recipes resolve through ``specforge train --plan``, and the launcher through
its DRY_RUN mode. Everything that needs the pinned target snapshot is skipped
when it is not cached.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ALGOS = ("eagle3", "dflash", "mmflash")
ARCHITECTURES = {
    "eagle3": "LlamaForCausalLMEagle3",
    "dflash": "DFlashDraftModel",
    "mmflash": "MMFlashDraftModel",
}
CAPTURE_LAYERS = {
    "eagle3": [1, 20, 38],
    "dflash": [1, 10, 20, 30, 39],
    "mmflash": [1, 10, 20, 30, 39],
}
# fc maps the concatenated captured layers (3 or 5 x 2560) back to 2560
FC_SHAPES = {"eagle3": (2560, 7680), "dflash": (2560, 12800), "mmflash": (2560, 12800)}
TARGET_LAYERS = 42
HIDDEN = 2560
VOCAB = 262144
MASK_TOKEN_ID = 4
EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
SPECFORGE_DATA = "/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge"
REGEN = (
    f"{SPECFORGE_DATA}/regen_data/"
    "gemma4-E4B-it_llava-ov15-1M-prompted_regen_first_turn.jsonl"
)
SMOKE_SUBSET = REGEN[: -len(".jsonl")] + ".smoke_every500.jsonl"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_mmflash_video_hpc.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_gemma4_spec_hpc.sh"


def _draft_path(algo: str) -> Path:
    return REPO_ROOT / "configs" / f"gemma4-e4b-{algo}.json"


def _recipe_path(algo: str) -> Path:
    return REPO_ROOT / "scripts" / "mmtraining_configs" / f"gemma4-e4b-{algo}_hpc.yaml"


def _draft(algo: str) -> dict:
    return json.loads(_draft_path(algo).read_text())


def _recipe(algo: str) -> dict:
    return yaml.safe_load(_recipe_path(algo).read_text())


def _capture_layers(draft: dict) -> list:
    if "eagle_config" in draft:
        return draft["eagle_config"]["eagle_aux_hidden_state_layer_ids"]
    return draft["dflash_config"]["target_layer_ids"]


# every recipe pins the same local snapshot (the regen's chat template and weights)
SNAPSHOT = _recipe("eagle3")["model"]["target_model_path"]
SNAPSHOT_CACHED = os.path.isfile(os.path.join(SNAPSHOT, "config.json"))


def _flag(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else None


def _flag_values(argv, flag):
    values = []
    for value in argv[argv.index(flag) + 1 :]:
        if value.startswith("--"):
            break
        values.append(value)
    return values


def _script_env(**values) -> dict:
    """A clean job-like environment with this interpreter's env first on PATH."""
    env_bin = os.path.dirname(sys.executable)
    env = {
        "HOME": os.environ.get("HOME", "/tmp"),
        "USER": os.environ.get("USER", "specforge"),
        "PATH": os.pathsep.join([env_bin, "/usr/bin", "/bin"]),
        "CONDA_PREFIX": sys.prefix,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for name in ("HF_HOME", "HF_HUB_OFFLINE"):
        if name in os.environ:
            env[name] = os.environ[name]
    env.update(values)
    return env


class GemmaDraftConfigTest(unittest.TestCase):
    def test_drafts_instantiate_on_meta_with_target_widths(self):
        import torch

        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

        for algo in ALGOS:
            with self.subTest(algo=algo):
                with contextlib.redirect_stdout(io.StringIO()):
                    config = AutoDraftModelConfig.from_file(str(_draft_path(algo)))
                with torch.device("meta"):
                    model = AutoDraftModel.from_config(config)
                self.assertEqual(type(model).__name__, ARCHITECTURES[algo])
                shapes = {
                    name: tuple(param.shape) for name, param in model.named_parameters()
                }
                self.assertEqual(shapes["fc.weight"], FC_SHAPES[algo])
                if algo == "eagle3":
                    self.assertEqual(shapes["embed_tokens.weight"], (VOCAB, HIDDEN))
                    self.assertEqual(shapes["lm_head.weight"], (32000, HIDDEN))

    def test_vocab_capture_layers_and_mask(self):
        for algo in ALGOS:
            with self.subTest(algo=algo):
                draft = _draft(algo)
                self.assertEqual(draft["vocab_size"], VOCAB)
                self.assertEqual(draft["hidden_size"], HIDDEN)
                layers = _capture_layers(draft)
                self.assertEqual(layers, CAPTURE_LAYERS[algo])
                self.assertLess(max(layers), TARGET_LAYERS)
                # SGLang captures the last layer before the final norm, HF after it
                self.assertNotIn(TARGET_LAYERS - 1, layers)
                if algo != "eagle3":
                    self.assertEqual(draft["dflash_config"]["mask_token_id"], MASK_TOKEN_ID)
                    self.assertEqual(draft["num_target_layers"], TARGET_LAYERS)
                    self.assertEqual(draft["block_size"], 16)

    def test_mmflash_draft_is_the_dflash_draft_but_its_class(self):
        dflash, mmflash = _draft("dflash"), _draft("mmflash")
        self.assertEqual(mmflash.pop("architectures"), ["MMFlashDraftModel"])
        self.assertEqual(mmflash.pop("auto_map"), {"AutoModel": "mmflash.MMFlashDraftModel"})
        del dflash["architectures"], dflash["auto_map"]
        self.assertEqual(mmflash, dflash)


@unittest.skipUnless(SNAPSHOT_CACHED, f"Gemma 4 snapshot not cached at {SNAPSHOT}")
class GemmaTargetSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoConfig, AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
        cls.text_config = AutoConfig.from_pretrained(
            SNAPSHOT, local_files_only=True
        ).text_config

    def test_mask_token_is_the_tokenizers(self):
        self.assertEqual(self.tokenizer.mask_token_id, MASK_TOKEN_ID)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(MASK_TOKEN_ID), "<mask>")
        for algo in ("dflash", "mmflash"):
            with self.subTest(algo=algo):
                self.assertEqual(
                    _draft(algo)["dflash_config"]["mask_token_id"],
                    self.tokenizer.mask_token_id,
                )
                self.assertEqual(
                    _recipe(algo)["model"]["mask_token_id"], self.tokenizer.mask_token_id
                )

    def test_drafts_match_the_target(self):
        text = self.text_config
        self.assertEqual(text.num_hidden_layers, TARGET_LAYERS)
        for algo in ALGOS:
            with self.subTest(algo=algo):
                draft = _draft(algo)
                self.assertEqual(draft["hidden_size"], text.hidden_size)
                self.assertEqual(draft["vocab_size"], text.vocab_size)
                self.assertLess(max(_capture_layers(draft)), text.num_hidden_layers)
                self.assertEqual(draft["bos_token_id"], self.tokenizer.bos_token_id)
                self.assertEqual(draft["pad_token_id"], self.tokenizer.pad_token_id)
                self.assertEqual(
                    draft["eos_token_id"], self.tokenizer.convert_tokens_to_ids("<turn|>")
                )

    def test_embedding_key_is_in_the_safetensors_header(self):
        from safetensors import safe_open

        with safe_open(os.path.join(SNAPSHOT, "model.safetensors"), framework="pt") as handle:
            self.assertIn(EMBEDDING_KEY, set(handle.keys()))
            shape = handle.get_slice(EMBEDDING_KEY).get_shape()
        self.assertEqual(list(shape), [VOCAB, HIDDEN])
        for algo in ALGOS:
            with self.subTest(algo=algo):
                self.assertEqual(_recipe(algo)["model"]["embedding_key"], EMBEDDING_KEY)


class GemmaRecipeTest(unittest.TestCase):
    def test_schema_has_the_mm_attention_field(self):
        # Plan C10. Without this ModelConfig field every Gemma recipe fails to
        # load (extra fields are forbidden), so it is checked on its own.
        from specforge.config.schema import ModelConfig

        self.assertIn(
            "sglang_mm_attention_backend",
            ModelConfig.model_fields,
            "ModelConfig.sglang_mm_attention_backend (plan C10) is missing; the "
            "Gemma recipes set it to sdpa",
        )

    def test_recipes_pin_the_gemma_capture(self):
        self.assertIn("models--google--gemma-4-E4B-it/snapshots/", SNAPSHOT)
        for algo in ALGOS:
            with self.subTest(algo=algo):
                recipe = _recipe(algo)
                model, data = recipe["model"], recipe["data"]
                self.assertEqual(model["target_model_path"], SNAPSHOT)
                self.assertEqual(model["draft_model_config"], f"configs/gemma4-e4b-{algo}.json")
                self.assertEqual(model["input_modality"], "image")
                self.assertTrue(model["trust_remote_code"])
                self.assertEqual(model["embedding_key"], EMBEDDING_KEY)
                # model level, so the launcher's capture_servers override keeps them
                self.assertEqual(model["sglang_attention_backend"], "triton")
                self.assertEqual(model["sglang_mm_attention_backend"], "sdpa")
                self.assertEqual(data["train_data_path"], REGEN)
                self.assertEqual(data["chat_template"], "gemma4")
                self.assertEqual(data["max_length"], 4096)
                self.assertEqual(data.get("image_max_tokens", 0), 0)
                self.assertEqual(recipe["training"]["strategy"], algo)
                run_id = f"gemma4-e4b-{algo}-llava-ov15-1M-prompted"
                self.assertEqual(recipe["run_id"], run_id)
                self.assertEqual(recipe["output_dir"], f"{SPECFORGE_DATA}/outputs/{run_id}")
                disaggregated = recipe["deployment"]["disaggregated"]
                self.assertEqual(disaggregated["control_dir"], f"{recipe['output_dir']}/control")
                self.assertEqual(
                    disaggregated["consumer_state_dir"],
                    f"{recipe['output_dir']}/consumer-state",
                )
                if algo == "eagle3":
                    self.assertEqual(
                        model["vocab_mapping_path"],
                        f"{SPECFORGE_DATA}/vocab_mapping/gemma4-e4b-eagle3.pt",
                    )
                else:
                    self.assertNotIn("vocab_mapping_path", model)
                    self.assertEqual(model["mask_token_id"], MASK_TOKEN_ID)
                if algo == "mmflash":
                    self.assertEqual(
                        data["visual_score_path"],
                        f"{SPECFORGE_DATA}/visual_kl/gemma4-E4B-it-llava-ov15-1M-prompted",
                    )
                    self.assertEqual(data["visual_score_transform"], "quantile")
                    self.assertTrue(data["visual_score_confidence_gate"])
                else:
                    self.assertNotIn("visual_score_path", data)

    def test_chat_template_is_registered(self):
        # only a prompt-cache label on the image path, but the text-path tools
        # (e.g. scripts/gates/select_overfit_sample.py) render with it
        from specforge.data.template import TEMPLATE_REGISTRY

        for algo in ALGOS:
            with self.subTest(algo=algo):
                template = TEMPLATE_REGISTRY.get(_recipe(algo)["data"]["chat_template"])
                self.assertEqual(template.assistant_header, "<|turn>model\n")

    def test_recipe_strategy_matches_the_draft_class(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry

        registry = builtin_algorithm_registry()
        for algo in ALGOS:
            with self.subTest(algo=algo):
                draft = _draft(algo)
                provider = registry.resolve(_recipe(algo)["training"]["strategy"])
                draft_config = provider.providers.model.draft_config
                self.assertEqual(draft["architectures"], [draft_config.architecture])
                if draft_config.expected_auto_map_model is not None:
                    self.assertEqual(
                        draft["auto_map"]["AutoModel"], draft_config.expected_auto_map_model
                    )


@unittest.skipUnless(SNAPSHOT_CACHED, f"Gemma 4 snapshot not cached at {SNAPSHOT}")
class GemmaRecipePlanTest(unittest.TestCase):
    """``specforge train --config <recipe> --plan``, with fresh run directories."""

    @classmethod
    def setUpClass(cls):
        from specforge.cli import main

        cls._tmp = tempfile.TemporaryDirectory(prefix="gemma4_plan_")
        cls.plans = {}
        for algo in ALGOS:
            out = os.path.join(cls._tmp.name, algo)
            argv = [
                "train",
                "--config",
                str(_recipe_path(algo)),
                "--plan",
                f"output_dir={out}",
                f"deployment.disaggregated.control_dir={out}/control",
                f"deployment.disaggregated.consumer_state_dir={out}/consumer-state",
            ]
            stdout = io.StringIO()
            # draft_model_config is relative to the repo root
            with contextlib.chdir(REPO_ROOT), contextlib.redirect_stdout(stdout):
                status = main(argv)
            if status != 0:
                raise AssertionError(f"--plan for {algo} exited {status}")
            cls.plans[algo] = json.loads(stdout.getvalue())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _capture_argv(self, algo):
        plan = self.plans[algo]
        self.assertEqual(plan["kind"], "managed_supervisor")
        servers = [s for s in plan["services"] if s["command"]["label"].startswith("capture-server")]
        self.assertEqual(len(servers), 1)
        return servers[0]["command"]["argv"]

    def test_capture_server_runs_gemma_on_triton(self):
        for algo in ALGOS:
            with self.subTest(algo=algo):
                argv = self._capture_argv(algo)
                self.assertEqual(_flag(argv, "--attention-backend"), "triton")
                self.assertEqual(_flag(argv, "--model-path"), SNAPSHOT)
                self.assertIn("--trust-remote-code", argv)

    def test_capture_server_runs_vision_on_sdpa(self):
        # Needs ModelConfig.sglang_mm_attention_backend (plan C10), which every
        # sglang_* field turns into a capture-server flag.
        for algo in ALGOS:
            with self.subTest(algo=algo):
                self.assertEqual(
                    _flag(self._capture_argv(algo), "--mm-attention-backend"), "sdpa"
                )

    def test_capture_layers_and_context(self):
        for algo in ALGOS:
            with self.subTest(algo=algo):
                argv = self._capture_argv(algo)
                # EAGLE3 image capture rides the dflash capture method by design
                self.assertEqual(_flag(argv, "--spec-capture-method"), "dflash")
                self.assertEqual(
                    _flag_values(argv, "--spec-capture-aux-layer-ids"),
                    [str(layer) for layer in CAPTURE_LAYERS[algo]],
                )
                # max_length 4096 + SGLang's 7-slot request headroom
                self.assertEqual(_flag(argv, "--context-length"), "4103")
                self.assertEqual(_flag(argv, "--chunked-prefill-size"), "-1")

    def test_trainer_topology_follows_the_qwen_recipes(self):
        expected = {"eagle3": ("1,2,3", "3"), "dflash": ("1,2,3", "3"), "mmflash": ("1", None)}
        for algo, (devices, nproc) in expected.items():
            with self.subTest(algo=algo):
                consumer = self.plans[algo]["commands"][1]
                self.assertEqual(consumer["env"]["CUDA_VISIBLE_DEVICES"], devices)
                self.assertEqual(_flag(consumer["argv"], "--nproc_per_node"), nproc)


class JobScriptSyntaxTest(unittest.TestCase):
    def test_scripts_parse(self):
        for script in (TRAIN_SCRIPT, EVAL_SCRIPT):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)], capture_output=True, text=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(SNAPSHOT_CACHED, f"Gemma 4 snapshot not cached at {SNAPSHOT}")
class TrainLauncherDryRunTest(unittest.TestCase):
    """scripts/train_mmflash_video_hpc.sh with DRY_RUN=1 on the EAGLE3 recipe,
    whose 3 trainers make the nproc_per_node override visible either way."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="gemma4_launcher_")
        self.root = Path(self._tmp.name)
        # never reached by a dry run; if one ever starts the stack, it stops here
        guard = self.root / "bin"
        guard.mkdir()
        (guard / "mooncake_master").write_text("#!/bin/sh\nexit 3\n")
        (guard / "mooncake_master").chmod(0o755)
        self.guard = str(guard)
        self.out = self.root / "run"

    def tearDown(self):
        self._tmp.cleanup()

    def _dry_run(self, visible_devices):
        extra = " ".join(
            [
                f"data.train_data_path={SMOKE_SUBSET}",
                "training.num_epochs=1",
                "training.log_interval=10",
                "run_id=gemma4-e4b-eagle3-dry-run",
                f"output_dir={self.out}",
                f"deployment.disaggregated.control_dir={self.out}/control",
                f"deployment.disaggregated.consumer_state_dir={self.out}/consumer-state",
            ]
        )
        env = _script_env(
            DRY_RUN="1",
            CONFIG=str(_recipe_path("eagle3")),
            CUDA_VISIBLE_DEVICES=visible_devices,
            EXTRA_OVERRIDES=extra,
            SPECFORGE_CACHE_ROOT=str(self.root / "cache"),
        )
        env["PATH"] = self.guard + os.pathsep + env["PATH"]
        result = subprocess.run(
            ["bash", str(TRAIN_SCRIPT)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
        self.assertIn("[train] dry run:", result.stdout)
        # nothing moved, created or started
        self.assertFalse(self.out.exists())
        self.assertFalse((self.root / "cache").exists())
        plan = json.loads(result.stdout[result.stdout.index("\n{\n") + 1 :])
        producer, consumer = plan["commands"]
        capture = plan["services"][1]["command"]
        for override in extra.split():
            self.assertIn(override, producer["argv"])
        self.assertEqual(plan["managed_ports"], [45000, 45001, 45002, 31000])
        self.assertEqual(_flag(capture["argv"], "--attention-backend"), "triton")
        return producer, consumer, capture

    def test_four_gpus_capture_on_the_first_and_train_on_three(self):
        uuids = [f"GPU-{index}a0b1c2d-0000-4000-8000-00000000000{index}" for index in range(4)]
        producer, consumer, capture = self._dry_run(",".join(uuids))
        self.assertEqual(capture["env"]["CUDA_VISIBLE_DEVICES"], uuids[0])
        self.assertEqual(_flag(capture["argv"], "--mem-fraction-static"), "0.8")
        self.assertEqual(consumer["env"]["CUDA_VISIBLE_DEVICES"], ",".join(uuids[1:]))
        self.assertEqual(_flag(consumer["argv"], "--nproc_per_node"), "3")
        self.assertIn("deployment.trainer.nproc_per_node=3", producer["argv"])
        self.assertNotIn(
            "deployment.disaggregated.managed_local.share_trainer_gpu=true",
            producer["argv"],
        )

    def test_one_gpu_is_shared_by_capture_and_one_trainer(self):
        uuid = "GPU-9f8e7d6c-0000-4000-8000-000000000000"
        producer, consumer, capture = self._dry_run(uuid)
        self.assertEqual(capture["env"]["CUDA_VISIBLE_DEVICES"], uuid)
        self.assertEqual(consumer["env"]["CUDA_VISIBLE_DEVICES"], uuid)
        # SHARED_CAPTURE_MEM's default leaves the rest of the card to the trainer
        self.assertEqual(_flag(capture["argv"], "--mem-fraction-static"), "0.3")
        self.assertNotIn("--nproc_per_node", consumer["argv"])
        self.assertIn("deployment.trainer.nproc_per_node=1", producer["argv"])
        self.assertIn(
            "deployment.disaggregated.managed_local.share_trainer_gpu=true",
            producer["argv"],
        )


class EvalScriptExportCheckTest(unittest.TestCase):
    """scripts/eval_gemma4_spec_hpc.sh refuses a wrong export before it touches
    SGLang. It runs from a copy whose benchmark_helper.sh is a stub, so the
    installed SGLang is never patched and no server is started."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="gemma4_eval_")
        self.root = Path(self._tmp.name)
        (self.root / "scripts").mkdir()
        (self.root / "benchmarks").mkdir()
        (self.root / "benchmarks" / "bench_mm.py").write_text("")
        shutil.copy(EVAL_SCRIPT, self.root / "scripts" / EVAL_SCRIPT.name)
        self.helper_called = self.root / "helper_called"
        (self.root / "scripts" / "benchmark_helper.sh").write_text(
            f"#!/bin/bash\ntouch '{self.helper_called}'\nexit 97\n"
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _export(self, algo, *, normalize):
        export = self.root / f"export-{algo}"
        export.mkdir()
        shutil.copy(_draft_path(algo), export / "config.json")
        if normalize:
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "gates" / "normalize_dflash_export.py"),
                    "--config",
                    str(export / "config.json"),
                    "--block-size",
                    "16",
                ],
                check=True,
            )
        return export

    def _run(self, **values):
        env = _script_env(SPECFORGE_CACHE_ROOT=str(self.root / "cache"), PORT="65531", **values)
        return subprocess.run(
            ["bash", str(self.root / "scripts" / EVAL_SCRIPT.name)],
            cwd=self.root,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )

    def test_unnormalized_dflash_family_export_is_refused(self):
        result = self._run(ALGO="dflash", DRAFT=str(self._export("mmflash", normalize=False)))
        self.assertEqual(result.returncode, 1)
        self.assertIn("normalize_dflash_export.py", result.stderr)
        self.assertFalse(self.helper_called.exists())

    def test_valid_exports_reach_the_sglang_helper(self):
        for algo, normalize, served_as in (
            ("mmflash", True, "dflash"),
            ("eagle3", False, "eagle3"),
        ):
            with self.subTest(algo=algo):
                if self.helper_called.exists():
                    self.helper_called.unlink()
                result = self._run(ALGO=served_as, DRAFT=str(self._export(algo, normalize=normalize)))
                # the stub helper fails the job right after the export check
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertTrue(self.helper_called.exists(), result.stdout + result.stderr)
                self.assertIn(f"target layers {CAPTURE_LAYERS[algo]}", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
