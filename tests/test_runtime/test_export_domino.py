# coding=utf-8
"""Domino / MMDomino checkpoints export to SGLang DFLASH draft directories.

A tiny ``DominoDraftModel`` is wrapped in the real online training model, its
checkpoint is written the way the trainer writes it (strategy checkpoint
filter + provider resume contract + ``CheckpointManager``), and exported.

``--to sglang`` must produce what SGLang's Domino DFLASH loader reads, the
layout of the Domino authors' checkpoints: ``architectures ==
["DFlashDraftModel"]``, the Domino head in ``dflash_config`` and the trained
``prefix_gru`` / ``embed_proj`` weights under their module names. ``--to hf``
stays the SpecForge reload format, and DFlash / MMFlash exports are untouched.
All of it runs on CPU.
"""

import hashlib
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch import nn

HIDDEN, VOCAB, GRU, EMB, BLOCK, MASK = 64, 256, 32, 16, 4, 7
LAYER_IDS = [1, 5]

#: dflash_config keys of the Domino authors' SGLang checkpoints
#: (Qwen3-4B-Domino-b16, Qwen3-8B-Domino-b16, qwen3.6-27B-domino).
AUTHOR_DFLASH_CONFIG_KEYS = {
    "projector_type",
    "mask_token_id",
    "shift_label",
    "target_layer_ids",
    "pure_draft_prefix_len",
    "gru_hidden_dim",
    "emb_dim",
}

DOMINO_HEAD = {
    "projector_type": "domino",
    "pure_draft_prefix_len": 1,
    "emb_dim": EMB,
    "gru_hidden_dim": GRU,
}


def draft_config(architecture, auto_map, **method):
    return {
        "architectures": [architecture],
        "auto_map": {"AutoModel": auto_map},
        "attention_bias": False,
        "block_size": BLOCK,
        "dtype": "bfloat16",
        "head_dim": 16,
        "hidden_act": "silu",
        "hidden_size": HIDDEN,
        "intermediate_size": 128,
        "max_position_embeddings": 512,
        "model_type": "qwen3",
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_target_layers": 8,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000000,
        "tie_word_embeddings": True,
        "vocab_size": VOCAB,
        "dflash_config": {
            "mask_token_id": MASK,
            "target_layer_ids": LAYER_IDS,
            **method,
        },
    }


def domino_config(shift_label, **overrides):
    method = {**DOMINO_HEAD, "shift_label": shift_label, **overrides}
    return draft_config("DominoDraftModel", "domino.DominoDraftModel", **method)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def read_safetensors(path):
    with safe_open(path, framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def tree_digest(root):
    digests = {}
    for name in sorted(os.listdir(root)):
        with open(os.path.join(root, name), "rb") as handle:
            digests[name] = hashlib.sha256(handle.read()).hexdigest()
    return digests


def save_trainer_checkpoint(workdir, config_path, strategy_name, **contract_overrides):
    """Write a run directory exactly as training does; returns (run_dir, draft state)."""
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
    from specforge.training.checkpoint import CheckpointManager

    torch.manual_seed(0)
    draft = AutoDraftModel.from_config(
        AutoDraftModelConfig.from_file(config_path), torch_dtype=torch.bfloat16
    )
    with torch.no_grad():  # trained weights differ from any fresh init
        for parameter in draft.parameters():
            parameter.add_(torch.randn_like(parameter))
    head = nn.Linear(HIDDEN, VOCAB, bias=False).to(torch.bfloat16)
    embed = nn.Embedding(VOCAB, HIDDEN).to(torch.bfloat16)
    common = dict(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=MASK,
        block_size=int(draft.block_size),
        attention_backend="sdpa",
        num_anchors=8,
    )
    run_config = SimpleNamespace(
        training=SimpleNamespace(lambda_base_start=1.0, lambda_base_decay_ratio=0.5)
    )
    if strategy_name == "domino":
        from specforge.algorithms.common.dflash_family_model import OnlineDominoModel
        from specforge.algorithms.domino.providers import resume_contract
        from specforge.training.strategies.base import DominoTrainStrategy

        model = OnlineDominoModel(**common, shift_label=draft.shift_label)
        strategy = DominoTrainStrategy(model)
    elif strategy_name == "mmdomino":
        from specforge.algorithms.common.mmdomino_model import OnlineMMDominoModel
        from specforge.algorithms.mmdomino.providers import resume_contract
        from specforge.training.strategies.base import MMDominoTrainStrategy

        model = OnlineMMDominoModel(**common, shift_label=draft.shift_label)
        strategy = MMDominoTrainStrategy(model)
    elif strategy_name == "dflash":
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
        from specforge.training.strategies.base import DFlashTrainStrategy

        model = OnlineDFlashModel(**common, loss_type="dflash")
        strategy = DFlashTrainStrategy(model)
        resume_contract = None  # the export reads no DFlash contract key
    else:
        from specforge.algorithms.common.mmflash_model import OnlineMMFlashModel
        from specforge.training.strategies.base import MMFlashTrainStrategy

        model = OnlineMMFlashModel(**common, loss_type="dflash")
        strategy = MMFlashTrainStrategy(model)
        resume_contract = None
    draft_state = strategy.checkpoint_state_filter(model.state_dict())
    shared = {"draft_state_dict": draft_state, "strategy": strategy.name}
    if resume_contract is not None:
        shared.update(resume_contract(run_config, draft, model))
    shared.update(contract_overrides)
    run_dir = os.path.join(workdir, f"{strategy_name}-run")
    CheckpointManager(run_dir, "tiny").save(shared, 2)
    return run_dir, draft_state


class _ExportCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="domino_export_")
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name

    def fresh_root(self):
        """A new empty work directory, for each subtest of a loop."""
        self.root = tempfile.mkdtemp(dir=self._tmp.name)

    def path(self, *parts):
        return os.path.join(self.root, *parts)


class TestDominoSGLangExport(_ExportCase):
    def export(self, strategy, shift_label):
        from specforge.export import export_to_sglang

        config_path = write_json(self.path("draft.json"), domino_config(shift_label))
        run_dir, trained = save_trainer_checkpoint(self.root, config_path, strategy)
        out = export_to_sglang(run_dir, config_path, self.path("sglang"))
        return out, trained

    def test_exports_the_sglang_dflash_domino_layout(self):
        for strategy, shift_label in (("domino", True), ("mmdomino", False)):
            with self.subTest(strategy=strategy, shift_label=shift_label):
                self.fresh_root()
                out, trained = self.export(strategy, shift_label)
                self.assertEqual(
                    sorted(os.listdir(out)), ["config.json", "model.safetensors"]
                )
                with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
                    config = json.load(handle)

                self.assertEqual(config["architectures"], ["DFlashDraftModel"])
                self.assertNotIn("auto_map", config)
                self.assertEqual(
                    config["dflash_config"],
                    {
                        "projector_type": "domino",
                        "shift_label": shift_label,
                        "pure_draft_prefix_len": 1,
                        "gru_hidden_dim": GRU,
                        "emb_dim": EMB,
                        "mask_token_id": MASK,
                        "target_layer_ids": LAYER_IDS,
                    },
                )
                self.assertEqual(
                    set(config["dflash_config"]), AUTHOR_DFLASH_CONFIG_KEYS
                )
                self.assertEqual(config["block_size"], BLOCK)
                self.assertEqual(config["num_hidden_layers"], 2)
                self.assertEqual(config["num_target_layers"], 8)
                self.assertEqual(config["vocab_size"], VOCAB)
                self.assertEqual(config["dtype"], "bfloat16")

                served = read_safetensors(os.path.join(out, "model.safetensors"))
                self.assertEqual(set(served), set(trained))
                # the shapes SGLang's DFlashDraftModel builds the Domino head with
                expected_head = {
                    "prefix_gru.weight_ih_l0": (3 * GRU, HIDDEN),
                    "prefix_gru.weight_hh_l0": (3 * GRU, GRU),
                    "embed_proj.0.weight": (EMB, HIDDEN + GRU),
                    "embed_proj.2.weight": (VOCAB, EMB),
                }
                for key, shape in expected_head.items():
                    self.assertEqual(tuple(served[key].shape), shape, msg=key)
                for key, value in trained.items():
                    self.assertEqual(served[key].dtype, torch.bfloat16, msg=key)
                    self.assertTrue(torch.equal(served[key], value), msg=key)
                self.assertFalse(
                    [k for k in served if k.startswith(("model.", "draft_model."))]
                )
                self.assertFalse([k for k in served if "embed_tokens" in k])
                self.assertFalse([k for k in served if "lm_head" in k])

    def test_cli_exports_domino_to_sglang(self):
        from specforge.cli import main

        config_path = write_json(self.path("draft.json"), domino_config(True))
        run_dir, _ = save_trainer_checkpoint(self.root, config_path, "domino")
        out = self.path("cli_export")
        self.assertEqual(
            main(
                [
                    "export",
                    "--to",
                    "sglang",
                    "--checkpoint",
                    run_dir,
                    "--draft-config",
                    config_path,
                    "--output-dir",
                    out,
                ]
            ),
            0,
        )
        with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["architectures"], ["DFlashDraftModel"])

    def test_hf_export_plus_normalize_serves_the_same_draft(self):
        import importlib.util

        from specforge.export import export_to_hf, export_to_sglang

        spec = importlib.util.spec_from_file_location(
            "normalize_dflash_export",
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "scripts",
                "gates",
                "normalize_dflash_export.py",
            ),
        )
        normalize = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(normalize)

        config_path = write_json(self.path("draft.json"), domino_config(True))
        run_dir, _ = save_trainer_checkpoint(self.root, config_path, "mmdomino")
        sglang_out = export_to_sglang(run_dir, config_path, self.path("sglang"))
        hf_out = export_to_hf(run_dir, config_path, self.path("hf"))
        normalized = normalize.normalize_export(
            os.path.join(hf_out, "config.json"), BLOCK
        )
        with open(os.path.join(sglang_out, "config.json"), encoding="utf-8") as handle:
            served = json.load(handle)

        self.assertEqual(normalized, served)
        self.assertEqual(
            tree_digest(sglang_out)["model.safetensors"],
            tree_digest(hf_out)["model.safetensors"],
        )

    def test_hf_export_still_reloads_as_a_domino_draft(self):
        from specforge.export import export_to_hf
        from specforge.modeling.auto import AutoDraftModel
        from specforge.modeling.draft.domino import DominoDraftModel

        config_path = write_json(self.path("draft.json"), domino_config(True))
        run_dir, trained = save_trainer_checkpoint(self.root, config_path, "domino")
        out = export_to_hf(run_dir, config_path, self.path("hf"))
        reloaded = AutoDraftModel.from_pretrained(out, torch_dtype=torch.bfloat16)

        self.assertIsInstance(reloaded, DominoDraftModel)
        self.assertTrue(reloaded.shift_label)
        state = reloaded.state_dict()
        for key, value in trained.items():
            self.assertTrue(torch.equal(state[key], value), msg=key)

    def test_draft_config_must_match_the_run(self):
        from specforge.export import export_to_sglang

        trained_config = write_json(self.path("trained.json"), domino_config(True))
        run_dir, _ = save_trainer_checkpoint(self.root, trained_config, "mmdomino")
        wrong = {
            "shift_label": domino_config(False),
            "mask_token_id": domino_config(True, mask_token_id=MASK + 1),
            "target_layer_ids": domino_config(True, target_layer_ids=[2, 6]),
        }
        for field, config in wrong.items():
            with self.subTest(field=field):
                config_path = write_json(self.path(f"{field}.json"), config)
                out = self.path(f"{field}_export")
                with self.assertRaisesRegex(ValueError, field):
                    export_to_sglang(run_dir, config_path, out)
                self.assertFalse(os.path.exists(out))

    def test_the_run_supplies_a_missing_mask_token_id(self):
        from specforge.export import export_to_sglang

        trained_config = write_json(self.path("trained.json"), domino_config(True))
        run_dir, _ = save_trainer_checkpoint(self.root, trained_config, "domino")
        config = domino_config(True)
        del config["dflash_config"]["mask_token_id"]
        config_path = write_json(self.path("no_mask.json"), config)
        out = export_to_sglang(run_dir, config_path, self.path("sglang"))
        with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["dflash_config"]["mask_token_id"], MASK)

    def test_refuses_what_sglang_cannot_serve(self):
        from specforge.export import export_to_sglang

        config_path = write_json(
            self.path("p0.json"), domino_config(True, pure_draft_prefix_len=0)
        )
        run_dir, _ = save_trainer_checkpoint(self.root, config_path, "domino")
        with self.assertRaisesRegex(ValueError, "pure_draft_prefix_len=1"):
            export_to_sglang(run_dir, config_path, self.path("p0_export"))
        self.assertFalse(os.path.exists(self.path("p0_export")))
        with self.assertRaisesRegex(ValueError, "vocab mapping"):
            export_to_sglang(
                run_dir,
                config_path,
                self.path("renamed"),
                weight_map={"fc.weight": "model.fc.weight"},
            )

    def test_a_checkpoint_without_the_domino_head_does_not_export(self):
        from specforge.export import export_to_hf, export_to_sglang

        config_path = write_json(self.path("draft.json"), domino_config(True))
        run_dir, trained = save_trainer_checkpoint(self.root, config_path, "domino")
        state_file = os.path.join(run_dir, "tiny-step2", "training_state.pt")
        state = torch.load(state_file, weights_only=False)
        for prefix in ("embed_proj.", "prefix_gru."):
            with self.subTest(dropped=prefix):
                state["draft_state_dict"] = {
                    k: v for k, v in trained.items() if not k.startswith(prefix)
                }
                torch.save(state, state_file)
                for export in (export_to_sglang, export_to_hf):
                    with self.assertRaisesRegex(ValueError, prefix.rstrip(".")):
                        export(run_dir, config_path, self.path("export"))


class TestDFlashFamilyExportsUnchanged(_ExportCase):
    def test_hf_export_is_the_plain_draft_save(self):
        from specforge.export import export_to_hf
        from specforge.export.checkpoint_io import (
            materialize_draft,
            resolve_training_state,
        )

        for strategy, architecture in (
            ("dflash", "DFlashDraftModel"),
            ("mmflash", "MMFlashDraftModel"),
        ):
            with self.subTest(strategy=strategy):
                self.fresh_root()
                source = draft_config(architecture, f"{strategy}.{architecture}")
                config_path = write_json(self.path("draft.json"), source)
                run_dir, _ = save_trainer_checkpoint(self.root, config_path, strategy)
                out = export_to_hf(run_dir, config_path, self.path("hf"))

                state = resolve_training_state(run_dir)
                # what the export wrote before Domino serving existed
                reference = materialize_draft(state, config_path)
                reference_state = dict(reference.state_dict())
                reference_state.update(state["draft_state_dict"])
                reference.save_pretrained(
                    self.path("reference"), state_dict=reference_state
                )
                self.assertEqual(tree_digest(out), tree_digest(self.path("reference")))
                with open(os.path.join(out, "config.json"), encoding="utf-8") as handle:
                    config = json.load(handle)
                self.assertEqual(config["architectures"], [architecture])
                self.assertEqual(config["auto_map"], source["auto_map"])
                self.assertEqual(config["dflash_config"], source["dflash_config"])

    def test_sglang_export_still_refuses_dflash_and_mmflash(self):
        from specforge.export import export_to_sglang

        for strategy, architecture in (
            ("dflash", "DFlashDraftModel"),
            ("mmflash", "MMFlashDraftModel"),
        ):
            with self.subTest(strategy=strategy):
                self.fresh_root()
                config_path = write_json(
                    self.path("draft.json"),
                    draft_config(architecture, f"{strategy}.{architecture}"),
                )
                run_dir, _ = save_trainer_checkpoint(self.root, config_path, strategy)
                with self.assertRaisesRegex(ValueError, "--to hf"):
                    export_to_sglang(run_dir, config_path, self.path("sglang"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
