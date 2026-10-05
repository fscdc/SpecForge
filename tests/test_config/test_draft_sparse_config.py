# coding=utf-8
"""``training.draft_sparse``: opt-in, MMFlash-only, and inert when unset."""

from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import ValidationError

from specforge.config import DraftSparseConfig, TrainingConfig, load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "scripts" / "mmtraining_configs"
SPARSE = {"sink": 4, "text": True, "stride": 32, "window": 2048}


class DraftSparseConfigTest(unittest.TestCase):
    def test_default_is_dense(self):
        self.assertIsNone(TrainingConfig(strategy="mmflash").draft_sparse)
        self.assertIsNone(TrainingConfig().draft_sparse)

    def test_valid_block_and_env_string(self):
        training = TrainingConfig(strategy="mmflash", draft_sparse=SPARSE)
        self.assertIsInstance(training.draft_sparse, DraftSparseConfig)
        self.assertEqual("sink=4,text=1,stride=32,window=2048", training.draft_sparse.to_sglang_env())
        training = TrainingConfig(strategy="mmflash", attention_backend="sdpa", draft_sparse={"window": 8})
        self.assertEqual("sink=0,text=1,stride=0,window=8", training.draft_sparse.to_sglang_env())

    def test_window_is_required(self):
        with self.assertRaises(ValidationError):
            TrainingConfig(strategy="mmflash", draft_sparse={"sink": 4, "text": True, "stride": 32})

    def test_rejects_negative_and_unknown_keys(self):
        for bad in ({**SPARSE, "sink": -1}, {**SPARSE, "window": -1}, {**SPARSE, "stride": -2},
                    {**SPARSE, "depth": 1}):
            with self.assertRaises(ValidationError, msg=str(bad)):
                TrainingConfig(strategy="mmflash", draft_sparse=bad)

    def test_mmflash_only(self):
        for strategy in ("dflash", "eagle3", "domino"):
            with self.assertRaises(ValidationError, msg=strategy):
                TrainingConfig(strategy=strategy, draft_sparse=SPARSE)

    def test_needs_a_masking_backend(self):
        for backend in ("eager", "fa", "usp"):
            with self.assertRaises(ValidationError, msg=backend):
                TrainingConfig(strategy="mmflash", attention_backend=backend, draft_sparse=SPARSE)

    def test_video_recipes_load(self):
        sparse = load_config(str(CONFIGS / "qwen3.5-4b-mmflash-video-sparse_hpc.yaml"))
        dense = load_config(str(CONFIGS / "qwen3.5-4b-mmflash-video-dense_hpc.yaml"))
        self.assertIsNotNone(sparse.training.draft_sparse)
        self.assertIsNone(dense.training.draft_sparse)
        for config in (sparse, dense):
            self.assertEqual("mmflash", config.training.strategy)
            self.assertIsNotNone(config.model.draft_checkpoint_path)
            self.assertIsNone(config.training.resume_from)
            # every video row (42.4k-45.0k tokens) fits; rows above are dropped
            self.assertGreaterEqual(config.data.max_length, 45056)
            # scored once by scripts/score_visual_kl_hpc.sh and shared by both runs
            self.assertTrue(config.data.visual_score_path.endswith("visual_kl/llava-video-2k"))
        # the two runs differ only where they must
        self.assertEqual(sparse.data.visual_score_path, dense.data.visual_score_path)
        self.assertNotEqual(sparse.output_dir, dense.output_dir)
        self.assertNotEqual(sparse.run_id, dense.run_id)
        sparse_ports = {server.port for server in sparse.deployment.disaggregated.managed_local.capture_servers}
        dense_ports = {server.port for server in dense.deployment.disaggregated.managed_local.capture_servers}
        self.assertFalse(sparse_ports & dense_ports)

    def test_existing_recipes_stay_dense(self):
        for path in sorted(CONFIGS.glob("*.yaml")):
            if "video" in path.name:
                continue
            config = load_config(str(path))
            self.assertIsNone(config.training.draft_sparse, path.name)


if __name__ == "__main__":
    unittest.main()
