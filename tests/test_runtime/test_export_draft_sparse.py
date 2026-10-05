# coding=utf-8
"""A sparse-trained checkpoint exports its context pattern; a dense one exports as before."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from specforge.export.to_hf import _stamp_draft_sparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import draft_sparse_env  # noqa: E402

CONFIG = {
    "architectures": ["MMFlashDraftModel"],
    "block_size": 16,
    "dflash_config": {"mask_token_id": 248070, "target_layer_ids": [1, 8, 15, 22, 29]},
}


class StampDraftSparseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.tmp.name, "config.json")
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump(CONFIG, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_dense_checkpoint_leaves_the_config_untouched(self):
        before = Path(self.config_path).read_bytes()
        _stamp_draft_sparse({"draft_state_dict": {}}, self.tmp.name)
        self.assertEqual(before, Path(self.config_path).read_bytes())

    def test_sparse_checkpoint_is_stamped(self):
        _stamp_draft_sparse({"mmflash_draft_sparse": "sink=4,text=1,stride=32,window=2048"}, self.tmp.name)
        config = json.loads(Path(self.config_path).read_text(encoding="utf-8"))
        self.assertEqual({"sink": 4, "text": 1, "stride": 32, "window": 2048}, config["dflash_config"]["draft_sparse"])
        # the rest of dflash_config survives
        self.assertEqual([1, 8, 15, 22, 29], config["dflash_config"]["target_layer_ids"])
        self.assertEqual(248070, config["dflash_config"]["mask_token_id"])

    def _env(self, *extra):
        out = io.StringIO()
        with redirect_stdout(out):
            status = draft_sparse_env.main([self.tmp.name, "--no-check-patch", *extra])
        return status, out.getvalue()

    def test_serving_flags_round_trip(self):
        _stamp_draft_sparse({"mmflash_draft_sparse": "sink=4,text=0,stride=0,window=2048"}, self.tmp.name)
        status, out = self._env()
        self.assertEqual(0, status)
        tag = "_" + os.path.basename(self.tmp.name)
        self.assertEqual(
            f"export DRAFT_MODEL={self.tmp.name}\n"
            "export DRAFT_SPARSE=sink=4,text=0,stride=0,window=2048\n"
            "export DRAFT_WINDOW=2048\n"
            f'export NAME_SUFFIX="${{NAME_SUFFIX:-{tag}}}"\n',
            out,
        )
        status, out = self._env("--format", "sglang")
        self.assertIn("export SGLANG_DFLASH_DRAFT_SPARSE=sink=4,text=0,stride=0,window=2048", out)
        self.assertIn("--speculative-draft-window-size 2048", out)

    def test_small_window_still_enables_the_compact_cache(self):
        _stamp_draft_sparse({"mmflash_draft_sparse": "sink=4,text=1,stride=32,window=0"}, self.tmp.name)
        self.assertIn("export DRAFT_WINDOW=16\n", self._env()[1])  # the block size

    def test_dense_export_serves_dense(self):
        status, out = self._env()
        self.assertEqual(0, status)
        self.assertIn("export DRAFT_SPARSE=''\n", out)
        self.assertIn("export DRAFT_WINDOW=\n", out)

    def _bash(self, script):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("DRAFT_", "NAME_SUFFIX"))}
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, cwd=str(ROOT))

    def test_child_processes_see_the_flags(self):
        """The benchmark scripts run as children of the shell that evaluated the flags."""
        _stamp_draft_sparse({"mmflash_draft_sparse": "sink=4,text=1,stride=32,window=2048"}, self.tmp.name)
        script = (
            f'flags=$({sys.executable} scripts/draft_sparse_env.py {self.tmp.name} --no-check-patch) || exit 9\n'
            'eval "$flags"\n'
            'bash -c \'printf "%s|%s|%s|%s" "$DRAFT_SPARSE" "$DRAFT_WINDOW" "$NAME_SUFFIX" "$DRAFT_MODEL"\''
        )
        result = self._bash(script)
        self.assertEqual(0, result.returncode, result.stderr)
        tag = "_" + os.path.basename(self.tmp.name)
        self.assertEqual(f"sink=4,text=1,stride=32,window=2048|2048|{tag}|{self.tmp.name}", result.stdout)
        # a NAME_SUFFIX chosen by the caller is kept
        result = self._bash("export NAME_SUFFIX=_mine\n" + script)
        self.assertTrue(result.stdout.split("|")[2] == "_mine", result.stdout)

    def test_failures_fail_closed(self):
        """A broken export must stop the job, even through a bare eval."""
        missing = os.path.join(self.tmp.name, "nope")
        result = self._bash(
            f'flags=$({sys.executable} scripts/draft_sparse_env.py {missing} --no-check-patch) || exit 9\necho served'
        )
        self.assertEqual(9, result.returncode)
        self.assertNotIn("served", result.stdout)
        result = self._bash(
            f'set -e\neval "$({sys.executable} scripts/draft_sparse_env.py {missing} --no-check-patch)"\necho served'
        )
        self.assertNotEqual(0, result.returncode)
        self.assertNotIn("served", result.stdout)


if __name__ == "__main__":
    unittest.main()
