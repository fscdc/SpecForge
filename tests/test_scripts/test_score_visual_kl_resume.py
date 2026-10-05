"""Resuming the visual-KL scorer with a different shard count never scores a row twice."""

import os
import unittest
from tempfile import TemporaryDirectory

from scripts import score_visual_kl


class TestScoreVisualKlResume(unittest.TestCase):
    def _write(self, directory, name, text):
        with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
            handle.write(text)

    def test_other_shard_files_are_skipped(self):
        with TemporaryDirectory() as tmp:
            # a 1-GPU run cut by walltime (torn last line), resumed on 4 GPUs
            self._write(tmp, "visual_kl.shard000-of-001.jsonl", '{"id": "a"}\n{"id": "b"}\n{"id": "c"')
            self._write(tmp, "visual_kl.shard001-of-004.jsonl", '{"id": "d"}\n')
            self._write(tmp, "quantiles.json", '{"points": 1000}')
            own = os.path.join(tmp, "visual_kl.shard000-of-004.jsonl")
            self.assertEqual({"a", "b", "d"}, score_visual_kl._ids_in_other_shards(tmp, own))
            # a shard's own file is resumed (and repaired) by _already_done, not here
            own = os.path.join(tmp, "visual_kl.shard001-of-004.jsonl")
            self.assertEqual({"a", "b"}, score_visual_kl._ids_in_other_shards(tmp, own))

    def test_empty_directory(self):
        with TemporaryDirectory() as tmp:
            own = os.path.join(tmp, "visual_kl.shard000-of-001.jsonl")
            self.assertEqual(set(), score_visual_kl._ids_in_other_shards(tmp, own))


if __name__ == "__main__":
    unittest.main()
