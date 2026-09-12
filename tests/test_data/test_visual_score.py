# coding=utf-8
"""The visual-dependency sidecar: loading, transforms, wire format."""

import json
import os
import tempfile
import unittest
from array import array

import numpy as np
import torch

from specforge.data.visual_score import (
    SCALE,
    TEXT_ONLY_SENTINEL,
    VisualScoreTable,
    dequantize_visual_score,
    expand_visual_score,
    fingerprint_input_ids,
    join_visual_scores,
    quantize,
    text_only_visual_score,
    transform_scores,
)


def _write_sidecar(directory, rows, name="visual_kl.shard000.jsonl"):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


class TestTransforms(unittest.TestCase):
    def test_quantile_rank_is_monotone_and_bounded(self):
        edges = np.quantile(np.linspace(0, 10, 101), np.linspace(0, 1, 1001))
        g = transform_scores(np.array([0.0, 2.5, 5.0, 7.5, 10.0, 50.0]), "quantile", edges=edges)
        self.assertTrue(np.all(np.diff(g) >= 0))
        self.assertTrue(np.all((g >= 0) & (g <= 1)))
        self.assertAlmostEqual(float(g[2]), 0.5, places=2)
        self.assertEqual(float(g[-1]), 1.0)

    def test_binary_uses_the_quantile_threshold(self):
        edges = np.quantile(np.linspace(0, 10, 101), np.linspace(0, 1, 1001))
        g = transform_scores(np.array([1.0, 8.0]), "binary", edges=edges, binary_threshold=0.75)
        self.assertEqual(g.tolist(), [0.0, 1.0])

    def test_saturate_puts_the_75th_percentile_at_one_half(self):
        edges = np.quantile(np.linspace(0, 10, 1001), np.linspace(0, 1, 1001))
        g = transform_scores(np.array([7.5, 0.0]), "saturate", edges=edges)
        self.assertAlmostEqual(float(g[0]), 0.5, places=2)
        self.assertEqual(float(g[1]), 0.0)

    def test_identity_clips(self):
        g = transform_scores(np.array([-1.0, 0.3, 4.0]), "identity")
        self.assertEqual(g.tolist(), [0.0, np.float32(0.3), 1.0])

    def test_unknown_transform_raises(self):
        with self.assertRaises(ValueError):
            transform_scores(np.array([1.0]), "nope")


class TestTable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rows = [
            {"id": "a#1", "n_tokens": 10, "n_loss": 3, "kl": [0.1, 5.0, 0.2], "entropy": [1, 1, 1]},
            {"id": "b#2", "n_tokens": 7, "n_loss": 2, "kl": [0.0, 0.05], "entropy": [1, 1]},
        ]
        self.path = _write_sidecar(self.tmp.name, self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_file_or_directory(self):
        for path in (self.path, self.tmp.name):
            with self.subTest(path=path):
                table = VisualScoreTable.load(path, transform="quantile")
                self.assertEqual(len(table), 2)
                n_tokens, fp, g = table.get("a#1")
                self.assertEqual(n_tokens, 10)
                self.assertEqual(g.shape, (3,))
                self.assertEqual(g.dtype, np.float32)
                # the 5.0 is the corpus maximum -> rank 1.0
                self.assertEqual(float(g[1]), 1.0)
                self.assertTrue(table.stats.describe())

    def test_quantile_cache_is_written_and_reused(self):
        VisualScoreTable.load(self.tmp.name, transform="quantile")
        cache = os.path.join(self.tmp.name, "quantiles.json")
        self.assertTrue(os.path.isfile(cache))
        with open(cache) as handle:
            payload = json.load(handle)
        self.assertEqual(payload["num_tokens"], 5)
        VisualScoreTable.load(self.tmp.name, transform="binary")  # must not choke on the cache

    def test_duplicate_or_ragged_rows_raise(self):
        _write_sidecar(self.tmp.name, [self.rows[0]], name="visual_kl.shard001.jsonl")
        with self.assertRaises(ValueError):
            VisualScoreTable.load(self.tmp.name)
        with tempfile.TemporaryDirectory() as other:
            path = _write_sidecar(other, [{**self.rows[0], "n_loss": 99}])
            with self.assertRaises(ValueError):
                VisualScoreTable.load(path)

    def test_join_counts_and_attaches_entries(self):
        table = VisualScoreTable.load(self.path, transform="identity")
        records = [
            {"id": "a#1", "image": "x.png"},
            {"id": "zzz", "image": "y.png"},
            {"id": "b#2", "image": None},
        ]
        joined, counts = join_visual_scores(records, table)
        self.assertEqual(counts, {"image_scored": 1, "image_missing": 1, "text_only": 1})
        self.assertIsNotNone(joined[0]["_visual_score"])
        self.assertIsNone(joined[1]["_visual_score"])
        self.assertIsNone(joined[2]["_visual_score"])
        # no table at all: every image row is "missing"
        _, counts = join_visual_scores(records, None)
        self.assertEqual(counts["image_missing"], 2)


class TestWireFormat(unittest.TestCase):
    def test_expand_scored_and_roundtrip(self):
        loss_mask = [0, 0, 1, 1, 0, 1]
        ids = [10, 11, 12, 13, 14, 15]
        g = np.array([0.25, 1.0, 0.0], dtype=np.float32)
        channel, status = expand_visual_score(
            loss_mask, (6, fingerprint_input_ids(ids), g), input_ids=ids
        )
        self.assertEqual(status, "scored")
        self.assertIsInstance(channel, array)
        self.assertEqual(list(channel), [0, 0, int(round(0.25 * SCALE)), SCALE, 0, 0])
        back = dequantize_visual_score(torch.tensor([list(channel)]))
        self.assertEqual(back.dtype, torch.float32)
        torch.testing.assert_close(back[0, [2, 3, 5]], torch.tensor([0.25, 1.0, 0.0]))
        self.assertTrue(bool((back >= 0).any()))

    def test_expand_missing_and_misaligned_fall_back_to_zero(self):
        loss_mask = [0, 1, 1]
        ids = [7, 8, 9]
        fp = fingerprint_input_ids(ids)
        channel, status = expand_visual_score(loss_mask, None, input_ids=ids)
        self.assertEqual((status, list(channel)), ("missing", [0, 0, 0]))
        channel, status = expand_visual_score(
            loss_mask, (3, fp, np.array([0.5])), input_ids=ids
        )
        self.assertEqual((status, list(channel)), ("misaligned", [0, 0, 0]))
        channel, status = expand_visual_score(
            loss_mask, (4, fp, np.array([0.5, 0.5])), input_ids=ids
        )
        self.assertEqual(status, "misaligned")

    def test_same_counts_but_a_different_regen_is_caught_by_the_fingerprint(self):
        """Two regens of one corpus share ids; only the response differs."""
        loss_mask = [0, 1, 1]
        prompted = [7, 8, 9]
        other_regen = [7, 80, 90]  # same id, same lengths, different answer
        entry = (3, fingerprint_input_ids(other_regen), np.array([0.5, 0.5], dtype=np.float32))
        channel, status = expand_visual_score(loss_mask, entry, input_ids=prompted)
        self.assertEqual((status, list(channel)), ("misaligned", [0, 0, 0]))
        # and the matching one still scores
        entry = (3, fingerprint_input_ids(prompted), np.array([0.5, 0.5], dtype=np.float32))
        _, status = expand_visual_score(loss_mask, entry, input_ids=prompted)
        self.assertEqual(status, "scored")

    def test_a_sidecar_without_a_fingerprint_still_loads(self):
        loss_mask = [0, 1, 1]
        entry = (3, "", np.array([0.5, 0.5], dtype=np.float32))
        _, status = expand_visual_score(loss_mask, entry, input_ids=[1, 2, 3])
        self.assertEqual(status, "scored")

    def test_text_only_sentinel_survives_dequantisation(self):
        channel = text_only_visual_score(4)
        self.assertEqual(list(channel), [TEXT_ONLY_SENTINEL] * 4)
        back = dequantize_visual_score(torch.tensor([list(channel)]))
        self.assertTrue(bool((back < 0).all()))
        self.assertFalse(bool((back >= 0).any(dim=1).item()))

    def test_quantize_clips_and_rounds(self):
        q = quantize(np.array([-0.1, 0.00004, 0.5, 1.2]))
        self.assertEqual(q.tolist(), [0, 0, SCALE // 2, SCALE])


if __name__ == "__main__":
    unittest.main()
