# coding=utf-8
"""The producer side of the visual-score channel (``_encode_worker``)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from specforge.algorithms.common import mm_server_input
from specforge.data.visual_score import (
    SCALE,
    TEXT_ONLY_SENTINEL,
    fingerprint_input_ids,
)

IDS = [11, 12, 13, 14, 15, 16]


def _config(strategy="mmflash"):
    return SimpleNamespace(
        training=SimpleNamespace(strategy=strategy, loss_type="dpace"),
        data=SimpleNamespace(
            image_root="",
            max_length=4096,
            train_only_last_turn=False,
            visual_score_path="",
            visual_score_transform="quantile",
            visual_score_binary_threshold=0.75,
            visual_score_confidence_gate=True,
        ),
    )


def _fake_encode(image):
    def encode(record, processor, **kwargs):
        return {
            "input_ids": list(IDS),
            "loss_mask": [0, 0, 0, 1, 1, 1],
            "image": image,
        }

    return encode


class TestEncodeWorkerVisualChannel(unittest.TestCase):
    def setUp(self):
        mm_server_input._WORKER.clear()
        mm_server_input._WORKER.update(
            {"config": _config(), "processor": object(), "header_ids": [1], "end_ids": {2}}
        )

    def tearDown(self):
        mm_server_input._WORKER.clear()

    def _run(self, record, image):
        with patch("specforge.data.mm_preprocessing.encode_mm_record", _fake_encode(image)):
            return mm_server_input._encode_worker(record)

    def test_text_only_row_carries_the_sentinel(self):
        payload, status = self._run({"id": "t#1", "image": None, "conversations": []}, None)
        self.assertEqual(status, "text_only")
        self.assertEqual(list(payload["visual_score"]), [TEXT_ONLY_SENTINEL] * 6)
        self.assertIsInstance(payload["visual_score"], array)

    def test_image_row_with_scores_is_expanded_onto_loss_positions(self):
        entry = (6, fingerprint_input_ids(IDS), np.array([0.5, 1.0, 0.0], dtype=np.float32))
        record = {"id": "i#1", "image": "x.png", "conversations": [], "_visual_score": entry}
        payload, status = self._run(record, "x.png")
        self.assertEqual(status, "scored")
        self.assertEqual(list(payload["visual_score"]), [0, 0, 0, SCALE // 2, SCALE, 0])

    def test_image_row_without_entry_or_misaligned_falls_back_to_zero(self):
        record = {"id": "i#2", "image": "x.png", "conversations": [], "_visual_score": None}
        payload, status = self._run(record, "x.png")
        self.assertEqual((status, list(payload["visual_score"])), ("missing", [0] * 6))
        record["_visual_score"] = (
            7,
            fingerprint_input_ids(IDS),
            np.array([0.5, 1.0, 0.0], dtype=np.float32),
        )
        payload, status = self._run(record, "x.png")
        self.assertEqual((status, list(payload["visual_score"])), ("misaligned", [0] * 6))

    def test_other_strategies_do_not_get_the_channel(self):
        mm_server_input._WORKER["config"] = _config(strategy="dflash")
        payload, status = self._run({"id": "i#3", "image": "x.png", "conversations": []}, "x.png")
        self.assertEqual(status, "off")
        self.assertNotIn("visual_score", payload)

    def test_dropped_rows_report_dropped(self):
        with patch("specforge.data.mm_preprocessing.encode_mm_record", lambda *a, **k: None):
            payload, status = mm_server_input._encode_worker({"id": "x", "image": None})
        self.assertEqual((payload, status), (None, "dropped"))

    def test_join_step_without_sidecar_marks_every_image_row_missing(self):
        adapter = mm_server_input.ImageServerInputAdapter
        records = [{"id": "a", "image": "x.png"}, {"id": "b", "image": None}]
        cfg = _config()
        table = adapter._load_visual_scores(cfg)
        self.assertIsNone(table)  # visual_score_path is empty in _config()
        joined, counts = adapter._attach_visual_scores(cfg, records, table)
        self.assertEqual(counts, {"image_scored": 0, "image_missing": 1, "text_only": 1})
        self.assertIn(mm_server_input.VISUAL_SCORE_RECORD_KEY, joined[0])
        # a non-mmflash run leaves the records alone and loads nothing
        other = _config(strategy="dflash")
        self.assertIsNone(adapter._load_visual_scores(other))
        untouched, counts = adapter._attach_visual_scores(other, records, None)
        self.assertIs(untouched, records)
        self.assertEqual(counts, {})


if __name__ == "__main__":
    unittest.main()
