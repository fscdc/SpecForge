# coding=utf-8
"""Multi-image (video frame) records through encoding and capture requests.

A record whose ``image`` is a list takes its own branch; single-image and
text-only records must come out exactly as before.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from specforge.algorithms.common import mm_server_input
from specforge.algorithms.common.mm_server_input import ImageServerInputAdapter
from specforge.data import mm_preprocessing
from specforge.data.mm_preprocessing import IMAGE_PLACEHOLDER, resolve_image_paths


def _task(image, ids=(1, 2, 3)):
    return SimpleNamespace(payload={"input_ids": list(ids), "image": image})


class BuildRequestInputsTest(unittest.TestCase):
    def setUp(self):
        self.adapter = ImageServerInputAdapter.__new__(ImageServerInputAdapter)

    def test_single_image_and_text_rows_unchanged(self):
        request = self.adapter.build_request_inputs([_task("/a.jpg"), _task(None)])
        self.assertEqual({"input_ids": [[1, 2, 3], [1, 2, 3]], "image_data": [["/a.jpg"], []]}, request)
        self.assertEqual({"input_ids": [[1, 2, 3]]}, self.adapter.build_request_inputs([_task(None)]))

    def test_frame_list_is_one_entry_per_row(self):
        frames = [f"/v/f{i:02d}.jpg" for i in range(48)]
        request = self.adapter.build_request_inputs([_task(frames), _task("/a.jpg"), _task(None)])
        self.assertEqual([frames, ["/a.jpg"], []], request["image_data"])
        self.assertIsInstance(request["image_data"][0], list)
        self.assertEqual(frames, request["image_data"][0])  # order preserved, not nested


class ResolveImagePathsTest(unittest.TestCase):
    def test_resolves_in_order(self):
        self.assertEqual(["/r/a.jpg", "/r/b.jpg"], resolve_image_paths(["a.jpg", "b.jpg"], "/r"))
        self.assertEqual(["/x/a.jpg"], resolve_image_paths(["/x/a.jpg"], "/r"))

    def test_rejects_malformed_lists(self):
        for bad in ([], [""], [None], ["a.jpg", 3], "a.jpg", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                resolve_image_paths(bad, "")


class EncodeWorkerTest(unittest.TestCase):
    """A frame-list payload is an image row for the visual-score channel."""

    def setUp(self):
        config = SimpleNamespace(
            training=SimpleNamespace(strategy="mmflash", loss_type="mmflash"),
            data=SimpleNamespace(
                image_root="", max_length=4096, train_only_last_turn=False,
                visual_score_path="", visual_score_transform="quantile",
                visual_score_binary_threshold=0.75, visual_score_confidence_gate=True,
            ),
        )
        mm_server_input._WORKER.clear()
        mm_server_input._WORKER.update({"config": config, "processor": object(), "header_ids": [1], "end_ids": {2}})

    def tearDown(self):
        mm_server_input._WORKER.clear()

    def test_frames_are_an_image_row(self):
        frames = ["/v/f0.jpg", "/v/f1.jpg"]

        def encode(record, processor, **kwargs):
            return {"input_ids": [11, 12, 13, 14], "loss_mask": [0, 0, 1, 1], "image": frames}

        with patch("specforge.data.mm_preprocessing.encode_mm_record", encode):
            payload, status = mm_server_input._encode_worker({"id": "v#0", "image": frames})
        self.assertNotEqual("text_only", status)
        self.assertEqual(frames, payload["image"])
        self.assertEqual(4, len(payload["visual_score"]))


def _processor_or_none():
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained("Qwen/Qwen3.5-4B", local_files_only=True)
    except Exception:  # not cached / offline machine
        return None


PROCESSOR = _processor_or_none()


@unittest.skipIf(PROCESSOR is None, "Qwen/Qwen3.5-4B processor not cached locally")
class EncodeMultiImageRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image

        cls.tmp = tempfile.TemporaryDirectory()
        cls.frames = []
        for index, size in enumerate([(320, 180), (320, 180), (240, 240)]):
            path = os.path.join(cls.tmp.name, f"f{index}.jpg")
            Image.new("RGB", size, (index * 40, 90, 160)).save(path, "JPEG")
            cls.frames.append(path)
        cls.header_ids = mm_preprocessing._assistant_header_ids(PROCESSOR)
        cls.end_ids = mm_preprocessing._end_token_ids(PROCESSOR)
        cls.image_token_id = PROCESSOR.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _encode(self, record, max_length=100000):
        return mm_preprocessing.encode_mm_record(
            record, PROCESSOR, image_root="", max_length=max_length, train_only_last_turn=False,
            header_ids=self.header_ids, end_ids=self.end_ids,
        )

    def _record(self, frames, placeholders=None):
        count = len(frames) if placeholders is None else placeholders
        return {
            "id": "v#0",
            "image": list(frames),
            "conversations": [
                {"role": "user", "content": IMAGE_PLACEHOLDER * count + "\nWhat happens?\nA. x\nB. y\n"},
                {"role": "assistant", "content": "Step by step, it is A."},
            ],
        }

    @staticmethod
    def _runs(ids, token):
        runs, start = [], None
        for index, value in enumerate(ids + [None]):
            if value == token and start is None:
                start = index
            elif value != token and start is not None:
                runs.append(index - start)
                start = None
        return runs

    def test_one_pad_run_per_frame_in_order(self):
        payload = self._encode(self._record(self.frames))
        self.assertEqual(self.frames, payload["image"])
        runs = self._runs(payload["input_ids"], self.image_token_id)
        self.assertEqual(3, len(runs))
        # each run is exactly what that frame expands to on its own
        for frame, run in zip(self.frames, runs):
            single = self._encode({
                "id": "s", "image": frame,
                "conversations": [{"role": "user", "content": IMAGE_PLACEHOLDER + "\nq"},
                                  {"role": "assistant", "content": "a"}],
            })
            self.assertEqual([run], self._runs(single["input_ids"], self.image_token_id))
        # the loss covers the answer only, after every frame
        first_loss = payload["loss_mask"].index(1)
        last_pad = max(i for i, t in enumerate(payload["input_ids"]) if t == self.image_token_id)
        self.assertGreater(first_loss, last_pad)
        self.assertEqual(len(payload["input_ids"]), len(payload["loss_mask"]))

    def test_placeholder_count_must_match(self):
        with self.assertRaises(ValueError):
            self._encode(self._record(self.frames, placeholders=2))
        with self.assertRaises(ValueError):
            self._encode(self._record(self.frames, placeholders=4))

    def test_too_long_rows_are_dropped(self):
        payload = self._encode(self._record(self.frames))
        self.assertIsNone(self._encode(self._record(self.frames), max_length=len(payload["input_ids"]) - 1))
        self.assertIsNotNone(self._encode(self._record(self.frames), max_length=len(payload["input_ids"])))

    def test_single_image_record_still_takes_the_old_path(self):
        record = {
            "id": "s", "image": self.frames[0],
            "conversations": [{"role": "user", "content": IMAGE_PLACEHOLDER + "\nq"},
                              {"role": "assistant", "content": "a"}],
        }
        with patch.object(mm_preprocessing, "_encode_multi_image_record") as multi:
            payload = self._encode(record)
        multi.assert_not_called()
        self.assertEqual(self.frames[0], payload["image"])


if __name__ == "__main__":
    unittest.main()
