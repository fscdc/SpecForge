# coding=utf-8
"""Gemma 4 E4B-it image records through the processor-driven encoder.

Gemma 4 closes a turn with ``<turn|>`` rather than its EOS, expands each image
into a variable number of soft tokens, and has no pixel budget the
``data.image_max_tokens`` cap could edit. Qwen3.5's end ids must not change.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

from specforge.data import mm_preprocessing
from specforge.data.mm_preprocessing import IMAGE_PLACEHOLDER

MODEL = "google/gemma-4-E4B-it"
BOS, EOS, UNK, TURN_OPEN, TURN_CLOSE, NEWLINE = 2, 1, 3, 105, 106, 107
BOI, IMAGE, EOI = 255999, 258880, 258882


def _processor_or_none(name):
    try:
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(name, local_files_only=True)
    except Exception:  # not cached / offline machine
        return None


def _snapshot_dir():
    """Local snapshot directory of the cached Gemma checkpoint, or None."""
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(MODEL, "config.json")
    except Exception:  # pragma: no cover - huggingface_hub without the helper
        return None
    return os.path.dirname(config) if isinstance(config, str) else None


PROCESSOR = _processor_or_none(MODEL)


def _fake_processor(**image_processor_fields):
    return SimpleNamespace(image_processor=SimpleNamespace(**image_processor_fields))


class ImageTokenCapTest(unittest.TestCase):
    def test_processor_without_pixel_budget_rejects_a_cap(self):
        # Gemma4ImageProcessor's shape: bounded by max_soft_tokens, no size/max_pixels
        processor = _fake_processor(
            patch_size=16,
            pooling_kernel_size=3,
            max_soft_tokens=280,
            size=None,
            max_pixels=None,
        )
        with self.assertRaisesRegex(ValueError, "image_max_tokens=140"):
            mm_preprocessing._limit_image_tokens(processor, 140)
        mm_preprocessing._limit_image_tokens(processor, 0)  # the default stays a no-op
        self.assertEqual(280, processor.image_processor.max_soft_tokens)

    def test_qwen_style_pixel_budgets_are_still_capped(self):
        sized = _fake_processor(
            patch_size=16,
            merge_size=2,
            size={"longest_edge": 16777216, "shortest_edge": 65536},
        )
        mm_preprocessing._limit_image_tokens(sized, 256)
        self.assertEqual(256 * 32 * 32, sized.image_processor.size["longest_edge"])
        self.assertEqual(65536, sized.image_processor.size["shortest_edge"])

        legacy = _fake_processor(patch_size=14, merge_size=2, max_pixels=12845056)
        mm_preprocessing._limit_image_tokens(legacy, 256)
        self.assertEqual(256 * 28 * 28, legacy.image_processor.max_pixels)

    @unittest.skipIf(PROCESSOR is None, f"{MODEL} processor not cached locally")
    def test_gemma4_target_rejects_a_cap_at_load(self):
        snapshot = _snapshot_dir()
        if snapshot is None:
            self.skipTest(f"{MODEL} snapshot directory not found")

        def config(cap):
            return SimpleNamespace(
                model=SimpleNamespace(
                    target_model_path=snapshot,
                    cache_dir=None,
                    trust_remote_code=False,
                    tokenizer_pad_token_id=None,
                ),
                data=SimpleNamespace(image_max_tokens=cap),
            )

        with self.assertRaisesRegex(ValueError, "Gemma4ImageProcessor"):
            mm_preprocessing.load_mm_processor(config(140))
        processor = mm_preprocessing.load_mm_processor(config(0))
        self.assertEqual(280, processor.image_processor.max_soft_tokens)


class QwenEndIdsUnchangedTest(unittest.TestCase):
    """Qwen3.5 already ends a reply at its EOS, <|im_end|>; nothing is added."""

    def test_qwen35_end_ids(self):
        processor = _processor_or_none("Qwen/Qwen3.5-4B")
        if processor is None:
            self.skipTest("Qwen/Qwen3.5-4B processor not cached locally")
        im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.assertEqual(im_end, processor.tokenizer.eos_token_id)
        self.assertEqual(im_end, mm_preprocessing._template_end_of_turn_id(processor))
        self.assertEqual({im_end}, mm_preprocessing._end_token_ids(processor))


def _record(image, turns):
    """A ShareGPT record; the first user turn carries one placeholder per image."""
    if image is None:
        count = 0
    else:
        count = len(image) if isinstance(image, list) else 1
    conversations = []
    for index, (question, answer) in enumerate(turns):
        prefix = IMAGE_PLACEHOLDER * count + "\n" if index == 0 else ""
        conversations += [
            {"role": "user", "content": prefix + question},
            {"role": "assistant", "content": answer},
        ]
    return {"id": "r#0", "image": image, "conversations": conversations}


@unittest.skipIf(PROCESSOR is None, f"{MODEL} processor not cached locally")
class Gemma4EncodeRecordTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image

        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.images = {}
        for width, height in [(640, 480), (224, 224), (300, 1200)]:
            path = os.path.join(tmp.name, f"{width}x{height}.png")
            Image.new("RGB", (width, height), (width % 255, 90, 160)).save(path)
            cls.images[(width, height)] = path
        cls.tokenizer = PROCESSOR.tokenizer
        cls.header_ids = mm_preprocessing._assistant_header_ids(PROCESSOR)
        cls.end_ids = mm_preprocessing._end_token_ids(PROCESSOR)

    def _encode(self, record, max_length=100000, train_only_last_turn=False):
        return mm_preprocessing.encode_mm_record(
            record,
            PROCESSOR,
            image_root="",
            max_length=max_length,
            train_only_last_turn=train_only_last_turn,
            header_ids=self.header_ids,
            end_ids=self.end_ids,
        )

    def _spans(self, payload):
        spans, run = [], []
        for token, masked in zip(
            payload["input_ids"] + [None], payload["loss_mask"] + [0]
        ):
            if masked:
                run.append(token)
            elif run:
                spans.append(self.tokenizer.decode(run))
                run = []
        return spans

    def _image_runs(self, ids):
        """Soft-token count of each '<|image>' + N x '<|image|>' + '<image|>' run."""
        runs = []
        for start, token in enumerate(ids):
            if token != BOI:
                continue
            end = ids.index(EOI, start)
            self.assertEqual({IMAGE}, set(ids[start + 1 : end]))
            runs.append(end - start - 1)
        self.assertEqual(
            sum(runs), ids.count(IMAGE), "image tokens outside a boi/eoi run"
        )
        return runs

    def _expected_count(self, width, height):
        counts = PROCESSOR._get_num_multimodal_tokens(image_sizes=[(height, width)])
        return int(counts.num_image_tokens[0])

    def test_special_token_ids(self):
        convert = self.tokenizer.convert_tokens_to_ids
        self.assertEqual(
            [BOI, IMAGE, EOI],
            convert([PROCESSOR.boi_token, PROCESSOR.image_token, PROCESSOR.eoi_token]),
        )
        self.assertEqual(
            [EOS, UNK, TURN_CLOSE],
            [
                self.tokenizer.eos_token_id,
                self.tokenizer.unk_token_id,
                convert("<turn|>"),
            ],
        )

    def test_assistant_header_and_end_ids(self):
        self.assertEqual([105, 4368, 107], self.header_ids)
        self.assertEqual("<|turn>model\n", self.tokenizer.decode(self.header_ids))
        self.assertEqual({EOS, TURN_CLOSE}, self.end_ids)
        # '<|im_end|>' and '<|eot_id|>' both resolve to <unk> on this tokenizer
        self.assertNotIn(UNK, self.end_ids)
        template_end = mm_preprocessing._template_end_of_turn_id(PROCESSOR)
        self.assertEqual(TURN_CLOSE, template_end)

    def test_single_turn_reply_stops_at_turn_close(self):
        payload = self._encode(
            _record(self.images[(640, 480)], [("What is it?", "A cat.")])
        )
        self.assertEqual(["A cat.<turn|>"], self._spans(payload))
        ids, mask = payload["input_ids"], payload["loss_mask"]
        last = max(i for i, masked in enumerate(mask) if masked)
        self.assertEqual(TURN_CLOSE, ids[last])
        # the template's trailing '\n' is not trained
        self.assertEqual([NEWLINE], ids[last + 1 :])
        self.assertEqual([0], mask[last + 1 :])
        self.assertGreater(mask.index(1), ids.index(EOI))

    def test_multi_turn_masks_each_model_reply(self):
        record = _record(
            self.images[(640, 480)], [("What is it?", "A cat."), ("Color?", "Black.")]
        )
        self.assertEqual(
            ["A cat.<turn|>", "Black.<turn|>"], self._spans(self._encode(record))
        )
        self.assertEqual(
            ["Black.<turn|>"],
            self._spans(self._encode(record, train_only_last_turn=True)),
        )

    def test_text_only_row(self):
        payload = self._encode(_record(None, [("Hi", "Hello!"), ("Bye", "Goodbye.")]))
        self.assertIsNone(payload["image"])
        self.assertEqual(["Hello!<turn|>", "Goodbye.<turn|>"], self._spans(payload))
        self.assertEqual(1, payload["input_ids"].count(BOS))
        self.assertEqual([], self._image_runs(payload["input_ids"]))

    def test_text_rows_match_the_gemma4_text_template(self):
        # the image path's end ids and the text path's 'gemma4' template must
        # mask a text-only row identically
        from specforge.data.preprocessing import preprocess_conversations
        from specforge.data.template import TEMPLATE_REGISTRY

        system = {"role": "system", "content": "Answer briefly."}
        for turns, with_system in [
            ([("What is 2+2?", "4.")], False),
            ([("Hi", "Hello!"), ("Bye", "Goodbye.")], False),
            ([("What is 2+2?", "4.")], True),
        ]:
            record = _record(None, turns)
            if with_system:
                record["conversations"].insert(0, system)
            with self.subTest(turns=len(turns), system=with_system):
                payload = self._encode(record)
                text = preprocess_conversations(
                    self.tokenizer,
                    [record["conversations"]],
                    TEMPLATE_REGISTRY.get("gemma4"),
                    100000,
                )
                self.assertEqual(text["input_ids"][0][0].tolist(), payload["input_ids"])
                self.assertEqual(text["loss_mask"][0][0].tolist(), payload["loss_mask"])

    def test_image_run_length_comes_from_the_processor(self):
        cap = PROCESSOR.image_processor.max_soft_tokens
        for (width, height), path in self.images.items():
            with self.subTest(size=(width, height)):
                payload = self._encode(_record(path, [("Describe.", "A block.")]))
                ids = payload["input_ids"]
                expected = self._expected_count(width, height)
                self.assertEqual([expected], self._image_runs(ids))
                self.assertTrue(0 < expected <= cap)
                self.assertEqual([BOS, TURN_OPEN], ids[:2])
                self.assertEqual(1, ids.count(BOS))

    def test_frame_list_has_one_run_per_image_in_order(self):
        sizes = [(640, 480), (224, 224)]
        frames = [self.images[size] for size in sizes]
        payload = self._encode(_record(frames, [("What changes?", "Its shape.")]))
        self.assertEqual(frames, payload["image"])
        self.assertEqual(
            [self._expected_count(*size) for size in sizes],
            self._image_runs(payload["input_ids"]),
        )
        self.assertEqual(["Its shape.<turn|>"], self._spans(payload))

    def test_rows_over_max_length_are_dropped(self):
        record = _record(self.images[(640, 480)], [("What is it?", "A cat.")])
        length = len(self._encode(record)["input_ids"])
        self.assertIsNone(self._encode(record, max_length=length - 1))
        self.assertIsNotNone(self._encode(record, max_length=length))

    def test_newline_closed_template_adds_no_end_id(self):
        # a template that closes a reply with plain text must not make '\n' an end id
        def render(messages, tokenize=False, add_generation_prompt=False):
            return "".join(f"{m['role']}: {m['content']}\n\n" for m in messages)

        processor = SimpleNamespace(
            tokenizer=self.tokenizer, apply_chat_template=render
        )
        self.assertIsNone(mm_preprocessing._template_end_of_turn_id(processor))
        # what remains are EOS and the tokenizer's eot_token
        self.assertEqual({EOS, TURN_CLOSE}, mm_preprocessing._end_token_ids(processor))


# Runs in a child process: importing sglang registers model configs with
# transformers and sets process-wide server args, which would leak into the
# other tests of a discover run.
_SGLANG_REEXPAND = """
import asyncio, json, sys
from types import SimpleNamespace

snapshot, requests_path, results_path = sys.argv[1:4]
try:
    import sglang.srt.server_args as server_args
    from sglang.srt.managers.tokenizer_manager import _determine_tensor_transport_mode
    from sglang.srt.multimodal.processors.gemma4 import Gemma4SGLangProcessor
except ImportError as exc:
    with open(results_path, "w") as handle:
        json.dump({"skip": f"sglang Gemma4 processor not importable: {exc}"}, handle)
    sys.exit(0)
from transformers import AutoConfig, AutoProcessor

args = server_args.ServerArgs(
    model_path=snapshot, skip_tokenizer_init=True, device="cpu",
    attention_backend="triton", disable_fast_image_processor=True,
)
server_args.set_global_server_args_for_scheduler(args)
processor = Gemma4SGLangProcessor(
    AutoConfig.from_pretrained(snapshot), args,
    AutoProcessor.from_pretrained(snapshot), _determine_tensor_transport_mode(args),
)
with open(requests_path) as handle:
    requests = json.load(handle)
results = []
for request in requests:
    # what the tokenizer manager does for a skip-tokenizer-init capture request
    output = asyncio.run(processor.process_mm_data_async(
        image_data=[request["image"]], input_text=request["input_ids"],
        request_obj=SimpleNamespace(video_data=None),
    ))
    results.append({
        "input_ids": list(output.input_ids),
        "offsets": [[list(span) for span in item.offsets] for item in output.mm_items],
    })
processor.io_executor.shutdown()
processor.cpu_executor.shutdown()
with open(results_path, "w") as handle:
    json.dump({"results": results}, handle)
"""


@unittest.skipIf(PROCESSOR is None, f"{MODEL} processor not cached locally")
@unittest.skipIf(importlib.util.find_spec("sglang") is None, "sglang not installed")
class SGLangGemma4ProcessorParityTest(unittest.TestCase):
    """The capture server re-expands ``image_data`` into the producer's exact ids.

    The producer sends pre-expanded ``input_ids`` plus image paths;
    ``Gemma4SGLangProcessor`` (CPU, slow image processor) decodes, collapses each
    run back to one ``<|image|>`` and re-expands it, so any count drift would
    make the server's sequence differ from the captured loss mask.
    """

    def test_server_ids_match_encode_mm_record(self):
        from PIL import Image

        snapshot = _snapshot_dir()
        if snapshot is None:
            self.skipTest(f"{MODEL} snapshot directory not found")
        header_ids = mm_preprocessing._assistant_header_ids(PROCESSOR)
        end_ids = mm_preprocessing._end_token_ids(PROCESSOR)
        with tempfile.TemporaryDirectory() as tmp:
            requests = []
            for width, height in [(640, 480), (224, 224), (1920, 1080), (300, 1200)]:
                # PNG: decoded by PIL on any host, never by nvJPEG
                path = os.path.join(tmp, f"{width}x{height}.png")
                Image.new("RGB", (width, height), (40, height % 255, 160)).save(path)
                payload = mm_preprocessing.encode_mm_record(
                    _record(path, [("Describe the picture.", "A plain block.")]),
                    PROCESSOR,
                    image_root="",
                    max_length=100000,
                    train_only_last_turn=False,
                    header_ids=header_ids,
                    end_ids=end_ids,
                )
                requests.append({"image": path, "input_ids": payload["input_ids"]})
            requests_path = os.path.join(tmp, "requests.json")
            results_path = os.path.join(tmp, "results.json")
            with open(requests_path, "w") as handle:
                json.dump(requests, handle)
            command = [sys.executable, "-c", _SGLANG_REEXPAND]
            child = subprocess.run(
                command + [snapshot, requests_path, results_path],
                capture_output=True,
                text=True,
                timeout=900,
            )
            self.assertEqual(0, child.returncode, child.stderr[-4000:])
            with open(results_path) as handle:
                outcome = json.load(handle)
        if "skip" in outcome:
            self.skipTest(outcome["skip"])
        self.assertEqual(len(requests), len(outcome["results"]))
        for request, result in zip(requests, outcome["results"]):
            with self.subTest(image=os.path.basename(request["image"])):
                ids = request["input_ids"]
                self.assertEqual(ids, result["input_ids"])
                run = [i for i, token in enumerate(ids) if token == IMAGE]
                self.assertEqual([[[run[0], run[-1]]]], result["offsets"])


if __name__ == "__main__":
    unittest.main()
