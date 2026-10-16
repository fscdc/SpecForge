# coding=utf-8
"""The ``gemma4`` text-path template against the real Gemma 4 E4B-it tokenizer.

The loss mask must cover each model reply up to and including ``<turn|>`` and
nothing else: no user turn, and not the ``\\n`` the template renders after the
turn closes. The Gemma 3 ``gemma`` entry must stay as it is, which on this
tokenizer means it finds no reply at all.
"""

from __future__ import annotations

import unittest

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY

MODEL = "google/gemma-4-E4B-it"
BOS, TURN_OPEN, TURN_CLOSE, NEWLINE = 2, 105, 106, 107


def _tokenizer_or_none():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except Exception:  # not cached / offline machine
        return None


TOKENIZER = _tokenizer_or_none()


def _encode(messages, template="gemma4", **kwargs):
    res = preprocess_conversations(
        TOKENIZER, [messages], TEMPLATE_REGISTRY.get(template), 4096, **kwargs
    )
    return res["input_ids"][0][0].tolist(), res["loss_mask"][0][0].tolist()


def _spans(input_ids, loss_mask):
    """Decoded runs of consecutive masked tokens, in order."""
    spans, run = [], []
    for token, masked in zip(input_ids + [None], loss_mask + [0]):
        if masked:
            run.append(token)
        elif run:
            spans.append(TOKENIZER.decode(run))
            run = []
    return spans


class Gemma4TemplateEntryTest(unittest.TestCase):
    def test_gemma4_entry(self):
        template = TEMPLATE_REGISTRY.get("gemma4")
        self.assertEqual("<|turn>model\n", template.assistant_header)
        self.assertEqual("<|turn>user\n", template.user_header)
        self.assertIsNone(template.system_prompt)
        self.assertEqual("<turn|>", template.end_of_turn_token)
        self.assertEqual("general", template.parser_type)
        self.assertFalse(template.enable_thinking)
        self.assertIsNone(template.ignore_token)

    def test_gemma3_entry_is_untouched(self):
        template = TEMPLATE_REGISTRY.get("gemma")
        self.assertEqual("<start_of_turn>model\n", template.assistant_header)
        self.assertEqual("<start_of_turn>user\n", template.user_header)
        self.assertEqual("You are a helpful assistant.", template.system_prompt)
        self.assertEqual("<end_of_turn>\n", template.end_of_turn_token)


@unittest.skipIf(TOKENIZER is None, f"{MODEL} tokenizer not cached locally")
class Gemma4TemplateSpansTest(unittest.TestCase):
    MULTI_TURN = [
        {"role": "user", "content": "Who are you?"},
        {"role": "assistant", "content": "My name is Gemma."},
        {"role": "user", "content": "How old are you?"},
        {"role": "assistant", "content": "11 years old."},
    ]

    def assert_only_model_turns_masked(self, input_ids, loss_mask):
        """Every masked token sits in a turn whose header is '<|turn>model\\n'."""
        role_ids = TOKENIZER("model\n", add_special_tokens=False)["input_ids"]
        model_header = [TURN_OPEN] + role_ids
        header_start = None
        for index, (token, masked) in enumerate(zip(input_ids, loss_mask)):
            if token == TURN_OPEN:
                header_start = index
            if masked:
                self.assertIsNotNone(
                    header_start, f"masked token {index} before any turn"
                )
                header = input_ids[header_start : header_start + len(model_header)]
                self.assertEqual(
                    model_header, header, f"masked token {index} outside a model turn"
                )

    def test_special_token_ids(self):
        self.assertEqual(BOS, TOKENIZER.bos_token_id)
        self.assertEqual(
            [TURN_OPEN, TURN_CLOSE],
            TOKENIZER.convert_tokens_to_ids(["<|turn>", "<turn|>"]),
        )
        self.assertEqual(
            [NEWLINE], TOKENIZER("\n", add_special_tokens=False)["input_ids"]
        )

    def test_single_turn_span_ends_at_turn_close(self):
        input_ids, loss_mask = _encode(
            [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."},
            ]
        )
        self.assertEqual(["The answer is 4.<turn|>"], _spans(input_ids, loss_mask))
        last = max(i for i, masked in enumerate(loss_mask) if masked)
        self.assertEqual(TURN_CLOSE, input_ids[last])
        # the '\n' after '<turn|>' is template text the target never generates
        self.assertEqual(NEWLINE, input_ids[last + 1])
        self.assertEqual(0, loss_mask[last + 1])

    def test_multi_turn_masks_only_the_model_replies(self):
        input_ids, loss_mask = _encode(self.MULTI_TURN)
        self.assertEqual(
            ["My name is Gemma.<turn|>", "11 years old.<turn|>"],
            _spans(input_ids, loss_mask),
        )
        self.assert_only_model_turns_masked(input_ids, loss_mask)

    def test_train_only_last_turn(self):
        input_ids, loss_mask = _encode(self.MULTI_TURN, train_only_last_turn=True)
        self.assertEqual(["11 years old.<turn|>"], _spans(input_ids, loss_mask))

    def test_exactly_one_bos_and_ids_match_the_official_template(self):
        input_ids, _ = _encode(self.MULTI_TURN)
        self.assertEqual(1, input_ids.count(BOS))
        self.assertEqual(BOS, input_ids[0])
        # no system turn is injected: the ids are the checkpoint template's own
        rendered = TOKENIZER.apply_chat_template(self.MULTI_TURN, tokenize=False)
        self.assertNotIn("<|turn>system", rendered)
        self.assertEqual(
            TOKENIZER(rendered, add_special_tokens=False)["input_ids"], input_ids
        )

    def test_data_system_turn_is_kept_but_not_trained(self):
        input_ids, loss_mask = _encode(
            [
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4."},
            ]
        )
        self.assertTrue(
            TOKENIZER.decode(input_ids).startswith(
                "<bos><|turn>system\nAnswer briefly.<turn|>\n"
            )
        )
        self.assertEqual(["4.<turn|>"], _spans(input_ids, loss_mask))
        self.assert_only_model_turns_masked(input_ids, loss_mask)

    def test_thought_channel_is_stripped_from_replies(self):
        input_ids, loss_mask = _encode(
            [
                {"role": "user", "content": "Hi"},
                {
                    "role": "assistant",
                    "content": "<|channel>thought\nGreet back.<channel|>Hello!",
                },
                {"role": "user", "content": "Bye"},
                {"role": "assistant", "content": "Goodbye."},
            ]
        )
        self.assertNotIn("<|channel>", TOKENIZER.decode(input_ids))
        self.assertEqual(
            ["Hello!<turn|>", "Goodbye.<turn|>"], _spans(input_ids, loss_mask)
        )

    def test_gemma3_template_finds_no_reply_on_gemma4(self):
        _, loss_mask = _encode(self.MULTI_TURN, template="gemma")
        self.assertEqual(0, sum(loss_mask))


if __name__ == "__main__":
    unittest.main()
