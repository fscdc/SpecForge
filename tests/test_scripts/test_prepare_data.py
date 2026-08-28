import io
import json
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import prepare_data


class TestPrepareData(unittest.TestCase):
    def test_parser_exposes_all_supported_presets(self):
        parser = prepare_data.build_parser()
        dataset_action = next(
            action for action in parser._actions if action.dest == "dataset"
        )

        self.assertEqual(prepare_data.SUPPORTED_DATASETS, tuple(dataset_action.choices))
        self.assertTrue(
            {
                "ultrachat",
                "sharegpt",
                "opc",
                "gsm8k",
                "hendrycks_math",
                "codealpaca-20k",
                "camel",
            }.issubset(dataset_action.choices)
        )
        self.assertTrue(
            prepare_data.UNSUPPORTED_VLM_DATASETS.isdisjoint(dataset_action.choices)
        )

    def test_vlm_presets_are_explicitly_unsupported(self):
        for dataset_name in prepare_data.UNSUPPORTED_VLM_DATASETS:
            with (
                self.subTest(dataset=dataset_name),
                self.assertRaisesRegex(ValueError, "VLM.*not supported"),
            ):
                prepare_data.load_dataset_preset(dataset_name)

    def test_parser_restores_eval_split_and_opc_subset(self):
        args = prepare_data.parse_args(
            [
                "--dataset",
                "opc",
                "--opc-subset",
                "all",
                "--split-eval",
            ]
        )

        self.assertEqual("all", args.opc_subset)
        self.assertTrue(args.split_eval)

    def test_custom_data_path_is_limited_to_sharegpt_json(self):
        for suffix in (".json", ".jsonl"):
            with self.subTest(suffix=suffix):
                args = prepare_data.parse_args(
                    [
                        "--dataset",
                        "sharegpt",
                        "--data-path",
                        f"custom{suffix}",
                    ]
                )
                self.assertEqual(Path(f"custom{suffix}"), args.data_path)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_data.parse_args(
                ["--dataset", "ultrachat", "--data-path", "custom.jsonl"]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            prepare_data.parse_args(
                ["--dataset", "sharegpt", "--data-path", "custom.csv"]
            )

    def test_custom_json_and_jsonl_use_the_hugging_face_json_loader(self):
        sentinel_dataset = object()
        for suffix in (".json", ".jsonl"):
            data_path = Path(f"custom{suffix}")
            with (
                self.subTest(suffix=suffix),
                patch.object(
                    prepare_data,
                    "_load_hf_dataset",
                    return_value=sentinel_dataset,
                ) as load_dataset,
            ):
                dataset = prepare_data.load_dataset_from_path(data_path)

                self.assertIs(sentinel_dataset, dataset)
                load_dataset.assert_called_once_with(
                    "json",
                    data_files=str(data_path),
                    split="train",
                )

    def test_sharegpt_conversion_normalizes_roles(self):
        row, skipped_count = prepare_data.process_sharegpt_row(
            {
                "id": 7,
                "conversations": [
                    {"from": "system", "value": "ignored"},
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                ],
            }
        )

        self.assertEqual(1, skipped_count)
        self.assertEqual(
            {
                "id": "7",
                "conversations": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ],
            },
            row,
        )

    def test_blend_conversion_keeps_the_source_column(self):
        row, _ = prepare_data.process_perfectblend_row(
            {
                "id": 7,
                "source": "meta-math/MetaMathQA",
                "conversations": [
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                ],
            }
        )

        # Regeneration reads the source back to find the math rows.
        self.assertEqual("meta-math/MetaMathQA", row["source"])

    def test_blend_conversion_tolerates_a_missing_source(self):
        row, _ = prepare_data.process_perfectblend_row(
            {
                "id": 7,
                "conversations": [{"from": "human", "value": "question"}],
            }
        )

        self.assertNotIn("source", row)

    def test_math_and_code_presets_produce_conversation_rows(self):
        math_row, skipped = prepare_data.process_gsm8k_row(
            {"question": "1 + 1?", "answer": "2"}
        )
        code_row, code_skipped = prepare_data.process_codealpaca_row(
            {"instruction": "return one", "output": "return 1"}
        )

        self.assertEqual(0, skipped)
        self.assertEqual(0, code_skipped)
        self.assertEqual("1 + 1?", math_row["conversations"][0]["content"])
        self.assertEqual("2", math_row["conversations"][1]["content"])
        self.assertEqual("return one", code_row["conversations"][0]["content"])
        self.assertEqual("return 1", code_row["conversations"][1]["content"])

    def test_gsm8k_preset_dispatches_to_its_hosted_dataset(self):
        sentinel_dataset = object()
        with patch.object(
            prepare_data,
            "_train_split",
            return_value=sentinel_dataset,
        ) as load_train:
            dataset, processor = prepare_data.load_dataset_preset("gsm8k")

        self.assertIs(sentinel_dataset, dataset)
        self.assertIs(prepare_data.process_gsm8k_row, processor)
        load_train.assert_called_once_with("openai/gsm8k", "main")

    def test_conversion_writes_only_the_training_jsonl(self):
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            output_path = prepare_data.process_and_save_dataset(
                [
                    {
                        "prompt_id": "prompt-1",
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"},
                        ],
                    }
                ],
                output_directory,
                prepare_data.process_ultrachat_row,
                "ultrachat",
            )

            self.assertEqual(output_directory / "ultrachat_train.jsonl", output_path)
            self.assertEqual([output_path], list(output_directory.iterdir()))
            self.assertEqual(
                {
                    "id": "prompt-1",
                    "conversations": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ],
                },
                json.loads(output_path.read_text()),
            )

    def test_conversion_can_write_the_restored_eval_split(self):
        with TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory)
            prepare_data.process_and_save_dataset(
                [
                    {
                        "prompt_id": "train",
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"},
                        ],
                    }
                ],
                output_directory,
                prepare_data.process_ultrachat_row,
                "ultrachat",
                eval_dataset=[
                    {
                        "prompt_id": "eval",
                        "messages": [
                            {"role": "user", "content": "eval question"},
                            {"role": "assistant", "content": "eval answer"},
                        ],
                    }
                ],
            )

            self.assertTrue((output_directory / "ultrachat_train.jsonl").exists())
            self.assertTrue((output_directory / "ultrachat_test.jsonl").exists())


class _SourceDataset:
    """The slice of the datasets API that the source selection actually uses."""

    def __init__(self, rows):
        self.rows = rows

    @property
    def column_names(self):
        return list(self.rows[0]) if self.rows else []

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def select(self, indices):
        return _SourceDataset([self.rows[index] for index in indices])


def _blended_dataset():
    """Stand-in for mlabonne/open-perfectblend's source column."""
    sizes = {
        "meta-math/MetaMathQA": 40,
        "openbmb/UltraInteract_sft": 30,
        "HuggingFaceH4/ultrafeedback_binarized": 20,
        "mlabonne/ultrachat_200k_sft": 10,
    }
    rows = []
    for source, count in sizes.items():
        rows.extend(
            {
                "id": f"{source}#{index}",
                "source": source,
                "conversations": [
                    {"from": "human", "value": f"q{index}"},
                    {"from": "gpt", "value": f"a{index}"},
                ],
            }
            for index in range(count)
        )
    return _SourceDataset(rows)


class TestSourceMixture(unittest.TestCase):
    def test_source_specs_parse_counts_and_all(self):
        self.assertEqual(
            prepare_data.parse_source_specs(["metamathqa=50000", "ultra", "x=all"]),
            [("metamathqa", 50000), ("ultra", None), ("x", None)],
        )

    def test_malformed_source_specs_are_rejected(self):
        for spec in ("=5", "x=abc", "x=0", "x=-3"):
            with self.assertRaises(ValueError):
                prepare_data.parse_source_specs([spec])

    def test_handles_match_sources_ignoring_punctuation_and_case(self):
        available = prepare_data.source_counts(_blended_dataset())
        self.assertEqual(
            prepare_data._resolve_source("metamathqa", available),
            "meta-math/MetaMathQA",
        )
        self.assertEqual(
            prepare_data._resolve_source("UltraInteract", available),
            "openbmb/UltraInteract_sft",
        )

    def test_unknown_and_ambiguous_handles_name_the_alternatives(self):
        available = prepare_data.source_counts(_blended_dataset())
        with self.assertRaisesRegex(ValueError, "matches none of"):
            prepare_data._resolve_source("nosuchthing", available)
        # "ultra" hits both ultrafeedback and ultrachat
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            prepare_data._resolve_source("ultra", available)

    def test_mixture_keeps_the_requested_count_per_source(self):
        dataset = _blended_dataset()
        selected = prepare_data.select_sources(
            dataset,
            prepare_data.parse_source_specs(["metamathqa=5", "ultrainteract=3"]),
            seed=42,
        )
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            prepare_data.source_counts(selected),
            {"meta-math/MetaMathQA": 5, "openbmb/UltraInteract_sft": 3},
        )

    def test_a_count_above_the_source_size_keeps_every_row(self):
        selected = prepare_data.select_sources(
            _blended_dataset(), [("ultrachat", 10_000)], seed=42
        )
        self.assertEqual(len(selected), 10)

    def test_the_draw_is_random_and_reproducible(self):
        dataset = _blended_dataset()
        first = [row["id"] for row in prepare_data.select_sources(
            dataset, [("metamathqa", 5)], seed=42
        )]
        again = [row["id"] for row in prepare_data.select_sources(
            dataset, [("metamathqa", 5)], seed=42
        )]
        other_seed = [row["id"] for row in prepare_data.select_sources(
            dataset, [("metamathqa", 5)], seed=7
        )]
        self.assertEqual(first, again)
        self.assertNotEqual(first, other_seed)
        # not simply the head of the source, which is what --sample-size gives
        self.assertNotEqual(first, [row["id"] for row in dataset.rows[:5]])

    def test_source_selection_needs_a_source_column(self):
        with self.assertRaisesRegex(ValueError, "no 'source' column"):
            prepare_data.source_counts(_SourceDataset([{"id": "1"}]))

    def test_existing_output_is_only_rewritten_with_overwrite(self):
        dataset = _blended_dataset()
        with TemporaryDirectory() as directory:
            output_directory = Path(directory)
            path = prepare_data.process_and_save_dataset(
                dataset, output_directory, prepare_data.process_sharegpt_row, "mix"
            )
            path.write_text("stale\n", encoding="utf-8")
            prepare_data.process_and_save_dataset(
                dataset, output_directory, prepare_data.process_sharegpt_row, "mix"
            )
            self.assertEqual(path.read_text(encoding="utf-8"), "stale\n")
            prepare_data.process_and_save_dataset(
                dataset,
                output_directory,
                prepare_data.process_sharegpt_row,
                "mix",
                overwrite=True,
            )
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
