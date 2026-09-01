import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts import regenerate_train_data
from tests.utils import (
    execute_shell_command,
    get_available_port,
    terminate_process_trees,
    wait_for_server,
)

CACHE_DIR = Path(__file__).parent.parent.parent.joinpath("cache")


class TestRegenerateTrainData(unittest.TestCase):
    def test_regenerate_sharegpt(self):
        port = get_available_port()
        data_process = execute_shell_command(
            "python scripts/prepare_data.py --dataset sharegpt"
        )
        data_process.wait()

        sglang_process = execute_shell_command(
            f"""python3 -m sglang.launch_server \
    --model unsloth/Llama-3.2-1B-Instruct \
    --tp 1 \
    --cuda-graph-bs 4 \
    --dtype bfloat16 \
    --mem-frac=0.8 \
    --port {port}
        """,
            disable_proxy=True,
            enable_hf_mirror=False,
            sglang_use_modelscope=True,
            start_new_session=True,
        )
        try:
            wait_for_server(
                f"http://localhost:{port}",
                timeout=300,
                disable_proxy=True,
                process=sglang_process,
            )
            regeneration_process = execute_shell_command(
                f"""python scripts/regenerate_train_data.py \
    --model unsloth/Llama-3.2-1B-Instruct \
    --concurrency 128 \
    --max-tokens 128 \
    --server-address localhost:{port} \
    --temperature 0.8 \
    --input-file-path ./cache/dataset/sharegpt_train.jsonl \
    --output-file-path ./cache/dataset/sharegpt_train_regen.jsonl \
    --num-samples 10
        """,
                disable_proxy=True,
                enable_hf_mirror=False,
            )
            regeneration_process.wait()
            self.assertEqual(regeneration_process.returncode, 0)
            self.assertTrue(
                CACHE_DIR.joinpath("dataset", "sharegpt_train_regen.jsonl").exists()
            )
        finally:
            terminate_process_trees(sglang_process, grace_s=30)


class TestMathCotPrompt(unittest.TestCase):
    """The math prompt regeneration uses must match what the benchmarks send."""

    def _row(self, source, prompt="What is 2+2?"):
        return {
            "id": "1",
            "source": source,
            "conversations": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "4"},
            ],
        }

    def _apply(self, row):
        return regenerate_train_data.apply_math_cot_prompt(
            row,
            regenerate_train_data.MATH_COT_SUFFIX,
            regenerate_train_data.DEFAULT_MATH_FAMILIES,
            regenerate_train_data.DEFAULT_MATH_SOURCES,
        )

    def test_suffix_matches_the_text_benchmarks(self):
        benchmark = Path(__file__).parent.parent.parent / "benchmarks" / "bench_text.py"
        declared = re.search(
            r"^COT_SUFFIX = (.+)$", benchmark.read_text(), re.MULTILINE
        )
        self.assertIsNotNone(declared, "benchmarks/bench_text.py lost COT_SUFFIX")
        self.assertEqual(
            eval(declared.group(1)),  # noqa: S307 - a string literal from our own repo
            regenerate_train_data.MATH_COT_SUFFIX,
        )

    def test_math_rows_get_the_benchmark_instruction(self):
        for source in (
            "meta-math/MetaMathQA",
            "HuggingFaceH4/orca-math-word-problems-200k",
        ):
            with self.subTest(source=source):
                row = self._row(source)
                self.assertTrue(self._apply(row))
                self.assertEqual(
                    "What is 2+2?" + regenerate_train_data.MATH_COT_SUFFIX,
                    row["conversations"][0]["content"],
                )

    def test_other_subsets_and_sourceless_rows_are_untouched(self):
        for source in (
            "openbmb/UltraInteract_sft",
            "mlabonne/ultrachat_200k_sft",
            "HuggingFaceH4/ultrafeedback_binarized",
            "theblackcat102/evol-codealpaca-v1",
            "Post-training-Data-Flywheel/AutoIF-instruct-61k",
            "mlabonne/lmsys-arena-human-preference-55k-sharegpt",
        ):
            with self.subTest(source=source):
                row = self._row(source)
                self.assertFalse(self._apply(row))
                self.assertEqual("What is 2+2?", row["conversations"][0]["content"])

        sourceless = {"id": "1", "conversations": [{"role": "user", "content": "hi"}]}
        self.assertFalse(self._apply(sourceless))

    def test_the_instruction_is_never_stacked_twice(self):
        row = self._row("meta-math/MetaMathQA")
        self.assertTrue(self._apply(row))
        self.assertFalse(self._apply(row))
        self.assertEqual(1, row["conversations"][0]["content"].count("step by step"))

    def test_only_the_first_user_turn_is_rewritten(self):
        row = {
            "id": "1",
            "source": "meta-math/MetaMathQA",
            "conversations": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ],
        }
        self.assertTrue(self._apply(row))
        self.assertEqual("You are helpful.", row["conversations"][0]["content"])
        self.assertTrue(row["conversations"][1]["content"].startswith("first\n"))
        self.assertEqual("second", row["conversations"][3]["content"])

    def test_an_input_without_sources_is_detected_up_front(self):
        with TemporaryDirectory() as directory:
            blend = Path(directory, "blend.jsonl")
            blend.write_text(
                json.dumps(self._row("meta-math/MetaMathQA")) + "\n", encoding="utf-8"
            )
            plain = Path(directory, "plain.jsonl")
            plain.write_text(
                json.dumps({"id": "1", "conversations": []}) + "\n", encoding="utf-8"
            )
            empty = Path(directory, "empty.jsonl")
            empty.write_text("", encoding="utf-8")

            # a text blend labels its math rows with `source`, a
            # LLaVA-OneVision file with the config in its id, and a file with
            # neither aligns nothing -- which is what the run header warns about
            self.assertEqual(
                "source", regenerate_train_data.input_math_annotation(str(blend))
            )
            self.assertIsNone(regenerate_train_data.input_math_annotation(str(plain)))
            self.assertIsNone(regenerate_train_data.input_math_annotation(str(empty)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
