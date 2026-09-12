"""
MathVerse benchmark evaluation script.

MathVerse (https://huggingface.co/datasets/AI4Math/MathVerse) asks how much of a
model's "visual" maths is actually read off the figure. Its `testmini` config
holds 788 problems, each written out in FIVE versions that move information out
of the text and into the diagram:

    Text Dominant -> Text Lite -> Vision Intensive -> Vision Dominant -> Vision Only

`Vision Only` carries no question text at all -- the problem is rendered into
the image -- which is why the prompt has a branch for it.

Two prompts:

* **default** -- the shared `STEP_BY_STEP_BOXED_PROMPT`, as every other
  benchmark in this suite sends, so the generation lengths stay comparable.
* **lmms_eval** (`mathverse-origin`) -- the dataset's own `query_cot`, which
  asks to "first conduct reasoning, and then answer the question and provide
  the correct option letter ... at the end". `lmms_eval_direct` sends `query_wo`
  instead, the answer-directly variant of the same column.

Scoring is RULE-BASED and is not the published metric. The lmms-eval task scores
MathVerse with a GPT judge (`MathVerseEvaluator.score_answer(..., quick_match=
False)`); there is no judge here, so the generation is reduced with MathVision's
rule-based extraction and compared to the reference both as a letter and, when
the choices can be parsed back out of the question, as the option's text. Read
the accuracy as a lower bound on the judged number, not as a replacement for it
-- the accept length and the throughput, which is what this suite exists to
measure, are unaffected either way.
"""

import ast
import os
import re
import shutil
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .mathvision import extract_answer as extract_math_answer
from .mathvision import score_answer as score_math_answer
from .registry import MM_BENCHMARKS
from .utils import (
    STEP_BY_STEP_BOXED_PROMPT,
    create_image_sgl_function,
    stratified_indices,
)

#: "default" is what we run; the other two are the dataset's own query columns.
PROMPT_VARIANTS = {
    "default": None,
    "lmms_eval": "query_cot",
    "lmms_eval_direct": "query_wo",
}

#: What a `Vision Only` question says instead of nothing at all. The dataset's
#: own queries open the same way ("According to the question shown in the
#: image, ..."), so the boxed variant has to say it too or the model is asked to
#: solve a problem it was never given.
VISION_ONLY_QUESTION = "The question is shown in the image."

#: "A:40°" / "A: 40°" / "A.40°", one option per line, after a "Choices:" line.
CHOICE_PATTERN = re.compile(r"^\s*([A-Z])\s*[:.]\s*(.*?)\s*$")


def build_prompt(question: str, variant: str = "default", **queries: str) -> str:
    """
    The prompt for one row, for one of the three variants.

    The two `lmms_eval` variants are the dataset's stored query columns, sent
    verbatim -- they already carry the question, the choices and the answer
    format, including the `Vision Only` wording.
    """
    column = PROMPT_VARIANTS[variant]
    if column is not None:
        return queries[column]
    question = (question or "").strip()
    if not question:
        return f"{VISION_ONLY_QUESTION}\n\n{STEP_BY_STEP_BOXED_PROMPT}"
    return f"{question}\n\n{STEP_BY_STEP_BOXED_PROMPT}"


def parse_choices(question_for_eval: str) -> List[str]:
    """
    The options of a multiple-choice question, read back out of its text.

    MathVerse writes them as "Choices:\\nA:40°\\nB:60°...", and the reference is
    the letter. Recovering the texts lets a generation that boxed "140°" instead
    of "D" still be credited. Returns [] when the block is absent or ragged.
    """
    _, separator, tail = question_for_eval.partition("Choices:")
    if not separator:
        return []
    options: List[str] = []
    expected = "A"
    for line in tail.splitlines():
        if not line.strip():
            continue
        match = CHOICE_PATTERN.match(line)
        # the options have to arrive in order and without gaps, otherwise the
        # index of the reference letter would point at the wrong text
        if not match or match.group(1) != expected:
            break
        options.append(match.group(2))
        expected = chr(ord(expected) + 1)
    return options


def usable_choices(question_for_eval: str, answer: str) -> List[str]:
    """
    The parsed choices, but only when the parse is self-consistent.

    A misparse would hand the scorer the text of the WRONG option as a second
    reference, crediting an incorrect generation -- worse than not parsing at
    all, since scoring by letter alone is already correct. Mirrors DynaMath's
    `_usable_options`.
    """
    options = parse_choices(question_for_eval)
    if not options or not all(option.strip() for option in options):
        return []
    answer = (answer or "").strip()
    if len(answer) != 1 or not answer.isalpha():
        return []
    index = ord(answer.upper()) - ord("A")
    return options if 0 <= index < len(options) else []


def _as_mapping(value: Any) -> Dict[str, Any]:
    """The `metadata` column, whether it arrives as a dict or as its repr."""
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


@MM_BENCHMARKS.register("mathverse")
class MathVerseBenchmarker(MMBenchmarker):
    """
    MathVerse benchmark implementation.

    Args:
        num_samples: number of rows to evaluate, all 3940 when not given.
            Truncation is group-aware, see `load_data`: it keeps whole problems,
            so every selected problem contributes all five of its versions.
        subset: restrict the rows to one or more problem versions or subjects,
            matched case-insensitively against the `problem_version` column and
            `metadata["subject"]`, e.g. `mathverse:200:"vision only"`. All of
            them when not given.
        split: the dataset split to evaluate. "testmini" is the only annotated
            one.
        prompt_variant: "default" sends the shared boxed instruction,
            "lmms_eval" the dataset's `query_cot`, "lmms_eval_direct" its
            `query_wo`.
        match: "lenient" also credits an answer that states the numeric
            reference without isolating it, "exact" is MathVision's rule-based
            comparison. Same meaning as MathVision's.
    """

    #: ``mathverse-origin``: the task's own "first conduct reasoning, then
    #: provide the correct option letter at the end" query, sent verbatim.
    ORIGINAL_PROMPT_KWARGS = {"prompt_variant": "lmms_eval"}

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "testmini",
        prompt_variant: str = "default",
        match: str = "lenient",
    ):
        super().__init__(num_samples, subset)
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt variant '{prompt_variant}', "
                f"expected any of {sorted(PROMPT_VARIANTS)}"
            )
        if match not in ("lenient", "exact"):
            raise ValueError(f"Unknown match mode '{match}', expected exact or lenient")
        self.split = split
        self.prompt_variant = prompt_variant
        self.match = match
        self.cache_dir = None
        # per-row metadata, kept aligned with the loaded questions
        self.problem_versions: List[str] = []
        self.subjects: List[str] = []
        self.problem_indexes: List[str] = []
        self.choices_list: List[List[str]] = []
        # per-row 1.0/0.0 of the last compute_accuracy() call
        self.hits: List[float] = []
        # problems whose versions were not all evaluated, see load_data
        self.partial_problems: int = 0
        # what the scoring actually compared, for --save-generations
        self.parsed_answers: List[Optional[str]] = []

    def default_max_new_tokens(self) -> int:
        """
        Room for the chain of thought both prompts ask for.

        A generation cut off before it reaches its answer is scored wrong, so
        this sits well above the 2048 of the base class. The caller keeps the
        last word through `--max-tokens`.
        """
        return 4096

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """
        Load MathVerse, keeping every problem's five versions together.

        The whole point of the benchmark is the comparison ACROSS versions of
        the same problem, so a truncation that left different problems in
        different versions would make the per-version breakdown a comparison of
        different question sets. Problems are therefore selected whole, and
        stratified over `metadata["subject"]` so the three subjects stay
        proportional; only a `num_samples` that is not a multiple of the version
        count can split the last one.
        """
        self.cache_dir = os.path.join(".cache", "mathverse_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        # the config and the split carry the same name here ("testmini"), but
        # the text-only config names its split differently, so fall back to the
        # single split a config ships rather than assuming the two match
        splits = load_dataset("AI4Math/MathVerse", self.split)
        dataset = (
            splits[self.split]
            if self.split in splits
            else next(iter(splits.values()))
        )

        indexes = (
            self._select_subset(dataset) if self.subset else list(range(len(dataset)))
        )
        indexes = self._truncate_by_problem(dataset, indexes)
        dataset = dataset.select(indexes)

        questions: List[Dict[str, Any]] = []
        answers: List[Optional[str]] = []
        self.problem_versions = []
        self.subjects = []
        self.problem_indexes = []
        self.choices_list = []
        for index, row in enumerate(dataset):
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            row["image"].convert("RGB").save(image_path, "PNG")

            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(
                        row["question"],
                        self.prompt_variant,
                        query_cot=row["query_cot"],
                        query_wo=row["query_wo"],
                    ),
                }
            )
            metadata = _as_mapping(row.get("metadata"))
            self.problem_versions.append(str(row["problem_version"]))
            self.subjects.append(str(metadata.get("subject", "unknown")))
            self.problem_indexes.append(str(row["problem_index"]))
            answer = row["answer"]
            answer = None if answer is None else str(answer).strip()
            answers.append(answer)
            # read off question_for_eval, which carries the full text for every
            # version -- `question` is empty on Vision Only
            self.choices_list.append(
                usable_choices(row["question_for_eval"], answer or "")
            )

        return questions, answers

    def _group_by_problem(self, dataset, indexes: List[int]) -> "OrderedDict":
        """Row indices grouped by problem, in dataset order."""
        problem_indexes = dataset["problem_index"]
        groups: "OrderedDict[str, List[int]]" = OrderedDict()
        for index in indexes:
            groups.setdefault(str(problem_indexes[index]), []).append(index)
        return groups

    def _truncate_by_problem(self, dataset, indexes: List[int]) -> List[int]:
        """Keep `num_samples` rows without splitting a problem, where possible."""
        self.partial_problems = 0
        if self.num_samples is None or self.num_samples >= len(indexes):
            return indexes

        groups = self._group_by_problem(dataset, indexes)
        names = list(groups)
        versions = len(groups[names[0]])
        # rounded UP, so that a `num_samples` which is not a multiple of the
        # version count still delivers the number of rows that was asked for --
        # the last problem is then cut, and counted in `partial_problems`
        wanted_problems = max(1, -(-self.num_samples // max(versions, 1)))

        metadata = dataset["metadata"]
        labels = [
            str(_as_mapping(metadata[groups[name][0]]).get("subject", "unknown"))
            for name in names
        ]
        chosen = [
            names[position]
            for position in stratified_indices(labels, wanted_problems)
        ]

        selected: List[int] = []
        for name in chosen:
            rows = groups[name]
            remaining = self.num_samples - len(selected)
            if remaining <= 0:
                break
            if remaining < len(rows):
                self.partial_problems += 1
                rows = rows[:remaining]
            selected.extend(rows)

        if self.partial_problems:
            print(
                f"Warning: {self.partial_problems} problem(s) were cut mid-way by "
                f"--num-samples {self.num_samples}; the per-version breakdown of "
                "those rows compares different problems. Use a multiple of "
                f"{versions} to avoid it."
            )
        return sorted(selected)

    def _select_subset(self, dataset) -> List[int]:
        """
        Indices of the rows whose version or subject matches the requested
        subset.

        Reads the metadata columns directly, filtering row by row would decode
        every image on the way.
        """
        wanted = {name.strip().lower() for name in self.subset}
        versions = [str(value).strip().lower() for value in dataset["problem_version"]]
        subjects = [
            str(_as_mapping(metadata).get("subject", "unknown")).strip().lower()
            for metadata in dataset["metadata"]
        ]
        unknown = wanted - set(versions) - set(subjects)
        if unknown:
            raise ValueError(
                f"Unknown MathVerse version/subject {sorted(unknown)}, expected "
                f"any of {sorted(set(versions) | set(subjects))}"
            )
        return [
            index
            for index, (version, subject) in enumerate(zip(versions, subjects))
            if version in wanted or subject in wanted
        ]

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """
        Keep the raw generation: the scoring needs the row's choices, which
        compute_accuracy() looks up by index.
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """
        Score every row with MathVision's rule-based comparison.

        NOT the published metric, which is GPT-judged; see the module docstring.
        """
        self.hits = []
        self.parsed_answers = []
        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if label is None or not isinstance(prediction, str):
                self.hits.append(0.0)
                self.parsed_answers.append(None)
                continue
            choices = self.choices_list[index] if index < len(self.choices_list) else []
            self.parsed_answers.append(extract_math_answer(prediction, choices))
            self.hits.append(
                float(score_math_answer(prediction, label, choices, self.match))
            )

        if not self.hits:
            return None

        boxed = sum(
            1
            for prediction in predictions
            if isinstance(prediction, str) and "oxed{" in prediction
        )
        accuracy = sum(self.hits) / len(self.hits)
        print(
            f"MathVerse: {boxed}/{len(predictions)} generations contain a "
            f"\\boxed{{}}, rule-based accuracy {accuracy:.4f} "
            "(the published metric is GPT-judged; this is a lower bound)"
        )
        return accuracy

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, the prompt variant and how the scoring was done."""
        description: Dict[str, Any] = {
            "split": self.split,
            "prompt_variant": self.prompt_variant,
            "match": self.match,
            "questions": len(self.problem_versions),
            "problems": len(set(self.problem_indexes)),
            # the published metric calls a GPT judge; this run never does
            "scoring": "rule-based (no judge), not comparable to published numbers",
        }
        if self.partial_problems:
            description["partial_problems"] = self.partial_problems
        return description

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every problem version.

        That is the axis MathVerse exists to measure, and it is also the one
        that moves the prompt length: `Text Dominant` states everything in text,
        `Vision Only` states none of it.

        The latency is the one of the whole run, so a version's throughput is
        its share of the aggregate rather than a figure it could reach on its
        own.
        """
        if not self.problem_versions:
            return None

        performance = {}
        for version in sorted(set(self.problem_versions)):
            indexes = [
                index
                for index, name in enumerate(self.problem_versions)
                if name == version and index < len(states)
            ]
            if not indexes:
                continue
            metrics = compute_metrics(
                [states[index] for index in indexes], latency, answer_key=answer_key
            )
            hits = [self.hits[index] for index in indexes if index < len(self.hits)]
            if hits:
                metrics.accuracy = sum(hits) / len(hits)
                metrics.num_valid_predictions = len(hits)
            performance[version] = metrics

        print(
            "MathVerse accuracy per version: "
            + ", ".join(
                f"{version}="
                + ("n/a" if metrics.accuracy is None else f"{metrics.accuracy:.4f}")
                for version, metrics in performance.items()
            )
        )
        return performance

    def create_sgl_function(self):
        """Create the SGL function for MathVerse (one figure per question)."""
        return create_image_sgl_function(
            function_name="get_mathverse_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            assistant_prefix=self.assistant_prefix,
        )

    def run(self, *args, **kwargs):
        """Run benchmark and clean up cache directory."""
        try:
            return super().run(*args, **kwargs)
        finally:
            if self.cache_dir and os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                print(f"Deleted temporary directory: {self.cache_dir}")
