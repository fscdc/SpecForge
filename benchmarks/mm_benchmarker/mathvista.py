"""
MathVista benchmark evaluation script.

MathVista (https://huggingface.co/datasets/AI4Math/MathVista) is a visual
mathematical-reasoning benchmark. Only the `testmini` split (1000 questions) is
annotated, so that is the one evaluated here. Every question is either
`multi_choice` with a list of options or `free_form` with an integer, float or
Python-list answer, and each carries a `metadata` block whose `task` field gives
the five-way breakdown the metrics are reported over.

Two prompts:

* **default** -- the shared `STEP_BY_STEP_BOXED_PROMPT`, as every other
  benchmark in this suite sends, so the generation lengths stay comparable.
  The question and its choices are rendered exactly as the task renders them;
  only the instruction differs, see `build_prompt`.
* **lmms_eval** (`mathvista-origin`) -- the dataset's own `query` column, which
  the MathVista authors built with the same `create_one_query` the lmms-eval
  task calls at `shot_num=0` with neither caption nor OCR. It opens with a
  `Hint:` line naming the answer format ("requiring an integer answer and
  provide the final value ... at the end").

Scoring is the rule-based half of the lmms-eval `mathvista` task:
`extract_answer` -> `normalize_extracted_answer` -> `safe_equal`. The task falls
back to a GPT judge whenever its quick rules find nothing; there is no judge
here, so two things stand in for it and both only ever fire when the rules
found nothing:

* the generation's `\\boxed{}` is read first, which is what our own prompt asks
  for and what the task never sees;
* for a numeric answer type, the last number in the generation is taken.

Both are documented departures, so a number produced here is a rule-based
approximation of the published one rather than the published one.
"""

import ast
import difflib
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import (
    STEP_BY_STEP_BOXED_PROMPT,
    create_image_sgl_function,
    extract_boxed,
    stratified_indices,
)

#: "default" is what we run, "lmms_eval" is the task's own query.
PROMPT_VARIANTS = ("default", "lmms_eval")

#: The first line of every stored `query`, naming the answer format. It is the
#: instruction the boxed prompt replaces, so it is cut rather than kept: keeping
#: it would ask for the answer twice, in two different places.
HINT_PREFIX = "Hint: "

#: The one shape the lmms-eval quick extraction looks for.
QUICK_ANSWER_PATTERN = re.compile(r'The answer is "(.*)"\.')

#: Last-resort numeric pull, see the module docstring.
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def strip_hint(query: str) -> str:
    """The stored query without its leading `Hint:` line."""
    head, separator, tail = query.partition("\n")
    if separator and head.startswith(HINT_PREFIX):
        return tail.strip()
    return query.strip()


def build_prompt(query: str, variant: str = "default") -> str:
    """
    The prompt for one question, for one of the two variants.

    Both are derived from the same stored `query`, so the question text and the
    choices block are rendered identically and the ONLY difference between a
    `mathvista` and a `mathvista-origin` run is the instruction. Reproducing the
    rendering by hand instead would risk drifting from the task on the spacing
    of the choices, which is exactly the kind of difference that moves a
    generation length without meaning anything.
    """
    if variant == "lmms_eval":
        return query
    return f"{strip_hint(query)}\n\n{STEP_BY_STEP_BOXED_PROMPT}"


def get_most_similar(prediction: str, choices: List[str]) -> str:
    """The option closest to the prediction, by difflib ratio, as the task does."""
    ratios = [
        difflib.SequenceMatcher(None, prediction, choice).ratio() for choice in choices
    ]
    return choices[ratios.index(max(ratios))]


def extract_answer(
    response: str,
    question_type: str,
    answer_type: str,
    choices: List[str],
    quick_extract: bool = True,
) -> str:
    """
    Reduce a generation to the answer it states.

    The rules of `MathVistaEvaluator.extract_answer`, with the boxed reading and
    the numeric fallback of the module docstring standing in for its GPT judge.
    """
    if not response:
        return ""

    # our own prompt asks for a box, so read it before anything else; the task
    # never sees one and has no rule for it
    boxed = extract_boxed(response)
    candidate = (boxed if boxed is not None else response).strip()

    if question_type == "multi_choice" and candidate in choices:
        return candidate

    if answer_type == "integer":
        try:
            return str(int(candidate))
        except ValueError:
            pass

    if answer_type == "float":
        try:
            return str(float(candidate))
        except ValueError:
            pass

    if quick_extract:
        match = QUICK_ANSWER_PATTERN.search(candidate)
        if match:
            return match.group(1)

    # Where the task would call its judge. For a numeric answer type the last
    # number of the generation is the best rule-based guess; for anything else
    # the text is handed on, and normalize_extracted_answer() matches it against
    # the choices.
    if answer_type in ("integer", "float"):
        numbers = NUMBER_PATTERN.findall(candidate.replace(",", ""))
        if numbers:
            return numbers[-1]
    return candidate


def normalize_extracted_answer(
    extraction: Any,
    choices: List[str],
    question_type: str,
    answer_type: str,
    precision: int,
) -> Optional[str]:
    """Put an extraction into the shape the reference answer is written in."""
    if question_type == "multi_choice":
        try:
            extraction = str(extraction).strip()
        except Exception:  # pragma: no cover - str() of a builtin type
            extraction = ""
        if not choices:
            return extraction or None

        # "(A) text" -> "A"
        letters = re.findall(r"\(([a-zA-Z])\)", extraction)
        if letters:
            extraction = letters[0].upper()

        sequential = [chr(ord("A") + index) for index in range(len(choices))]
        if extraction.upper() in sequential:
            return choices[sequential.index(extraction.upper())]
        if not extraction:
            return None
        return get_most_similar(extraction, choices)

    if answer_type == "integer":
        try:
            return str(int(float(extraction)))
        except (TypeError, ValueError):
            return None

    if answer_type == "float":
        try:
            return str(round(float(extraction), int(precision)))
        except (TypeError, ValueError):
            return None

    if answer_type == "list":
        try:
            return str(extraction)
        except Exception:  # pragma: no cover - str() of a builtin type
            return None

    return None


def safe_equal(prediction: Any, answer: Any) -> bool:
    """Whether the two answers are the same, never raising on an odd type."""
    try:
        return prediction == answer
    except Exception:  # pragma: no cover - comparison of exotic types
        return False


def score_answer(
    generation: str,
    answer: str,
    question_type: str,
    answer_type: str,
    choices: List[str],
    precision: int,
) -> Tuple[bool, Optional[str]]:
    """Whether a generation answers the question, and what it was reduced to."""
    extraction = extract_answer(generation, question_type, answer_type, choices)
    prediction = normalize_extracted_answer(
        extraction, choices, question_type, answer_type, precision
    )
    return safe_equal(prediction, answer), prediction


def _as_mapping(value: Any) -> Dict[str, Any]:
    """The `metadata` column, whether it arrives as a dict or as its repr."""
    if isinstance(value, dict):
        return value
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


@MM_BENCHMARKS.register("mathvista")
class MathVistaBenchmarker(MMBenchmarker):
    """
    MathVista benchmark implementation.

    Args:
        num_samples: number of questions to evaluate, all 1000 when not given.
            The selection is stratified over `task` x `question_type`, see
            `load_data`.
        subset: restrict the questions to one or more tasks, matched
            case-insensitively against `metadata["task"]`, e.g.
            `mathvista:200:"geometry problem solving"`. All of them when not
            given.
        split: the dataset split to evaluate. "testmini" is the only annotated
            one; "test" ships no answers and would score 0.
        prompt_variant: "default" sends the shared boxed instruction,
            "lmms_eval" the dataset's own `query`.
    """

    #: ``mathvista-origin``: the task's own "Hint: ... provide the final value
    #: ... at the end" query, sent verbatim.
    ORIGINAL_PROMPT_KWARGS = {"prompt_variant": "lmms_eval"}

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "testmini",
        prompt_variant: str = "default",
    ):
        super().__init__(num_samples, subset)
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt variant '{prompt_variant}', "
                f"expected any of {sorted(PROMPT_VARIANTS)}"
            )
        self.split = split
        self.prompt_variant = prompt_variant
        self.cache_dir = None
        # per-question metadata, kept aligned with the loaded questions
        self.tasks: List[str] = []
        self.question_types: List[str] = []
        self.answer_types: List[str] = []
        self.choices_list: List[List[str]] = []
        self.precisions: List[int] = []
        # per-question 1.0/0.0 of the last compute_accuracy() call
        self.hits: List[float] = []
        # what the scoring actually compared, for --save-generations
        self.parsed_answers: List[Optional[str]] = []

    def default_max_new_tokens(self) -> int:
        """
        Room for the chain of thought the boxed prompt asks for.

        A generation cut off before it reaches its box is scored wrong, so this
        sits well above the 2048 of the base class. The caller keeps the last
        word through `--max-tokens`.
        """
        return 4096

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """
        Load and preprocess MathVista.

        The rows are stored grouped by source dataset, so a plain head of the
        split would evaluate a handful of sources and call it the benchmark.
        The subset is stratified over `task` x `question_type` instead: `task`
        is the axis the metrics are broken down over, and `question_type`
        decides which scoring path a question takes, so both have to stay
        proportional for the accuracy to mean anything.
        """
        self.cache_dir = os.path.join(".cache", "mathvista_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("AI4Math/MathVista")[self.split]

        if self.subset:
            dataset = dataset.select(self._select_subset(dataset))
        if self.num_samples is not None:
            labels = [
                f"{_as_mapping(metadata).get('task', 'unknown')}|{question_type}"
                for metadata, question_type in zip(
                    dataset["metadata"], dataset["question_type"]
                )
            ]
            dataset = dataset.select(stratified_indices(labels, self.num_samples))

        questions: List[Dict[str, Any]] = []
        answers: List[Optional[str]] = []
        self.tasks = []
        self.question_types = []
        self.answer_types = []
        self.choices_list = []
        self.precisions = []
        for index, row in enumerate(dataset):
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            row["decoded_image"].convert("RGB").save(image_path, "PNG")

            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(row["query"], self.prompt_variant),
                }
            )
            metadata = _as_mapping(row.get("metadata"))
            self.tasks.append(str(metadata.get("task", "unknown")))
            self.question_types.append(str(row["question_type"]))
            self.answer_types.append(str(row["answer_type"]))
            self.choices_list.append([str(choice) for choice in (row["choices"] or [])])
            try:
                self.precisions.append(int(float(row["precision"] or 0)))
            except (TypeError, ValueError):
                self.precisions.append(0)
            answer = row["answer"]
            answers.append(None if answer is None else str(answer).strip())

        return questions, answers

    def _select_subset(self, dataset) -> List[int]:
        """
        Indices of the rows whose task matches the requested subset.

        Reads the metadata column directly, filtering row by row would decode
        every image on the way.
        """
        wanted = {name.strip().lower() for name in self.subset}
        tasks = [
            str(_as_mapping(metadata).get("task", "unknown")).strip().lower()
            for metadata in dataset["metadata"]
        ]
        unknown = wanted - set(tasks)
        if unknown:
            raise ValueError(
                f"Unknown MathVista task(s) {sorted(unknown)}, "
                f"expected any of {sorted(set(tasks))}"
            )
        return [index for index, task in enumerate(tasks) if task in wanted]

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """
        Keep the raw generation: the scoring needs the question's type, choices
        and precision, which compute_accuracy() looks up by index.
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Score every question, and report the overall accuracy."""
        self.hits = []
        self.parsed_answers = []
        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if label is None or not isinstance(prediction, str):
                self.hits.append(0.0)
                self.parsed_answers.append(None)
                continue
            hit, parsed = score_answer(
                prediction,
                label,
                self.question_types[index],
                self.answer_types[index],
                self.choices_list[index],
                self.precisions[index],
            )
            self.hits.append(float(hit))
            self.parsed_answers.append(parsed)

        if not self.hits:
            return None

        boxed = sum(
            1
            for prediction in predictions
            if isinstance(prediction, str) and "oxed{" in prediction
        )
        accuracy = sum(self.hits) / len(self.hits)
        print(
            f"MathVista: {boxed}/{len(predictions)} generations contain a "
            f"\\boxed{{}}, rule-based accuracy {accuracy:.4f}"
        )
        return accuracy

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, the prompt variant and how the scoring was done."""
        return {
            "split": self.split,
            "prompt_variant": self.prompt_variant,
            "questions": len(self.tasks),
            # the published metric calls a GPT judge whenever its rules find
            # nothing; this run never does, see the module docstring
            "scoring": "rule-based (no judge), boxed-first",
        }

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every task.

        The latency is the one of the whole run, so a task's throughput is its
        share of the aggregate rather than a figure it could reach on its own.
        """
        if not self.tasks:
            return None

        performance = {}
        for task in sorted(set(self.tasks)):
            indexes = [
                index
                for index, name in enumerate(self.tasks)
                if name == task and index < len(states)
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
            performance[task] = metrics

        print(
            "MathVista accuracy per task: "
            + ", ".join(
                f"{task}="
                + ("n/a" if metrics.accuracy is None else f"{metrics.accuracy:.4f}")
                for task, metrics in performance.items()
            )
        )
        return performance

    def create_sgl_function(self):
        """Create the SGL function for MathVista (one figure per question)."""
        return create_image_sgl_function(
            function_name="get_mathvista_answer",
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
