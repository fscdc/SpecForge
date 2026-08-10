"""
RealWorldQA benchmark evaluation script.

RealWorldQA (https://huggingface.co/datasets/xai-org/RealworldQA) asks about
real-world driving and indoor scenes. Part of the questions are four-way multiple
choice, the rest expect a short open answer (a number, "Downhill", …), and the
metrics are reported overall and per question type.

The prompt and the scoring are ported from the lmms-eval `realworldqa` task: the
question is sent as-is, since it already carries its own answer instruction, and
the comparison depends on the reference:

- a reference among A/B/C/D means the generation is reduced to an option letter;
- anything else is compared as text, lowercased and without its trailing period.

That text comparison is lenient by default: a generation also counts when it
states the reference ("the road goes downhill" for "Downhill") or spells a number
out ("two" for "2"). The strict equality the task uses is reported alongside as
`exact_match`, and can be made the metric with `match="exact"`.
"""

import os
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import (
    CHOICES,
    create_image_sgl_function,
    extract_choice,
    reference_in_prediction,
)

# the questions carry this instruction themselves, which the task strips whenever
# it appends a prompt of its own
REPLACE_PROMPT = (
    "Please answer directly with only the letter of the correct option and nothing else."
)

# from the task's own NumberWordsToDigitsFilter, which its yaml never wires up
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

MULTIPLE_CHOICE = "multiple-choice"
OPEN_ENDED = "open-ended"


def build_prompt(question: str, pre_prompt: str = "", post_prompt: str = "") -> str:
    """
    The prompt of the lmms-eval task.

    With the default (empty) prompts the question is sent as-is, keeping the
    answer instruction it already contains. A task variant that appends its own
    instruction strips that one first.
    """
    question = question.strip()
    if post_prompt:
        question = question.replace(REPLACE_PROMPT, "")
    return f"{pre_prompt}{question}{post_prompt}"


def is_multiple_choice(answer: str) -> bool:
    """Whether the reference is an option letter rather than an open answer."""
    return str(answer).strip().upper() in CHOICES


def normalize_open_answer(text: str) -> str:
    """Lowercase and drop the trailing period, as the task does."""
    return str(text).lower().strip().rstrip(".")


def score_answer(generation: str, answer: str, match: str = "lenient") -> float:
    """
    Score one generation against its reference.

    "exact" is the metric of the lmms-eval task. "lenient" additionally accepts an
    open answer that is spelled out as a number word, or stated inside a longer
    sentence.
    """
    reference = str(answer).strip()
    if is_multiple_choice(reference):
        # a letter is a letter, there is nothing to loosen here
        return float(extract_choice(generation, CHOICES) == reference.upper())

    prediction = normalize_open_answer(generation)
    reference = reference.lower()
    if prediction == reference:
        return 1.0
    if match != "lenient":
        return 0.0

    if NUMBER_WORDS.get(prediction, prediction) == NUMBER_WORDS.get(
        reference, reference
    ):
        return 1.0
    return float(reference_in_prediction(prediction, reference))


@MM_BENCHMARKS.register("realworldqa")
class RealWorldQABenchmarker(MMBenchmarker):
    """
    RealWorldQA benchmark implementation.

    Args:
        num_samples: number of questions to evaluate, all of them when not given.
        subset: restrict the questions to "multiple-choice" or "open-ended", which
            is the only grouping the dataset allows. Both when not given.
        split: the dataset split to evaluate, "test" is the only one.
        match: "lenient" also credits an open answer stated inside a sentence or
            spelled out as a number word, "exact" is the metric of the lmms-eval
            task. Both numbers are reported either way.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "test",
        match: str = "lenient",
    ):
        super().__init__(num_samples, subset)
        if match not in ("lenient", "exact"):
            raise ValueError(f"Unknown match mode '{match}', expected exact or lenient")
        unknown = {name.strip().lower() for name in (subset or [])} - {
            MULTIPLE_CHOICE,
            OPEN_ENDED,
        }
        if unknown:
            raise ValueError(
                f"Unknown RealWorldQA subset(s) {sorted(unknown)}, "
                f"expected {MULTIPLE_CHOICE} or {OPEN_ENDED}"
            )
        self.match = match
        self.split = split
        self.cache_dir = None
        # per-question type, kept aligned with the loaded questions
        self.question_types: List[str] = []
        # per-question 1.0/0.0 of the last compute_accuracy() call, under the
        # selected mode and under the strict one
        self.hits: List[float] = []
        self.exact_hits: List[float] = []
        self.exact_match: Optional[float] = None

    def default_max_new_tokens(self) -> int:
        """The lmms-eval task answers in at most 16 tokens."""
        return 16

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess the RealWorldQA dataset."""
        self.cache_dir = os.path.join(".cache", "realworldqa_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("xai-org/RealworldQA")[self.split]

        if self.subset:
            wanted = {name.strip().lower() for name in self.subset}
            keep = [
                index
                for index, answer in enumerate(dataset["answer"])
                if (MULTIPLE_CHOICE if is_multiple_choice(answer) else OPEN_ENDED)
                in wanted
            ]
            dataset = dataset.select(keep)
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels = []
        self.question_types = []
        for index, row in enumerate(dataset):
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            row["image"].convert("RGB").save(image_path, "PNG")

            questions.append(
                {"image_path": image_path, "question": build_prompt(row["question"])}
            )
            answer = str(row["answer"]).strip()
            labels.append(answer or None)
            self.question_types.append(
                MULTIPLE_CHOICE if is_multiple_choice(answer) else OPEN_ENDED
            )

        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """
        Keep the raw generation: how it has to be read depends on the reference,
        which compute_accuracy() looks up by index.
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Score every question, overall and per question type."""
        self.hits = []
        self.exact_hits = []
        by_type: Dict[str, List[float]] = defaultdict(list)

        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if label is None or not isinstance(prediction, str):
                self.hits.append(0.0)
                self.exact_hits.append(0.0)
                continue
            hit = score_answer(prediction, label, self.match)
            self.hits.append(hit)
            self.exact_hits.append(score_answer(prediction, label, "exact"))
            question_type = (
                self.question_types[index]
                if index < len(self.question_types)
                else OPEN_ENDED
            )
            by_type[question_type].append(hit)

        if not self.hits:
            return None

        accuracy = sum(self.hits) / len(self.hits)
        self.exact_match = sum(self.exact_hits) / len(self.exact_hits)
        print(
            f"RealWorldQA {self.match} match over {len(self.hits)} questions: "
            f"{accuracy:.4f}, exact {self.exact_match:.4f}"
        )
        print(
            "  "
            + ", ".join(
                f"{name}={sum(scores) / len(scores):.4f} ({len(scores)} questions)"
                for name, scores in sorted(by_type.items())
            )
        )
        return accuracy

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of the multiple choice and the open questions, the only
        grouping this dataset carries.

        The latency is the one of the whole run, so a group's throughput is its
        share of the aggregate rather than a figure it could reach on its own.
        """
        if not self.question_types:
            return None

        performance = {}
        for question_type in (MULTIPLE_CHOICE, OPEN_ENDED):
            indexes = [
                index
                for index, name in enumerate(self.question_types)
                if name == question_type and index < len(states)
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
            performance[question_type] = metrics
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, the match mode and the strict accuracy alongside."""
        description: Dict[str, Any] = {
            "split": self.split,
            "match": self.match,
            "questions": len(self.question_types),
        }
        if self.exact_match is not None:
            description["exact_match"] = self.exact_match
        return description

    def create_sgl_function(self):
        """Create the SGL function for RealWorldQA (image-based Q&A)."""
        return create_image_sgl_function(
            function_name="get_realworldqa_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            assistant_prefix=self.assistant_prefix,
        )

    def run(self, *args, **kwargs):
        """Run benchmark and clean up cache directory."""
        try:
            return super().run(*args, **kwargs)
        finally:
            # clean up cache directory
            if self.cache_dir and os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                print(f"Deleted temporary directory: {self.cache_dir}")
