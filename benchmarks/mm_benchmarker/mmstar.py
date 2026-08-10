"""
MMStar benchmark evaluation script.

MMStar (https://huggingface.co/datasets/Lin-Chen/MMStar) is a vision-indispensable
benchmark: every sample was picked so that the question cannot be answered from
the text alone. It is evaluated on the `val` split, the only one that ships the
answers, and every question is multiple choice with a letter as the reference.

The prompt and the scoring are ported from the lmms-eval `mmstar` task:

- the options are already part of the `question` field, so only the instruction
  "Answer with the option's letter from the given choices directly" is appended,
  after removing the " Please answer yes or no." some questions end with;
- the generation is reduced to one of A/B/C/D and compared to the reference;
- the score is a macro average over the 18 l2 categories, not a plain mean over
  the questions, and each of the 6 categories is aggregated the same way over the
  l2 categories it contains.

The answers are short, so the throughput is more prefill-bound here than on a
chain-of-thought benchmark like `chartqa`, while the accept length stays
meaningful.
"""

import os
import re
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import CHOICES, create_image_sgl_function, extract_choice

# lmms_eval_specific_kwargs of the task: "default" is what published numbers use,
# "qwen3_vl" is the variant the task ships for the Qwen3-VL family
PROMPT_VARIANTS = {
    "default": {
        "pre_prompt": "",
        "post_prompt": "\nAnswer with the option's letter from the given choices directly",
    },
    "qwen3_vl": {
        "pre_prompt": "Question: ",
        "post_prompt": "Answer with the option letter only.",
    },
}

# some questions end with this, which the task strips before adding its own
REPLACE_PROMPT = " Please answer yes or no."

# the l2 categories of every category, used for the macro average
EVAL_TYPE_DICT = {
    "coarse perception": [
        "image scene and topic",
        "image style & quality",
        "image emotion",
    ],
    "fine-grained perception": ["object counting", "recognition", "localization"],
    "instance reasoning": [
        "single-instance reasoning",
        "cross-instance attribute reasoning",
        "cross-instance relation reasoning",
    ],
    "logical reasoning": [
        "code & sequence reasoning",
        "diagram reasoning",
        "common reasoning",
    ],
    "science & technology": [
        "biology & chemistry & physics",
        "electronics & energy & mechanical eng.",
        "geography & earth science & agriculture",
    ],
    "math": [
        "geometry",
        "numeric commonsense and calculation",
        "statistical reasoning",
    ],
}

# the fields a --benchmark-list subset is matched against, e.g. mmstar:200:math
SUBSET_COLUMNS = ("category", "l2_category")


def build_prompt(question: str, variant: str = "default") -> str:
    """The prompt of the lmms-eval task, for one of its prompt variants."""
    pre_prompt = PROMPT_VARIANTS[variant]["pre_prompt"]
    post_prompt = PROMPT_VARIANTS[variant]["post_prompt"]

    question = question.strip()
    if pre_prompt:
        question = question.replace(REPLACE_PROMPT, "")
        question = f"{pre_prompt}{question}"
    if post_prompt:
        question = question.replace(REPLACE_PROMPT, "")
        question = f"{question}{post_prompt}"
    return question


def macro_average(scores_by_group: Dict[str, List[float]]) -> Optional[float]:
    """Mean of the per-group means, which is how the task aggregates."""
    if not scores_by_group:
        return None
    group_means = [sum(scores) / len(scores) for scores in scores_by_group.values()]
    return sum(group_means) / len(group_means)


@MM_BENCHMARKS.register("mmstar")
class MMStarBenchmarker(MMBenchmarker):
    """
    MMStar benchmark implementation.

    Args:
        num_samples: number of samples to evaluate, all of them when not given.
        subset: restrict the questions to one or more categories, matched
            case-insensitively against the `category` and `l2_category` columns,
            e.g. `mmstar:200:math`. All of them when not given.
        split: the dataset split to evaluate, "val" is the only annotated one.
        prompt_variant: which `lmms_eval_specific_kwargs` block to use, "default"
            (what published numbers use) or "qwen3_vl".
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "val",
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
        self.categories: List[str] = []
        self.l2_categories: List[str] = []
        # per-question 1.0/0.0 of the last compute_accuracy() call
        self.hits: List[float] = []
        # macro average per category and per l2 category of that call
        self.category_scores: Dict[str, float] = {}
        self.l2_category_scores: Dict[str, float] = {}

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess the MMStar dataset."""
        self.cache_dir = os.path.join(".cache", "mmstar_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("Lin-Chen/MMStar")[self.split]

        if self.subset:
            dataset = dataset.select(self._select_subset(dataset))
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels = []
        self.categories = []
        self.l2_categories = []
        for index, row in enumerate(dataset):
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            row["image"].convert("RGB").save(image_path, "PNG")

            # the options are part of the question, so the whole field is kept:
            # dropping them would hide the choices from the model
            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(row["question"], self.prompt_variant),
                }
            )

            answer = str(row["answer"]).strip().upper()
            labels.append(answer or None)
            self.categories.append(str(row.get("category", "unknown")))
            self.l2_categories.append(str(row.get("l2_category", "unknown")))

        return questions, labels

    def _select_subset(self, dataset) -> List[int]:
        """
        Indices of the rows whose category matches the requested subset.

        Reads the label columns directly, filtering row by row would decode every
        image on the way.
        """
        wanted = {name.strip().lower() for name in self.subset}
        columns = [
            [str(value).strip().lower() for value in dataset[column]]
            for column in SUBSET_COLUMNS
            if column in dataset.column_names
        ]
        available = {value for column in columns for value in column}
        unknown = wanted - available
        if unknown:
            raise ValueError(
                f"Unknown MMStar subset(s) {sorted(unknown)}, "
                f"expected any of {sorted(available)}"
            )
        return [
            index
            for index in range(len(dataset))
            if any(column[index] in wanted for column in columns)
        ]

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """Reduce the generation to the option letter it settles on."""
        return extract_choice(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """
        Score every question and report the macro average over the l2 categories,
        which is the "average" metric of the task.
        """
        self.hits = []
        by_l2: Dict[str, List[float]] = defaultdict(list)
        by_category_l2: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if label is None:
                self.hits.append(0.0)
                continue
            hit = float(
                prediction is not None
                and str(prediction).strip().upper() == str(label).strip().upper()
            )
            self.hits.append(hit)

            l2_category = (
                self.l2_categories[index]
                if index < len(self.l2_categories)
                else "unknown"
            )
            category = (
                self.categories[index] if index < len(self.categories) else "unknown"
            )
            by_l2[l2_category].append(hit)
            by_category_l2[category][l2_category].append(hit)

        if not by_l2:
            return None

        self.l2_category_scores = {
            l2_category: sum(scores) / len(scores)
            for l2_category, scores in sorted(by_l2.items())
        }
        self.category_scores = {
            category: macro_average(l2_scores)
            for category, l2_scores in sorted(by_category_l2.items())
        }

        print(
            "MMStar accuracy per l2 category: "
            + ", ".join(
                f"{name}={score:.4f}" for name, score in self.l2_category_scores.items()
            )
        )
        print(
            "MMStar accuracy per category: "
            + ", ".join(
                f"{name}={score:.4f}" for name, score in self.category_scores.items()
            )
        )
        return macro_average(by_l2)

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every category, its accuracy being the macro average
        over the l2 categories it contains, as the task computes it.

        The latency is the one of the whole run, so a category's throughput is its
        share of the aggregate rather than a figure it could reach on its own.
        """
        if not self.categories:
            return None

        performance = {}
        for category in sorted(set(self.categories)):
            indexes = [
                index
                for index, name in enumerate(self.categories)
                if name == category and index < len(states)
            ]
            if not indexes:
                continue
            metrics = compute_metrics(
                [states[index] for index in indexes], latency, answer_key=answer_key
            )
            if category in self.category_scores:
                metrics.accuracy = self.category_scores[category]
                metrics.num_valid_predictions = len(
                    [index for index in indexes if index < len(self.hits)]
                )
            performance[category] = metrics
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, the prompt variant and the l2 category breakdown."""
        description: Dict[str, Any] = {
            "split": self.split,
            "prompt_variant": self.prompt_variant,
            "questions": len(self.categories),
            "aggregation": "macro average over the l2 categories",
        }
        if self.l2_category_scores:
            description["accuracy_per_l2_category"] = self.l2_category_scores
        return description

    def create_sgl_function(self):
        """Create the SGL function for MMStar (image-based Q&A)."""
        return create_image_sgl_function(
            function_name="get_mmstar_answer",
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
