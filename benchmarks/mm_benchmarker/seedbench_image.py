"""
SEED-Bench (image part) benchmark evaluation script.

SEED-Bench (https://huggingface.co/datasets/lmms-lab-encoder/SEED-Bench) is a
four-way multiple-choice benchmark. Its `test` split holds both halves of the
suite: 9 image dimensions, one picture per question, and 3 video dimensions,
eight sampled frames per question. This benchmarker evaluates the image half
only -- rows whose `data_type` is "image" -- which is the part reported as
"SEED-Bench Image".

Ported from the lmms-eval `seedbench` task, with one deliberate difference: the
answer instruction is the step-by-step boxed prompt the other benchmarks here
send, not the task's "Answer with the option's letter from the given choices
directly". Keeping one prompt across benchmarks is what makes their generation
lengths, and therefore their speculative-decoding numbers, comparable. Pass
``post_prompt`` to send the question the way the task does; only then are the
accuracies comparable to published lmms-eval numbers.

Scoring is the task's: the generation is reduced to an option letter and
compared with the reference, case-insensitively. Where the task takes the first
character of the response -- enough for a model told to answer with a letter and
nothing else -- this reads the letter out of a reasoned answer, which is what the
boxed prompt asks for.
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
    STEP_BY_STEP_BOXED_PROMPT,
    create_image_sgl_function,
    extract_choice,
)

#: The task's own instruction, kept so a run can reproduce its numbers.
DIRECT_ANSWER_PROMPT = "\nAnswer with the option's letter from the given choices directly."

#: The nine image dimensions of SEED-Bench, by `question_type_id`. Ten to twelve
#: are the video half and never appear here.
DIMENSIONS = {
    1: "scene-understanding",
    2: "instance-identity",
    3: "instance-attributes",
    4: "instance-location",
    5: "instance-counting",
    6: "spatial-relation",
    7: "instance-interaction",
    8: "visual-reasoning",
    9: "text-understanding",
}


def build_prompt(
    row: Dict[str, Any],
    pre_prompt: str = "",
    post_prompt: str = "\n" + STEP_BY_STEP_BOXED_PROMPT,
) -> str:
    """The task's question-and-choices block with this repo's answer instruction."""
    question = str(row["question"]).strip()
    options = "\n".join(
        f"{letter}. {row[f'choice_{letter.lower()}']}" for letter in CHOICES
    )
    return f"{pre_prompt}{question}\n{options}{post_prompt}"


@MM_BENCHMARKS.register("seedbench-image")
class SEEDBenchImageBenchmarker(MMBenchmarker):
    """
    SEED-Bench image-half benchmark implementation.

    Args:
        num_samples: number of questions to evaluate, all of them when not given.
        subset: restrict to one or more of the nine dimensions, by name
            ("instance-counting") or by `question_type_id` ("5").
        split: the dataset split to evaluate, "test" is the only one.
        post_prompt: the instruction appended to every question. Defaults to the
            shared step-by-step boxed prompt; pass the task's own
            DIRECT_ANSWER_PROMPT for numbers comparable to lmms-eval.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "test",
        post_prompt: Optional[str] = None,
    ):
        super().__init__(num_samples, subset)
        names = set(DIMENSIONS.values())
        unknown = {
            name.strip().lower()
            for name in (subset or [])
            if name.strip().lower() not in names
            and name.strip() not in {str(key) for key in DIMENSIONS}
        }
        if unknown:
            raise ValueError(
                f"Unknown SEED-Bench dimension(s) {sorted(unknown)}, expected "
                f"{sorted(names)} or an id in {sorted(DIMENSIONS)}"
            )
        self.split = split
        self.post_prompt = (
            "\n" + STEP_BY_STEP_BOXED_PROMPT if post_prompt is None else post_prompt
        )
        self.cache_dir = None
        # per-question dimension, kept aligned with the loaded questions
        self.dimensions: List[str] = []
        self.question_ids: List[str] = []
        self.hits: List[float] = []

    def default_max_new_tokens(self) -> int:
        """Room for the reasoning the shared prompt asks for before the box.

        The lmms-eval task answers in a single letter, which is all its
        answer-directly instruction needs.
        """
        return 2048

    def _wanted_dimensions(self) -> Optional[set]:
        """The `question_type_id`s the subset selects, or None for all nine."""
        if not self.subset:
            return None
        by_name = {name: key for key, name in DIMENSIONS.items()}
        wanted = set()
        for entry in self.subset:
            entry = entry.strip().lower()
            wanted.add(by_name[entry] if entry in by_name else int(entry))
        return wanted

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load SEED-Bench, keep the image half, and write its pictures out."""
        self.cache_dir = os.path.join(".cache", "seedbench_image_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("lmms-lab-encoder/SEED-Bench", split=self.split)

        # The split interleaves both halves, and the video rows carry eight
        # frames each, so they are dropped before anything is decoded.
        wanted = self._wanted_dimensions()
        keep = [
            index
            for index, (data_type, type_id) in enumerate(
                zip(dataset["data_type"], dataset["question_type_id"])
            )
            if data_type == "image" and (wanted is None or type_id in wanted)
        ]
        dataset = dataset.select(keep)
        print(f"SEED-Bench {self.split}: {len(dataset)} image questions selected")

        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels: List[Optional[str]] = []
        self.dimensions = []
        self.question_ids = []
        for index, row in enumerate(dataset):
            images = row["image"]
            if not images:
                continue
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            # One picture per image-half question; the column is a list because
            # the video half shares it.
            images[0].convert("RGB").save(image_path, "PNG")

            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(row, post_prompt=self.post_prompt),
                }
            )
            labels.append(str(row["answer"]).strip().upper() or None)
            self.dimensions.append(
                DIMENSIONS.get(row["question_type_id"], "unknown")
            )
            self.question_ids.append(str(row["question_id"]))

        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """The option letter the generation settles on."""
        if not isinstance(output, str):
            return None
        return extract_choice(output, CHOICES)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Score every question, overall and per dimension."""
        self.hits = []
        by_dimension: Dict[str, List[float]] = defaultdict(list)

        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            hit = float(
                isinstance(prediction, str)
                and label is not None
                and prediction.strip().upper() == str(label).strip().upper()
            )
            self.hits.append(hit)
            dimension = (
                self.dimensions[index] if index < len(self.dimensions) else "unknown"
            )
            by_dimension[dimension].append(hit)

        if not self.hits:
            return None

        accuracy = sum(self.hits) / len(self.hits)
        print(
            f"SEED-Bench image accuracy over {len(self.hits)} questions: "
            f"{accuracy:.4f}"
        )
        for name, scores in sorted(by_dimension.items()):
            print(
                f"  {name:<22}{sum(scores) / len(scores):.4f} "
                f"({len(scores)} questions)"
            )
        return accuracy

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """Report the metrics of every dimension that was evaluated.

        The latency is the one of the whole run, so a dimension's throughput is
        its share of the aggregate rather than a figure it could reach alone.
        """
        if not self.dimensions:
            return None

        performance = {}
        for dimension in sorted(set(self.dimensions)):
            indexes = [
                index
                for index, name in enumerate(self.dimensions)
                if name == dimension and index < len(states)
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
            performance[dimension] = metrics
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, the dimensions covered, and the prompt used."""
        return {
            "split": self.split,
            "questions": len(self.question_ids),
            "dimensions": sorted(set(self.dimensions)),
            "post_prompt": self.post_prompt,
        }

    def create_sgl_function(self):
        """Create the SGL function for SEED-Bench (image-based multiple choice)."""
        return create_image_sgl_function(
            function_name="get_seedbench_image_answer",
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
