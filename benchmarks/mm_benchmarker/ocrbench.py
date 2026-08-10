"""
OCRBench benchmark evaluation script.

OCRBench (https://huggingface.co/datasets/echo840/OCRBench) is 1000 questions
over 25 source datasets, grouped into 10 question types, that probe the OCR
abilities of a VLM. An answer counts as correct when one of its references
appears anywhere in the generation, so the metrics are reported both overall and
per question type.

The prompt and the scoring are ported from the lmms-eval `ocrbench` task: the
question is sent as-is, and the comparison is a containment check, case
insensitive and with the fullwidth ASCII folded, except for HME100k where the
handwritten formulas are compared with every space removed instead.

Because the benchmark has exactly 1000 one-point questions, the accuracy over the
full split is the official score divided by 1000. The raw score of each question
type, i.e. what the OCRBench report normally quotes, is written to the results
file as well.
"""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_image_sgl_function

# how many questions each type contributes to the 1000 points of the benchmark
QUESTION_TYPE_TOTALS = {
    "Regular Text Recognition": 50,
    "Irregular Text Recognition": 50,
    "Artistic Text Recognition": 50,
    "Handwriting Recognition": 50,
    "Digit String Recognition": 50,
    "Non-Semantic Text Recognition": 50,
    "Scene Text-centric VQA": 200,
    "Doc-oriented VQA": 200,
    "Key Information Extraction": 200,
    "Handwritten Mathematical Expression Recognition": 100,
}
# the six types the report sums up as "Text Recognition", out of 300
RECOGNITION_TYPES = tuple(
    name for name, total in QUESTION_TYPE_TOTALS.items() if total == 50
)
MAX_SCORE = sum(QUESTION_TYPE_TOTALS.values())

# the source dataset scored with its own rule, see score_answer()
HANDWRITTEN_MATHS_DATASET = "HME100k"

# the columns a --benchmark-list subset is matched against
SUBSET_COLUMNS = ("dataset", "question_type")


def fold_fullwidth_ascii(text: str) -> str:
    """Fold fullwidth ASCII and ideographic space without broader NFKC changes."""
    folded = []
    for char in text:
        codepoint = ord(char)
        if 0xFF01 <= codepoint <= 0xFF5E:
            folded.append(chr(codepoint - 0xFEE0))
        elif codepoint == 0x3000:
            folded.append(" ")
        else:
            folded.append(char)
    return "".join(folded)


def score_answer(prediction: str, answers: Any, dataset_name: str) -> int:
    """
    1 when one of the reference answers appears in the generation, 0 otherwise.

    Handwritten formulas are compared with every space removed and without
    folding the case, every other source dataset case insensitively with the
    fullwidth ASCII folded.
    """
    references = answers if isinstance(answers, (list, tuple)) else [answers]

    for reference in references:
        if dataset_name == HANDWRITTEN_MATHS_DATASET:
            answer = str(reference).strip().replace("\n", " ").replace(" ", "")
            predict = prediction.strip().replace("\n", " ").replace(" ", "")
        else:
            answer = fold_fullwidth_ascii(str(reference)).lower().strip()
            answer = answer.replace("\n", " ")
            predict = fold_fullwidth_ascii(prediction).lower().strip()
            predict = predict.replace("\n", " ")
        if answer in predict:
            return 1
    return 0


@MM_BENCHMARKS.register("ocrbench")
class OCRBenchBenchmarker(MMBenchmarker):
    """
    OCRBench benchmark implementation.

    Args:
        num_samples: number of questions to evaluate, all 1000 when not given.
        subset: restrict the questions to one or more source datasets or question
            types, matched case-insensitively against the `dataset` and
            `question_type` columns, e.g. `ocrbench:100:HME100k`. All of them when
            not given.
        split: the dataset split to evaluate, "test" is the only one.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "test",
    ):
        super().__init__(num_samples, subset)
        self.split = split
        self.cache_dir = None
        # per-question metadata, kept aligned with the loaded questions
        self.question_types: List[str] = []
        self.source_datasets: List[str] = []
        # per-question 1/0 of the last compute_accuracy() call
        self.hits: List[int] = []

    def default_max_new_tokens(self) -> int:
        """The lmms-eval task answers in at most 128 tokens."""
        return 128

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[List[str]]]]:
        """Load and preprocess the OCRBench dataset."""
        self.cache_dir = os.path.join(".cache", "ocrbench_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("echo840/OCRBench")[self.split]

        if self.subset:
            dataset = dataset.select(self._select_subset(dataset))
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels = []
        self.question_types = []
        self.source_datasets = []
        for index, row in enumerate(dataset):
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            row["image"].convert("RGB").save(image_path, "PNG")

            # the question is sent as-is, the task adds no instruction to it
            questions.append(
                {"image_path": image_path, "question": row["question"].strip()}
            )
            self.question_types.append(str(row["question_type"]))
            self.source_datasets.append(str(row["dataset"]))

            answers = [str(answer) for answer in row["answer"]]
            labels.append(answers or None)

        return questions, labels

    def _select_subset(self, dataset) -> List[int]:
        """
        Indices of the rows whose source dataset or question type matches the
        requested subset.

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
                f"Unknown OCRBench subset(s) {sorted(unknown)}, "
                f"expected any of {sorted(available)}"
            )
        return [
            index
            for index in range(len(dataset))
            if any(column[index] in wanted for column in columns)
        ]

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """
        Keep the raw generation: the reference answer only has to appear in it,
        and the rule depends on the source dataset, which compute_accuracy() looks
        up by index.
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """
        Score every question and report the accuracy.

        Over the full split this is the official score out of 1000, since every
        question is worth exactly one point.
        """
        self.hits = []
        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if not label or not isinstance(prediction, str):
                self.hits.append(0)
                continue
            source = (
                self.source_datasets[index]
                if index < len(self.source_datasets)
                else ""
            )
            self.hits.append(score_answer(prediction, label, source))

        if not self.hits:
            return None
        self._print_report()
        return sum(self.hits) / len(self.hits)

    def _score_per_question_type(self) -> Dict[str, Dict[str, int]]:
        """Raw score and question count of every type that was evaluated."""
        scores: Dict[str, Dict[str, int]] = {}
        for index, question_type in enumerate(self.question_types):
            if index >= len(self.hits):
                break
            counts = scores.setdefault(
                question_type,
                {
                    "score": 0,
                    "questions": 0,
                    "total": QUESTION_TYPE_TOTALS.get(question_type, 0),
                },
            )
            counts["score"] += self.hits[index]
            counts["questions"] += 1
        return scores

    def _print_report(self) -> None:
        """Print the score table the OCRBench report normally quotes."""
        scores = self._score_per_question_type()
        recognition = sum(
            counts["score"]
            for name, counts in scores.items()
            if name in RECOGNITION_TYPES
        )
        final_score = sum(counts["score"] for counts in scores.values())
        evaluated = sum(counts["questions"] for counts in scores.values())

        print(f"OCRBench score: {final_score}/{evaluated} questions evaluated")
        print(f"  Text Recognition (of 300): {recognition}")
        for name in QUESTION_TYPE_TOTALS:
            if name in scores:
                counts = scores[name]
                print(
                    f"  {name} (of {counts['total']}): "
                    f"{counts['score']}/{counts['questions']}"
                )
        if evaluated == MAX_SCORE:
            print(f"  Final Score (of {MAX_SCORE}): {final_score}")

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every question type.

        The latency is the one of the whole run, so a type's throughput is its
        share of the aggregate rather than a figure it could reach on its own.
        """
        if not self.question_types:
            return None

        performance = {}
        for question_type in QUESTION_TYPE_TOTALS:
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
        """Report the split and the raw OCRBench score of every question type."""
        scores = self._score_per_question_type()
        if not scores:
            return {"split": self.split, "questions": len(self.question_types)}

        recognition = sum(
            counts["score"]
            for name, counts in scores.items()
            if name in RECOGNITION_TYPES
        )
        return {
            "split": self.split,
            "questions": len(self.question_types),
            "ocrbench_score": {
                "final_score": sum(counts["score"] for counts in scores.values()),
                "max_score": MAX_SCORE,
                "text_recognition": recognition,
                "per_question_type": scores,
            },
        }

    def create_sgl_function(self):
        """Create the SGL function for OCRBench (image-based Q&A)."""
        return create_image_sgl_function(
            function_name="get_ocrbench_answer",
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
