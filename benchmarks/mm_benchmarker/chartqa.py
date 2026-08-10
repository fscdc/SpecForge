"""
ChartQA benchmark evaluation script.

The prompt and the scoring are ported from the lm-evaluation-harness `chartqa`
task (which itself adapts https://github.com/mistralai/mistral-evals) so that the
numbers stay comparable:

- exact_match: the text after "Final Answer:" has to match the reference exactly.
- relaxed_accuracy: same parsing, but numeric answers are accepted when they are
  within 5% of the reference (see https://arxiv.org/pdf/2203.10244.pdf, 5.1).
- anywhere_accuracy: falls back to searching the reference anywhere in the
  generation. Overly generous, reported for reference only.

`relaxed_accuracy` is the headline metric, i.e. the one reported as `accuracy`.
"""

import os
import re
import shutil
import string
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import MMBenchmarker
from .utils import create_image_sgl_function
from .registry import MM_BENCHMARKS

# doc_to_text of the lm-eval task, minus the <image> placeholder (the image is
# prepended by the SGL function instead).
CHARTQA_INSTRUCTION = """Analyze the image and question carefully, using step-by-step reasoning.
First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
Final Answer: <answer>
where <answer> follows the following instructions:
- <answer> should should be a single phrase or number.
- <answer> should not paraphrase or reformat the text in the image.
- If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
- If the question is a Yes/No question, <answer> should be Yes/No.
- If <answer> is a number, it should not contain any units.
- If <answer> is a percentage, it should include a % sign.
- If <answer> is an entity, it should include the full label from the graph.
IMPORTANT: Remember, to end your answer with Final Answer: <answer>."""

# the two ways a question was produced, usable as `chartqa:<n>:human` subsets
SUBSET_TO_LABEL_ID = {"human": 0, "machine": 1}


def _normalize_string(s: str) -> str:
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def _remove_end_punctuation(unnormalized_string: str) -> str:
    while (
        unnormalized_string
        and (
            unnormalized_string[-1] in string.punctuation
            or unnormalized_string[-1].isspace()
        )
        and unnormalized_string[-1] != "%"
    ):
        unnormalized_string = unnormalized_string[:-1]
    return unnormalized_string


def _preprocess_text(text: str) -> str:
    if not any(char.isdigit() for char in text):
        return _normalize_string(text)
    return _remove_end_punctuation(text).replace(",", "").replace("$", "")


def _to_float(text: str) -> Tuple[Optional[float], bool]:
    text = text.strip()
    is_percent = text.endswith("%")
    try:
        return float(text.rstrip("%")), is_percent
    except ValueError:
        return None, False


def _compare_numeric_values(
    prediction: float, target: float, max_relative_change: float
) -> float:
    relative_change = abs(prediction - target) / max(abs(target), 1e-10)
    return 1.0 if relative_change <= max_relative_change else 0.0


def _compare_numeric_with_percent(
    prediction: float,
    prediction_is_percent: bool,
    target: float,
    target_is_percent: bool,
    max_relative_change: float,
) -> float:
    def to_decimal(value: float, is_percent: bool) -> float:
        return value / 100 if is_percent else value

    # compare as-is
    value = _compare_numeric_values(prediction, target, max_relative_change)

    # if not equal and one is a percentage, try the other comparisons
    if value != 1.0 and (prediction_is_percent or target_is_percent):
        value = max(
            value,
            _compare_numeric_values(
                to_decimal(prediction, prediction_is_percent),
                target,
                max_relative_change,
            ),
            _compare_numeric_values(
                prediction,
                to_decimal(target, target_is_percent),
                max_relative_change,
            ),
        )
    return value


def _compare_text_values(prediction: str, target: str) -> float:
    while prediction and prediction[-1] in string.punctuation:
        prediction = prediction[:-1]
    return 1.0 if prediction.lower() == target.lower() else 0.0


def relaxed_correctness(
    prediction: str, targets: List[str], max_relative_change: float = 0.05
) -> float:
    """
    Relaxed correctness: an answer counts as correct when it is within
    `max_relative_change` of a numeric reference, or matches a textual one.
    """
    prediction = _preprocess_text(prediction)
    prediction_float, prediction_is_percent = _to_float(prediction)

    value_list = []
    for target in targets:
        target = _preprocess_text(target)
        target_float, target_is_percent = _to_float(target)

        if prediction_float is not None and target_float is not None:
            # compare as numeric values
            value = _compare_numeric_with_percent(
                prediction_float,
                prediction_is_percent,
                target_float,
                target_is_percent,
                max_relative_change,
            )
        elif target.isalpha() and len(target) == 1 and len(prediction) > 0:
            # compare as multiple choice options: take first letter of prediction
            value = 1.0 if prediction[0].lower() == target.lower() else 0.0
        else:
            # compare as text values
            value = _compare_text_values(prediction, target)

        value_list.append(value)

    return max(value_list) if value_list else 0.0


def extract_final_answer(generation: str) -> str:
    """
    Take the text that follows the last "answer:" of the generation, or "" when
    the model did not follow the requested format.
    """
    # strip extraneous markdown around the answer
    generation = re.sub(r"([aA]nswer)\**:\**", "\\1:", generation)

    final_answer_index = generation.lower().rfind("answer:")
    if final_answer_index == -1:
        return ""

    # find the first non-empty line after "answer:"
    start_index = final_answer_index + len("answer:")
    lines = generation[start_index:].split("\n")
    final_answer = next((line.strip() for line in lines if line.strip()), "")

    # remove any markdown formatting
    return re.sub(r"[*_\[\]\(\)]", "", final_answer)


def exact_match_score(generation: str, targets: List[str]) -> float:
    """Strict match of the "Final Answer: ..." text against a reference."""
    # like the lm-eval task, the answer has to be on the last line of the
    # generation ($ without re.MULTILINE)
    match = re.search(
        r"(?:Final Answer|FINAL ANSWER): (.+)$", generation, re.IGNORECASE
    )
    if not match:
        return 0.0
    prediction = match.group(1).strip().lower().removesuffix(".")
    return (
        1.0 if any(prediction == target.strip().lower() for target in targets) else 0.0
    )


def relaxed_accuracy_score(generation: str, targets: List[str]) -> float:
    """Relaxed correctness on the answer parsed out of the requested format."""
    prediction = extract_final_answer(generation)
    if not prediction:
        # parsing failed
        return 0.0
    return relaxed_correctness(prediction, targets)


def anywhere_accuracy_score(generation: str, targets: List[str]) -> float:
    """
    Relaxed correctness, falling back to looking for the reference anywhere in
    the generation when the requested answer format is missing.

    NOTE: this is an overly generous metric and is likely to falsely inflate
    scores. It is reported to tell "the model was right but did not follow the
    format" apart from "the model was wrong".
    """
    prediction = extract_final_answer(generation)
    if prediction:
        return relaxed_correctness(prediction, targets)

    for target in targets:
        try:
            number = float(target)
        except ValueError:
            # the reference is a text string, so we search for the typical
            # patterns instead: searching for the reference directly is a bad
            # idea for letter-option questions. This stays heuristic and can
            # produce both false positives and false negatives.
            candidates = []
            for candidate_target in targets:
                candidates.extend(
                    [
                        f"is {candidate_target}",
                        f"was {candidate_target}",
                        f" {candidate_target}.",
                        f"are {candidate_target}",
                        f"\n\n{candidate_target}",
                    ]
                )
            if any(candidate.lower() in generation for candidate in candidates):
                return 1.0
            continue

        # revert to int if the reference is actually an int
        if int(number) == number:
            number = int(number)
        # with commas (e.g. 1,000), without them (e.g. 1000), or as a percentage
        if (
            format(number, ",") in generation
            or str(number) in generation
            or f"{number}%" in generation
        ):
            return 1.0

    return 0.0


@MM_BENCHMARKS.register("chartqa")
class ChartQABenchmarker(MMBenchmarker):
    """
    ChartQA benchmark implementation.

    Args:
        num_samples: number of samples to evaluate, all of them when not given.
        subset: restrict the questions to how they were written, one or both of
            "human" and "machine". All of them when not given.
        split: the dataset split to evaluate, "test" like the lm-eval task.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "test",
    ):
        super().__init__(num_samples, subset)
        unknown = set(subset or []) - set(SUBSET_TO_LABEL_ID)
        if unknown:
            raise ValueError(
                f"Unknown ChartQA subset(s) {sorted(unknown)}, "
                f"expected any of {sorted(SUBSET_TO_LABEL_ID)}"
            )
        self.split = split
        self.cache_dir = None
        # all three metrics of the last compute_accuracy() call, the headline one
        # (relaxed accuracy) is what gets reported as BenchmarkMetrics.accuracy
        self.scores: Dict[str, float] = {}

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[List[str]]]]:
        """Load and preprocess the ChartQA dataset."""
        self.cache_dir = os.path.join(".cache", "chartqa_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("HuggingFaceM4/ChartQA")[self.split]

        if self.subset:
            wanted = self._resolve_subset_ids(dataset)
            # read the label column directly, filtering row by row would decode
            # every image on the way
            keep = [
                idx
                for idx, label_id in enumerate(dataset["human_or_machine"])
                if label_id in wanted
            ]
            dataset = dataset.select(keep)
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels = []
        for idx, q in enumerate(dataset):
            # charts are full of small text, so keep them lossless
            image_path = os.path.join(image_dir, f"{idx:06d}.png")
            q["image"].convert("RGB").save(image_path, "PNG")

            questions.append(
                {
                    "image_path": image_path,
                    "question": f"{q['query']}\n{CHARTQA_INSTRUCTION}",
                }
            )

            answers = [
                str(answer).strip() for answer in q["label"] if str(answer).strip()
            ]
            labels.append(answers or None)

        return questions, labels

    def _resolve_subset_ids(self, dataset) -> set:
        """Map the requested subset names onto the `human_or_machine` class ids."""
        feature = dataset.features["human_or_machine"]
        ids = set()
        for name in self.subset:
            try:
                ids.add(feature.str2int(name))
            except (AttributeError, ValueError, KeyError):
                # the column is a plain int in some revisions of the dataset
                ids.add(SUBSET_TO_LABEL_ID[name])
        return ids

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """
        Keep the raw generation.

        Every metric parses the "Final Answer:" part itself, and the anywhere
        metric even needs the untouched generation to fall back on.
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Compute the three ChartQA metrics, and report the relaxed accuracy."""
        scorers = {
            "exact_match": exact_match_score,
            "relaxed_accuracy": relaxed_accuracy_score,
            "anywhere_accuracy": anywhere_accuracy_score,
        }
        totals = {name: 0.0 for name in scorers}

        valid_count = 0
        for prediction, label in zip(predictions, labels):
            if not label:
                continue
            valid_count += 1
            if not isinstance(prediction, str):
                continue
            for name, scorer in scorers.items():
                totals[name] += scorer(prediction, label)

        if valid_count == 0:
            return None

        self.scores = {name: total / valid_count for name, total in totals.items()}
        print(
            f"ChartQA scores over {valid_count} questions: "
            + ", ".join(f"{name}={score:.4f}" for name, score in self.scores.items())
        )
        return self.scores["relaxed_accuracy"]

    def default_max_new_tokens(self) -> int:
        """The lm-eval task generates at most 512 tokens per question."""
        return 512

    def create_sgl_function(self):
        """Create the SGL function for ChartQA (image-based Q&A)."""
        return create_image_sgl_function(
            function_name="get_chartqa_answer",
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
