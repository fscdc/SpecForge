"""
DynaMath benchmark evaluation script.

DynaMath (https://huggingface.co/datasets/kcz358/DynaMath) is a dynamic visual
maths benchmark: 501 seed questions, each instantiated as 10 variants that keep
the question but change the figure and the numbers. Answers are floats, letters
of an embedded multiple choice, or short text.

Two metrics are reported, both averaged over the SEED QUESTIONS rather than over
the rows, which is what makes them differ:

* **average** -- the mean accuracy over a question's variants. This is the
  headline `accuracy`.
* **worst** -- 1.0 only when a question's every variant is answered correctly.
  This is the metric DynaMath exists to measure: whether a model actually solved
  the problem or just happened to land on one instance of it.

The prompt is MathVision's `BOXED_PROMPT`, and the scoring is MathVision's
rule-based `score_answer`, so the two benchmarks stay directly comparable. That
is a deliberate departure from the lmms-eval `dynamath` task, which prompts for
`<think>`/`<answer>` tags and scores through `lmms_eval`'s `compute_score`;
neither is available here, and a second prompt would make the generation lengths
-- and therefore the speculative-decoding numbers -- incomparable.

Like MathVision, symbolic comparison of LaTeX answers needs `latex2sympy2`.
Without it answers still match as text and as plain numbers, so the accuracy is
a lower bound: `pip install latex2sympy2` to score exactly.
"""

import os
import re
import shutil
import string as string_module
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .mathvision import BOXED_PROMPT, extract_answer, score_answer
from .registry import MM_BENCHMARKS
from .utils import create_image_sgl_function

#: Variants per seed question in the `test` split. Only used to explain the
#: layout below and to size the truncation warning; the grouping itself counts
#: the rows it actually loaded.
VARIANTS_PER_QUESTION = 10


def build_prompt(question: str) -> str:
    """
    MathVision's prompt, applied to a DynaMath question.

    A DynaMath question already carries its choices inline ("... choice: (A) yes
    (B) no"), so unlike MathVision there is no separate options block to append.
    The missing separator between the instruction and the question is
    MathVision's, and is reproduced so that both benchmarks send the same shape.
    """
    return f"{BOXED_PROMPT}{question}"


def parse_options(question: str) -> List[str]:
    """
    The choices written into a multiple-choice question, or [] when there are
    none to be read.

    DynaMath has no options column: the choices live in the question text, with
    an inconsistent lead-in ("Choices:", "choice:", or nothing at all) and option
    bodies that may themselves contain parentheses ("f(x)=g(x)-5"). So rather
    than matching a delimiter or a body pattern, this walks the (A), (B), (C)...
    markers in ascending order and slices between them, which the parenthesised
    bodies cannot confuse.

    The last occurrence of each marker is taken, so a question that mentions
    "(A)" in its prose before listing the choices still splits at the list.
    Callers must treat the result as advisory: `_usable_options` below is what
    decides whether a parse is trustworthy enough to score against.
    """
    spans: List[Tuple[int, int]] = []
    for letter in string_module.ascii_uppercase:
        match = None
        for match in re.finditer(r"\(%s\)" % letter, question):
            pass  # the last one, see the docstring
        if match is None:
            break
        if spans and match.start() < spans[-1][1]:
            break  # markers out of textual order, so this is not a choice list
        spans.append((match.start(), match.end()))

    if len(spans) < 2:
        return []

    options = []
    for index, (_, end) in enumerate(spans):
        stop = spans[index + 1][0] if index + 1 < len(spans) else len(question)
        options.append(question[end:stop].strip().rstrip(".,;").strip())
    return options


def _usable_options(question: str, answer: str) -> List[str]:
    """
    The parsed choices, but only when the parse is self-consistent.

    A misparse would hand `score_answer` the text of the WRONG option as the
    reference, crediting an incorrect generation -- worse than not parsing at
    all, since scoring by letter alone is already correct. So the options are
    kept only when every one of them is non-empty and the reference letter
    indexes into them; otherwise the question is scored by letter only.
    """
    options = parse_options(question)
    if not options or not all(options):
        return []
    answer = answer.strip()
    if len(answer) != 1 or not answer.isalpha():
        return []
    index = ord(answer.upper()) - ord("A")
    return options if 0 <= index < len(options) else []


@MM_BENCHMARKS.register("dynamath")
class DynaMathBenchmarker(MMBenchmarker):
    """
    DynaMath benchmark implementation.

    Args:
        num_samples: number of variants to evaluate, all 5010 when not given.
            Truncation is group-aware, see `load_data`.
        subset: restrict the questions to one or more subjects, matched
            case-insensitively against the `subject` column, e.g.
            `dynamath:200:algebra`. All of them when not given.
        split: the dataset split to evaluate, "test" is the only one.
        match: "lenient" also credits an answer that states the numeric reference
            without isolating it, "exact" is the rule-based metric. Both numbers
            are reported either way. Same meaning as MathVision's.
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
        self.match = match
        self.split = split
        self.cache_dir = None
        # per-variant metadata, kept aligned with the loaded questions
        self.subjects: List[str] = []
        self.question_ids: List[str] = []
        self.options_list: List[List[str]] = []
        # per-variant 1.0/0.0 of the last compute_accuracy() call, under the
        # selected mode and under the strict one
        self.hits: List[float] = []
        self.exact_hits: List[float] = []
        # question-averaged scores, which are the reported metrics
        self.average_score: Optional[float] = None
        self.worst_score: Optional[float] = None
        self.exact_average: Optional[float] = None
        self.exact_worst: Optional[float] = None
        # questions whose variants were not all evaluated, see load_data
        self.partial_questions: int = 0
        # what the scoring actually compared, for --save-generations
        self.parsed_answers: List[Optional[str]] = []

    def default_max_new_tokens(self) -> int:
        """
        Room for the chain of thought the boxed prompt asks for.

        MathVision's default, because this sends MathVision's prompt. The caller
        keeps the last word: `--max-tokens` overrides this through
        `MMBenchmarker.get_max_new_tokens()`.
        """
        return 16384

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """
        Load DynaMath, keeping every seed question's variants together.

        The split stores the variants strided rather than adjacent: a question's
        ten rows are 501 apart. Taking the first `num_samples` rows would
        therefore return that many DIFFERENT questions with ONE variant each, and
        the worst-case metric -- a minimum over a question's variants -- would
        silently collapse onto the average. So the rows are regrouped by question
        first, and the truncation then keeps whole questions.
        """
        self.cache_dir = os.path.join(".cache", "dynamath_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("kcz358/DynaMath")[self.split]

        indexes = (
            self._select_subset(dataset) if self.subset else list(range(len(dataset)))
        )

        # group by seed question, preserving the order the questions first appear
        # in, so that a truncated run is a prefix of the full one
        grouped: Dict[str, List[int]] = defaultdict(list)
        for index in indexes:
            grouped[str(dataset["id"][index])].append(index)

        ordered: List[int] = []
        for rows in grouped.values():
            ordered.extend(rows)

        if self.num_samples is not None:
            ordered = ordered[: self.num_samples]

        # a question whose variants were cut in half still scores, but its worst
        # case is a minimum over fewer variants and so is biased upwards
        kept = defaultdict(int)
        for index in ordered:
            kept[str(dataset["id"][index])] += 1
        self.partial_questions = sum(
            1 for name, count in kept.items() if count < len(grouped[name])
        )
        if self.partial_questions:
            print(
                f"DynaMath: {self.partial_questions} of {len(kept)} questions are "
                "only partly covered by --benchmark-list dynamath:<n>, so their "
                "worst-case score is a minimum over fewer variants. Use a "
                f"multiple of {VARIANTS_PER_QUESTION} to avoid this."
            )

        questions = []
        labels: List[Optional[str]] = []
        self.subjects = []
        self.question_ids = []
        self.options_list = []
        for position, index in enumerate(ordered):
            row = dataset[index]
            image_path = os.path.join(image_dir, f"{position:06d}.png")
            row["decoded_image"].convert("RGB").save(image_path, "PNG")

            question = str(row["question"])
            answer = str(row["ground_truth"]).strip()
            questions.append(
                {"image_path": image_path, "question": build_prompt(question)}
            )
            labels.append(answer or None)
            self.options_list.append(_usable_options(question, answer))
            self.subjects.append(str(row.get("subject", "unknown")))
            self.question_ids.append(str(row["id"]))

        print(
            f"DynaMath {self.split}: {len(questions)} variants over "
            f"{len(set(self.question_ids))} questions"
        )
        return questions, labels

    def _select_subset(self, dataset) -> List[int]:
        """
        Indices of the rows whose subject matches the requested subset.

        Reads the subject column directly; filtering row by row would decode
        every image on the way.
        """
        wanted = {name.strip().lower() for name in self.subset}
        subjects = [str(subject).strip().lower() for subject in dataset["subject"]]
        unknown = wanted - set(subjects)
        if unknown:
            raise ValueError(
                f"Unknown DynaMath subject(s) {sorted(unknown)}, "
                f"expected any of {sorted(set(subjects))}"
            )
        return [index for index, subject in enumerate(subjects) if subject in wanted]

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """
        Keep the raw generation: the scoring needs the options of the question,
        which compute_accuracy() looks up by index.
        """
        return output

    def _aggregate(self, hits: List[float]) -> Tuple[float, float]:
        """
        Question-averaged (average, worst) over per-variant scores.

        Both metrics average over the seed questions, so a question contributes
        once however many variants it has -- which is what keeps a partially
        covered question from outweighing a fully covered one.
        """
        by_question: Dict[str, List[float]] = defaultdict(list)
        for index, hit in enumerate(hits):
            name = (
                self.question_ids[index]
                if index < len(self.question_ids)
                else f"unknown-{index}"
            )
            by_question[name].append(hit)

        averages = [sum(scores) / len(scores) for scores in by_question.values()]
        worsts = [min(scores) for scores in by_question.values()]
        return sum(averages) / len(averages), sum(worsts) / len(worsts)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Score every variant, and report the question-averaged accuracy."""
        self.hits = []
        self.exact_hits = []
        self.parsed_answers = []
        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if label is None or not isinstance(prediction, str):
                self.hits.append(0.0)
                self.exact_hits.append(0.0)
                self.parsed_answers.append(None)
                continue
            options = self.options_list[index] if index < len(self.options_list) else []
            self.parsed_answers.append(extract_answer(prediction, options))
            self.hits.append(float(score_answer(prediction, label, options, self.match)))
            # the strict metric is tracked alongside, so that a lenient run stays
            # comparable with a published rule-based number
            self.exact_hits.append(
                float(score_answer(prediction, label, options, "exact"))
            )

        if not self.hits:
            return None

        self.average_score, self.worst_score = self._aggregate(self.hits)
        self.exact_average, self.exact_worst = self._aggregate(self.exact_hits)

        # a generation that never reaches its \boxed{} is either truncated or not
        # following the prompt, and then the answer is parsed out of prose
        boxed = sum(
            1
            for prediction in predictions
            if isinstance(prediction, str) and "oxed{" in prediction
        )
        print(
            f"DynaMath: {boxed}/{len(predictions)} generations contain a "
            f"\\boxed{{}}, {self.match} average {self.average_score:.4f}, "
            f"worst {self.worst_score:.4f} "
            f"(exact average {self.exact_average:.4f}, "
            f"worst {self.exact_worst:.4f})"
        )
        return self.average_score

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, the mode, and both metrics under both modes."""
        description: Dict[str, Any] = {
            "split": self.split,
            "match": self.match,
            "variants": len(self.question_ids),
            "questions": len(set(self.question_ids)),
        }
        if self.partial_questions:
            description["partially_covered_questions"] = self.partial_questions
        for name, value in (
            ("dynamath_average", self.average_score),
            ("dynamath_worst", self.worst_score),
            ("dynamath_average_exact", self.exact_average),
            ("dynamath_worst_exact", self.exact_worst),
        ):
            if value is not None:
                description[name] = value
        return description

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every subject.

        The accuracy of a subject is its question-averaged average score, the
        same quantity the headline reports. The latency is the one of the whole
        run, so a subject's throughput is its share of the aggregate rather than
        a figure it could reach on its own.
        """
        if not self.subjects:
            return None

        performance = {}
        for subject in sorted(set(self.subjects)):
            indexes = [
                index
                for index, name in enumerate(self.subjects)
                if name == subject and index < len(states)
            ]
            if not indexes:
                continue
            metrics = compute_metrics(
                [states[index] for index in indexes], latency, answer_key=answer_key
            )
            scored = [index for index in indexes if index < len(self.hits)]
            if scored:
                by_question: Dict[str, List[float]] = defaultdict(list)
                for index in scored:
                    by_question[self.question_ids[index]].append(self.hits[index])
                metrics.accuracy = sum(
                    sum(values) / len(values) for values in by_question.values()
                ) / len(by_question)
                metrics.num_valid_predictions = len(scored)
            performance[subject] = metrics

        print(
            "DynaMath average accuracy per subject: "
            + ", ".join(
                f"{subject}="
                + ("n/a" if metrics.accuracy is None else f"{metrics.accuracy:.4f}")
                for subject, metrics in performance.items()
            )
        )
        return performance

    def create_sgl_function(self):
        """Create the SGL function for DynaMath (one figure per question)."""
        return create_image_sgl_function(
            function_name="get_dynamath_answer",
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
