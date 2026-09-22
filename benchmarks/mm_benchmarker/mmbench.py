"""
MMBench (English) benchmark for speculative-decoding measurement.

Data: `lmms-lab-encoder/MMBench`, config ``en`` -- the dataset lmms-eval's
`mmbench_en_*` tasks read, public, images as a proper Image feature. It is the
same 4,329 dev / 6,666 test rows as `HuggingFaceM4/MMBench` (which stores the
images as base64 strings). ``test`` answers are not public; ``dev`` has them.

CircularEval rows. Every base question (``index`` below 1e6) is stored again
with its options rotated (``index + k*1e6``): dev is 1,164 base questions in
4,329 rows, test is 1,784 in 6,666. The official metric scores a base question
as correct only when every rotation is answered correctly. This port keeps the
BASE QUESTIONS ONLY by default (one request per distinct image/question, a
"vanilla" single-pass accuracy on dev) -- the ``allrows`` subset token keeps the
rotated copies as independent questions instead. Rows are stored grouped by
source (the first few hundred are all ScienceQA two-option questions), so the
``num_samples`` subset is stratified over the 20 categories.

Prompts, selected by ``prompt_variant`` (``-origin`` picks the official one):

- ``default``: the official question text and options block, then the shared
  step-by-step boxed instruction every MCQ benchmark here uses, so the
  generation is long enough to measure decoding and `extract_choice()` reads
  the letter back out of the box.
- ``lmms_eval``: the official lmms-eval prompt verbatim --
  ``{hint} {question} There are several options:\\nA. ...\\nB. ...`` followed by
  ``Answer with the option's letter from the given choices directly.``

Scoring is rule-based (no judge): the option letter via `extract_choice()`,
falling back to the single option whose text appears in the generation -- the
rule part of lmms-eval's `MMBench_Evaluator.can_infer()`, whose GPT fallback
for the remainder is never called here.
"""

import os
import re
import shutil
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import (
    STEP_BY_STEP_BOXED_PROMPT,
    create_image_sgl_function,
    extract_choice,
    stratified_indices,
    strip_reasoning,
)

DATASET_PATH = "lmms-lab-encoder/MMBench"
DATASET_NAME = "en"
SPLITS = ("test", "dev")
SPLIT_ALIASES = {"test": "test", "dev": "dev", "val": "dev", "validation": "dev"}
OPTION_LETTERS = ("A", "B", "C", "D")
# lmms-eval's MMBench_Evaluator default `sys_prompt`, which heads the options
OPTIONS_HEADER = "There are several options:"
PROMPT_VARIANTS = {
    "default": "\n" + STEP_BY_STEP_BOXED_PROMPT,
    "lmms_eval": "\nAnswer with the option's letter from the given choices directly.",
}
# the dataset stores every missing value as one of these strings
MISSING = {"", "nan", "none", "null"}


def present(value: Any) -> Optional[str]:
    """The cell as text, or None when the dataset marks it missing."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING else text


def options_of(row: Dict[str, Any]) -> Dict[str, str]:
    """The options a row actually has, in letter order."""
    return {
        letter: text
        for letter in OPTION_LETTERS
        for text in [present(row.get(letter))]
        if text is not None
    }


def build_options_block(options: Dict[str, str]) -> str:
    """Port of lmms-eval `MMBench_Evaluator.create_options_prompt`."""
    lines = [OPTIONS_HEADER] + [f"{letter}. {text}" for letter, text in options.items()]
    return "\n".join(lines)


def build_prompt(
    question: str, hint: Optional[str], options: Dict[str, str], prompt_variant: str
) -> str:
    """Official question text and options block, then the variant's instruction."""
    body = f"{hint} {question}" if hint else question
    return f"{body} {build_options_block(options)}{PROMPT_VARIANTS[prompt_variant]}"


def infer_from_option_text(generation: str, options: Dict[str, str]) -> Optional[str]:
    """
    The letter whose option text is the only one contained in the generation,
    the rule lmms-eval applies before falling back to a judge.
    """
    haystack = generation.lower()
    candidates = [
        letter
        for letter, text in options.items()
        if text and text.lower() in haystack
    ]
    return candidates[0] if len(candidates) == 1 else None


@MM_BENCHMARKS.register("mmbench")
class MMBenchBenchmarker(MMBenchmarker):
    """
    MMBench (en), base questions only unless asked otherwise.

    Args:
        num_samples: number of questions, all of them when not given; the
            subset is stratified over the 20 categories.
        subset: optional tokens: ``test`` / ``dev`` (``val``) pick the split;
            ``allrows`` keeps the CircularEval rotated copies as questions of
            their own; anything else is matched case-insensitively against
            the ``category`` and ``L2-category`` names to filter the questions,
            e.g. ``mmbench:200:test,ocr,object_localization``.
        split: the split when the subset names none, ``test`` by default.
        prompt_variant: ``default`` (shared boxed prompt) or ``lmms_eval``
            (the official prompt, what ``mmbench-origin`` runs).
    """

    ORIGINAL_PROMPT_KWARGS = {"prompt_variant": "lmms_eval"}

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "test",
        prompt_variant: str = "default",
    ):
        super().__init__(num_samples, subset)
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt variant {prompt_variant!r}, "
                f"expected one of {sorted(PROMPT_VARIANTS)}"
            )
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}, expected one of {SPLITS}")
        self.prompt_variant = prompt_variant
        self.split = split
        self.base_only = True
        # category / L2-category filters, resolved against the data in load_data()
        self.category_filters: List[str] = []
        for token in subset or []:
            key = token.strip().lower()
            if key in SPLIT_ALIASES:
                self.split = SPLIT_ALIASES[key]
            elif key == "allrows":
                self.base_only = False
            elif key:
                self.category_filters.append(key)
        self.cache_dir: Optional[str] = None
        # per-question metadata, aligned with the loaded questions
        self.indices: List[int] = []
        self.options: List[Dict[str, str]] = []
        self.categories: List[str] = []
        self.l2_categories: List[str] = []
        # from the last compute_accuracy() call (empty on the test split)
        self.hits: List[float] = []
        self.parsed_answers: List[Optional[str]] = []
        self.scores: Dict[str, float] = {}

    # ------------------------------------------------------------------ data
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        self.cache_dir = os.path.join(".cache", "mmbench_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset(DATASET_PATH, DATASET_NAME)[self.split]
        # everything but the image, so the selection below decodes no pixels
        meta = dataset.remove_columns(["image"])
        indices = [int(value) for value in meta["index"]]
        categories = [str(value) for value in meta["category"]]
        l2_categories = [str(value) for value in meta["L2-category"]]

        keep = [
            position
            for position, index in enumerate(indices)
            if not self.base_only or index < 1_000_000
        ]
        if self.category_filters:
            known = {name.lower() for name in categories} | {
                name.lower() for name in l2_categories
            }
            unknown = sorted(set(self.category_filters) - known)
            if unknown:
                raise ValueError(
                    f"Unknown MMBench subset token(s) {unknown}; expected a split "
                    f"({sorted(SPLIT_ALIASES)}), 'allrows', or a category among "
                    f"{sorted(known)}"
                )
            wanted = set(self.category_filters)
            keep = [
                position
                for position in keep
                if categories[position].lower() in wanted
                or l2_categories[position].lower() in wanted
            ]
        if not keep:
            raise ValueError("No MMBench rows left after applying the subset")
        chosen = [
            keep[slot]
            for slot in stratified_indices(
                [categories[position] for position in keep], self.num_samples
            )
        ]
        selected = dataset.select(chosen)

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        self.indices, self.options = [], []
        self.categories, self.l2_categories = [], []
        for slot, row in enumerate(selected):
            image_path = os.path.join(image_dir, f"{slot:06d}.png")
            row["image"].convert("RGB").save(image_path, "PNG")

            options = options_of(row)
            question = present(row.get("question")) or ""
            hint = present(row.get("hint"))
            answer = present(row.get("answer"))
            answer = answer.upper() if answer and answer.upper() in options else None

            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(
                        question, hint, options, self.prompt_variant
                    ),
                }
            )
            labels.append(answer)
            self.indices.append(int(row["index"]))
            self.options.append(options)
            self.categories.append(str(row["category"]))
            self.l2_categories.append(str(row["L2-category"]))

        with_answers = sum(1 for label in labels if label)
        print(
            f"MMBench {self.split}: {len(questions)} questions "
            f"({'base only' if self.base_only else 'all rows'}, "
            f"{with_answers} with a reference), prompt={self.prompt_variant}, "
            f"options={dict(sorted(Counter(len(o) for o in self.options).items()))}"
        )
        return questions, labels

    # --------------------------------------------------------------- scoring
    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """The option letter the generation settles on; None when unreadable."""
        if not isinstance(output, str):
            return None
        return extract_choice(strip_reasoning(output), choices=OPTION_LETTERS)

    def _resolve(self, prediction: Optional[str], generation_hint: Any, slot: int) -> Optional[str]:
        """Restrict a letter to the question's options, else infer from option text."""
        options = self.options[slot] if slot < len(self.options) else {}
        if prediction in options:
            return prediction
        if isinstance(generation_hint, str):
            return infer_from_option_text(generation_hint, options)
        return None

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Rule-based accuracy; None on the test split, whose answers are not public."""
        generations = getattr(self, "generations", None) or []
        self.hits, self.parsed_answers = [], []
        scored: List[float] = []
        per_l2: Dict[str, List[float]] = {}
        per_category: Dict[str, List[float]] = {}
        for slot, (prediction, label) in enumerate(zip(predictions, labels)):
            generation = generations[slot] if slot < len(generations) else None
            resolved = self._resolve(prediction, generation, slot)
            self.parsed_answers.append(resolved)
            if not label:
                self.hits.append(0.0)
                continue
            hit = 1.0 if resolved == label else 0.0
            self.hits.append(hit)
            scored.append(hit)
            per_l2.setdefault(self.l2_categories[slot], []).append(hit)
            per_category.setdefault(self.categories[slot], []).append(hit)
        if not scored:
            return None
        self.scores = {
            name: sum(values) / len(values) for name, values in sorted(per_l2.items())
        }
        accuracy = sum(scored) / len(scored)
        print(
            f"MMBench rule-based accuracy over {len(scored)} questions: {accuracy:.4f} "
            "(single-pass, no CircularEval, no judge) -- "
            + ", ".join(f"{name}={value:.3f}" for name, value in self.scores.items())
        )
        return accuracy

    # ------------------------------------------------------------ reporting
    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """Metrics per L2 category; the latency is the whole run's, as elsewhere."""
        if not self.l2_categories:
            return None
        performance: Dict[str, BenchmarkMetrics] = {}
        for name in sorted(set(self.l2_categories)):
            slots = [
                slot
                for slot, value in enumerate(self.l2_categories)
                if value == name and slot < len(states)
            ]
            if not slots:
                continue
            metrics = compute_metrics(
                [states[slot] for slot in slots], latency, answer_key=answer_key
            )
            if self.scores and name in self.scores:
                hits = [self.hits[slot] for slot in slots if slot < len(self.hits)]
                metrics.accuracy = sum(hits) / len(hits)
                metrics.num_valid_predictions = len(hits)
            performance[name] = metrics
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        return {
            "dataset": f"{DATASET_PATH}:{DATASET_NAME}",
            "split": self.split,
            "rows": "base questions only (index < 1e6)" if self.base_only else "all rows incl. circular copies",
            "prompt_variant": self.prompt_variant,
            "questions": len(self.indices),
            "questions_per_l2_category": dict(sorted(Counter(self.l2_categories).items())),
            "scoring": (
                "none: test answers are not public"
                if self.split == "test"
                else "rule-based single-pass (no CircularEval, no judge)"
            ),
        }

    def default_max_new_tokens(self) -> int:
        """lmms-eval's setting for the task."""
        return 1024

    def create_sgl_function(self):
        return create_image_sgl_function(
            function_name="get_mmbench_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            assistant_prefix=self.assistant_prefix,
        )

    def run(self, *args, **kwargs):
        try:
            return super().run(*args, **kwargs)
        finally:
            if self.cache_dir and os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                print(f"Deleted temporary directory: {self.cache_dir}")
