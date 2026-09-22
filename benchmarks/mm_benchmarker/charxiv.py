"""
CharXiv benchmark (reasoning questions) for speculative-decoding measurement.

CharXiv (https://arxiv.org/abs/2406.18521) pairs 2,323 arXiv charts with one
open-vocabulary reasoning question each. The Hub dataset `princeton-nlp/CharXiv`
has two splits:

- ``validation``: 1,000 charts, with answers;
- ``test``: 1,323 charts, WITHOUT answers -- `reasoning_a` is null on the Hub
  and in the official repo's `data/reasoning_test.json` alike (checked row by
  row). The default split here is ``test``, which therefore yields throughput
  and accept length only; on ``validation`` the rule-based scorer below also
  reports an accuracy.

Prompt variants, selected by ``prompt_variant`` (the ``-origin`` suffix of
`bench_mm.py` picks the official one through `ORIGINAL_PROMPT_KWARGS`):

- ``default``: the question followed by the shared step-by-step prompt this
  repo already sends to ChartQA (describe the chart, reason, then
  ``Final Answer: <answer>``). Long generations that the decoding can be
  measured on, and an answer line that needs no judge to parse.
- ``lmms_eval``: the official CharXiv reasoning prompt as ported by lmms-eval
  (`lmms_eval/tasks/charxiv`): the question plus the answer-type instruction
  block (`REASONING_RESP_INST`), nothing else. Two notes on that port:
  * lmms-eval indexes the instruction by `reasoning_q_source`; the official
    codebase indexes it by `inst_category`, which on the Hub is
    `reasoning_a_type` (verified on the data: rows typed 3 carry numeric
    in-chart answers, rows typed 1 carry label names, while `q_source`
    varies independently). This port follows the official codebase.
  * type 4 (number-in-general) officially appends an instruction derived
    from the ground-truth answer ("exact integer" / "N decimal places").
    Without an answer -- the test split -- a generic "must be a number" line
    is used instead, see `GENERIC_NUMBER_INSTRUCTION`.

Scoring is rule-based per answer type and is a LOWER BOUND on the published
metric, which extracts the answer with GPT-4o and grades it with the rubrics in
lmms-eval's `constant.py`; no judge is ever called here (same stance as
mathverse.py / mathvista.py). Sampling with ``num_samples`` is stratified over
the four answer types, which the split is not stored in order of.
"""

import math
import os
import re
import shutil
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .chartqa import CHARTQA_INSTRUCTION
from .registry import MM_BENCHMARKS
from .utils import create_image_sgl_function, stratified_indices, strip_reasoning

DATASET_PATH = "princeton-nlp/CharXiv"
SPLITS = ("test", "validation")
SPLIT_ALIASES = {"val": "validation", "validation": "validation", "test": "test"}
PROMPT_VARIANTS = ("default", "lmms_eval")

# `reasoning_a_type` / official `inst_category`
ANSWER_TYPES = {
    1: "text-in-chart",
    2: "text-in-general",
    3: "number-in-chart",
    4: "number-in-general",
}

# Verbatim from lmms-eval `lmms_eval/tasks/charxiv/constant.py`
# (REASONING_RESP_INST), itself the official CharXiv prompt. The stray
# "exlicitly" is theirs.
REASONING_RESP_INST = {
    1: """{}
    * Your final answer must be grounded to some text that is explicitly written and relevant to the question in the chart.
    * If you need to answer multiple terms, separate them with commas.
    * Unless specified in the question (such as answering with a letter), you are required to answer the full names of subplots and/or labels by default.
    """,
    2: """{}
    * If there are options in the question, your final answer must conform to one of the options.
    * If there are additional instructions in the question, follow them accordingly.
    * If there are neither options nor additional instructions, you are allowed to respond with a short phrase only.
    """,
    3: """{}
    * Your final answer must be grounded to a number that is exlicitly written and relevant to the question in the chart, even if it's an approximate value.
    * You are allowed to extract numbers within some text when needed.
    """,
    4: """{}
    {}
    """,
}

# what the official prompt says for a type-4 question when no ground-truth
# answer is available to derive the decimal places from
GENERIC_NUMBER_INSTRUCTION = "* Your final answer must be a number."


def get_number_instruction(answer: str) -> str:
    """Port of lmms-eval `reasoning_utils.get_number_instruction`."""
    whole, _, decimal = str(answer).strip().partition(".")
    if not decimal:
        return "* Your final answer must be an exact integer."
    return f"* Your final answer must be a number with {len(decimal)} decimal places."


def build_official_prompt(question: str, answer_type: int, answer: Optional[str]) -> str:
    """The official reasoning prompt for one question, as lmms-eval sends it."""
    if answer_type in (1, 2, 3):
        prompt = REASONING_RESP_INST[answer_type].format(question)
    elif answer_type == 4:
        instruction = (
            get_number_instruction(answer)
            if answer is not None and str(answer).strip()
            else GENERIC_NUMBER_INSTRUCTION
        )
        prompt = REASONING_RESP_INST[4].format(question, instruction)
    else:
        raise ValueError(f"Invalid CharXiv answer type: {answer_type!r}")
    # lmms-eval sends `question.strip()`
    return prompt.strip()


def build_prompt(
    question: str, answer_type: int, answer: Optional[str], prompt_variant: str
) -> str:
    if prompt_variant == "lmms_eval":
        return build_official_prompt(question, answer_type, answer)
    return f"{question}\n{CHARTQA_INSTRUCTION}"


# ----------------------------------------------------------------------------
# rule-based scoring
# ----------------------------------------------------------------------------

_POW10_RE = re.compile(r"10\s*\^\s*\{?\s*([-+]?\d+(?:\.\d+)?)\s*\}?")
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?"
)


def parse_number(text: Optional[str]) -> Optional[float]:
    """The first number in `text`, accepting 1,500 / 1.5e3 / 10^-2 / 94% / $94."""
    if text is None:
        return None
    text = str(text).replace("$", "").replace("\\", "").strip()
    if not text:
        return None
    match = _POW10_RE.search(text)
    if match:
        try:
            return 10.0 ** float(match.group(1))
        except (ValueError, OverflowError):
            return None
    match = _NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def normalize_text(text: Optional[str]) -> str:
    """Case/markdown/punctuation-insensitive form of a short textual answer."""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"[*`]", "", text)
    text = text.strip().strip("\"'").strip()
    # "(b)" and "b" are the same option letter
    if re.fullmatch(r"\(\s*[A-Za-z]\s*\)", text):
        text = text.strip("() ")
    text = re.sub(r"\s+", " ", text)
    while text and text[-1] in ".!?;:" :
        text = text[:-1]
    return text.strip().lower()


def _terms(text: str) -> List[str]:
    return [t for t in (normalize_text(part) for part in text.split(",")) if t]


def score_reasoning(prediction: Optional[str], target: str, answer_type: int) -> float:
    """
    1.0 / 0.0 for one prediction against the reference, by answer type.

    Follows the official rubrics as far as rules can: numeric types compare
    values with notation differences forgiven (1500 == 1.5e3 == 10^3.176...),
    type 4 after rounding to the reference's decimal places; textual types
    compare normalized strings, with comma-separated term lists compared as
    sets (the type-1 rubric allows any order). Semantic paraphrases the judge
    would accept for type 2 are not, hence "lower bound".
    """
    if prediction is None or not str(prediction).strip():
        return 0.0
    target = str(target).strip()
    if answer_type in (3, 4):
        predicted = parse_number(prediction)
        reference = parse_number(target)
        if predicted is not None and reference is not None:
            _, _, decimals = target.partition(".")
            if answer_type == 4 and decimals and decimals.isdigit():
                predicted = round(predicted, len(decimals))
            return (
                1.0
                if math.isclose(predicted, reference, rel_tol=1e-6, abs_tol=1e-9)
                else 0.0
            )
        # fall through: a non-numeric reference such as "Not Applicable"
    normalized_prediction = normalize_text(prediction)
    normalized_target = normalize_text(target)
    if not normalized_target:
        return 0.0
    if normalized_prediction == normalized_target:
        return 1.0
    prediction_terms, target_terms = _terms(prediction), _terms(target)
    if len(target_terms) > 1 and set(prediction_terms) == set(target_terms):
        return 1.0
    if normalized_target in ("yes", "no") and normalized_prediction.startswith(
        normalized_target
    ):
        return 1.0
    return 0.0


_FINAL_ANSWER_RE = re.compile(r"final\s*answer\s*\**\s*:\s*\**", re.IGNORECASE)


def extract_marked_answer(generation: str) -> Optional[str]:
    """
    The text after the last "Final Answer:" marker, first non-empty line.

    ChartQA's extractor also deletes ()[]_* from the answer, which is right for
    ChartQA's plain labels but mangles CharXiv references such as
    "JeVois (408MHz)", "[50, 100]" or "A_v^t"; only markdown emphasis is
    removed here.
    """
    matches = list(_FINAL_ANSWER_RE.finditer(generation))
    if not matches:
        return None
    tail = generation[matches[-1].end():]
    for line in tail.split("\n"):
        line = re.sub(r"[*`]", "", line).strip()
        if line:
            return line
    return None


def last_line_answer(generation: str) -> Optional[str]:
    """
    The last non-empty line of a generation, for the official prompt which asks
    for no answer marker. Stands in for the judge's answer extraction.
    """
    for line in reversed(generation.split("\n")):
        line = re.sub(r"[*`]", "", line).strip()
        if line:
            return line
    return None


@MM_BENCHMARKS.register("charxiv")
class CharXivBenchmarker(MMBenchmarker):
    """
    CharXiv reasoning questions.

    Args:
        num_samples: number of questions, all of them when not given; the
            subset is stratified over the four answer types.
        subset: optional tokens, any of ``test`` / ``val`` / ``validation`` to
            pick the split, and ``type1`` .. ``type4`` to keep only those
            answer types (1 text-in-chart, 2 text-in-general, 3 number-in-chart,
            4 number-in-general). E.g. ``charxiv:200:val,type3,type4``.
        split: the split when the subset names none, ``test`` by default.
        prompt_variant: ``default`` (shared ChartQA-style prompt) or
            ``lmms_eval`` (the official prompt, what ``charxiv-origin`` runs).
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
                f"expected one of {PROMPT_VARIANTS}"
            )
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}, expected one of {SPLITS}")
        self.prompt_variant = prompt_variant
        self.split = split
        self.wanted_types: Optional[set] = None
        self._parse_subset(subset or [])
        self.cache_dir: Optional[str] = None
        # per-question metadata, aligned with the loaded questions
        self.answer_types: List[int] = []
        # `categories` is a side channel dump_generations() writes per record
        self.categories: List[str] = []
        self.raw_questions: List[str] = []
        # per-question 1.0/0.0 and the answers that were compared, from the
        # last compute_accuracy() call (empty on the test split)
        self.hits: List[float] = []
        self.parsed_answers: List[Optional[str]] = []
        self.scores: Dict[str, float] = {}

    def _parse_subset(self, tokens: List[str]) -> None:
        types = set()
        for token in tokens:
            key = token.strip().lower()
            if key in SPLIT_ALIASES:
                self.split = SPLIT_ALIASES[key]
            elif re.fullmatch(r"type[1-4]", key):
                types.add(int(key[-1]))
            else:
                raise ValueError(
                    f"Unknown CharXiv subset token {token!r}, expected one of "
                    f"{sorted(SPLIT_ALIASES)} or type1..type4"
                )
        self.wanted_types = types or None

    # ------------------------------------------------------------------ data
    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        self.cache_dir = os.path.join(".cache", "charxiv_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset(DATASET_PATH)[self.split]

        # read the type column on its own: filtering row by row would decode
        # every image on the way
        types = [int(value) for value in dataset["reasoning_a_type"]]
        keep = [
            index
            for index, answer_type in enumerate(types)
            if self.wanted_types is None or answer_type in self.wanted_types
        ]
        if not keep:
            raise ValueError(
                f"No CharXiv {self.split} rows of answer type(s) "
                f"{sorted(self.wanted_types or [])}"
            )
        chosen = [
            keep[position]
            for position in stratified_indices(
                [types[index] for index in keep], self.num_samples
            )
        ]
        dataset = dataset.select(chosen)

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        self.answer_types, self.categories, self.raw_questions = [], [], []
        for position, row in enumerate(dataset):
            # charts are full of small text, keep them lossless
            image_path = os.path.join(image_dir, f"{position:06d}.png")
            row["image"].convert("RGB").save(image_path, "PNG")

            answer_type = int(row["reasoning_a_type"])
            answer = row.get("reasoning_a")
            answer = str(answer).strip() if answer is not None else None
            question = str(row["reasoning_q"]).strip()

            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(
                        question, answer_type, answer, self.prompt_variant
                    ),
                }
            )
            labels.append(answer or None)
            self.answer_types.append(answer_type)
            self.categories.append(ANSWER_TYPES[answer_type])
            self.raw_questions.append(question)

        with_answers = sum(1 for label in labels if label)
        print(
            f"CharXiv {self.split}: {len(questions)} reasoning questions "
            f"({with_answers} with a reference), prompt={self.prompt_variant}, "
            f"types={dict(sorted(Counter(self.categories).items()))}"
        )
        return questions, labels

    # --------------------------------------------------------------- scoring
    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """The answer line of the generation, or its last line for the official prompt."""
        if not isinstance(output, str):
            return None
        generation = strip_reasoning(output)
        if self.prompt_variant == "default":
            return extract_marked_answer(generation)
        return last_line_answer(generation)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """
        Rule-based accuracy over the questions that have a reference.

        None on the test split, whose references are not public, so that
        `metrics.accuracy` stays unset rather than reading 0.
        """
        self.hits, self.parsed_answers = [], []
        scored: List[float] = []
        per_type: Dict[str, List[float]] = {}
        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            self.parsed_answers.append(prediction if isinstance(prediction, str) else None)
            if not label:
                self.hits.append(0.0)
                continue
            answer_type = self.answer_types[index] if index < len(self.answer_types) else 1
            hit = score_reasoning(prediction, label, answer_type)
            self.hits.append(hit)
            scored.append(hit)
            per_type.setdefault(ANSWER_TYPES[answer_type], []).append(hit)
        if not scored:
            return None
        self.scores = {
            name: sum(values) / len(values) for name, values in sorted(per_type.items())
        }
        accuracy = sum(scored) / len(scored)
        print(
            f"CharXiv rule-based accuracy over {len(scored)} questions: "
            f"{accuracy:.4f} (lower bound; the published metric is GPT-judged), "
            + ", ".join(f"{name}={value:.4f}" for name, value in self.scores.items())
        )
        return accuracy

    # ------------------------------------------------------------ reporting
    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """Metrics per answer type; the latency is the whole run's, as elsewhere."""
        if not self.answer_types:
            return None
        performance: Dict[str, BenchmarkMetrics] = {}
        for answer_type, name in sorted(ANSWER_TYPES.items()):
            indexes = [
                index
                for index, value in enumerate(self.answer_types)
                if value == answer_type and index < len(states)
            ]
            if not indexes:
                continue
            metrics = compute_metrics(
                [states[index] for index in indexes], latency, answer_key=answer_key
            )
            if self.scores and name in self.scores:
                hits = [self.hits[index] for index in indexes if index < len(self.hits)]
                metrics.accuracy = sum(hits) / len(hits)
                metrics.num_valid_predictions = len(hits)
            performance[name] = metrics
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        scoring = (
            "none: the test split has no public references"
            if self.split == "test"
            else "rule-based (no judge), a lower bound on the GPT-judged metric"
        )
        return {
            "dataset": DATASET_PATH,
            "split": self.split,
            "task": "reasoning",
            "prompt_variant": self.prompt_variant,
            "questions": len(self.answer_types),
            "questions_per_type": dict(sorted(Counter(self.categories).items())),
            "type4_number_instruction": (
                "from the reference answer" if self.split != "test" else "generic"
            ),
            "scoring": scoring,
        }

    def default_max_new_tokens(self) -> int:
        """Room for the describe-then-answer prompt; the official one is terse."""
        return 1024 if self.prompt_variant == "default" else 512

    def create_sgl_function(self):
        return create_image_sgl_function(
            function_name="get_charxiv_answer",
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
