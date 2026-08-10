"""
SimpleVQA benchmark evaluation script.

SimpleVQA (https://huggingface.co/datasets/m-a-p/SimpleVQA) asks short factual
questions about an image, in English and in Chinese, over 106 categories. The
answer is a short phrase, and the metrics are reported overall, per language and
per category.

The prompt and the normalization are ported from the lmms-eval `simplevqa` task:
the question is sent with the "Answer the question using a short phrase."
instruction, and both the generation and the reference go through
`EvalAIAnswerProcessor` (lowercasing, punctuation stripping, number words,
articles and contractions) before being compared.

The comparison itself is lenient by default: a generation counts when it states
the reference, with or without surrounding context ("the eiffel tower in paris"
for "eiffel tower"). The strict equality the lmms-eval task uses is reported
alongside as `exact_match`, and can be made the metric with `match="exact"`.
"""

import base64
import io
import os
import re
import shutil
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset
from PIL import Image

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_image_sgl_function, reference_in_prediction

# lmms_eval_specific_kwargs of the task: "default" is what published numbers use,
# "qwen3_vl" is the variant the task ships for the Qwen3-VL family
PROMPT_VARIANTS = {
    "default": {
        "pre_prompt": "",
        "post_prompt": "\nAnswer the question using a short phrase.",
    },
    "qwen3_vl": {
        "pre_prompt": "Question: ",
        "post_prompt": "\nAnswer with a short phrase only.",
    },
}

# the columns a --benchmark-list subset is matched against
SUBSET_COLUMNS = ("language", "original_category")


class EvalAIAnswerProcessor:
    """
    Processes an answer similar to Eval AI

    Ported from the lmms-eval task utils, which copied it from
    https://github.com/facebookresearch/mmf/blob/c46b3b3/pythia/tasks/processors.py#L897
    """

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }

    NUMBER_MAP = {
        "none": "0",
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
    ARTICLES = ["a", "an", "the"]
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(\,)+(?=\d)")
    PUNCTUATIONS = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    ]

    def __init__(self, *args, **kwargs):
        pass

    def word_tokenize(self, word):
        word = word.lower()
        word = word.replace(",", "").replace("?", "").replace("'s", " 's")
        return word.strip()

    def process_punctuation(self, in_text):
        out_text = in_text
        for p in self.PUNCTUATIONS:
            if (p + " " in in_text or " " + p in in_text) or (
                re.search(self.COMMA_STRIP, in_text) is not None
            ):
                out_text = out_text.replace(p, "")
            else:
                out_text = out_text.replace(p, " ")
        # the reference passes re.UNICODE as the third argument, which re.sub
        # reads as a count of 32 replacements: kept as-is to score identically
        out_text = self.PERIOD_STRIP.sub("", out_text, re.UNICODE)
        return out_text

    def process_digit_article(self, in_text):
        out_text = []
        temp_text = in_text.lower().split()
        for word in temp_text:
            # the reference uses setdefault, which also grows the shared map;
            # get() returns the same value without that side effect
            word = self.NUMBER_MAP.get(word, word)
            if word not in self.ARTICLES:
                out_text.append(word)
            else:
                pass
        for word_id, word in enumerate(out_text):
            if word in self.CONTRACTIONS:
                out_text[word_id] = self.CONTRACTIONS[word]
        out_text = " ".join(out_text)
        return out_text

    def __call__(self, item):
        item = self.word_tokenize(item)
        item = item.replace("\n", " ").replace("\t", " ").strip()
        item = self.process_punctuation(item)
        item = self.process_digit_article(item)
        return item


EVAL_AI_PROCESSOR = EvalAIAnswerProcessor()


def build_prompt(question: str, variant: str = "default") -> str:
    """The prompt of the lmms-eval task, for one of its prompt variants."""
    pre_prompt = PROMPT_VARIANTS[variant]["pre_prompt"]
    post_prompt = PROMPT_VARIANTS[variant]["post_prompt"]
    return f"{pre_prompt}{question.strip()}{post_prompt}"


def score_answer(prediction: str, reference: str, match: str = "lenient") -> float:
    """
    Score one answer, both sides already normalized by `EvalAIAnswerProcessor`.

    "exact" is the metric of the lmms-eval task, "lenient" also accepts a
    generation that states the reference along with some context, e.g.
    "the eiffel tower in paris" for "eiffel tower".
    """
    if prediction == reference:
        return 1.0
    if match == "lenient" and reference_in_prediction(prediction, reference):
        return 1.0
    return 0.0


def decode_image(image: Any) -> Image.Image:
    """
    The image column holds a base64 string, unlike the other benchmarks where it
    is already decoded. Both are accepted, as in the reference implementation.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")


@MM_BENCHMARKS.register("simplevqa")
class SimpleVQABenchmarker(MMBenchmarker):
    """
    SimpleVQA benchmark implementation.

    Args:
        num_samples: number of questions to evaluate, all of them when not given.
        subset: restrict the questions to one or more languages or categories,
            matched case-insensitively against the `language` and
            `original_category` columns, e.g. `simplevqa:200:en`. All of them when
            not given.
        split: the dataset split to evaluate, "test" is the only one.
        prompt_variant: which `lmms_eval_specific_kwargs` block to use, "default"
            (what published numbers use) or "qwen3_vl".
        match: "lenient" also credits a generation that states the reference along
            with some context, "exact" is the metric of the lmms-eval task. Both
            numbers are reported either way.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "test",
        prompt_variant: str = "default",
        match: str = "lenient",
    ):
        super().__init__(num_samples, subset)
        if match not in ("lenient", "exact"):
            raise ValueError(f"Unknown match mode '{match}', expected exact or lenient")
        self.match = match
        if prompt_variant not in PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt variant '{prompt_variant}', "
                f"expected any of {sorted(PROMPT_VARIANTS)}"
            )
        self.split = split
        self.prompt_variant = prompt_variant
        self.cache_dir = None
        # per-question metadata, kept aligned with the loaded questions
        self.languages: List[str] = []
        self.question_categories: List[str] = []
        # per-question 1.0/0.0 of the last compute_accuracy() call, under the
        # selected mode and under the strict one
        self.hits: List[float] = []
        self.exact_hits: List[float] = []
        self.category_scores: Dict[str, float] = {}
        self.exact_match: Optional[float] = None
        self.empty_generations: Optional[int] = None

    def default_max_new_tokens(self) -> int:
        """The lmms-eval task answers in at most 32 tokens."""
        return 32

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess the SimpleVQA dataset."""
        self.cache_dir = os.path.join(".cache", "simplevqa_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("m-a-p/SimpleVQA")[self.split]

        if self.subset:
            dataset = dataset.select(self._select_subset(dataset))
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels = []
        self.languages = []
        self.question_categories = []
        for index, row in enumerate(dataset):
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            decode_image(row["image"]).save(image_path, "PNG")

            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(row["question"], self.prompt_variant),
                }
            )
            self.languages.append(str(row.get("language", "unknown")))
            self.question_categories.append(
                str(row.get("original_category", "unknown"))
            )

            answer = str(row["answer"]).strip()
            labels.append(answer or None)

        return questions, labels

    def _select_subset(self, dataset) -> List[int]:
        """
        Indices of the rows whose language or category matches the requested
        subset.

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
                f"Unknown SimpleVQA subset(s) {sorted(unknown)}, "
                f"expected a language or a category among {len(available)} values"
            )
        return [
            index
            for index in range(len(dataset))
            if any(column[index] in wanted for column in columns)
        ]

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """Normalize the generation the way the VQA metric expects."""
        return EVAL_AI_PROCESSOR(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Score the normalized answers, overall and per category."""
        self.hits = []
        self.exact_hits = []
        by_category: Dict[str, List[float]] = defaultdict(list)

        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if label is None or not isinstance(prediction, str):
                self.hits.append(0.0)
                self.exact_hits.append(0.0)
                continue
            reference = EVAL_AI_PROCESSOR(str(label))
            hit = score_answer(prediction, reference, self.match)
            self.hits.append(hit)
            # the strict metric of the task is tracked alongside, so that a
            # lenient run stays comparable with a published exact-match number
            self.exact_hits.append(score_answer(prediction, reference, "exact"))
            category = (
                self.question_categories[index]
                if index < len(self.question_categories)
                else "unknown"
            )
            by_category[category].append(hit)

        if not self.hits:
            return None

        self.category_scores = {
            category: sum(scores) / len(scores)
            for category, scores in sorted(by_category.items())
        }
        accuracy = sum(self.hits) / len(self.hits)
        self.exact_match = sum(self.exact_hits) / len(self.exact_hits)
        self.empty_generations = sum(
            1 for prediction in predictions if not str(prediction or "").strip()
        )

        print(
            f"SimpleVQA {self.match} match over {len(self.hits)} questions: "
            f"{accuracy:.4f} across {len(self.category_scores)} categories"
        )
        print(
            f"  exact match {self.exact_match:.4f}, "
            f"{self.empty_generations} empty generations"
        )
        return accuracy

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every language, which is also where the token rates
        differ the most.

        The latency is the one of the whole run, so a language's throughput is its
        share of the aggregate rather than a figure it could reach on its own.
        """
        if not self.languages:
            return None

        performance = {}
        for language in sorted(set(self.languages)):
            indexes = [
                index
                for index, name in enumerate(self.languages)
                if name == language and index < len(states)
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
            performance[language] = metrics
        return performance

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, the prompt variant and the per-category accuracy."""
        description: Dict[str, Any] = {
            "split": self.split,
            "prompt_variant": self.prompt_variant,
            "questions": len(self.languages),
        }
        description["match"] = self.match
        if self.exact_match is not None:
            description["exact_match"] = self.exact_match
            description["empty_generations"] = self.empty_generations
        if self.category_scores:
            description["accuracy_per_category"] = self.category_scores
        return description

    def create_sgl_function(self):
        """Create the SGL function for SimpleVQA (image-based Q&A)."""
        return create_image_sgl_function(
            function_name="get_simplevqa_answer",
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
