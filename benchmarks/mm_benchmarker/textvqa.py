"""
TextVQA benchmark evaluation script.

TextVQA (https://huggingface.co/datasets/lmms-lab-encoder/textvqa) asks questions
whose answer is written somewhere in the photo -- a brand on a phone, a number on
a scoreboard -- so answering means reading the scene rather than recognising it.

Ported from the lmms-eval `textvqa` tasks, with one deliberate difference: the
prompt is the step-by-step boxed instruction the other benchmarks here send,
not the task's "Answer the question using a single word or phrase". Keeping one
prompt across benchmarks is what makes their generation lengths, and therefore
their speculative-decoding numbers, comparable. Pass ``post_prompt=""`` to send
the question the way the task does; only then are the accuracies comparable to
published lmms-eval numbers.

Split matters here. The `test` split ships ten empty strings in place of its
answers -- the labels live on the evaluation server -- so nothing can be scored
locally, which is why the lmms-eval `textvqa_test` task only writes a submission
file. This benchmark does the same: on `test` it reports throughput and latency
and leaves accuracy unset. Use ``split="validation"`` for a scored run.

Scoring on `validation` is the VQA metric the task uses: each of the ten
references is scored against the other nine, an answer matching at least three of
them earns full credit, and the ten scores are averaged.
"""

import os
import re
import shutil
import statistics
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import (
    STEP_BY_STEP_BOXED_PROMPT,
    create_image_sgl_function,
    extract_boxed,
)

#: Splits that carry usable references; `test` withholds them.
SCORED_SPLITS = ("train", "validation")

_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hes": "he's", "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Im": "I'm", "isnt": "isn't", "itd": "it'd", "itll": "it'll", "its": "it's",
    "lets": "let's", "maam": "ma'am", "mightve": "might've", "mustve": "must've",
    "shant": "shan't", "shed": "she'd", "shes": "she's", "shouldve": "should've",
    "shouldnt": "shouldn't", "thats": "that's", "thered": "there'd",
    "therere": "there're", "theres": "there's", "theyd": "they'd",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "wasnt": "wasn't",
    "wed": "we'd", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
    "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve": "would've",
    "wouldnt": "wouldn't", "yall": "y'all", "youd": "you'd", "youll": "you'll",
    "youre": "you're", "youve": "you've",
}
_NUMBERS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
_COMMA_STRIP = re.compile(r"(\d)(,)(\d)")
_PUNCTUATION = [
    ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-",
    ">", "<", "@", "`", ",", "?", "!",
]


def normalize_answer(text: str) -> str:
    """The EvalAI normalisation the VQA metric is defined on.

    Punctuation, articles, contractions and spelled-out digits are all folded
    away, so "The Nokia." and "nokia" score as the same answer.
    """
    source = str(text).replace("\n", " ").replace("\t", " ").strip()
    # Each test reads the original string, as EvalAI's processor does: dropping
    # a mark must not change how the next one is judged.
    text = source
    for punctuation in _PUNCTUATION:
        surrounded = f"{punctuation} " in source or f" {punctuation}" in source
        if surrounded or _COMMA_STRIP.search(source) is not None:
            text = text.replace(punctuation, "")
        else:
            text = text.replace(punctuation, " ")
    text = _PERIOD_STRIP.sub("", text, re.UNICODE)

    words = []
    for word in text.lower().split():
        word = _NUMBERS.get(word, word)
        if word in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(word, word))
    return " ".join(words)


def score_answer(generation: str, answers: List[str]) -> float:
    """The VQA metric: full credit once three of the ten references agree.

    Each reference is scored against the other nine rather than all ten, which
    is what keeps a single annotator's typo from capping the score.
    """
    references = [normalize_answer(answer) for answer in answers]
    # the prompt asks for a box, so read that first and fall back to the whole
    # generation, exactly as the other benchmarks here do
    prediction = normalize_answer(extract_boxed(generation) or generation)
    if not references:
        return 0.0
    scores = []
    for index in range(len(references)):
        others = [references[j] for j in range(len(references)) if j != index]
        matching = [item for item in others if item == prediction]
        scores.append(min(1.0, len(matching) / 3.0))
    return statistics.mean(scores)


def build_prompt(
    question: str,
    ocr_tokens: Optional[List[str]] = None,
    pre_prompt: str = "",
    post_prompt: str = "\n" + STEP_BY_STEP_BOXED_PROMPT,
) -> str:
    """The lmms-eval prompt with this repo's shared answer instruction.

    ``capitalize()`` is the task's own doing: the dataset stores questions
    lowercased. ``ocr_tokens`` reproduces the task's optional OCR reference,
    which its default configuration leaves off.
    """
    reference = ""
    if ocr_tokens:
        reference = f"\nReference OCR token: {', '.join(ocr_tokens)}"
    return f"{pre_prompt}{str(question).capitalize()}{reference}{post_prompt}"


@MM_BENCHMARKS.register("textvqa")
class TextVQABenchmarker(MMBenchmarker):
    """
    TextVQA benchmark implementation.

    Args:
        num_samples: number of questions to evaluate, all of them when not given.
        subset: unused, TextVQA carries no grouping to select on.
        split: "test" (the default, unscored -- its references are withheld),
            "validation" or "train".
        ocr: send the dataset's OCR tokens alongside the question, as the task's
            `ocr: true` variant does.
        post_prompt: the instruction appended to every question. Defaults to the
            shared step-by-step boxed prompt; pass "" for the task's own
            single-word instruction, which is the only form comparable to
            published lmms-eval numbers.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "test",
        ocr: bool = False,
        post_prompt: Optional[str] = None,
    ):
        super().__init__(num_samples, subset)
        self.split = split
        self.ocr = ocr
        self.post_prompt = (
            "\n" + STEP_BY_STEP_BOXED_PROMPT if post_prompt is None else post_prompt
        )
        self.cache_dir = None
        self.question_ids: List[int] = []
        self.hits: List[float] = []

    def default_max_new_tokens(self) -> int:
        """Room for the reasoning the shared prompt asks for before the box.

        The lmms-eval task answers in a handful of tokens, which is all its
        single-word instruction needs.
        """
        return 2048

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[List[str]]]]:
        """Load TextVQA and write its images out for the image-based SGL call."""
        self.cache_dir = os.path.join(".cache", "textvqa_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        # Asked for by split: the repo carries train, validation and test,
        # and indexing the whole DatasetDict would fetch all three.
        dataset = load_dataset("lmms-lab-encoder/textvqa", split=self.split)
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels: List[Optional[List[str]]] = []
        self.question_ids = []
        for index, row in enumerate(dataset):
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            row["image"].convert("RGB").save(image_path, "PNG")

            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(
                        row["question"],
                        row.get("ocr_tokens") if self.ocr else None,
                        post_prompt=self.post_prompt,
                    ),
                }
            )
            answers = [a for a in (row.get("answers") or []) if str(a).strip()]
            labels.append(answers or None)
            self.question_ids.append(int(row.get("question_id", index)))

        scored = sum(1 for label in labels if label)
        if not scored:
            print(
                f"TextVQA {self.split}: references are withheld on this split, "
                "so throughput and latency are reported without an accuracy."
            )
        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """Keep the raw generation: the VQA metric normalises it against the
        ten references, which compute_accuracy() looks up by index."""
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Average VQA accuracy, or None when the split withholds its answers."""
        self.hits = []
        for prediction, label in zip(predictions, labels):
            if not label or not isinstance(prediction, str):
                continue
            self.hits.append(score_answer(prediction, label))

        if not self.hits:
            return None
        accuracy = sum(self.hits) / len(self.hits)
        print(
            f"TextVQA {self.split} VQA accuracy over {len(self.hits)} "
            f"questions: {accuracy:.4f}"
        )
        return accuracy

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, whether OCR tokens were sent, and the prompt used."""
        return {
            "split": self.split,
            "questions": len(self.question_ids),
            "scored_questions": len(self.hits),
            "ocr_tokens": self.ocr,
            "post_prompt": self.post_prompt,
        }

    def create_sgl_function(self):
        """Create the SGL function for TextVQA (image-based Q&A)."""
        return create_image_sgl_function(
            function_name="get_textvqa_answer",
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
