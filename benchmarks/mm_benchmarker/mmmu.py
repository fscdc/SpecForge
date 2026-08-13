"""
MMMU benchmark evaluation script.

MMMU (https://huggingface.co/datasets/MMMU/MMMU) is a college-level benchmark of
30 subjects, each one a separate dataset config, mixing multiple-choice and open
questions and sometimes referring to several images in one question. Every subset
is loaded separately and the metrics are reported both overall and per subset.

The prompts and the answer parsing are ported from the lm-evaluation-harness
`mmmu` task, which reuses the code of the MMMU authors:
https://github.com/MMMU-Benchmark/MMMU/blob/main/eval/utils/eval_utils.py

Note on the split: the answers of the `test` split are withheld for the
leaderboard, so accuracy cannot be computed on it and is reported as null. Use
`validation` (900 annotated questions) when the accuracy is what you are after;
the throughput and accept length are meaningful on either.
"""

import ast
import os
import random
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_interleaved_sgl_function

# Prompt formats of the MMMU repository, as used by the lm-eval task, with the
# answer-directly instruction replaced by an explanation plus a \boxed{}. The
# scoring reads the box first (see `extract_boxed`), and only falls back to the
# repository's own parsers when a generation carries none.
MULTI_CHOICE_EXAMPLE_FORMAT = """{}

{}

Answer with an explanation, then put the letter of the correct option in \\boxed{{}}."""

SHORT_ANS_EXAMPLE_FORMAT = """{}

Answer with an explanation, then put your final answer in \\boxed{{}}."""

START_CHR = "A"
OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

# the 30 subjects of the benchmark, one dataset config each
SUBSETS = (
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
)

# questions reference their images as "<image 1>" ... "<image 7>"
IMAGE_PLACEHOLDER = re.compile(r"<image ([1-7])>")
MAX_IMAGES = 7

# the parser falls back on a random option when it finds nothing, seeded so that
# a run stays reproducible
_RANDOM = random.Random(42)


def build_prompt(question: str, question_type: str, options: str) -> str:
    """The MMMU prompt, with the "<image i>" placeholders still in place."""
    if question_type == "multiple-choice":
        choices_str = ""
        for i, choice in enumerate(ast.literal_eval(options)):
            # add (A) {choice1}\n , (B) {choice2}\n , and so on
            choices_str += f"\n({chr(ord(START_CHR) + i)}) {choice}"
        # remove the extraneous prepended \n that we added
        return MULTI_CHOICE_EXAMPLE_FORMAT.format(question, choices_str.lstrip())
    return SHORT_ANS_EXAMPLE_FORMAT.format(question)


def split_into_parts(prompt: str, image_paths: Dict[int, str]) -> List[Tuple[str, str]]:
    """
    Cut the prompt into ("text", str) / ("image", path) parts, so that each image
    is sent where its placeholder was.

    A question may reference the same image several times (e.g. validation_Math_19
    reads <image 1>, <image 2>, <image 1>), which is preserved here. A placeholder
    whose column is empty is dropped.
    """
    parts: List[Tuple[str, str]] = []
    position = 0
    for match in IMAGE_PLACEHOLDER.finditer(prompt):
        text = prompt[position : match.start()]
        if text:
            parts.append(("text", text))
        path = image_paths.get(int(match.group(1)))
        if path:
            parts.append(("image", path))
        position = match.end()
    tail = prompt[position:]
    if tail:
        parts.append(("text", tail))
    return parts


# ----------- Process Multi-choice -------------
def extract_boxed(response: str) -> Optional[str]:
    """
    The content of the last ``\\boxed{...}``, or None when there is none.

    Matched on "oxed{" so that a single- and a double-escaped backslash both
    hit, and closed by brace counting so that a boxed ``\\frac{1}{2}`` survives.
    A generation truncated inside its box yields what it had written so far.
    """
    start = response.rfind("oxed{")
    if start == -1:
        return None
    depth = 1
    collected = []
    for char in response[start + len("oxed{") :]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                break
        collected.append(char)
    return "".join(collected).strip() or None


def _unwrap_boxed_text(text: str) -> str:
    """Drop a LaTeX text wrapper and the punctuation a box is decorated with."""
    text = re.sub(r"\\(?:text|mathrm)\s*\{([^}]*)\}", r"\1", text)
    # one pass over both ends: "(a)." has to lose the period before the paren
    return text.strip().strip("()[]{}$ .,:;").strip()


def parse_boxed_choice(
    boxed: str, all_choices: List[str], index2ans: Dict[str, str]
) -> Optional[str]:
    """
    The option letter a ``\\boxed{}`` names, either directly or by option text.

    Returns None when the box holds neither, leaving the caller to fall back to
    the repository's parser rather than guessing.
    """
    text = _unwrap_boxed_text(boxed)
    if not text:
        return None
    if text.upper() in all_choices:
        return text.upper()
    for letter, option in index2ans.items():
        # both sides normalized the same way, so a boxed "dog." still matches
        if letter in all_choices and _unwrap_boxed_text(str(option)).lower() == text.lower():
            return letter
    return None


def parse_multi_choice_response(
    response: str, all_choices: List[str], index2ans: Dict[str, str]
) -> str:
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f" {choice} " in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5
    # tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        return _RANDOM.choice(all_choices)
    if len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        # get the last one
        return candidates[start_indexes.index(max(start_indexes))]
    # if only one candidate, use it.
    return candidates[0]


# ----------- Process Open -------------
def check_is_number(string: str) -> bool:
    """Check if the given string a number."""
    try:
        float(string.replace(",", ""))
        return True
    except ValueError:
        # check if there's comma inside
        return False


def normalize_str(string: str) -> List[Any]:
    """Normalize the str to lower case and make them float numbers if possible."""
    string = string.strip()

    if check_is_number(string):
        string = string.replace(",", "")
        # leave 2 decimal
        return [round(float(string), 2)]

    # it's likely to be a string, lower it
    string = string.lower()
    if len(string) == 1:
        return [" " + string, string + " "]  # avoid trivial matches
    return [string]


def extract_numbers(string: str) -> List[str]:
    """Exact all forms of numbers from a string with regex."""
    # Pattern for numbers with commas
    pattern_commas = r"-?\b\d{1,3}(?:,\d{3})+\b"
    # Pattern for scientific notation
    pattern_scientific = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    # Pattern for simple numbers without commas
    pattern_simple = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"

    return (
        re.findall(pattern_commas, string)
        + re.findall(pattern_scientific, string)
        + re.findall(pattern_simple, string)
    )


def parse_open_response(response: str) -> List[Any]:
    """
    Parse the prediction from the generated response.
    Return a list of predicted strings or numbers.
    """

    def get_key_subresponses(response: str) -> List[str]:
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators_of_keys = [
            "could be ",
            "so ",
            "is ",
            "thus ",
            "therefore ",
            "final ",
            "answer ",
            "result ",
        ]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            # if last one, accept it's an equation (the entire response can be
            # just one sentence with equation)
            if index == len(sub_responses) - 1:
                indicators_of_keys.extend(["="])
            # the shortest response that may contain the answer (tail part of it)
            shortest_key_response = None
            for indicator in indicators_of_keys:
                if indicator in resp:
                    if not shortest_key_response:
                        shortest_key_response = resp.split(indicator)[-1].strip()
                    elif len(resp.split(indicator)[-1].strip()) < len(
                        shortest_key_response
                    ):
                        shortest_key_response = resp.split(indicator)[-1].strip()

            if shortest_key_response:
                # and it's not trivial
                if shortest_key_response.strip() not in [
                    ":",
                    ",",
                    ".",
                    "!",
                    "?",
                    ";",
                    ":",
                    "'",
                ]:
                    key_responses.append(shortest_key_response)
        if len(key_responses) == 0:  # did not found any
            return [response]
        return key_responses

    key_responses = get_key_subresponses(response)

    pred_list = key_responses.copy()  # keep the original string response
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))

    tmp_pred_list = []
    for i in range(len(pred_list)):
        tmp_pred_list.extend(normalize_str(pred_list[i]))

    # remove duplicates
    return list(set(tmp_pred_list))


# ----------- Evaluation -------------
def eval_multi_choice(gold_i: Any, pred_i: str) -> bool:
    """Evaluate a multiple choice instance."""
    if isinstance(gold_i, list):
        # only they are exactly the same, we consider it as correct
        return any(answer == pred_i for answer in gold_i)
    return gold_i == pred_i


def eval_open(gold_i: Any, pred_i: List[Any]) -> bool:
    """Evaluate an open question instance."""
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)

    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    return True
        elif pred in norm_answers:  # it's a float number
            return True
    return False


@MM_BENCHMARKS.register("mmmu")
class MMMUBenchmarker(MMBenchmarker):
    """
    MMMU benchmark implementation.

    Args:
        num_samples: total number of questions to evaluate, spread as evenly as
            possible over the selected subsets. All of them when not given.
        subset: the subjects to evaluate, e.g. `mmmu:300:Math,Physics`, matched
            case-insensitively against the 30 dataset configs. All of them when
            not given.
        split: the dataset split to evaluate. The answers of the default "test"
            split are withheld, so accuracy is only available on "validation"
            and "dev".
        single_image_only: drop the questions that would send more than one image,
            which SGLang's frontend refuses (`assert len(s.images_) == 1, "Only
            support one image."` in `RuntimeEndpoint._add_images`). What was
            dropped is reported per subset in the results file.
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        subset: Optional[List[str]] = None,
        split: str = "test",
        single_image_only: bool = True,
    ):
        super().__init__(num_samples, subset)
        self.subsets = self._resolve_subsets(subset)
        self.split = split
        self.single_image_only = single_image_only
        self.cache_dir = None
        # per-question metadata, kept aligned with the loaded questions
        self.question_subsets: List[str] = []
        self.question_types: List[str] = []
        self.options_list: List[List[str]] = []
        # per-question 1.0/0.0 of the last compute_accuracy() call, None for
        # the questions whose reference answer the split withholds
        self.hits: List[Optional[float]] = []
        # subset -> {"skipped": n, "kept": n, "total": n} of the multi-image filter
        self.multi_image_filter: Dict[str, Dict[str, int]] = {}

    def default_max_new_tokens(self) -> int:
        """
        Room for the explanation the prompt asks for before the \\boxed{}.

        A generation cut off before it reaches its box falls back to parsing the
        explanation, which is exactly what the box is there to avoid.
        """
        return 4096

    @staticmethod
    def _resolve_subsets(subset: Optional[List[str]]) -> List[str]:
        """Map the requested subject names onto the dataset config names."""
        if not subset:
            return list(SUBSETS)
        by_lowercase = {name.lower(): name for name in SUBSETS}
        resolved = []
        for name in subset:
            key = name.strip().lower()
            if key not in by_lowercase:
                raise ValueError(
                    f"Unknown MMMU subset '{name}', expected any of {', '.join(SUBSETS)}"
                )
            resolved.append(by_lowercase[key])
        return resolved

    def _samples_per_subset(self) -> Dict[str, Optional[int]]:
        """
        Spread `num_samples` over the subsets, so that a capped run still covers
        every subject instead of exhausting the first ones.
        """
        if self.num_samples is None:
            return {name: None for name in self.subsets}
        quota, remainder = divmod(self.num_samples, len(self.subsets))
        return {
            name: quota + (1 if idx < remainder else 0)
            for idx, name in enumerate(self.subsets)
        }

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess the MMMU dataset, one config per subject."""
        self.cache_dir = os.path.join(".cache", "mmmu_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        questions: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        self.question_subsets = []
        self.question_types = []
        self.options_list = []
        self.multi_image_filter = {}

        quotas = self._samples_per_subset()
        for name in self.subsets:
            quota = quotas[name]
            if quota == 0:
                continue
            dataset = load_dataset("MMMU/MMMU", name=name)[self.split]
            # filter before the quota, so that a capped run still gets the number
            # of questions it asked for
            if self.single_image_only:
                keep = self._single_image_indices(dataset)
                if len(keep) < len(dataset):
                    self.multi_image_filter[name] = {
                        "skipped": len(dataset) - len(keep),
                        "kept": len(keep),
                        "total": len(dataset),
                    }
                dataset = dataset.select(keep)
            if quota is not None:
                dataset = dataset.select(range(min(quota, len(dataset))))

            for row in dataset:
                index = len(questions)
                prompt = build_prompt(
                    row["question"], row["question_type"], row["options"]
                )
                image_paths = self._materialize_images(row, index, image_dir)
                questions.append({"parts": split_into_parts(prompt, image_paths)})

                self.question_subsets.append(name)
                self.question_types.append(row["question_type"])
                self.options_list.append(
                    ast.literal_eval(row["options"])
                    if row["question_type"] == "multiple-choice"
                    else []
                )
                labels.append(self._parse_label(row["answer"]))
            print(f"Loaded subset '{name}' with {len(dataset)} questions")

        if self.multi_image_filter:
            skipped = sum(
                counts["skipped"] for counts in self.multi_image_filter.values()
            )
            print(
                f"Skipped {skipped} multi-image questions over "
                f"{len(self.multi_image_filter)} subsets, which SGLang's frontend "
                "cannot send: " + self._format_filter()
            )

        if labels and all(label is None for label in labels):
            print(
                f"The '{self.split}' split of MMMU has no public answers, so only "
                "the throughput and the accept length are measured. Use "
                "'validation' to also get the accuracy."
            )
        return questions, labels

    @staticmethod
    def _single_image_indices(dataset) -> List[int]:
        """
        Indices of the questions that send at most one image.

        Counted from the "<image i>" placeholders of the prompt, since that is
        what decides how many images the request carries: a question referring to
        the same image twice sends it twice. Only the text columns are read, so
        no image is decoded on the way.
        """
        prompts = (
            build_prompt(question, question_type, options)
            for question, question_type, options in zip(
                dataset["question"], dataset["question_type"], dataset["options"]
            )
        )
        return [
            index
            for index, prompt in enumerate(prompts)
            if len(IMAGE_PLACEHOLDER.findall(prompt)) <= 1
        ]

    def _format_filter(self) -> str:
        return ", ".join(
            f"{name} {counts['skipped']}/{counts['total']}"
            for name, counts in self.multi_image_filter.items()
        )

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split and what the multi-image filter removed."""
        description: Dict[str, Any] = {
            "split": self.split,
            "single_image_only": self.single_image_only,
            "questions": len(self.question_subsets),
        }
        if self.multi_image_filter:
            description["skipped_multi_image"] = sum(
                counts["skipped"] for counts in self.multi_image_filter.values()
            )
            # only the subsets that actually lost questions
            description["skipped_multi_image_per_subset"] = self.multi_image_filter
        return description

    @staticmethod
    def _parse_label(answer: Any) -> Optional[str]:
        """The reference answer, or None when the split withholds it."""
        answer = str(answer).strip()
        return answer if answer and answer != "?" else None

    def _materialize_images(
        self, row: Dict[str, Any], index: int, image_dir: str
    ) -> Dict[int, str]:
        """Save the images of one question, keyed by their placeholder number."""
        image_paths = {}
        for number in range(1, MAX_IMAGES + 1):
            image = row.get(f"image_{number}")
            if image is None:
                continue
            path = os.path.join(image_dir, f"{index:06d}_{number}.png")
            image.convert("RGB").save(path, "PNG")
            image_paths[number] = path
        return image_paths

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """
        Keep the raw generation: the parsing needs the question type and the
        options, which compute_accuracy() looks up by index.
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Score every question, and report the overall accuracy."""
        self.hits = []
        for index, (prediction, label) in enumerate(zip(predictions, labels)):
            if label is None:
                # the split withholds the reference, so the question is unscorable
                self.hits.append(None)
            elif not isinstance(prediction, str):
                self.hits.append(0.0)
            else:
                self.hits.append(float(self._score(prediction, label, index)))

        scored = [hit for hit in self.hits if hit is not None]
        if not scored:
            return None
        return sum(scored) / len(scored)

    def _score(self, prediction: str, label: str, index: int) -> bool:
        """
        Parse one generation the way its question type asks for, and score it.

        The prompt asks for the answer in a ``\\boxed{}``, so that is read
        first. Falling back to the repository's parsers matters: theirs scan the
        whole generation, which now carries an explanation, and the
        multiple-choice one picks a letter at random when it finds none.
        """
        boxed = extract_boxed(prediction)
        if self.question_types[index] == "multiple-choice":
            option_strs = self.options_list[index]
            all_choices = OPTION_LETTERS[: len(option_strs)]
            index2ans = dict(zip(OPTION_LETTERS, option_strs))
            parsed = (
                parse_boxed_choice(boxed, all_choices, index2ans)
                if boxed is not None
                else None
            )
            if parsed is None:
                parsed = parse_multi_choice_response(prediction, all_choices, index2ans)
            return eval_multi_choice(label, parsed)
        # An open answer is scored against the box alone when there is one:
        # parsing the explanation too would credit any number it mentions.
        return eval_open(
            label, parse_open_response(boxed if boxed is not None else prediction)
        )

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every subset.

        The latency is the one of the whole run, so a subset's throughput is its
        share of the aggregate rather than a figure it could reach on its own.
        """
        if not self.question_subsets:
            return None

        performance = {}
        for name in self.subsets:
            indexes = [
                index
                for index, subset in enumerate(self.question_subsets)
                if subset == name and index < len(states)
            ]
            if not indexes:
                continue
            metrics = compute_metrics(
                [states[index] for index in indexes], latency, answer_key=answer_key
            )
            hits = [
                self.hits[index]
                for index in indexes
                if index < len(self.hits) and self.hits[index] is not None
            ]
            if hits:
                metrics.accuracy = sum(hits) / len(hits)
                metrics.num_valid_predictions = len(hits)
            performance[name] = metrics

        print("MMMU accuracy per subset: " + self._format_breakdown(performance))
        return performance

    @staticmethod
    def _format_breakdown(performance: Dict[str, BenchmarkMetrics]) -> str:
        return ", ".join(
            f"{name}="
            + ("n/a" if metrics.accuracy is None else f"{metrics.accuracy:.4f}")
            for name, metrics in performance.items()
        )

    def create_sgl_function(self):
        """Create the SGL function for MMMU (text interleaved with images)."""
        return create_interleaved_sgl_function(
            function_name="get_mmmu_answer",
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
