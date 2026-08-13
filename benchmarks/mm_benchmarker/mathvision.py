"""
MathVision benchmark evaluation script.

MathVision (https://huggingface.co/datasets/MathLLMs/MathVision) is a set of
competition-style maths problems stated over a figure, mixing multiple choice and
open answers, tagged with a subject and a difficulty level. The metrics are
reported both overall and per subject.

The prompt and the scoring are ported from the lmms-eval `mathvision_test` task,
i.e. the rule-based `mathvision_standard_eval` metric, not the GPT-judge variant:
the generation is stripped down to its `\\boxed{}` (or to the text after "the
answer is"), normalized, and compared to the reference both as a letter and as
the option's text.

That comparison falls back on symbolic evaluation for LaTeX answers, which needs
`latex2sympy2`. Without it, answers still match as text and as plain numbers, so
the accuracy is a lower bound: install it (`pip install latex2sympy2`) to score
exactly like lmms-eval.
"""

import math
import os
import re
import shutil
import string as string_module
from typing import Any, Dict, List, Optional, Tuple

from benchmarker.utils import BenchmarkMetrics, compute_metrics
from datasets import load_dataset

from .base import MMBenchmarker
from .registry import MM_BENCHMARKS
from .utils import create_image_sgl_function

try:
    from latex2sympy2 import latex2sympy
except ImportError:  # scored without symbolic evaluation, see the module docstring
    latex2sympy = None

# prompt of the lmms-eval task
BOXED_PROMPT = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'

# the "flag" phrases the reference implementation cuts the answer out of
ANSWER_FLAGS = (
    "the final answer is",
    "the answer is",
    "the correct answer is",
    "the answer should be",
)

# only these names are available to the expression evaluator below, which is what
# `from math import *` gives the reference implementation
_MATH_NAMESPACE = {
    name: getattr(math, name) for name in dir(math) if not name.startswith("_")
}


def _eval_math_expr(expression: str) -> Any:
    """
    Evaluate a sympy-printed expression, without exposing the builtins.

    The reference implementation calls a bare eval() on text derived from the
    model output; restricting the namespace keeps the maths working while a
    generation cannot reach anything else.
    """
    return eval(expression, {"__builtins__": {}}, dict(_MATH_NAMESPACE))


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def eval_tuple(s: str) -> str:
    """
    Evaluate the mathematical expressions inside a tuple or list written as a
    string, e.g. "(2*3, 5+2)" -> "(6,7)".

    Returns the original string when it is not a tuple/list or when anything
    fails to evaluate.
    """
    if latex2sympy is None or len(s) < 2:
        return s

    sl = s[1:-1].split(",")
    try:
        if s[0] == "(" and s[-1] == ")" and len(sl) > 1:
            # skip the elements that are not meant to be evaluated
            body = ",".join(
                (
                    str(round(_eval_math_expr(str(latex2sympy(sub))), 2))
                    if "infty" not in sub and sub not in ["a", "-a"]
                    else sub
                )
                for sub in sl
            )
            return f"({body})"
        if s[0] == "[" and s[-1] == "]" and len(sl) > 1:
            body = ",".join(
                (
                    str(round(_eval_math_expr(str(latex2sympy(sub))), 2))
                    if "infty" not in sub and sub not in ["a", "-a"]
                    else sub
                )
                for sub in sl
            )
            return f"[{body}]"
    except Exception:
        return s
    return s


def is_equal(gt_answer: str, model_answer: str) -> bool:
    """
    Judge whether the two answers are equivalent, as text, as a tuple, or after
    symbolic evaluation.
    """
    model_answer = model_answer.lower()
    gt_answer = gt_answer.lower()

    if model_answer.replace(" ", "") == "" or gt_answer.replace(" ", "") == "":
        return False
    if gt_answer.strip() == model_answer.strip():
        return True

    model_answer = eval_tuple(model_answer)
    gt_answer = eval_tuple(gt_answer)
    if gt_answer == model_answer:
        return True

    if latex2sympy is None:
        # no symbolic evaluation available, still catch the numeric answers
        try:
            return round(float(gt_answer), 2) == round(float(model_answer), 2)
        except ValueError:
            return False

    try:
        return round(_eval_math_expr(str(latex2sympy(gt_answer))), 2) == round(
            _eval_math_expr(str(latex2sympy(model_answer))), 2
        )
    except Exception:
        return False


def _remove_right_units(string: str) -> str:
    return string.split("\\text{ ")[0]


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string

    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) > 0 and split[0] != "{":
            # the argument of the sqrt is not enclosed in braces
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]

    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
                continue
            # the numerator and the denominator have to be there
            if len(substr) < 2:
                return string
            numerator, denominator = substr[0], substr[1]
            rest = substr[2:] if len(substr) > 2 else ""
            if denominator != "{":
                new_str += "{" + numerator + "}{" + denominator + "}" + rest
            else:
                new_str += "{" + numerator + "}" + denominator + rest
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        a, b = int(a), int(b)
        assert string == "{}/{}".format(a, b)
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except Exception:
        return string


def _strip_string(string: str) -> str:
    """Normalize the LaTeX of an answer, as the reference implementation does."""
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # floating numbers written without their leading zero
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # only the value after an equality or an approximation matters
    if len(string.split("=")) == 2:
        string = string.split("=")[-1]
    if len(string.split("\\approx")) == 2:
        string = string.split("\\approx")[-1]

    if "sqrt" in string:
        string = _fix_sqrt(string)
    string = string.replace(" ", "")
    if "sqrt" in string:
        string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)


def find_math_answer(s: str) -> str:
    """Cut the answer out of a generation and normalize it."""
    s = s.lower()
    if "{}" in s:
        s = s.replace("{}", "")

    try:
        ans = re.compile("oxed{(.*)}", flags=re.S).findall(s)[-1]
    except Exception:
        # no \boxed{}, so the whole string is the answer
        ans = s

    # a closing brace with no opening one before it ends the answer
    if ans.find("}") != -1 and (ans.find("{") == -1 or ans.find("}") < ans.find("{")):
        ans = ans.split("}")[0]

    ans = ans.split("=")[-1]
    ans = ans.split("\\approx")[-1]

    ans = ans.replace(" ", "").replace("\\,", "").replace("∞", "\\infty")
    ans = ans.replace("+\infty", "\infty").replace("\\\\", "\\").replace("\n", "")
    ans = ans.replace("\\text", "").replace("\\mbox", "").replace("bmatrix", "pmatrix")
    ans = ans.replace("\\left", "").replace("\\right", "").replace("^{\\circ}", "")
    ans = ans.replace("^\\circ", "").replace("{m}^3", "").replace("m^3", "")
    ans = ans.replace("{units}", "").replace("units", "")
    ans = ans.replace("{km}", "").replace("km", "")

    return _strip_string(ans)


def build_prompt(question: str, options: List[str]) -> str:
    """
    The prompt of the lmms-eval task, minus one contradiction.

    The task appends "Answer the question with the option's letter from the
    given choices directly." to a multiple-choice question, which tells the
    model the opposite of the step-by-step instruction BOXED_PROMPT already
    opened with. Only the boxed instruction is kept, so both question types ask
    for the same thing; `score_answer` credits a box holding either the letter
    or the option text.

    The missing separator between the instruction and the question is the
    task's, and is reproduced as-is.
    """
    letters = [chr(ord("A") + index) for index in range(len(options))]
    choices_str = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, options)
    )
    if choices_str:
        return f"{BOXED_PROMPT}{question}\nChoices: {choices_str}"
    return f"{BOXED_PROMPT}{question}"


def extract_answer(generation: str, options: List[str]) -> str:
    """Reduce a generation to the answer it states, as the reference does."""
    model_answer = generation.strip()

    for letter in "ABCDE":
        if (
            model_answer.endswith(f" {letter}.")
            or model_answer.endswith(f" ({letter}).")
            or model_answer.startswith(f"{letter}\n")
            or model_answer.startswith(f"({letter})\n")
            or model_answer.startswith(f"({letter}) {letter}\n")
        ):
            model_answer = letter

    if is_number(model_answer.split("is ")[-1].rstrip(".")):
        model_answer = model_answer.split("is ")[-1].rstrip(".")

    if "oxed{" not in model_answer:
        for flag in ANSWER_FLAGS:
            raw_model_answer = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw_model_answer:
                model_answer = model_answer.split("\n")[0].split(". ")[0]
            flag = flag.replace("the", "The")
            raw_model_answer = model_answer
            model_answer = model_answer.split(flag)[-1].strip()
            if flag in raw_model_answer:
                model_answer = model_answer.split("\n")[0].split(". ")[0]
    elif model_answer.count("oxed{") > 1:
        model_answer = "\\boxed{" + model_answer.split("oxed{")[-1]

    model_answer = find_math_answer(model_answer)
    for letter in "abcde":
        model_answer = model_answer.replace(f"({letter})", letter)
        model_answer = model_answer.replace(f"{{{letter}}}", letter)
    return model_answer.rstrip(".").lstrip(":").strip()


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def numeric_match(model_answer: str, reference: str) -> bool:
    """
    Accept a numeric reference that the answer states but did not isolate.

    Two shapes only, both of which the strict comparison misses while leaving no
    room to credit a different number:

    - the answer starts with it and trails something else, e.g. "12cm^2" for 12,
      which `_remove_right_units` only strips for a handful of units;
    - it is the single number of an answer the extraction could not cut down,
      e.g. "theperimeterequals24units" for 24, because the sentence used none of
      the four phrases the reference implementation looks for.
    """
    if not is_number(reference):
        return False

    target = round(float(reference), 2)
    numbers = NUMBER_PATTERN.findall(model_answer.replace(",", ""))
    if not numbers:
        return False
    if model_answer.startswith(numbers[0]) and round(float(numbers[0]), 2) == target:
        return True
    return len(numbers) == 1 and round(float(numbers[0]), 2) == target


def score_answer(
    generation: str, answer: str, options: List[str], match: str = "lenient"
) -> bool:
    """
    Whether a generation answers the question, by letter or by option text.

    "exact" is the rule-based metric of the lmms-eval task. "lenient" adds the
    numeric shapes above, which the task counts as wrong even though the answer
    is stated.
    """
    gt_answer = str(answer)
    gt_answer_value = ""
    is_letter = (
        len(gt_answer) == 1 and gt_answer.upper() in string_module.ascii_uppercase
    )
    if options and is_letter:
        index = ord(gt_answer.upper()) - ord("A")
        if 0 <= index < len(options):
            gt_answer_value = str(options[index])

    model_answer = extract_answer(generation, options)
    if is_equal(gt_answer, model_answer) or is_equal(gt_answer_value, model_answer):
        return True
    if match != "lenient":
        return False
    return numeric_match(model_answer, gt_answer) or numeric_match(
        model_answer, gt_answer_value
    )


@MM_BENCHMARKS.register("mathvision")
class MathVisionBenchmarker(MMBenchmarker):
    """
    MathVision benchmark implementation.

    Args:
        num_samples: number of questions to evaluate, all of them when not given.
        subset: restrict the questions to one or more subjects, matched
            case-insensitively against the `subject` column, e.g.
            `mathvision:200:algebra`. All of them when not given.
        split: the dataset split to evaluate, "test" (3040 questions) or
            "testmini" (304).
        match: "lenient" also credits an answer that states the numeric reference
            without isolating it, "exact" is the rule-based metric of the
            lmms-eval task. Both numbers are reported either way.
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
        # per-question metadata, kept aligned with the loaded questions
        self.subjects: List[str] = []
        self.options_list: List[List[str]] = []
        # per-question 1.0/0.0 of the last compute_accuracy() call, under the
        # selected mode and under the strict one
        self.hits: List[float] = []
        self.exact_hits: List[float] = []
        self.exact_match: Optional[float] = None
        # what the scoring actually compared, for --save-generations
        self.parsed_answers: List[Optional[str]] = []

    def default_max_new_tokens(self) -> int:
        """The lmms-eval task allows a long chain of thought before the answer."""
        return 16384

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess the MathVision dataset."""
        if latex2sympy is None:
            print(
                "latex2sympy2 is not installed, so LaTeX answers are only compared as "
                "text and as plain numbers: the accuracy is a lower bound. "
                "Install it with `pip install latex2sympy2` to score like lmms-eval."
            )

        self.cache_dir = os.path.join(".cache", "mathvision_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("MathLLMs/MathVision")[self.split]

        if self.subset:
            dataset = dataset.select(self._select_subset(dataset))
        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels = []
        self.subjects = []
        self.options_list = []
        for index, row in enumerate(dataset):
            image_path = os.path.join(image_dir, f"{index:06d}.png")
            row["decoded_image"].convert("RGB").save(image_path, "PNG")

            options = [str(option) for option in (row["options"] or [])]
            questions.append(
                {
                    "image_path": image_path,
                    "question": build_prompt(row["question"], options),
                }
            )
            self.options_list.append(options)
            self.subjects.append(str(row.get("subject", "unknown")))
            answer = str(row["answer"]).strip()
            labels.append(answer or None)

        return questions, labels

    def _select_subset(self, dataset) -> List[int]:
        """
        Indices of the rows whose subject matches the requested subset.

        Reads the subject column directly, filtering row by row would decode every
        image on the way.
        """
        wanted = {name.strip().lower() for name in self.subset}
        subjects = [str(subject).strip().lower() for subject in dataset["subject"]]
        unknown = wanted - set(subjects)
        if unknown:
            raise ValueError(
                f"Unknown MathVision subject(s) {sorted(unknown)}, "
                f"expected any of {sorted(set(subjects))}"
            )
        return [index for index, subject in enumerate(subjects) if subject in wanted]

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """
        Keep the raw generation: the scoring needs the options of the question,
        which compute_accuracy() looks up by index.
        """
        return output

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Score every question, and report the overall accuracy."""
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
            self.hits.append(
                float(score_answer(prediction, label, options, self.match))
            )
            # the strict metric is tracked alongside, so that a lenient run stays
            # comparable with a published rule-based number
            self.exact_hits.append(
                float(score_answer(prediction, label, options, "exact"))
            )

        if not self.hits:
            return None

        # a generation that never reaches its \boxed{} is either truncated or not
        # following the prompt, and then the answer is parsed out of prose
        boxed = sum(
            1
            for prediction in predictions
            if isinstance(prediction, str) and "oxed{" in prediction
        )
        self.exact_match = sum(self.exact_hits) / len(self.exact_hits)
        accuracy = sum(self.hits) / len(self.hits)
        print(
            f"MathVision: {boxed}/{len(predictions)} generations contain a \\boxed{{}}, "
            f"{self.match} accuracy {accuracy:.4f}, exact {self.exact_match:.4f}"
        )
        return accuracy

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """Report the split, the match mode and the strict accuracy alongside."""
        description: Dict[str, Any] = {
            "split": self.split,
            "match": self.match,
            "questions": len(self.subjects),
        }
        if self.exact_match is not None:
            description["exact_match"] = self.exact_match
        return description

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, BenchmarkMetrics]]:
        """
        Report the metrics of every subject.

        The latency is the one of the whole run, so a subject's throughput is its
        share of the aggregate rather than a figure it could reach on its own.
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
            hits = [self.hits[index] for index in indexes if index < len(self.hits)]
            if hits:
                metrics.accuracy = sum(hits) / len(hits)
                metrics.num_valid_predictions = len(hits)
            performance[subject] = metrics

        print(
            "MathVision accuracy per subject: "
            + ", ".join(
                f"{subject}="
                + ("n/a" if metrics.accuracy is None else f"{metrics.accuracy:.4f}")
                for subject, metrics in performance.items()
            )
        )
        return performance

    def create_sgl_function(self):
        """Create the SGL function for MathVision (one figure per question)."""
        return create_image_sgl_function(
            function_name="get_mathvision_answer",
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
