"""
Utility functions for the multimodal benchmarks.
"""

import re
from typing import Any, Callable, List, Optional, Tuple

import sglang as sgl

# the option letters of a four-way multiple choice question
CHOICES = ("A", "B", "C", "D")

# What the Qwen3-family chat template emits after the assistant marker when
# thinking is turned off (enable_thinking=False). The SGL frontend builds the
# prompt itself and never runs the HF Jinja template, so the block has to be
# prefilled by hand to keep the model out of its reasoning mode.
NO_THINKING_PREFIX = "<think>\n\n</think>\n\n"


def strip_reasoning(text: str) -> str:
    """Drop a <think> block, whose stray letters would confuse the extraction."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    return re.sub(r"<think>.*", " ", text, flags=re.DOTALL | re.IGNORECASE)


_DEFAULT_CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H"]

_ANSWER_PHRASES = [
    "the answer is",
    "answer is",
    "the correct answer is",
    "correct answer is",
    "the best answer is",
    "best answer is",
    "the correct option is",
    "correct option is",
    "the best option is",
    "best option is",
    "the choice is",
    "choice is",
    "the correct choice is",
    "correct choice is",
    "i choose",
    "i select",
    "i pick",
    "my answer is",
    "my choice is",
    # Korean
    "옵션",
    "정답은",
    "답은",
    "답:",
    # Chinese
    "答案是",
    "答案为",
    "选",
    # Japanese
    "答えは",
]

# Higher = more confident that this is the intended answer.
_FORMAT_PRIORITY = {
    "start": 10,
    "end": 9,
    "phrase": 7,
    "parentheses": 6,
    "period": 5,
    "colon": 4,
    "right_paren": 3,
    "space": 2,
    "fallback": 0,
}


def extract_mcq_answer(response: str, choices: Optional[List[str]] = None) -> str:
    """
    Extract a multiple-choice answer letter from model output.

    Ported verbatim from `lmms_eval.tasks._task_utils.mcq_extract`, so that the
    letter a benchmark scores is the one lmms-eval would have scored. Candidates
    are collected in every supported format, then the highest-priority format
    wins, and within a format the last occurrence.

    Returns an uppercase letter, or "" when nothing matched. The response is
    expected to have its <think> block already stripped, which is what
    `extract_choice()` below takes care of.
    """
    if not response or not response.strip():
        return ""

    all_choices = choices or _DEFAULT_CHOICES

    text = response.strip()
    for char in [",", ".", "!", "?", ";", ":", "'", '"']:
        text = text.strip(char)
    # Pad with spaces for boundary matching.
    text = " " + text + " "

    candidates: list = []  # (letter, position, format_name)

    for ch in all_choices:  # (A)
        if f"({ch})" in text:
            candidates.append((ch, text.rfind(f"({ch})"), "parentheses"))

    for ch in all_choices:  # A.
        if f"{ch}." in text:
            candidates.append((ch, text.rfind(f"{ch}."), "period"))

    for ch in all_choices:  # A:
        if f"{ch}:" in text:
            candidates.append((ch, text.rfind(f"{ch}:"), "colon"))

    for ch in all_choices:  # A)
        if f"{ch})" in text:
            candidates.append((ch, text.rfind(f"{ch})"), "right_paren"))

    for ch in all_choices:  # A followed by a space
        if f"{ch} " in text:
            candidates.append((ch, text.rfind(f"{ch} "), "space"))

    # common answer phrases ("the answer is A", ...)
    text_lower = text.lower()
    for phrase in _ANSWER_PHRASES:
        idx = text_lower.find(phrase)
        if idx != -1:
            after = idx + len(phrase)
            for ch in all_choices:
                ch_pos = text.find(ch, after)
                if ch_pos != -1:
                    candidates.append((ch, ch_pos, "phrase"))

    # starts with a standalone choice letter (not part of a word)
    stripped = text.strip()
    for ch in all_choices:
        if stripped.startswith(ch) and (
            len(stripped) == 1 or not stripped[1].isalpha()
        ):
            candidates.append((ch, 0, "start"))

    # ends with a standalone choice letter
    for ch in all_choices:
        if stripped.endswith(ch) and (
            len(stripped) == 1 or not stripped[-2].isalpha()
        ):
            candidates.append((ch, len(text) - 1, "end"))

    if not candidates:  # any occurrence, lowest priority
        for ch in all_choices:
            if ch in text:
                candidates.append((ch, text.rfind(ch), "fallback"))

    if not candidates:
        return ""

    # highest-priority format wins; within a format, the later position wins
    candidates.sort(key=lambda x: (_FORMAT_PRIORITY.get(x[2], 0), x[1]), reverse=True)
    return candidates[0][0]


# One instruction for every benchmark that asks for a worked answer, so the
# generations stay comparable across tasks and the box is always what scoring
# reads. Kept verbatim from the MathVision task, which is where it comes from.
STEP_BY_STEP_BOXED_PROMPT = (
    'Please solve the problem step by step and put your answer in one "\\boxed{}".'
)


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


def extract_choice(
    response: str, choices: Tuple[str, ...] = CHOICES
) -> Optional[str]:
    """
    The option letter a generation settles on, or None. This is what the
    benchmarks score against.

    Deliberately not `extract_mcq_answer()` above, which is what lmms-eval runs:
    that one ranks a letter at the start or the end of the response above an
    explicitly announced answer, so "A teddy bear is shown" scores as A and
    "C is wrong, the answer is B" scores as C. Here the explicit statement wins,
    and an isolated letter is only read as an answer when it cannot be the
    article of a sentence.

    The patterns are tried from the most explicit to the loosest, and the last
    match of a pattern wins, since a model that reasons before concluding states
    its answer at the end.
    """
    text = strip_reasoning(response).strip()
    if not text:
        return None
    letters = "".join(choices)

    # the whole answer is the letter, e.g. "C", "(C)", "**C**", "C."
    match = re.fullmatch(rf"\**\(?([{letters}])\)?\**\s*[.):,]?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # an explicitly announced answer, e.g. "Answer: C", "the answer is (C)"
    for pattern in (
        rf"\\boxed\{{\s*\(?([{letters}])\)?\s*\}}",
        rf"(?:answer|option|choice)\s*(?:is|:|：)?\s*\**\(?([{letters}])\)?\b",
        rf"答案\s*[:：是]?\s*\(?([{letters}])\)?",
        rf"选择\s*[:：是]?\s*\(?([{letters}])\)?",
    ):
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # a letter that opens a line or is followed by its option text, e.g. "C. bird"
    matches = re.findall(
        rf"(?:^|\n)\s*\**\(?([{letters}])\)?\**\s*[.):,]", text, re.IGNORECASE
    )
    if matches:
        return matches[-1].upper()

    # an isolated letter that cannot be the article "A" of a sentence
    matches = re.findall(rf"(?<![\w])([{letters}])(?![\w])(?!\s+[a-z])", text)
    if matches:
        return matches[-1].upper()

    return None


def reference_in_prediction(prediction: str, reference: str) -> bool:
    """
    Whether the reference answer appears in the generation, both normalized.

    Only this direction is accepted: crediting a prediction because it is
    contained in the reference would score "lake" against "Kaptai Lake", i.e.
    reward an answer that did not identify anything.

    An ASCII reference has to sit on token boundaries, so that "3" does not match
    "35" nor "cat" match "category". A CJK reference is matched as a plain
    substring, since there are no word boundaries to anchor to.
    """
    if not reference or not prediction:
        return False
    if reference.isascii():
        pattern = rf"(?<!\w){re.escape(reference)}(?!\w)"
        return re.search(pattern, prediction) is not None
    return reference in prediction


def create_image_sgl_function(
    function_name: str = "get_image_answer",
    answer_key: str = "answer",
    max_tokens: int = 2048,
    assistant_prefix: Optional[str] = None,
) -> Callable:
    """
    Create an SGL function for image-based Q&A.

    Args:
        function_name: Name of the function
        answer_key: Key for storing the answer
        max_tokens: Maximum tokens to generate
        assistant_prefix: Text to prefill the assistant turn with, e.g.
            NO_THINKING_PREFIX. It is part of the prompt, not of the answer, so
            it never shows up in the generation that gets scored.

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, image_path, question, **kwargs):
        """
        The body of the SGL function: constructs a multimodal conversation flow.

        Note: sgl.image() automatically encodes the image into a format supported
        by the model for multimodal input, using the image token of the chat
        template the backend was created with.
        """
        # User input: Image + Text question
        s += sgl.user(sgl.image(image_path) + question)
        _generate_answer(s, answer_key, max_tokens, assistant_prefix)

    sgl_func.__name__ = function_name
    return sgl_func


def create_interleaved_sgl_function(
    function_name: str = "get_interleaved_answer",
    answer_key: str = "answer",
    max_tokens: int = 2048,
    assistant_prefix: Optional[str] = None,
) -> Callable:
    """
    Create an SGL function for a question that interleaves text and images.

    The question is passed as a list of ("text", str) / ("image", path) parts, so
    that every image lands exactly where its placeholder was in the original
    question, and an image referenced twice is sent twice.

    Args:
        function_name: Name of the function
        answer_key: Key for storing the answer
        max_tokens: Maximum tokens to generate
        assistant_prefix: Text to prefill the assistant turn with

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, parts: List[Tuple[str, str]], **kwargs):
        # the turn is opened by hand because the number of parts varies
        s += sgl.user_begin()
        for kind, value in parts:
            s += sgl.image(value) if kind == "image" else value
        s += sgl.user_end()
        _generate_answer(s, answer_key, max_tokens, assistant_prefix)

    sgl_func.__name__ = function_name
    return sgl_func


def _generate_answer(
    s: Any, answer_key: str, max_tokens: int, assistant_prefix: Optional[str]
) -> None:
    """Append the assistant turn that holds the generation."""
    if assistant_prefix:
        # open the turn by hand so that the prefix lands in the prompt while only
        # the generation is bound to `answer_key`
        s += sgl.assistant_begin()
        s += assistant_prefix
        s += sgl.gen(answer_key, max_tokens=max_tokens)
        s += sgl.assistant_end()
    else:
        s += sgl.assistant(sgl.gen(answer_key, max_tokens=max_tokens))
