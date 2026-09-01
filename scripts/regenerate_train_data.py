import argparse
import base64
import json
import mimetypes
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

from tqdm import tqdm

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:
    OpenAI = None
    _OPENAI_IMPORT_ERROR = exc
else:
    _OPENAI_IMPORT_ERROR = None

try:
    from scripts.conversation_validation import has_think_marker, validate_conversation
except ModuleNotFoundError:
    from conversation_validation import has_think_marker, validate_conversation

try:
    from scripts.prepare_data_mm import (
        LLAVA_OV_FAMILIES,
        _base_config,
        leaks_answer,
    )
except ModuleNotFoundError:
    from prepare_data_mm import LLAVA_OV_FAMILIES, _base_config, leaks_answer

IMAGE_PLACEHOLDER = "<image>"

# The instruction the math benchmarks append to a question, character for
# character in sync with scripts/regenerate_train_data.py::MATH_COT_SUFFIX and
# benchmarks/bench_text.py::COT_SUFFIX. Regenerating math rows behind this
# wording is what keeps the draft model's training prompts in the same
# distribution as its benchmark prompts.
MATH_COT_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# LLaVA-OneVision families rewritten as math. `geometry` is deliberately not
# here: its prompts already carry their own answer-format instruction ("select
# the correct option letter"), so a \boxed{} suffix would contradict it. Pass
# --math-family geometry to include it anyway.
DEFAULT_MATH_FAMILIES = ("text-math",)

# Text blends label their rows with a `source` handle instead of an id prefix,
# matched the way scripts/prepare_data.py matches its --source handles: case
# and punctuation are folded away, so "orcamath" still finds
# "HuggingFaceH4/orca-math-word-problems-200k".
DEFAULT_MATH_SOURCES = ("metamathqa", "orca-math-word-problems")


def _normalize_source(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def is_math_source(source: Any, handles: Sequence[str]) -> bool:
    """Whether a row's source field names one of the math subsets."""
    if not isinstance(source, str):
        return False
    normalized = _normalize_source(source)
    return any(_normalize_source(handle) in normalized for handle in handles)


def input_math_annotation(input_file_path: str) -> str | None:
    """Which of the two math labellings the input carries, if either.

    The blends annotate their rows differently -- LLaVA-OneVision writes the
    config into the id, text blends carry a `source` column -- and neither is
    an error, but a file with no usable label would silently align nothing.
    """
    with open(input_file_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return None
            if row_family(data) is not None:
                return "id"
            if isinstance(data.get("source"), str):
                return "source"
            return None
    return None


def row_family(data: Any) -> str | None:
    """Which LLaVA-OneVision family a row came from, read off its id.

    The records carry no `source` column -- they match the ShareGPT4V schema
    exactly -- but prepare_data_mm.py writes the config into the id as
    ``<config>#<row>-<pair>``, which is enough to recover the family.
    """
    if not isinstance(data, dict):
        return None
    identifier = data.get("id")
    if not isinstance(identifier, str) or "#" not in identifier:
        return None
    return LLAVA_OV_FAMILIES.get(_base_config(identifier.rsplit("#", 1)[0]))


def sanitize_regen_row(data: Any) -> str | None:
    """Drop turns a regeneration cannot use, or say why the row is unusable.

    Three fixes, all for quirks of the LLaVA-OneVision blend:

    * Blank system turns are dropped. `orca_agentinstruct` ships every row with
      an empty system field, which would otherwise render as an empty system
      block and, worse, fail validation and skip the row outright.
    * A blank user turn kills the row: there is no question to answer.
    * Blank assistant turns are dropped rather than rejected. The original
      answers are discarded during regeneration anyway, so an empty one is no
      reason to lose an otherwise good prompt.
    """
    if not isinstance(data, dict):
        return None
    conversations = data.get("conversations")
    if not isinstance(conversations, list):
        return None

    kept: List[Dict[str, Any]] = []
    for message in conversations:
        if not isinstance(message, dict):
            kept.append(message)
            continue
        role = message.get("role")
        content = message.get("content")
        blank = not isinstance(content, str) or not content.strip()
        if role == "user":
            if not isinstance(content, str):
                return "User turn has no text"
            if not content.replace(IMAGE_PLACEHOLDER, "").strip():
                return "User turn is empty apart from the image placeholder"
        elif blank and role in {"system", "assistant"}:
            continue
        kept.append(message)
    data["conversations"] = kept
    return None


def is_math_row(data: Any, families: Sequence[str], sources: Sequence[str]) -> bool:
    """Whether a row is a math problem, under either blend's labelling.

    LLaVA-OneVision rows are identified by the config in their id; text blends
    such as perfectblend carry a `source` handle instead. Checking both lets one
    script regenerate either without the caller saying which it has.
    """
    if row_family(data) in set(families):
        return True
    return is_math_source(data.get("source"), sources)


def apply_math_cot_prompt(
    data: Any, suffix: str, families: Sequence[str], sources: Sequence[str]
) -> bool:
    """Append the benchmark's reasoning instruction to a math row's prompt.

    Only the first user message is rewritten: these rows are one question each,
    and what matters is that the target model answers the exact prompt the
    draft model will later be benchmarked on. An already-suffixed prompt is
    left alone, so a resumed or re-regenerated file never stacks it twice.
    """
    if not suffix or not isinstance(data, dict) or not is_math_row(
        data, families, sources
    ):
        return False
    for message in data["conversations"]:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return False
        if content.rstrip().endswith(suffix.strip()):
            return False
        message["content"] = content.rstrip() + suffix
        return True
    return False

#: Answer-format instructions that pin the response down to a word, a phrase,
#: a yes/no or an option letter. `--align-prompts` REPLACES these with
#: MATH_COT_SUFFIX instead of appending to it: the two contradict, and the
#: boxed wording is what the benchmarks send, so a row that kept its terse
#: instruction would train the draft model on a prompt shape it never meets.
#:
#: The list is exact, closed, and was read off the corpus rather than guessed:
#: every sentence that ends a prompt in >= 20 rows of
#: llava-ov15-1M_train.jsonl (293 of them) was reviewed, and these are the ones
#: that constrain the ANSWER.
#:
#: Exact strings matter more than they look. The blend is full of wording that
#: reads like a terse instruction but IS the task -- "Please introduce the
#: image in a concise, factual manner." alone is 96,902 rows, and "Describe the
#: image concisely.", "Provide a brief description of the given image.",
#: "Express your answer as a common fraction." and "Answer with detailed
#: steps." are all tasks too. Matching on "concise"/"short"/"brief", or on any
#: regex looser than these literals, would rewrite a tenth of the corpus into
#: maths prompts. None of those near misses ever co-occurs with an instruction
#: below (measured: 0 rows), so exact matching alone keeps them safe.
#:
#: Removed longest-first, so an instruction that contains a shorter one cannot
#: leave its tail behind.
TERSE_ANSWER_INSTRUCTIONS = (
    # -- answer reduced to a word or a phrase
    "Answer the question with a short phrase.",
    "Answer the question using a single word or phrase.",
    "Answer the question with a single word or phrase.",
    "Answer the question with a single word or short phrase.",
    "Answer the question with Yes or No.",
    "Answer this question using the text in the image directly.",
    "Reply using only a word or a phrase.",
    "Provide a single-word or phrase answer.",
    "Give your answer in one word or a phrase.",
    "Respond with just one word or a short phrase.",
    "Answer concisely with one word or phrase.",
    "Please respond briefly.",
    "Please answer in short and concise manner.",
    "Answer in brief.",
    "Answer in a concise manner.",
    "Answer in a concise, factual manner.",
    "Provide a concise answer.",
    # -- answer reduced to an option letter. Replaced for the same reason
    # benchmarks/mm_benchmarker/mathvision.py drops the task's own "answer with
    # the option's letter" line: a boxed letter satisfies both, an unboxed one
    # does not.
    "Answer with the option's letter from the given choices directly.",
    "Please select the correct answer by letter.",
    "Answer with the letter.",
)

#: The same class of instruction, written as a preamble that WRAPS the question
#: ("Hint: <instruction>\nQuestion: <question>\nChoices: ...") instead of
#: trailing it. The whole line goes, not just the sentence, or the "Hint:"
#: label would dangle.
TERSE_ANSWER_PREAMBLES = (
    "Hint: Please answer the question and provide the correct option letter, "
    "e.g., A, B, C, D, at the end.",
    "Hint: Please answer the question and provide the final answer at the end.",
    "First conduct reasoning for the text-only mathemtical problem and then "
    "provide the corret option letter at the end.",
    "Answer the mathemtical geometry problem and directly provide the correct "
    "option letter.",
)

#: The label that pairs with a preamble. In all 11,997 rows carrying one, the
#: next line is "Question: ...", so once the preamble goes the label is a
#: vestige -- and dropping it leaves the bare question the benchmarks send.
#: "Choices:" is NOT dropped: a benchmark prompt keeps its choice block.
PREAMBLE_QUESTION_LABEL = "Question:"

#: Prompts that prescribe a shape for the WHOLE response -- llava_cot's
#: <SUMMARY>/<CAPTION> scaffold, ifeval's length/keyword/casing/postscript
#: constraints, geometry's "Answer: xxx" form. Rewriting these would delete the
#: task, so a prompt carrying any of these markers is left exactly as it is
#: even when it also carries a terse instruction (711 rows do).
STRUCTURED_OUTPUT_MARKERS = (
    "You are tasked with analyzing images and providing structured responses",
    "First perform reasoning, then finally select the question from the "
    "choices in the following format:",
    "According to the question shown in the image, please first perform "
    "reasoning, then finally select the right answer from the choices",
    "Your ENTIRE response should be in",
    "Your entire response should be in English, and in all lowercase",
    "Your response should contain at least",
    "Your answer must contain",
    "Highlight at least",
    "Finish your response with this exact phrase",
    "Include keywords",
    "At the end of your response, please explicitly add a postscript",
    "Answer with one of the following options:",
    "Answer with at least",
)

#: Case-insensitive, because ifeval ships a lowercased copy of this one.
STRUCTURED_OUTPUT_MARKERS_CASEFOLD = ("No other words should follow this phrase.",)


def carries_structured_template(text: str) -> bool:
    """Whether a prompt prescribes the shape of the whole response."""
    if any(marker in text for marker in STRUCTURED_OUTPUT_MARKERS):
        return True
    lowered = text.lower()
    return any(
        marker.lower() in lowered for marker in STRUCTURED_OUTPUT_MARKERS_CASEFOLD
    )


def strip_terse_answer_instructions(text: str) -> tuple[str, list[str]]:
    """Remove every terse answer-format instruction, and say which ones went.

    Returns the remaining prompt and the instructions removed. The prompt comes
    back unchanged (and the list empty) when it carries none, so the caller can
    tell a rewrite from a no-op.

    Both shapes are handled. A preamble owns a whole line, but not always the
    first one -- an image row leads with the <image> placeholder, so the
    instruction sits on line two -- hence matching line by line; the
    "Question:" label it left behind goes with it. A trailing or inline
    instruction is cut where it stands, every occurrence of it: some rows stack
    two with no separator ("...single word or phrase.Provide a single-word or
    phrase answer.").
    """
    removed: list[str] = []

    for preamble in TERSE_ANSWER_PREAMBLES:
        if preamble not in text:
            continue
        lines: list[str] = []
        dropped = False
        for line in text.split("\n"):
            stripped = line.strip()
            if not dropped and stripped.startswith(preamble):
                remainder = stripped[len(preamble) :].strip()
                # geo3k writes the label on the SAME line as its preamble
                if remainder.startswith(PREAMBLE_QUESTION_LABEL):
                    remainder = remainder[len(PREAMBLE_QUESTION_LABEL) :].strip()
                dropped = True
                if remainder:
                    lines.append(remainder)
                continue
            if dropped and stripped.startswith(PREAMBLE_QUESTION_LABEL):
                # the label only existed to pair with the preamble
                lines.append(stripped[len(PREAMBLE_QUESTION_LABEL) :].strip())
                continue
            lines.append(line)
        if dropped:
            text = "\n".join(lines)
            removed.append(preamble)

    # longest first: no instruction may leave the tail of another behind
    for instruction in sorted(TERSE_ANSWER_INSTRUCTIONS, key=len, reverse=True):
        if instruction in text:
            text = text.replace(instruction, " ")
            removed.append(instruction)

    return text, removed


def rewrite_terse_answer_prompt(data: Any, suffix: str) -> bool:
    """Replace a row's terse answer-format instruction with the boxed one.

    Only the first user message is rewritten, for the reason
    apply_math_cot_prompt gives: these rows are one question each, and what
    matters is that the target answers the prompt the draft is later
    benchmarked on.

    Returns whether the row was rewritten. It is not when the prompt carries no
    terse instruction, when it prescribes a response shape
    (`carries_structured_template`), when the boxed suffix is already there --
    so a resumed run never stacks it -- or when removing the instruction would
    leave nothing to ask, which would turn the row into a bare image.

    What was removed is recorded on the row under `prompt_rewrite`, so the
    output file carries the evidence that the replacement was complete and not
    merely its result.
    """
    if not suffix or not isinstance(data, dict):
        return False
    conversations = data.get("conversations")
    if not isinstance(conversations, list):
        return False

    for message in conversations:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return False
        if carries_structured_template(content):
            return False
        if content.rstrip().endswith(suffix.strip()):
            return False

        stripped, removed = strip_terse_answer_instructions(content)
        if not removed:
            return False

        # tidy the whitespace the removals left, without touching the
        # question's own line breaks
        stripped = "\n".join(line.rstrip() for line in stripped.split("\n"))
        while "\n\n\n" in stripped:
            stripped = stripped.replace("\n\n\n", "\n\n")
        stripped = stripped.strip()

        # a prompt that was nothing but its instruction has no question left,
        # and MATH_COT_SUFFIX alone is not one
        if not stripped.replace(IMAGE_PLACEHOLDER, "").strip():
            return False

        message["content"] = stripped + suffix
        data["prompt_rewrite"] = {"rule": "terse-answer", "removed": removed}
        return True
    return False


#: LLaVA-OneVision configs whose questions expect a short factual answer -- a
#: value read off a chart, a count, a table cell, a geometry result -- but whose
#: prompts carry no answer-format instruction at all. `--align-prompts` APPENDS
#: MATH_COT_SUFFIX to their question rows, so they train the draft model on the
#: same prompt shape as the rows whose terse instruction was replaced.
#:
#: The list is by config, not by text: these rows have no instruction sentence
#: to match. It was read off the corpus config by config -- generation-style
#: subsets (captions, chart summaries, OCR transcription, TikZ/HTML, region
#: grounding) are deliberately absent.
SHORT_ANSWER_CONFIGS = frozenset(
    {
        # chart values and labels
        "tinychart_train",
        "plotqa",
        "mapqa",
        "cambrian",
        # document / text-in-image QA
        "Docmatix",
        "allenai_pixmo_docs",
        "textvqa",
        "st_vqa",
        "llavar",
        "visualmrc",
        # tables
        "robut_wikisql",
        "robut_wtq",
        "robut_sqa",
        "hitab",
        "finqa",
        "tat_qa",
        # counting and synthetic attributes
        "CLEVR",
        "tallyqa",
        # geometry and figures
        "geo170k_align",
        "geomverse",
        "intergps",
        "arxiv_figs",
        # general knowledge / VQA
        "aokvqa",
        "viquae",
    }
)

#: The subset of SHORT_ANSWER_CONFIGS whose questions are imperative
#: computations ("Compute the diagonal ...", "Find y.") rather than questions:
#: not one of their rows ends with a question mark, so the gate below would
#: exclude exactly the rows the suffix fits best.
SHORT_ANSWER_IMPERATIVE_CONFIGS = frozenset({"geomverse", "intergps"})

#: Question-phrased requests for a description, which a boxed final answer does
#: not fit ("Could you describe the environment shown in the picture?" --
#: cambrian carries a few hundred). Used only to SKIP appending: a false
#: positive here merely leaves a row unchanged, which is why this one may be a
#: heuristic while the replace/append selections must be exact.
DESCRIPTION_QUESTION = re.compile(
    r"^(?:can|could|would) you (?:describe|elaborate|discuss|tell me (?:something|more) about)"
    r"|^what (?:do you (?:see|think is going on)|is happening|are the key elements)"
    r"|^how would you describe",
    re.IGNORECASE,
)


def row_config(data: Any) -> str | None:
    """Which LLaVA-OneVision config a row came from, read off its id."""
    if not isinstance(data, dict):
        return None
    identifier = data.get("id")
    if not isinstance(identifier, str) or "#" not in identifier:
        return None
    return _base_config(identifier.rsplit("#", 1)[0])


def append_short_answer_prompt(
    data: Any, suffix: str, configs: frozenset | set
) -> bool:
    """Append the boxed instruction to a short-answer row that carries none.

    Runs AFTER the terse rewrite and the math rule: a row either of them
    already suffixed ends with `suffix` and is skipped here, so nothing stacks.

    A row is appended to when its config is one of `configs` and its prompt is
    a question (ends with "?"), or unconditionally for the imperative-math
    configs. Rows prescribing a response shape, question-phrased description
    requests, and rows with no text beyond the image placeholder are left
    alone.

    The append is recorded on the row under `prompt_rewrite`, like the terse
    rule records its removals.
    """
    if not suffix or not isinstance(data, dict):
        return False
    config = row_config(data)
    if config is None or config not in configs:
        return False
    conversations = data.get("conversations")
    if not isinstance(conversations, list):
        return False

    for message in conversations:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            return False
        if carries_structured_template(content):
            return False
        if content.rstrip().endswith(suffix.strip()):
            return False

        body = content.replace(IMAGE_PLACEHOLDER, " ").strip()
        if not body:
            return False
        if (
            not body.endswith("?")
            and config not in SHORT_ANSWER_IMPERATIVE_CONFIGS
        ):
            # statements in these configs are the generation-style remainder
            # ("Generate underlying data table of the chart."), not questions
            return False
        if DESCRIPTION_QUESTION.search(body.split("\n")[-1].strip()):
            return False

        message["content"] = content.rstrip() + suffix
        data["prompt_rewrite"] = {"rule": "short-answer-config", "config": config}
        return True
    return False


# Chat/Instruct Qwen3.5 checkpoints only. "-Base" checkpoints are not
# chat-aligned and cannot reliably serve /v1/chat/completions requests.
SUPPORTED_MM_MODELS = (
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
    "Qwen/Qwen3.5-27B-FP8",
    "Qwen/Qwen3.5-27B-GPTQ-Int4",
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-35B-A3B-FP8",
    "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    "Qwen/Qwen3.5-122B-A10B",
    "Qwen/Qwen3.5-122B-A10B-FP8",
    "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4",
    "Qwen/Qwen3.5-397B-A17B",
    "Qwen/Qwen3.5-397B-A17B-FP8",
    "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4",
)


def validate_regen_input(data: Any) -> str | None:
    """Return why a ShareGPT row cannot be regenerated, or ``None``."""
    if not isinstance(data, dict):
        return "Expected a JSON object"

    conversation_error = validate_conversation(
        data.get("conversations"),
        error_style="regeneration",
    )
    if conversation_error is not None:
        return conversation_error

    image_path = data.get("image")
    if image_path is not None:
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            return f"Image file not found: {image_path!r}"
        has_placeholder = any(
            isinstance(message.get("content"), str)
            and IMAGE_PLACEHOLDER in message["content"]
            for message in data["conversations"]
        )
        if not has_placeholder:
            return "Row has an `image` field but no `<image>` placeholder in conversations"

    return None


def set_skipped(data: Any, error: str) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {"status": "skipped", "error": error, "data": data}
    data["status"] = "skipped"
    data["error"] = error
    return data


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def input_has_image_field(path: str) -> bool:
    """Peek the first non-empty row to decide whether this input is
    multimodal. A single preparation run produces a homogeneous file (either
    every row carries an `image` field or none do), so checking the first
    row is sufficient without scanning the whole file.
    """
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            return isinstance(data, dict) and "image" in data
    return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Re-generate training data using sglang model server"
    )

    # model related arguments
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, required=True)
    model_group.add_argument(
        "--reasoning",
        choices=["none", "save", "disable"],
        default="none",
        help=(
            "Reasoning mode: 'none' for standard models, 'save' to store "
            "reasoning_content, or 'disable' to disable thinking via extra_body"
        ),
    )
    model_group.add_argument(
        "--is-gpt-oss",
        action="store_true",
        help="Whether the model is a GPT-OSS model",
    )

    # sampling params
    sampling_params_group = parser.add_argument_group("sampling parameters")
    sampling_params_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sglang model server",
    )
    sampling_params_group.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling top_p",
    )
    sampling_params_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling value sent via extra_body",
    )
    sampling_params_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Mapped to presence_penalty in the OpenAI API",
    )
    sampling_params_group.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens (default: 4096)",
    )

    # optimization
    optimization_group = parser.add_argument_group("optimization")
    optimization_group.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="The number of requests to send to a single server concurrently, the total number of concurrent requests is concurrency * number of server addresses",
    )

    # data related arguments
    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--input-file-path", type=str, required=True, help="Path to the input file"
    )
    data_group.add_argument(
        "--output-file-path", type=str, required=True, help="Path to the output file"
    )
    data_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="The number of samples to regenerate, if not provided, all samples will be regenerated",
    )
    data_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file, skip already processed samples",
    )

    # prompt alignment
    prompt_group = parser.add_argument_group("prompt alignment")
    data_group.add_argument(
        "--no-filter",
        action="store_true",
        help=(
            "Skip the pass that runs once regeneration finishes and drops rows "
            "the target left unusable -- an empty answer, a thinking marker, a "
            "decoding loop, or an answer that just repeats its prompt. The "
            "dropped rows are written to <output>_rejected.jsonl with the "
            "reason, never deleted."
        ),
    )

    prompt_group.add_argument(
        "--align-prompts",
        action="store_true",
        help=(
            "Rewrite prompts to the wording the benchmarks use before asking "
            "the target model, instead of regenerating against the prompt the "
            "JSONL already carries. Two rules run, in this order. Math: a row "
            "gets --math-cot-suffix APPENDED when its LLaVA-OneVision family "
            "is a --math-family, or its `source` handle is a --math-source. "
            "Terse answers: a row whose prompt pins the response to a word, a "
            "phrase, a yes/no or an option letter has that instruction "
            "REPLACED by --math-cot-suffix, because the two contradict; "
            "prompts that prescribe a whole response shape (llava_cot's "
            "<SUMMARY> scaffold, ifeval's constraints, geometry's "
            "'Answer: xxx') are left alone, and --no-terse-rewrite turns the "
            "rule off. Short answers: a question row of a --short-answer-config "
            "(chart/table/document/counting/geometry subsets whose answers are "
            "short but whose prompts carry no instruction) gets the suffix "
            "APPENDED. Off by default for every blend, so the default run "
            "reproduces the dataset's own prompts."
        ),
    )
    prompt_group.add_argument(
        "--math-cot-suffix",
        type=str,
        default=MATH_COT_SUFFIX,
        help=(
            "With --align-prompts, the instruction appended to a math row's "
            "prompt (default: the gsm8k/math500/aime suffix)."
        ),
    )
    prompt_group.add_argument(
        "--no-terse-rewrite",
        action="store_true",
        help=(
            "With --align-prompts, keep the terse answer-format instructions "
            "(the 'short phrase' / 'single word or phrase' suffixes, the "
            "'Hint: ... provide the final answer at the end.' preamble, and "
            "the option-letter ones) instead of replacing them with "
            "--math-cot-suffix. The math rule still runs."
        ),
    )
    prompt_group.add_argument(
        "--short-answer-config",
        type=str,
        nargs="*",
        default=None,
        help=(
            "With --align-prompts, LLaVA-OneVision configs whose "
            "instruction-less question rows get --math-cot-suffix appended "
            "(default: the built-in short-answer list, see "
            "SHORT_ANSWER_CONFIGS). Pass with no values to disable the rule."
        ),
    )
    prompt_group.add_argument(
        "--math-source",
        type=str,
        nargs="+",
        default=list(DEFAULT_MATH_SOURCES),
        help=(
            "With --align-prompts, `source` handles treated as math for text "
            "blends, matched case- and punctuation-insensitively (default: "
            f"{' '.join(DEFAULT_MATH_SOURCES)})."
        ),
    )
    prompt_group.add_argument(
        "--math-family",
        action="append",
        default=None,
        help=(
            "LLaVA-OneVision family treated as math by --align-prompts; "
            "repeatable. `geometry` is excluded by default because its "
            "prompts already specify their own answer format (default: "
            f"{' '.join(DEFAULT_MATH_FAMILIES)})."
        ),
    )

    # sglang server
    server_group = parser.add_argument_group("sglang server")
    server_group.add_argument(
        "--server-address",
        type=str,
        nargs="+",
        help="Server address and port for sglang model server",
    )
    args = parser.parse_args()
    if args.math_family is None:
        args.math_family = list(DEFAULT_MATH_FAMILIES)
    if args.short_answer_config is None:
        args.short_answer_config = sorted(SHORT_ANSWER_CONFIGS)
    unknown_configs = sorted(
        set(args.short_answer_config) - set(LLAVA_OV_FAMILIES)
    )
    if unknown_configs:
        parser.error(
            f"--short-answer-config: unknown config(s) {unknown_configs}, "
            "expected LLaVA-OneVision config names (see "
            "scripts/prepare_data_mm.py::LLAVA_OV_FAMILIES)"
        )
    unknown = sorted(set(args.math_family) - set(LLAVA_OV_FAMILIES.values()))
    if unknown:
        parser.error(
            f"unknown --math-family {unknown}; choose from "
            f"{sorted(set(LLAVA_OV_FAMILIES.values()))}"
        )
    return args


def get_random_reasoning_effort() -> str:
    """Get a random reasoning effort level for the model with weighted probabilities."""
    # usage example: https://huggingface.co/openai/gpt-oss-20b/discussions/28
    # Reasoning effort levels with weights: LOW(4), MEDIUM(4), HIGH(2)
    reasoning_efforts = [
        "low",
        "medium",
        "high",
    ]
    weights = [4, 4, 2]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]


def compute_context_length(conversations: List[Dict[str, Any]]) -> int:
    """
    This is a rough estimate of the context length measured in untokenized
    tokens.
    """
    length = 0
    for message in conversations:
        content = message.get("content")
        if isinstance(content, str):
            # {"role": "assistant", "content": "Hi, how can I help?"}
            length += len(content.split())
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        length += len(text.split())
    return length


def _image_to_data_url(image_path: str) -> str:
    """Read a local image file and encode it as an OpenAI-style data URL."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def to_multimodal_messages(
    messages: List[Dict[str, Any]], image_data_url: str
) -> List[Dict[str, Any]]:
    """Return a copy of ``messages`` with the first `<image>` placeholder
    replaced by an OpenAI-style image content part.

    The stored message list (plain-string content with the placeholder
    intact) is left untouched; this copy exists only to be sent over the
    wire, so every regenerated row keeps writing the placeholder back to
    disk.
    """
    converted = []
    image_injected = False
    for message in messages:
        content = message.get("content")
        if (
            not image_injected
            and isinstance(content, str)
            and IMAGE_PLACEHOLDER in content
        ):
            text = content.replace(IMAGE_PLACEHOLDER, "").strip()
            converted_message = dict(message)
            converted_message["content"] = [
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": text},
            ]
            converted.append(converted_message)
            image_injected = True
        else:
            converted.append(message)
    return converted


def build_query_kwargs(args, messages, max_tokens=None, image_data_url=None):
    effective_max_tokens = max_tokens if max_tokens is not None else args.max_tokens

    query_messages = messages
    if args.reasoning == "save":
        query_messages = []
        for message in messages:
            query_message = dict(message)
            if query_message.get("role") == "assistant":
                query_message.pop("reasoning_content", None)
            query_messages.append(query_message)

    if image_data_url is not None:
        query_messages = to_multimodal_messages(query_messages, image_data_url)

    query_kwargs = dict(
        model=args.model,
        messages=query_messages,
        max_tokens=effective_max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.reasoning == "disable":
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    elif args.reasoning == "save":
        extra_body["chat_template_kwargs"] = {"enable_thinking": True}
    if extra_body:
        query_kwargs["extra_body"] = extra_body
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    return query_kwargs


def call_sglang(
    args,
    server_address: str,
    data: List[Dict[str, Any]],
    max_tokens=None,
) -> str:
    """Send a batch of prompts to sglang /v1/completions."""
    if OpenAI is None:
        raise ModuleNotFoundError(
            "dataset regeneration requires the OpenAI client; install "
            "SpecForge's data extra with `pip install 'specforge[data]'`"
        ) from _OPENAI_IMPORT_ERROR
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")

    messages = data["conversations"]
    regenerated_messages = []

    image_path = data.get("image")
    image_data_url = _image_to_data_url(image_path) if image_path is not None else None

    # ignore data which starts with an assistant message
    if messages[0]["role"] == "assistant":
        data["status"] = "error"
        data["error"] = "Data starts with an assistant message"
        return data

    for message in messages:
        if message["role"] == "system":
            regenerated_messages.append(message)
        elif message["role"] == "assistant":
            continue
        elif message["role"] == "user":
            regenerated_messages.append(message)

            query_kwargs = build_query_kwargs(
                args, regenerated_messages, max_tokens, image_data_url=image_data_url
            )

            try:
                resp = client.chat.completions.create(**query_kwargs)
            except Exception as e:
                data["status"] = "error"
                data["error"] = str(e)
                return data
            response_text = resp.choices[0].message.content
            if args.reasoning == "disable" and (
                not isinstance(response_text, str)
                or not response_text.strip()
                or has_think_marker(response_text)
            ):
                return set_skipped(
                    data,
                    "Non-reasoning assistant response is empty or contains a thinking marker",
                )
            resp_msg = {
                "role": "assistant",
                "content": response_text,
            }
            if args.reasoning == "save":
                response_message = resp.choices[0].message
                reasoning_content = getattr(response_message, "reasoning_content", None)
                if reasoning_content is None:
                    model_extra = getattr(response_message, "model_extra", None)
                    if isinstance(model_extra, dict):
                        reasoning_content = model_extra.get("reasoning_content")
                if max_tokens is None and (
                    not isinstance(response_text, str)
                    or not response_text.strip()
                    or not isinstance(reasoning_content, str)
                    or not reasoning_content.strip()
                ):
                    data["status"] = "error"
                    data["error"] = (
                        "Reasoning generation requires non-empty assistant content "
                        "and reasoning_content"
                    )
                    return data
                if max_tokens is None and (
                    has_think_marker(response_text)
                    or has_think_marker(reasoning_content)
                ):
                    return set_skipped(
                        data,
                        "Reasoning response contains a residual thinking marker",
                    )
                resp_msg["reasoning_content"] = reasoning_content
            regenerated_messages.append(resp_msg)
        else:
            data["status"] = "error"
            data["error"] = f"Invalid message role: {message['role']}"
            return data
    data["conversations"] = regenerated_messages
    data["status"] = "success"
    return data


#: A response this long that keeps restating the same chunk is a decoding loop,
#: not an answer; below it the repetition test is too noisy to trust.
DEGENERATE_MIN_CHARS = 4000
#: Fraction of distinct chunks under which a long response counts as looping.
DEGENERATE_MAX_UNIQUE = 0.25


def is_degenerate(text: str, chunk: int = 64) -> bool:
    """Whether a long response is a decoding loop rather than an answer.

    Chunked rather than compressed or diffed because it has to run over a
    million rows: a loop repeats the same window over and over, so the share of
    distinct windows collapses while ordinary prose stays near one.
    """
    if len(text) < DEGENERATE_MIN_CHARS:
        return False
    chunks = [text[i : i + chunk] for i in range(0, len(text), chunk)]
    return len(set(chunks)) / len(chunks) < DEGENERATE_MAX_UNIQUE


def filter_regenerated(output_file_path: str) -> tuple[int, "Counter"]:
    """Drop rows the regeneration left unusable, keeping the rejects on disk.

    Runs once the whole file exists rather than inside the request loop: a row
    can only be judged against the answer that came back, and one extra pass is
    cheaper than threading this through the futures.
    """
    kept_path = f"{output_file_path}.filtered"
    rejected_path = output_file_path.replace(".jsonl", "_rejected.jsonl")
    reasons: Counter = Counter()
    kept = 0
    with (
        open(output_file_path, encoding="utf-8") as source,
        open(kept_path, "w", encoding="utf-8") as keep,
        open(rejected_path, "w", encoding="utf-8") as reject,
    ):
        for line in source:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            conversations = data.get("conversations") or []
            answers = [
                message
                for message in conversations
                if isinstance(message, dict) and message.get("role") == "assistant"
            ]
            questions = [
                message
                for message in conversations
                if isinstance(message, dict) and message.get("role") == "user"
            ]
            answer = (answers[-1].get("content") or "") if answers else ""
            question = (questions[-1].get("content") or "") if questions else ""

            if not answers:
                reason = "no assistant turn"
            elif not answer.strip():
                reason = "empty answer"
            elif has_think_marker(answer):
                reason = "thinking marker in answer"
            elif leaks_answer(question, answer):
                reason = "answer repeats the prompt"
            elif is_degenerate(answer):
                reason = "degenerate repetition"
            else:
                reason = None

            if reason is None:
                keep.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept += 1
            else:
                data["filtered_reason"] = reason
                reject.write(json.dumps(data, ensure_ascii=False) + "\n")
                reasons[reason] += 1
    os.replace(kept_path, output_file_path)
    if not reasons:
        os.unlink(rejected_path)
    else:
        print(f"  rejects written to {rejected_path}")
    return kept, reasons


def summarise_regenerated(output_file_path: str) -> None:
    """Print what the regenerated training file actually contains.

    Answer length is the number worth watching: a subset whose answers are two
    characters long fills most of a draft block with padding, which flatters
    every acceptance number measured on it.
    """

    def quantile(values, fraction):
        return values[min(int(len(values) * fraction), len(values) - 1)]

    total = 0
    with_image = 0
    with_system = 0
    prompts: Dict[str, List[int]] = defaultdict(list)
    answers: Dict[str, List[int]] = defaultdict(list)
    with open(output_file_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            total += 1
            with_image += data.get("image") is not None
            conversations = data["conversations"]
            with_system += any(m.get("role") == "system" for m in conversations)
            family = row_family(data) or "unlabelled"
            prompts[family].append(len(conversations[-2]["content"]))
            answers[family].append(len(conversations[-1]["content"]))
    if not total:
        print("No rows to summarise.")
        return

    print(f"\nRegenerated training data: {output_file_path}")
    print(
        f"  {total:,} rows -- {with_image:,} with an image, "
        f"{total - with_image:,} text-only, {with_system:,} with a system turn"
    )
    print(
        f"  {'family':<18}{'rows':>10}{'share':>8}"
        f"{'prompt p50':>12}{'answer p50':>12}{'p90':>9}{'<=3 chars':>11}"
    )
    for family, lengths in sorted(answers.items(), key=lambda kv: -len(kv[1])):
        lengths.sort()
        heads = sorted(prompts[family])
        tiny = sum(1 for value in lengths if value <= 3) / len(lengths)
        print(
            f"  {family:<18}{len(lengths):>10,}{len(lengths) / total:>8.1%}"
            f"{quantile(heads, 0.5):>12,}{quantile(lengths, 0.5):>12,}"
            f"{quantile(lengths, 0.9):>9,}{tiny:>11.0%}"
        )


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Validate parameters
    if not (0.0 <= args.temperature <= 1.0):
        raise ValueError("Temperature must be between 0.0 and 1.0")

    if args.max_tokens <= 0:
        raise ValueError("Max tokens must be greater than 0")

    if input_has_image_field(args.input_file_path) and args.model not in SUPPORTED_MM_MODELS:
        raise ValueError(
            f"Input file {args.input_file_path!r} contains multimodal rows "
            f"(an `image` field), but --model {args.model!r} is not a "
            "supported Qwen3.5 chat/instruct checkpoint. Supported models: "
            f"{', '.join(SUPPORTED_MM_MODELS)}"
        )

    print(f"Configuration:")
    print(f"  Model path: {args.model}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Temperature: {args.temperature}")
    print(f"  API URL: {args.server_address}")
    print(f"  Input file: {args.input_file_path}")
    print(f"  Output file: {args.output_file_path}")
    print(f"  Resume mode: {args.resume}")
    print(f"  Align prompts: {args.align_prompts}")
    if args.align_prompts:
        print(f"    math families: {', '.join(args.math_family)}")
        print(f"    math sources : {' '.join(args.math_source)}")
        print(f"    math suffix  : {args.math_cot_suffix.strip()!r}")
        print(
            "    terse rewrite: "
            + ("off (--no-terse-rewrite)" if args.no_terse_rewrite else "on")
        )
        print(
            "    short answers: "
            + (
                f"{len(args.short_answer_config)} configs"
                if args.short_answer_config
                else "off (--short-answer-config with no values)"
            )
        )
        labelling = input_math_annotation(args.input_file_path)
        if labelling is None:
            print(
                "    WARNING: the input rows carry neither a LLaVA-OneVision "
                "id nor a 'source' field, so no math prompt will be aligned. "
                "Rebuild the input with scripts/prepare_data.py "
                "--dataset perfectblend (which keeps 'source'), or drop "
                "--align-prompts to silence this."
            )
        else:
            print(f"    math labelled by: {labelling}")
    print("-" * 50)
    total_lines = count_lines(args.input_file_path)

    skip_lines = 0
    error_file_path = args.output_file_path.replace(".jsonl", "_error.jsonl")
    skipped_file_path = args.output_file_path.replace(".jsonl", "_skipped.jsonl")

    if args.resume and os.path.exists(args.output_file_path):
        existing_success = count_lines(args.output_file_path)
        existing_error = 0
        if os.path.exists(error_file_path):
            existing_error = count_lines(error_file_path)
        existing_skipped = 0
        if os.path.exists(skipped_file_path):
            existing_skipped = count_lines(skipped_file_path)
        skip_lines = existing_success + existing_error + existing_skipped
        print(f"Resume mode enabled:")
        print(f"  Found {existing_success} successful samples in output file")
        print(f"  Found {existing_error} error samples in error file")
        print(f"  Found {existing_skipped} skipped samples in skipped file")
        print(f"  Skipping first {skip_lines} input samples")
        print("-" * 50)

        if skip_lines >= total_lines:
            print(f"All {total_lines} samples already processed. Nothing to do.")
            return

    # test all server addresses
    valid_server_addresses = []
    for server_address in args.server_address:
        dummy_data = dict(
            conversations=[{"role": "user", "content": "Hello, how are you?"}]
        )
        result = call_sglang(
            args,
            server_address,
            dummy_data,
            max_tokens=1,
        )
        if result is not None and result.get("status") == "success":
            valid_server_addresses.append(server_address)
        else:
            print(f"Server {server_address} is not available")

    if len(valid_server_addresses) == 0:
        raise ValueError("No server address is available")
    print(
        f"Using {len(valid_server_addresses)} server addresses: {valid_server_addresses}"
    )
    print("-" * 50)

    # Determine file open mode based on resume flag
    file_mode = "a" if (args.resume and skip_lines > 0) else "w"
    print(
        f"Regenerating dataset and saving the output to {args.output_file_path} and error log to {error_file_path}"
    )
    print(
        f"File open mode: {file_mode} ({'append' if file_mode == 'a' else 'overwrite'})"
    )
    print("-" * 50)
    context_token_sum = 0
    context_token_min = None
    context_token_max = 0
    success_samples = 0
    error_samples = 0
    skipped_samples = 0
    submitted_samples = 0
    aligned_samples = 0
    terse_rewritten_samples = 0
    terse_removed: Counter[str] = Counter()
    short_answer_samples = 0
    short_answer_configs_hit: Counter[str] = Counter()

    # Create progress bar
    with (
        open(args.input_file_path, "r") as input_file,
        open(args.output_file_path, file_mode) as output_file_handle,
        open(error_file_path, file_mode) as error_file_handle,
        open(skipped_file_path, file_mode, encoding="utf-8") as skipped_file_handle,
    ):
        executor = ThreadPoolExecutor(
            max_workers=args.concurrency * len(valid_server_addresses)
        )
        waiting_queue = {
            server_address: [] for server_address in valid_server_addresses
        }
        pbar = tqdm(total=total_lines, desc="Processing", initial=skip_lines)
        start_server_index = 0

        if skip_lines > 0:
            print(f"Skipping {skip_lines} already processed samples...")
            for _ in range(skip_lines):
                next(input_file, None)
            print(f"Resuming from sample {skip_lines + 1}")

        for line in input_file:
            if args.num_samples is not None and submitted_samples >= args.num_samples:
                break

            data = json.loads(line.strip())
            unusable = sanitize_regen_row(data)
            if unusable is None and args.align_prompts:
                # The terse rule runs FIRST. A text-math row can carry a terse
                # preamble ("First conduct reasoning ... provide the corret
                # option letter at the end."), and the math rule would append
                # the boxed suffix to it; the terse rule then bails on the
                # suffix it finds and leaves the contradicting preamble in
                # place. Rewriting first strips the preamble and appends the
                # suffix itself, after which the math rule no-ops on it.
                if not args.no_terse_rewrite:
                    rewritten = rewrite_terse_answer_prompt(
                        data, args.math_cot_suffix
                    )
                    terse_rewritten_samples += rewritten
                    if rewritten:
                        for instruction in data["prompt_rewrite"]["removed"]:
                            terse_removed[instruction] += 1
                aligned_samples += apply_math_cot_prompt(
                    data, args.math_cot_suffix, args.math_family, args.math_source
                )
                if args.short_answer_config:
                    appended = append_short_answer_prompt(
                        data, args.math_cot_suffix, set(args.short_answer_config)
                    )
                    short_answer_samples += appended
                    if appended:
                        short_answer_configs_hit[
                            data["prompt_rewrite"]["config"]
                        ] += 1
            invalid_reason = unusable or validate_regen_input(data)
            if invalid_reason is not None:
                skipped_file_handle.write(
                    json.dumps(set_skipped(data, invalid_reason), ensure_ascii=False)
                    + "\n"
                )
                skipped_samples += 1
                pbar.update(1)
                continue

            # find server address with the least waiting requests
            server_address = valid_server_addresses[start_server_index]
            start_server_index = (start_server_index + 1) % len(valid_server_addresses)

            # submit prompt to sglang
            while len(waiting_queue[server_address]) >= args.concurrency:
                finished_on_request = False
                # check if any future is done, if so, write the result to the output file
                for req_future in waiting_queue[server_address]:
                    if req_future.done():
                        regen_data = req_future.result()

                        if regen_data["status"] == "error":
                            error_file_handle.write(
                                json.dumps(regen_data, ensure_ascii=False) + "\n"
                            )
                            error_samples += 1
                        elif regen_data["status"] == "skipped":
                            skipped_file_handle.write(
                                json.dumps(regen_data, ensure_ascii=False) + "\n"
                            )
                            skipped_samples += 1
                        else:
                            ctx_len = compute_context_length(
                                regen_data.get("conversations", [])
                            )
                            context_token_sum += ctx_len
                            if context_token_min is None:
                                context_token_min = ctx_len
                            else:
                                context_token_min = min(context_token_min, ctx_len)
                            context_token_max = max(context_token_max, ctx_len)

                            output_file_handle.write(
                                json.dumps(regen_data, ensure_ascii=False) + "\n"
                            )
                            success_samples += 1
                        waiting_queue[server_address].remove(req_future)
                        finished_on_request = True

                if finished_on_request:
                    break

            req_future = executor.submit(
                call_sglang,
                args,
                server_address,
                data,
            )
            waiting_queue[server_address].append(req_future)
            submitted_samples += 1
            pbar.update(1)

        # deal with all the remaining requests
        for server_address, waiting_queue_items in waiting_queue.items():
            for req_future in waiting_queue_items:
                regen_data = req_future.result()
                if regen_data["status"] == "error":
                    error_file_handle.write(
                        json.dumps(regen_data, ensure_ascii=False) + "\n"
                    )
                    error_samples += 1
                elif regen_data["status"] == "skipped":
                    skipped_file_handle.write(
                        json.dumps(regen_data, ensure_ascii=False) + "\n"
                    )
                    skipped_samples += 1
                else:
                    ctx_len = compute_context_length(
                        regen_data.get("conversations", [])
                    )
                    context_token_sum += ctx_len
                    if context_token_min is None:
                        context_token_min = ctx_len
                    else:
                        context_token_min = min(context_token_min, ctx_len)
                    context_token_max = max(context_token_max, ctx_len)

                    output_file_handle.write(
                        json.dumps(regen_data, ensure_ascii=False) + "\n"
                    )
                    success_samples += 1

    print(f"\nProcessing completed!")
    if success_samples > 0:
        avg_len = context_token_sum / success_samples
        print("Context length statistics (token count over conversations):")
        print(f"Number of successful examples: {success_samples}")
        print(f"Shortest context length: {context_token_min}")
        print(f"Longest context length: {context_token_max}")
        print(f"Average context length: {avg_len:.2f}")
    else:
        print("No successful examples to compute context length statistics.")

    total_processed = success_samples + error_samples + skipped_samples
    if skip_lines > 0:
        print(f"\nResume processing completed!")
        print(f"  Previously processed: {skip_lines}")
        print(
            f"  Newly processed: {total_processed} "
            f"({success_samples} success, {error_samples} failed, "
            f"{skipped_samples} skipped)"
        )
        print(f"  Total: {skip_lines + total_processed}")
    else:
        print(
            f"\nProcessing completed! {success_samples} samples regenerated, "
            f"{error_samples} samples failed, {skipped_samples} samples skipped."
        )
    if args.align_prompts:
        print(f"  Prompts rewritten as math: {aligned_samples}")
        if not args.no_terse_rewrite:
            print(
                "  Terse answer instructions replaced: "
                f"{terse_rewritten_samples}"
            )
            for instruction, count in terse_removed.most_common():
                print(f"    {count:>8,}  {instruction!r}")
        if args.short_answer_config:
            print(
                "  Short-answer prompts appended: "
                f"{short_answer_samples}"
            )
            for config, count in short_answer_configs_hit.most_common():
                print(f"    {count:>8,}  {config}")

    if args.num_samples is not None:
        # A partial run's file is not the training set, so rewriting it in
        # place would be surprising.
        print("\nSkipping the filter pass: --num-samples wrote a partial file.")
    elif args.no_filter:
        print("\nSkipping the filter pass (--no-filter).")
    else:
        print("\nFiltering unusable rows ...")
        kept, reasons = filter_regenerated(args.output_file_path)
        dropped = sum(reasons.values())
        print(f"  kept {kept:,}, dropped {dropped:,}")
        for reason, count in reasons.most_common():
            print(f"    {count:>8,}  {reason}")
    summarise_regenerated(args.output_file_path)


if __name__ == "__main__":
    main()
