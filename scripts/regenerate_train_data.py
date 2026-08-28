"""
This script will re-generate the dataset from target model,
which better aligns the draft model with the target model’s output distribution.

Usage:
1. Set up one or more SGLang servers for the target model.

python3 -m sglang.launch_server \
	--model Qwen/Qwen3.5-4B \
	--mem-fraction-static 0.7 \
	--tp 1 \
	--trust-remote-code \
    --cuda-graph-max-bs 128 \
	--host 0.0.0.0 \
	--port 30000 \
	--dtype bfloat16 \
    --reasoning-parser qwen3


2. Regenerate the dataset using the `regenerate_train_data.py` script.
python scripts/regenerate_train_data.py \
    --model Qwen/Qwen3.5-4B \
    --concurrency 128 \
    --max-tokens 4096 \
    --server-address localhost:30000 localhost:30010 localhost:30020 localhost:30030 localhost:30040 localhost:30050 localhost:30060 localhost:30070 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --input-file-path /local_home1/fengsicheng/specforge/data/sharegpt4v_train.jsonl \
    --output-file-path /local_home1/fengsicheng/specforge/regen_data/sharegpt4v_regen_first_turn.jsonl \
    --resume \
    --reasoning disable

Multimodal rows (produced by prepare_data_mm.py) carry a top-level `image`
path and an `<image>` placeholder inside a user turn's text. Multimodal
regeneration is restricted to Qwen3.5 chat/instruct checkpoints (see
SUPPORTED_MM_MODELS below); "-Base" checkpoints are not chat-aligned and are
rejected.
"""

import argparse
import base64
import json
import mimetypes
import os
import random
from concurrent.futures import ThreadPoolExecutor
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
    from scripts.prepare_data_mm import LLAVA_OV_FAMILIES, _base_config
except ModuleNotFoundError:
    from prepare_data_mm import LLAVA_OV_FAMILIES, _base_config

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
    prompt_group.add_argument(
        "--align-prompts",
        action="store_true",
        help=(
            "Rewrite prompts to the wording the benchmarks use before asking "
            "the target model, instead of regenerating against the prompt the "
            "JSONL already carries. Today the only rule is math: a row gets "
            "--math-cot-suffix appended when its LLaVA-OneVision family is a "
            "--math-family, or its `source` handle is a --math-source. Off by "
            "default for every blend, so the default run reproduces the "
            "dataset's own prompts."
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
                aligned_samples += apply_math_cot_prompt(
                    data, args.math_cot_suffix, args.math_family, args.math_source
                )
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


if __name__ == "__main__":
    main()
