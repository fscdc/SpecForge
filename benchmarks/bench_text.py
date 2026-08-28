#!/usr/bin/env python3
"""
Text benchmarks driven through the OpenAI chat-completions API, DFlash-style.

Why this exists
---------------
Its predecessor (bench_mmflash.py, now bench_mm.py) drove the text benchmarks
through the SGL frontend, which posts a raw prompt string to SGLang's native
`/generate`. For gsm8k that string was the 5-shot completion scaffold
`benchmarker/gsm8k.py` builds, so the model never saw a chat template and never
entered its reasoning mode -- a different workload from the one DFlash reports
on, whose accept length is not comparable.

This script keeps the `benchmarker` package's datasets, questions and scoring
but swaps the transport and the metric definition for DFlash's
(`dflash/benchmark.py::_run_openai`). `bench_mm.py` is its multimodal twin and
uses the same transport, so the two report comparable numbers:

* one non-streaming `POST /v1/chat/completions` per turn. `messages` are built
  here, the chat template is applied server-side, and the reasoning switch
  travels in `chat_template_kwargs` -- so `--reasoning` works the way it does
  for DFlash instead of being prefilled into the prompt.
* the prompt is the raw question plus the benchmark's instruction, 0-shot. No
  few-shot examples, no `Question:`/`Answer:` scaffold.
* `accept length` is the mean over REQUESTS of the server's own
  `spec_accept_length`, which is what DFlash reports. The token-weighted
  `sum(completion_tokens) / sum(spec_verify_ct)` that
  `benchmarker/utils.py::compute_metrics` uses is printed next to it as a
  cross-check; the DFlash number is the one in `metrics.accept_length`.
* a warmup of `concurrency` requests capped at 64 new tokens runs first, from
  the questions just past the evaluated slice, and is excluded from the timing.
* `--max-tokens` actually takes effect. On the SGL path it was silently
  overridden by the `max_tokens=512` baked into `create_few_shot_sgl_function`
  (`sglang/lang/interpreter.py::_resolve_sampling_params` lets the `gen()`
  value win over the `run_batch()` default).

Unchanged from the SGL-frontend original: the sampling-parameter flags and their
defaults, and the dataset sampling -- `<benchmark>:<n>` still takes the FIRST n
questions in dataset order, not a shuffled sample.

The server is expected to be running already; this script never launches one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from benchmarker import BENCHMARKS
from benchmarker.utils import (
    BenchmarkMetrics,
    format_length_summary,
    format_throughput_summary,
    length_summary,
    throughput_summary,
)
from tqdm import tqdm

# The instruction appended to the raw question, per benchmark. These are the
# `user_prefix` the SGL functions already used, plus DFlash's own wording for
# the ones whose prompt lived in the few-shot scaffold instead
# (dflash/benchmark.py::DATASETS).
COT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
PROMPT_SUFFIX: Dict[str, str] = {
    "gsm8k": COT_SUFFIX,
    "math500": COT_SUFFIX,
    "aime": COT_SUFFIX,
}

# Benchmarks whose load_data() wraps the raw question in a completion-style
# scaffold that a chat prompt must not carry. gsm8k builds
# "Question: <raw>\nAnswer:" in get_one_example(); everything else in
# `benchmarker` stores the question verbatim.
QUESTION_WRAPPER: Dict[str, Tuple[str, str]] = {
    "gsm8k": ("Question: ", "\nAnswer:"),
}

# System prompt the benchmark's own SGL function installed, which building
# `messages` here would otherwise drop.
SYSTEM_PROMPT: Dict[str, str] = {}
try:  # keep the import failure of one benchmark from breaking the others
    from benchmarker.mtbench import SYSTEM_PROMPT as _MTBENCH_SYSTEM_PROMPT

    SYSTEM_PROMPT["mtbench"] = _MTBENCH_SYSTEM_PROMPT
except ImportError:  # pragma: no cover
    pass

# Stop strings the benchmark's answer format relies on, carried over from the
# `stop=` its SGL function passed to gen().
STOP_STRINGS: Dict[str, List[str]] = {
    "mbpp": ["[DONE]"],
}

# Sampling flags that are forwarded verbatim when given. Same set and same
# names as bench_mm.py; SGLang accepts all of them on /v1/chat/completions
# (srt/entrypoints/openai/protocol.py::ChatCompletionRequest).
OPTIONAL_SAMPLING_PARAMS = (
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
)

# Per-sample context printed in the accept-length report, in this order. The
# generation goes last because it is the only unbounded one.
REPORT_FIELD_ORDER = (
    "question",
    "prompt",
    "label",
    "prediction",
    "finish_reason",
    "reasoning",
    "generation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    server_group = parser.add_argument_group("server")
    server_group.add_argument(
        "--base-url",
        type=str,
        nargs="+",
        default=["http://127.0.0.1:30000"],
        help=(
            "Base URL of the running server, or several for a data-parallel "
            "deployment (one per GPU). Questions are spread over them "
            "round-robin and each gets `--concurrency` requests in flight, so "
            "N servers see N * concurrency at once."
        ),
    )
    server_group.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to put in the request body, as the server reports it.",
    )
    server_group.add_argument("--timeout-s", type=int, default=3600)

    benchmark_group = parser.add_argument_group("benchmark")
    benchmark_group.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        default=["gsm8k:200"],
        help=(
            "Benchmarks to run, as <name>:<num-prompts>:<subset>,<subset>. The "
            "first <num-prompts> questions in dataset order are evaluated. "
            f"Available: {', '.join(sorted(BENCHMARKS.benchmarks))}"
        ),
    )
    benchmark_group.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Requests in flight per server.",
    )
    benchmark_group.add_argument(
        "--warmup",
        type=int,
        default=-1,
        help=(
            "Warmup requests to send before timing, capped at 64 new tokens and "
            "excluded from every metric. -1 (the default) means "
            "concurrency * number of servers, which is what DFlash does. 0 "
            "disables it."
        ),
    )
    benchmark_group.add_argument(
        "--no-flush-cache",
        action="store_true",
        default=False,
        help=(
            "Keep the prefix cache the warmup filled. By default the cache is "
            "flushed between the warmup and the timed run so that the first "
            "questions do not decode against a warmer cache than the rest; "
            "DFlash does not flush, so pass this to match it exactly."
        ),
    )
    benchmark_group.add_argument(
        "--prompt-suffix",
        type=str,
        default=None,
        help=(
            "Override the instruction appended to every question (see "
            "PROMPT_SUFFIX). Pass an empty string to append nothing."
        ),
    )
    benchmark_group.add_argument(
        "--reasoning",
        type=str,
        default=None,
        help=(
            "off/on, or a model-supported reasoning level (low/medium/high/"
            "xhigh). Sent as chat_template_kwargs, so the server's chat "
            "template decides what it means. This replaces the old "
            "--disable-thinking, which prefilled <think></think> into the "
            "prompt because the SGL path never ran the chat template."
        ),
    )
    benchmark_group.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Alias for --reasoning off.",
    )
    benchmark_group.add_argument("--output-dir", type=str, default="./results")
    benchmark_group.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of this run, added to the output file names.",
    )
    benchmark_group.add_argument(
        "--save-generations",
        action="store_true",
        default=False,
        help=(
            "Write what the model generated for every question next to the "
            "results, as <benchmark>_generations_<timestamp>.jsonl, together "
            "with its prompt and per-sample decoding counters."
        ),
    )
    benchmark_group.add_argument(
        "--accept-length-report",
        type=int,
        default=5,
        help=(
            "How many of the worst and best questions to print in full after "
            "each benchmark, ranked by their own accept length. The "
            "distribution summary is printed regardless; 0 prints only that."
        ),
    )
    benchmark_group.add_argument(
        "--report-max-chars",
        type=int,
        default=800,
        help="Truncate each field of that report to this many characters, 0 disables.",
    )
    benchmark_group.add_argument(
        "--block-size",
        type=int,
        default=None,
        help=(
            "The speculative block size the server was launched with. Recorded "
            "in the results file only -- this script does not configure the "
            "server -- so that runs can be told apart afterwards."
        ),
    )

    sampling_group = parser.add_argument_group("sampling")
    sampling_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature, 0 (the default) is greedy decoding.",
    )
    sampling_group.add_argument("--top-p", type=float, default=None)
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="0 or a negative value disables it and is not sent.",
    )
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument("--presence-penalty", type=float, default=None)
    sampling_group.add_argument("--frequency-penalty", type=float, default=None)
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Divide the logits of tokens already seen, in (0, 2], 1.0 disables it.",
    )
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum new tokens per turn. Defaults to what the benchmark asks "
            "for (2048 for most, 1024 for humaneval/mbpp, 32768 for aime)."
        ),
    )
    return parser.parse_args()


def reasoning_kwargs(reasoning: Optional[str]) -> Dict[str, Any]:
    """
    The chat_template_kwargs that carry a reasoning switch.

    Verbatim from dflash/benchmark.py::_reasoning_kwargs with no template to
    inspect: the client cannot see which key this model's template reads, so
    every spelling is sent and the template picks the one it knows.
    """
    if reasoning is None:
        return {}
    if reasoning in {"on", "off"}:
        return {"enable_thinking": reasoning == "on"}
    return {
        "enable_thinking": True,
        "reasoning_effort": reasoning,
        "reasoning_strength": reasoning,
    }


def build_body(
    messages: List[Dict[str, Any]],
    *,
    model: str,
    max_new_tokens: int,
    sampling_params: Dict[str, Any],
    stop: Optional[List[str]],
    reasoning: Optional[str],
) -> Dict[str, Any]:
    """The request body, shaped like dflash/benchmark.py::send_openai."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": reasoning_kwargs(reasoning),
        # SGLang extension: without it the response carries no meta_info and
        # there is no accept length to read
        "return_meta_info": True,
    }
    body.update(sampling_params)
    if stop:
        body["stop"] = stop
    return body


def send_chat(base_url: str, body: Dict[str, Any], timeout_s: int) -> Dict[str, Any]:
    response = requests.post(
        base_url.rstrip("/") + "/v1/chat/completions", json=body, timeout=timeout_s
    )
    response.raise_for_status()
    return response.json()


def response_meta_info(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    The server's meta_info for the first choice.

    SGLang attaches it per choice (`ChatCompletionResponseChoice.meta_info`),
    not at the top level, so DFlash's own `output.get("meta_info")` reads
    nothing on a current server. The top level is still tried as a fallback for
    the versions and backends that put it there.
    """
    choices = payload.get("choices") or []
    if choices:
        meta = choices[0].get("meta_info")
        if meta:
            return meta
    return payload.get("meta_info") or {}


def flush_cache(base_urls: Sequence[str]) -> None:
    for base_url in base_urls:
        try:
            requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
        except requests.RequestException as error:
            print(f"[warning] could not flush the cache of {base_url}: {error}")


def question_turns(question: Any) -> List[str]:
    """
    The user turns of one question, in order.

    Single-turn benchmarks store the text under "question"; the multi-turn ones
    (mt-bench) under "question_1", "question_2", ...
    """
    if isinstance(question, str):
        return [question]
    if not isinstance(question, dict):
        raise TypeError(f"Cannot read a prompt out of a {type(question).__name__}")
    if "question" in question:
        return [question["question"]]
    numbered = sorted(
        (
            (int(match.group(1)), key)
            for key, match in (
                (key, re.fullmatch(r"question_(\d+)", key)) for key in question
            )
            if match is not None
        )
    )
    if not numbered:
        raise KeyError(f"No question field in {sorted(question)}")
    return [question[key] for _, key in numbered]


def chat_prompt(benchmark_name: str, text: str, suffix_override: Optional[str]) -> str:
    """The raw question with its instruction, without any completion scaffold."""
    prefix, suffix = QUESTION_WRAPPER.get(benchmark_name, ("", ""))
    if prefix and text.startswith(prefix):
        text = text[len(prefix) :]
    if suffix and text.endswith(suffix):
        text = text[: -len(suffix)]
    instruction = (
        suffix_override
        if suffix_override is not None
        else PROMPT_SUFFIX.get(benchmark_name, "")
    )
    return text + instruction


def build_prompts(
    benchmark_name: str, questions: Sequence[Any], suffix_override: Optional[str]
) -> List[List[str]]:
    """One list of user turns per question, ready to be put into `messages`."""
    return [
        [chat_prompt(benchmark_name, turn, suffix_override) for turn in question_turns(q)]
        for q in questions
    ]


def run_sample(
    index: int,
    turns: Sequence[str],
    base_url: str,
    *,
    model: str,
    system_prompt: Optional[str],
    max_new_tokens: int,
    sampling_params: Dict[str, Any],
    stop: Optional[List[str]],
    reasoning: Optional[str],
    timeout_s: int,
) -> Dict[str, Any]:
    """
    Run one question to completion and return its generation and counters.

    A multi-turn question is one request per turn, with the assistant's answer
    fed back in, so the counters are summed the way
    `benchmarker/utils.py::per_sample_spec_stats` sums them: generations add up,
    the prompt side is maxed because turn 2 re-sends turn 1.
    """
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    contents: List[str] = []
    reasonings: List[str] = []
    accept_lengths: List[float] = []
    completion_tokens = 0
    spec_verify_ct = 0
    prompt_tokens = 0
    cached_tokens = 0
    finish_reason = None
    rid = None
    is_speculative = False
    timings: Dict[str, float] = {}

    for turn in turns:
        messages.append({"role": "user", "content": turn})
        body = build_body(
            messages,
            model=model,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            stop=stop,
            reasoning=reasoning,
        )
        payload = send_chat(base_url, body, timeout_s)

        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning_text = message.get("reasoning_content") or ""
        contents.append(content)
        reasonings.append(reasoning_text)
        # only the visible answer is fed back, which is what the chat templates
        # of the reasoning models do with a previous turn anyway
        messages.append({"role": "assistant", "content": content})

        usage = payload.get("usage") or {}
        completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens", 0) or 0))

        meta = response_meta_info(payload)
        rid = rid or meta.get("id") or payload.get("id")
        cached_tokens = max(cached_tokens, int(meta.get("cached_tokens", 0) or 0))
        # server-side phase timings, summed over the turns so they stay
        # comparable with completion_tokens, which is also summed
        for key in ("e2e_latency", "first_token_latency", "decode_latency"):
            value = meta.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timings[key] = timings.get(key, 0.0) + float(value)
        if "spec_verify_ct" in meta:
            is_speculative = True
            spec_verify_ct += int(meta.get("spec_verify_ct", 0) or 0)
        if "spec_accept_length" in meta:
            try:
                accept_lengths.append(float(meta["spec_accept_length"]))
            except (TypeError, ValueError):
                pass
        finish_reason = choice.get("finish_reason") or finish_reason

    return {
        "index": index,
        "rid": rid,
        # DFlash's definition: what the server itself reported for this request,
        # averaged over the turns of a multi-turn question
        "accept_length": statistics.mean(accept_lengths) if accept_lengths else None,
        # the token-weighted definition benchmarker/utils.py::compute_metrics
        # uses, kept for the cross-check line in the report
        "accept_length_token_weighted": (
            completion_tokens / spec_verify_ct
            if is_speculative and spec_verify_ct
            else (1.0 if is_speculative else None)
        ),
        "completion_tokens": completion_tokens,
        "spec_verify_ct": spec_verify_ct if is_speculative else None,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
        # absent unless the server carries the request-timing-split patch
        "e2e_latency": timings.get("e2e_latency"),
        "first_token_latency": timings.get("first_token_latency"),
        "decode_latency": timings.get("decode_latency"),
        "finish_reason": finish_reason,
        "generation": contents[-1] if len(contents) == 1 else contents,
        "reasoning": reasonings[-1] if len(reasonings) == 1 else reasonings,
    }


def run_requests(
    prompts: Sequence[Sequence[str]],
    base_urls: Sequence[str],
    concurrency: int,
    desc: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Run every question, `concurrency` in flight per server, results in order.

    Question i goes to server i % len(base_urls), so the load is spread the way
    `mm_benchmarker.base.ShardedSGLFunction` spreads it. A request that fails
    aborts the run: metrics computed over only the requests that succeeded
    would be meaningless.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(prompts)
    failures: List[Tuple[int, BaseException]] = []
    workers = max(concurrency, 1) * len(base_urls)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_sample, index, turns, base_urls[index % len(base_urls)], **kwargs
            ): index
            for index, turns in enumerate(prompts)
        }
        for future in tqdm(futures, total=len(futures), desc=desc):
            index = futures[future]
            try:
                results[index] = future.result()
            except BaseException as error:  # noqa: BLE001 - reported below
                failures.append((index, error))

    if failures:
        index, error = failures[0]
        raise RuntimeError(
            f"{len(failures)} of {len(prompts)} requests failed, so the metrics "
            f"of this run would be meaningless. First failure (question "
            f"{index}): {type(error).__name__}: {error}"
        )
    return [result for result in results if result is not None]


def truncate_for_report(value: Any, max_chars: int) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated, {len(text)} chars total]"


def print_histogram(
    values: np.ndarray,
    label: str = "distribution of the per-question accept length",
    bins: int = 10,
    width: int = 40,
) -> None:
    counts, edges = np.histogram(values, bins=bins)
    peak = max(int(counts.max()), 1)
    print(f"\n{label}:")
    for count, low, high in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * int(count) / peak))
        print(f"  {low:6.3f} - {high:6.3f} | {bar:<{width}} {int(count)}")


def print_sample_report(
    rank: int, stat: Dict[str, Any], context: Dict[str, Any], max_chars: int
) -> None:
    print(
        f"  [{rank}] index={stat['index']}  "
        f"accept_length={stat['accept_length']:.3f}  "
        f"completion_tokens={stat['completion_tokens']}  "
        f"spec_verify_ct={stat['spec_verify_ct']}  "
        f"prompt_tokens={stat['prompt_tokens']}  "
        f"cached_tokens={stat['cached_tokens']}"
    )
    ordered = [key for key in REPORT_FIELD_ORDER if context.get(key) not in (None, "")]
    ordered += [
        key
        for key in context
        if key not in REPORT_FIELD_ORDER and context.get(key) not in (None, "")
    ]
    for key in ordered:
        text = truncate_for_report(context[key], max_chars)
        if "\n" in text:
            print(f"      {key}:")
            for line in text.splitlines():
                print(f"        {line}")
        else:
            print(f"      {key}: {text}")
    print()


def report_accept_length(
    stats: List[Dict[str, Any]],
    contexts: List[Dict[str, Any]],
    benchmark_name: str,
    top_k: int,
    max_chars: int,
) -> Optional[Dict[str, Any]]:
    """
    Break the run-level accept length down per question and show its extremes.

    The headline number is DFlash's: the mean over requests of the server's own
    `spec_accept_length`. The token-weighted mean is printed next to it because
    the two diverge as soon as generation lengths differ, and every other
    SpecForge report quotes the token-weighted one.
    """
    if not stats:
        return None

    print(f"\n{f' {benchmark_name}: accept length per question ':=^78}")
    measured = [stat for stat in stats if stat["accept_length"] is not None]
    if not measured:
        print(
            "the responses carry no spec_accept_length: either this run decoded "
            "without speculation, or the server did not return meta_info (it "
            "needs return_meta_info support on /v1/chat/completions)."
        )
        print("=" * 78)
        return None

    values = np.array([stat["accept_length"] for stat in measured], dtype=float)
    completion_tokens = sum(stat["completion_tokens"] for stat in measured)
    verify_count = sum(stat["spec_verify_ct"] or 0 for stat in measured)
    token_weighted = completion_tokens / verify_count if verify_count else float("nan")
    percentiles = {
        f"p{q}": float(np.percentile(values, q)) for q in (10, 25, 50, 75, 90)
    }

    print(f"questions measured:        {len(measured)} / {len(stats)}")
    print(
        f"per-question mean:         {values.mean():.3f}   (what this run reports, "
        f"DFlash's definition)"
    )
    print(
        f"token-weighted mean:       {token_weighted:.3f}   (what compute_metrics "
        f"in benchmarker/utils.py would report)"
    )
    print(
        f"spread:                    std {values.std(ddof=0):.3f}   "
        f"min {values.min():.3f}   max {values.max():.3f}"
    )
    print(
        "percentiles:               "
        + "   ".join(f"{name} {value:.3f}" for name, value in percentiles.items())
    )
    print_histogram(values)

    if top_k > 0:
        ordered = sorted(measured, key=lambda stat: stat["accept_length"])
        count = min(top_k, len(ordered))
        if 2 * count > len(ordered):
            print(
                f"\nnote: only {len(ordered)} measured questions, so the two lists "
                "below overlap"
            )
        print(f"\n{f' {count} lowest accept length ':-^78}")
        for rank, stat in enumerate(ordered[:count], start=1):
            print_sample_report(rank, stat, contexts[stat["index"]], max_chars)
        print(f"{f' {count} highest accept length ':-^78}")
        for rank, stat in enumerate(reversed(ordered[-count:]), start=1):
            print_sample_report(rank, stat, contexts[stat["index"]], max_chars)
    print("=" * 78)

    return {
        "num_questions": len(stats),
        "num_measured": len(measured),
        "mean": float(values.mean()),
        "token_weighted_mean": token_weighted,
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        **percentiles,
    }


def print_summary(
    benchmark_name: str,
    metrics: BenchmarkMetrics,
    num_prompts: int,
    num_output_tokens: int,
    concurrency: int,
    num_servers: int,
    accept_summary: Optional[Dict[str, Any]],
    length_stats: Optional[Dict[str, Any]] = None,
    throughput_stats: Optional[Dict[str, Any]] = None,
) -> None:
    print(f"\n{'=' * 50}")
    print(f"Benchmark:        {benchmark_name}")
    print(f"Num prompts:      {num_prompts}")
    print(f"Concurrency:      {concurrency} per server x {num_servers} server(s)")
    print(f"Latency:          {metrics.latency:.1f}s")
    print(f"Output tokens:    {num_output_tokens}")
    throughput_lines = format_throughput_summary(throughput_stats)
    if throughput_lines:
        for line in throughput_lines:
            print(line)
    else:
        print(f"Throughput:       {metrics.output_throughput:,.2f} tok/s")
    for line in format_length_summary(length_stats):
        print(line)
    if accept_summary is not None:
        print(f"Accept length:    {accept_summary['mean']:.3f}")
        print(f"Spec verify ct:   {int(accept_summary.get('verify_count', 0))}")
    if metrics.accuracy is not None:
        print(
            f"Accuracy:         {metrics.accuracy:.4f}   "
            f"({metrics.num_valid_predictions}/{metrics.num_questions} parsed)"
        )
    print(f"{'=' * 50}")


def dump_generations(
    path: str,
    stats: List[Dict[str, Any]],
    contexts: List[Dict[str, Any]],
) -> int:
    with open(path, "w") as handle:
        for stat in stats:
            record = dict(contexts[stat["index"]])
            record.update(
                {
                    key: stat[key]
                    for key in (
                        "index",
                        "rid",
                        "accept_length",
                        "accept_length_token_weighted",
                        "completion_tokens",
                        "spec_verify_ct",
                        "prompt_tokens",
                        "total_tokens",
                        "cached_tokens",
                        "e2e_latency",
                        "first_token_latency",
                        "decode_latency",
                        "finish_reason",
                    )
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return len(stats)


def parse_benchmark_list(items: Sequence[str]) -> List[Tuple[str, Optional[int], Optional[List[str]]]]:
    parsed = []
    for item in items:
        splits = item.split(":")
        if len(splits) == 1:
            name, num_prompts, subset = splits[0], None, None
        elif len(splits) == 2:
            (name, num_prompts), subset = splits, None
        elif len(splits) == 3:
            name, num_prompts, subset = splits[0], splits[1], splits[2].split(",")
        else:
            raise ValueError(f"Invalid benchmark list format: {item}")
        if name not in BENCHMARKS.benchmarks:
            raise KeyError(
                f"Unknown benchmark {name!r}. Available: "
                f"{', '.join(sorted(BENCHMARKS.benchmarks))}"
            )
        parsed.append((name, int(num_prompts) if num_prompts else None, subset))
    return parsed


def instantiate(name: str, num_samples: Optional[int], subset: Optional[List[str]]):
    cls = BENCHMARKS.get(name)
    if subset is None:
        return cls(num_samples=num_samples)
    return cls(num_samples=num_samples, subset=subset)


def main() -> None:
    args = parse_args()

    if args.disable_thinking:
        if args.reasoning not in (None, "off"):
            raise ValueError(
                f"--disable-thinking conflicts with --reasoning {args.reasoning}"
            )
        args.reasoning = "off"

    benchmark_list = parse_benchmark_list(args.benchmark_list)
    base_urls = [url.rstrip("/") for url in args.base_url]
    concurrency = max(args.concurrency, 1)
    warmup_count = (
        concurrency * len(base_urls) if args.warmup < 0 else max(args.warmup, 0)
    )

    # every sampling argument that was actually given, greedy decoding otherwise
    sampling_params: Dict[str, Any] = {"temperature": args.temperature}
    for name in OPTIONAL_SAMPLING_PARAMS:
        value = getattr(args, name)
        if value is None:
            continue
        # SGLang disables top_k with -1; sending 0 is rejected, and DFlash only
        # sends it when it is positive
        if name == "top_k" and value <= 0:
            continue
        sampling_params[name] = value

    results: Dict[str, Any] = {
        "model": args.model,
        "base_urls": base_urls,
        "concurrency": concurrency,
        "block_size": args.block_size,
        "sampling_params": sampling_params,
        "max_tokens": args.max_tokens,
        "reasoning": args.reasoning,
        "warmup": warmup_count,
        "transport": "openai-chat-completions",
    }

    for benchmark_name, num_prompts, subset in benchmark_list:
        print(
            f"Running benchmark {benchmark_name} with {num_prompts} prompts, "
            f"concurrency {concurrency} x {len(base_urls)} server(s), "
            f"reasoning {args.reasoning}, subset {subset}"
        )

        # load the evaluated slice plus the warmup questions that follow it, so
        # that the evaluated questions stay the FIRST num_prompts of the dataset
        load_count = None if num_prompts is None else num_prompts + warmup_count
        benchmarker = instantiate(benchmark_name, load_count, subset)
        questions, labels = benchmarker.load_data()
        if not questions:
            print("No valid questions found. Please check the dataset format.")
            continue

        eval_count = len(questions) if num_prompts is None else min(num_prompts, len(questions))
        prompts = build_prompts(benchmark_name, questions, args.prompt_suffix)
        eval_prompts = prompts[:eval_count]
        # the questions just past the evaluated slice, wrapping around when the
        # dataset is too small to spare any
        warmup_prompts = prompts[eval_count : eval_count + warmup_count]
        if warmup_count and len(warmup_prompts) < warmup_count:
            warmup_prompts += [
                prompts[i % len(prompts)]
                for i in range(warmup_count - len(warmup_prompts))
            ]

        max_new_tokens = args.max_tokens or benchmarker.get_max_new_tokens()
        request_kwargs = dict(
            model=args.model,
            system_prompt=SYSTEM_PROMPT.get(benchmark_name),
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            stop=STOP_STRINGS.get(benchmark_name),
            reasoning=args.reasoning,
            timeout_s=args.timeout_s,
        )

        if warmup_prompts:
            print(f"[warmup] {len(warmup_prompts)} requests ...")
            run_requests(
                warmup_prompts,
                base_urls,
                concurrency,
                "Warmup",
                **{**request_kwargs, "max_new_tokens": min(64, max_new_tokens)},
            )
            if not args.no_flush_cache:
                flush_cache(base_urls)

        print(
            f"Running benchmark: {len(eval_prompts)} prompts, "
            f"max_new_tokens={max_new_tokens} ..."
        )
        start = time.perf_counter()
        stats = run_requests(
            eval_prompts, base_urls, concurrency, "Benchmarking", **request_kwargs
        )
        latency = time.perf_counter() - start

        # score with the benchmark's own extractor, on the visible answer: with
        # a reasoning parser installed the thinking text is in reasoning_content
        # and is not what the answer should be read out of
        predictions = []
        for stat in stats:
            output = stat["generation"]
            if isinstance(output, list):
                output = output[-1]
            index = stat["index"]
            predictions.append(
                benchmarker.extract_answer(
                    output, labels[index] if labels and index < len(labels) else None
                )
                if isinstance(output, str)
                else output
            )
        eval_labels = list(labels[:eval_count]) if labels else labels

        accuracy = None
        if eval_labels:
            accuracy = benchmarker.compute_accuracy(predictions, eval_labels)
            valid_count = sum(1 for p in predictions if p is not None)
            if accuracy is not None and valid_count < len(predictions):
                print(
                    f"Warning: {len(predictions) - valid_count} predictions could "
                    "not be extracted."
                )

        contexts = [
            {
                "question": questions[stat["index"]],
                "prompt": (
                    eval_prompts[stat["index"]][0]
                    if len(eval_prompts[stat["index"]]) == 1
                    else eval_prompts[stat["index"]]
                ),
                "label": eval_labels[stat["index"]] if eval_labels else None,
                "prediction": predictions[position],
                "generation": stat["generation"],
                "reasoning": stat["reasoning"],
                "finish_reason": stat["finish_reason"],
            }
            for position, stat in enumerate(stats)
        ]
        num_output_tokens = sum(stat["completion_tokens"] for stat in stats)
        accept_summary = report_accept_length(
            stats,
            contexts,
            benchmark_name,
            args.accept_length_report,
            args.report_max_chars,
        )
        if accept_summary is not None:
            accept_summary["verify_count"] = sum(
                stat["spec_verify_ct"] or 0 for stat in stats
            )

        metrics = BenchmarkMetrics(
            latency=latency,
            output_throughput=num_output_tokens / latency if latency > 0 else 0.0,
            # DFlash's definition, 1.0 when the run decoded without speculation
            accept_length=accept_summary["mean"] if accept_summary else 1.0,
            accuracy=accuracy,
            num_questions=len(stats),
            num_valid_predictions=sum(1 for p in predictions if p is not None),
        )
        length_stats = length_summary(stats)
        throughput_stats = throughput_summary(stats, latency)
        print_summary(
            benchmark_name,
            metrics,
            len(eval_prompts),
            num_output_tokens,
            concurrency,
            len(base_urls),
            accept_summary,
            length_stats,
            throughput_stats,
        )

        if args.save_generations:
            os.makedirs(args.output_dir, exist_ok=True)
            dump_path = os.path.join(
                args.output_dir,
                f"{args.name + '_' if args.name else ''}{benchmark_name}"
                f"_generations_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
            )
            written = dump_generations(dump_path, stats, contexts)
            print(f"Saved {written} generations to {dump_path}")

        results.setdefault(benchmark_name, []).append(
            dict(
                num_samples=num_prompts,
                num_questions=len(stats),
                max_tokens=max_new_tokens,
                metrics=[asdict(metrics)],
                accept_length_summary=accept_summary,
                length_summary=length_stats,
                throughput_summary=throughput_stats,
                per_sample_stats=[
                    {
                        key: stat[key]
                        for key in (
                            "index",
                            "rid",
                            "accept_length",
                            "accept_length_token_weighted",
                            "completion_tokens",
                            "spec_verify_ct",
                            "prompt_tokens",
                            "total_tokens",
                            "cached_tokens",
                            "e2e_latency",
                            "first_token_latency",
                            "decode_latency",
                            "finish_reason",
                        )
                    }
                    for stat in stats
                ],
            )
        )

    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(
        args.output_dir,
        f"{args.name + '_' if args.name else ''}results_"
        f"{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    with open(result_file, "w") as handle:
        json.dump(results, handle, indent=4, default=str)
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
