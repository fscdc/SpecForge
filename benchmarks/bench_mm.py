#!/usr/bin/env python3
"""
Multimodal benchmarks driven through the OpenAI chat-completions API.

This is `bench_mmflash.py` rewritten onto the transport `bench_text.py` uses,
so the two report comparable numbers. What changed:

* **Transport.** One non-streaming `POST /v1/chat/completions` per question
  instead of the SGL frontend's `/generate`. `messages` are built here as an
  OpenAI content array, and the chat template is applied server-side -- which
  is why `--chat-template-name` is gone: the server's own template (or its
  `--chat-template` launch flag) decides the prompt now.
* **Reasoning.** `--reasoning off/on/<level>` travels in `chat_template_kwargs`.
  It replaces `--disable-thinking`, which had to prefill `<think></think>` into
  the prompt because the SGL path never ran the chat template. The two produce
  the same prompt bytes for the Qwen3 family, so runs stay comparable.
* **Accept length.** The headline number is the mean over REQUESTS of the
  server's own `spec_accept_length`, which is what DFlash reports. The
  token-weighted `sum(completion_tokens) / sum(spec_verify_ct)` that
  `benchmarker/utils.py::compute_metrics` uses is printed beside it.
  NOTE: the per-category breakdown in `metrics.categorical_performance` is
  still computed by each benchmark's own `compute_categorical_performance`,
  which calls `compute_metrics`, so the accept length inside it stays
  token-weighted. Compare categories against each other, not against the
  headline number.
* **Warmup.** `concurrency` requests capped at 64 new tokens run first, from
  the questions just past the evaluated slice, and are excluded from the
  timing. The prefix cache is flushed afterwards unless `--no-flush-cache`.
* **Images.** Sent as base64 `data:` URIs in `image_url` content parts, which
  is what `sgl.image()` did client-side. They are encoded ONCE before the
  warmup, so no encoding lands inside the measured latency.
* **Multi-image.** The SGL frontend refused more than one image per request
  (`RuntimeEndpoint._add_images` asserts it), which is why MMMU dropped its
  multi-image questions. A content array has no such limit, so the full
  benchmark now runs; pass `--single-image-only` to restore the old filter and
  compare against older MMMU results.
* **No server launching.** `--config-list` and the `ServerArgs` group are gone;
  point `--base-url` at servers that are already up. `--block-size` is recorded
  in the results file as a label only.

Unchanged from bench_mmflash.py: the sampling-parameter flags and their
defaults, the dataset sampling (`<benchmark>:<n>` takes the FIRST n questions),
every benchmark's own prompt/scoring/reporting, and both entropy reports.
"""

from __future__ import annotations

import argparse
import base64
import collections
import inspect
import json
import mimetypes
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from benchmarker.utils import (
    BenchmarkMetrics,
    format_length_summary,
    format_throughput_summary,
    length_summary,
    load_results,
    load_verify_entropy_dump,
    per_sample_entropy_stats,
    results_path,
    save_results,
    throughput_summary,
)
from mm_benchmarker import MM_BENCHMARKS
from tqdm import tqdm

# Sampling flags forwarded verbatim when given. Same set and names as
# bench_mmflash.py; SGLang accepts all of them on /v1/chat/completions
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
# generation goes last because it is the only unbounded one. Anything else the
# benchmark exposes per index is appended after these.
REPORT_FIELD_ORDER = (
    "image_path",
    "question",
    "category",
    "l2_category",
    "label",
    "prediction",
    "parsed",
    "score",
    "finish_reason",
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
            "round-robin and each gets `--concurrency` requests in flight."
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
        default=["chartqa:200"],
        help=(
            "Benchmarks to run, as <name>:<num-prompts>:<subset>,<subset>. The "
            "first <num-prompts> questions in dataset order are evaluated. "
            f"Available: {', '.join(sorted(MM_BENCHMARKS.benchmarks))}. Text-only "
            "benchmarks live in bench_text.py."
        ),
    )
    benchmark_group.add_argument(
        "--concurrency", type=int, default=1, help="Requests in flight per server."
    )
    benchmark_group.add_argument(
        "--warmup",
        type=int,
        default=-1,
        help=(
            "Warmup requests to send before timing, capped at 64 new tokens and "
            "excluded from every metric. -1 (the default) means "
            "concurrency * number of servers. 0 disables it."
        ),
    )
    benchmark_group.add_argument(
        "--no-flush-cache",
        action="store_true",
        default=False,
        help=(
            "Keep the prefix cache the warmup filled. By default it is flushed "
            "between the warmup and the timed run."
        ),
    )
    benchmark_group.add_argument(
        "--reasoning",
        type=str,
        default=None,
        help=(
            "off/on, or a model-supported reasoning level. Sent as "
            "chat_template_kwargs, so the server's chat template decides what it "
            "means. Replaces bench_mmflash.py's --disable-thinking."
        ),
    )
    benchmark_group.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help="Alias for --reasoning off.",
    )
    benchmark_group.add_argument(
        "--single-image-only",
        action="store_true",
        default=False,
        help=(
            "Drop the questions that carry more than one image (MMMU only). The "
            "SGL frontend could not send them, so bench_mmflash.py always did "
            "this; the chat API can, so the default is now to run them. Pass "
            "this to compare against results produced before the switch."
        ),
    )
    benchmark_group.add_argument("--output-dir", type=str, default="./results")
    benchmark_group.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Name of this run, added to the output file names. Reusing it "
            "resumes that run's results file: the benchmarks already recorded "
            "in it are skipped."
        ),
    )
    benchmark_group.add_argument(
        "--save-generations",
        action="store_true",
        default=False,
        help=(
            "Write what the model generated for every question next to the "
            "results, as <benchmark>_generations.jsonl."
        ),
    )
    benchmark_group.add_argument(
        "--accept-length-report",
        type=int,
        default=5,
        help=(
            "How many of the worst and best questions to print in full after "
            "each benchmark, ranked by their own accept length. 0 prints only "
            "the distribution summary."
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
            "server."
        ),
    )
    benchmark_group.add_argument(
        "--verify-entropy-dump",
        type=str,
        default=None,
        help=(
            "Path of the DFlash verify-entropy dump: the target's entropy at "
            "every position it verified, split by accepted / rejected / "
            "discarded. Needs the server started with "
            "SGLANG_DFLASH_ENTROPY_DUMP=<path> and "
            "patches/sglang/<ver>/dflash-verify-entropy.patch applied."
        ),
    )
    benchmark_group.add_argument(
        "--token-entropy",
        type=int,
        default=0,
        metavar="TOP_K",
        help=(
            "Measure the entropy of the target model's next-token distribution "
            "by asking for this many top logprobs per generated token (20 is a "
            "good default, 0 disables). REQUIRES a target-only server: DFlash "
            "rejects return_logprob. Costs response bandwidth, so do not "
            "compare throughput against a run without it."
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
    sampling_group.add_argument("--repetition-penalty", type=float, default=None)
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum new tokens per question. Defaults to what the benchmark "
            "asks for (512 for chartqa, 2048 for mmstar, 16384 for mathvision)."
        ),
    )
    return parser.parse_args()


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------


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
    reasoning: Optional[str],
    top_logprobs: Optional[int],
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
    if top_logprobs:
        # serving_chat.py maps these onto return_logprob / top_logprobs_num, so
        # meta_info comes back with the output_top_logprobs that
        # benchmarker/utils.py::token_entropy_series expects
        body["logprobs"] = True
        body["top_logprobs"] = int(top_logprobs)
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
    not at the top level. The top level is tried as a fallback for the versions
    and backends that put it there.
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


class MetaInfoState:
    """
    The bit of an SGL state that `benchmarker/utils.py` and the benchmarks'
    `compute_categorical_performance` actually read.

    Wrapping the chat responses in this keeps their scoring and their entropy
    maths reusable without a copy of either.
    """

    __slots__ = ("_meta", "_text")

    def __init__(self, meta: Dict[str, Any], text: str):
        self._meta = meta
        self._text = text

    def get_meta_info(self, key: str = "answer") -> Dict[str, Any]:
        return self._meta

    def __getitem__(self, key: str) -> str:
        return self._text

    def error(self):
        return None


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def image_paths_of(question: Any) -> List[str]:
    """Every image one question references, in order, duplicates included."""
    if not isinstance(question, dict):
        return []
    if "parts" in question:
        return [value for kind, value in question["parts"] if kind == "image"]
    path = question.get("image_path")
    return [path] if path else []


def encode_image(path: str) -> str:
    """
    One image as a `data:` URI.

    This is what `sgl.image()` did client-side; doing it here keeps the bytes
    off the server's disk-read path and, because every image is encoded before
    the timer starts, keeps the encoding out of the measured latency.
    """
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def encode_images(questions: Sequence[Any]) -> Dict[str, str]:
    """Pre-encode every distinct image the questions reference."""
    paths = sorted({path for q in questions for path in image_paths_of(q)})
    if not paths:
        return {}
    cache: Dict[str, str] = {}
    total = 0
    for path in tqdm(paths, desc="Encoding images"):
        cache[path] = encode_image(path)
        total += len(cache[path])
    print(f"Encoded {len(paths)} images, {total / 1e6:.1f} MB of base64 held in memory")
    return cache


def question_content(question: Any, images: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    One question as an OpenAI content array.

    The interleaved benchmarks (MMMU) carry their own ordered parts so that
    every image lands where its placeholder was; the rest put the single image
    first and the text after it, which is the order `sgl.image(path) + question`
    produced.
    """

    def image_part(path: str) -> Dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": images[path]}}

    if isinstance(question, dict) and "parts" in question:
        return [
            image_part(value) if kind == "image" else {"type": "text", "text": value}
            for kind, value in question["parts"]
        ]
    if isinstance(question, str):
        return [{"type": "text", "text": question}]
    if not isinstance(question, dict):
        raise TypeError(f"Cannot build a prompt out of a {type(question).__name__}")

    content: List[Dict[str, Any]] = []
    if question.get("image_path"):
        content.append(image_part(question["image_path"]))
    text = question.get("question")
    if text is None:
        raise KeyError(f"No question field in {sorted(question)}")
    content.append({"type": "text", "text": text})
    return content


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def run_sample(
    index: int,
    content: List[Dict[str, Any]],
    base_url: str,
    *,
    model: str,
    max_new_tokens: int,
    sampling_params: Dict[str, Any],
    reasoning: Optional[str],
    top_logprobs: Optional[int],
    timeout_s: int,
) -> Dict[str, Any]:
    """Run one question and return its generation, counters and raw meta_info."""
    body = build_body(
        [{"role": "user", "content": content}],
        model=model,
        max_new_tokens=max_new_tokens,
        sampling_params=sampling_params,
        reasoning=reasoning,
        top_logprobs=top_logprobs,
    )
    payload = send_chat(base_url, body, timeout_s)

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    meta = response_meta_info(payload)

    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    spec_verify_ct = int(meta.get("spec_verify_ct", 0) or 0)
    is_speculative = "spec_verify_ct" in meta

    accept_length = None
    if "spec_accept_length" in meta:
        try:
            accept_length = float(meta["spec_accept_length"])
        except (TypeError, ValueError):
            accept_length = None

    return {
        "index": index,
        "rid": meta.get("id") or payload.get("id"),
        # DFlash's definition: what the server itself reported for this request
        "accept_length": accept_length,
        # the token-weighted definition compute_metrics uses, for the cross-check
        "accept_length_token_weighted": (
            completion_tokens / spec_verify_ct
            if is_speculative and spec_verify_ct
            else (1.0 if is_speculative else None)
        ),
        "completion_tokens": completion_tokens,
        "spec_verify_ct": spec_verify_ct if is_speculative else None,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "total_tokens": (
            int(usage.get("prompt_tokens", 0) or 0) + completion_tokens
        ),
        "cached_tokens": int(meta.get("cached_tokens", 0) or 0),
        # server-side timing; present only with the request-timing-split patch,
        # except e2e_latency which stock SGLang already returns
        "e2e_latency": meta.get("e2e_latency"),
        "first_token_latency": meta.get("first_token_latency"),
        "decode_latency": meta.get("decode_latency"),
        "finish_reason": choice.get("finish_reason"),
        "generation": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or "",
        "meta_info": meta,
    }


def run_requests(
    contents: Sequence[List[Dict[str, Any]]],
    base_urls: Sequence[str],
    concurrency: int,
    desc: str,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Run every question, `concurrency` in flight per server, results in order.

    Question i goes to server i % len(base_urls), the way
    `mm_benchmarker.base.ShardedSGLFunction` sharded them. A request that fails
    aborts the run: metrics over only the requests that succeeded would be
    meaningless.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(contents)
    failures: List[Tuple[int, BaseException]] = []
    workers = max(concurrency, 1) * len(base_urls)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_sample, index, content, base_urls[index % len(base_urls)], **kwargs
            ): index
            for index, content in enumerate(contents)
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
            f"{len(failures)} of {len(contents)} requests failed, so the metrics "
            f"of this run would be meaningless. First failure (question "
            f"{index}): {type(error).__name__}: {error}"
        )
    return [result for result in results if result is not None]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def truncate_for_report(value: Any, max_chars: int) -> str:
    """Render one report field, bounded so a long generation stays readable."""
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, default=str)
    )
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated, {len(text)} chars total]"


def sample_context(benchmarker: Any, index: int) -> Dict[str, Any]:
    """
    Everything the benchmark knows about one question, keyed for the report.

    The question dict is whatever the benchmark built (for the image
    benchmarks: image_path + question), and the rest are the per-index lists a
    benchmarker fills in while scoring.
    """
    context: Dict[str, Any] = {}
    questions = getattr(benchmarker, "questions", None) or []
    if index < len(questions):
        question = questions[index]
        if isinstance(question, dict):
            context.update(question)
        else:
            context["question"] = question
    for attribute, key in (
        ("labels", "label"),
        ("predictions", "prediction"),
        ("parsed_answers", "parsed"),
        ("hits", "score"),
        ("categories", "category"),
        ("l2_categories", "l2_category"),
        ("generations", "generation"),
    ):
        values = getattr(benchmarker, attribute, None)
        if isinstance(values, list) and index < len(values):
            context[key] = values[index]
    return context


def print_sample_report(
    rank: int, stat: Dict[str, Any], benchmarker: Any, max_chars: int
) -> None:
    """Print one question in full: its decoding counters, prompt and answer."""
    print(
        f"  [{rank}] index={stat['index']}  "
        f"accept_length={stat['accept_length']:.3f}  "
        f"completion_tokens={stat['completion_tokens']}  "
        f"spec_verify_ct={stat['spec_verify_ct']}  "
        f"prompt_tokens={stat['prompt_tokens']}  "
        f"cached_tokens={stat['cached_tokens']}"
    )
    context = sample_context(benchmarker, stat["index"])
    context.setdefault("finish_reason", stat["finish_reason"])
    # the known fields first, in a readable order, then whatever else the
    # benchmark happened to put in its question dict
    ordered = [key for key in REPORT_FIELD_ORDER if key in context]
    ordered += [key for key in context if key not in REPORT_FIELD_ORDER]
    for key in ordered:
        text = truncate_for_report(context[key], max_chars)
        if "\n" in text:
            print(f"      {key}:")
            for line in text.splitlines():
                print(f"        {line}")
        else:
            print(f"      {key}: {text}")
    print()


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


def report_accept_length(
    stats: List[Dict[str, Any]],
    benchmarker: Any,
    benchmark_name: str,
    top_k: int,
    max_chars: int,
) -> Optional[Dict[str, Any]]:
    """
    Break the run-level accept length down per question and show its extremes.

    The headline number is DFlash's: the mean over requests of the server's own
    `spec_accept_length`. The token-weighted mean is printed beside it because
    the two diverge as soon as generation lengths differ, and it is what every
    older SpecForge report quotes.
    """
    if not stats:
        return None

    print(f"\n{f' {benchmark_name}: accept length per question ':=^78}")
    measured = [stat for stat in stats if stat["accept_length"] is not None]
    if not measured:
        print(
            "the responses carry no spec_accept_length: either this run decoded "
            "without speculation (block size 0), or the server returned no "
            "meta_info (it needs return_meta_info on /v1/chat/completions)."
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
        f"per-question mean:         {values.mean():.3f}   "
        f"(what this run reports, DFlash's definition)"
    )
    print(
        f"token-weighted mean:       {token_weighted:.3f}   "
        f"(what compute_metrics would report)"
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
            print_sample_report(rank, stat, benchmarker, max_chars)
        print(f"{f' {count} highest accept length ':-^78}")
        for rank, stat in enumerate(reversed(ordered[-count:]), start=1):
            print_sample_report(rank, stat, benchmarker, max_chars)
    print("=" * 78)

    return {
        "num_questions": len(stats),
        "num_measured": len(measured),
        "mean": float(values.mean()),
        "token_weighted_mean": token_weighted,
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "verify_count": verify_count,
        **percentiles,
    }


def report_token_entropy(
    entropy_stats: List[Dict[str, Any]],
    entropy_series: List[List[float]],
    benchmark_name: str,
) -> Optional[Dict[str, Any]]:
    """
    Summarize the target model's per-token entropy over one benchmark.

    Entropy is in nats and is the top-k estimate described in
    ``benchmarker.utils.token_entropy_series``: 0 means the target was certain
    of its next token, so a draft only has to be right about something the
    target itself considers obvious.
    """
    measured = [s for s in entropy_stats if s.get("n_tokens")]
    if not measured:
        return None

    pooled = np.concatenate([np.asarray(s) for s in entropy_series if len(s)])
    print(f"\n{f' {benchmark_name}: target-model token entropy (nats) ':=^78}")
    print(f"questions measured:        {len(measured)} / {len(entropy_stats)}")
    print(f"tokens measured:           {len(pooled)}")
    q = np.percentile(pooled, [10, 25, 50, 75, 90, 99])
    print(f"token-pooled mean:         {pooled.mean():.4f}   std {pooled.std():.4f}")
    print(
        "percentiles:               "
        + "   ".join(f"p{p} {v:.4f}" for p, v in zip((10, 25, 50, 75, 90, 99), q))
    )
    print(
        f"near-deterministic (<0.1): {100 * (pooled < 0.1).mean():5.1f}% of tokens     "
        f"uncertain (>1.0): {100 * (pooled > 1.0).mean():5.1f}%"
    )
    print(
        f"mean top-1 probability:    "
        f"{np.mean([s['top1_prob_mean'] for s in measured]):.4f}     "
        f"mean tail mass outside top-k: "
        f"{np.mean([s['tail_mass_mean'] for s in measured]):.5f}"
    )
    print_histogram(pooled, label="distribution over all generated tokens")

    # does the target get less certain further into a generation?
    deciles = [s["entropy_by_decile"] for s in measured if s.get("entropy_by_decile")]
    profile = np.asarray(deciles).mean(axis=0) if deciles else None
    if profile is not None:
        print("\nmean entropy by decile of the generation (start -> end):")
        print("  " + "  ".join(f"{v:.3f}" for v in profile))

    print("=" * 78)
    return {
        "unit": "nats",
        "num_questions": len(measured),
        "num_tokens": int(len(pooled)),
        "mean": float(pooled.mean()),
        "std": float(pooled.std()),
        **{f"p{p}": float(v) for p, v in zip((10, 25, 50, 75, 90, 99), q)},
        "frac_below_0.1": float((pooled < 0.1).mean()),
        "frac_above_1.0": float((pooled > 1.0).mean()),
        "top1_prob_mean": float(np.mean([s["top1_prob_mean"] for s in measured])),
        "tail_mass_mean": float(np.mean([s["tail_mass_mean"] for s in measured])),
        "entropy_by_decile": profile.tolist() if profile is not None else None,
    }


def report_verify_entropy(
    stats: List[Dict[str, Any]], benchmark_name: str, dump_path: str
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Report the target's entropy at DFlash verify time, split by what happened.

    This is the measurement a target-only run cannot give. The entropy at an
    accepted position is the same number plain decoding would produce -- spec
    decoding preserves the distribution -- so what is new here is the
    alignment: how uncertain the target was at the exact position where it
    rejected the draft. A high rejection-point entropy means the draft is
    losing on tokens the target itself was unsure about (little headroom); a
    low one means the draft is missing tokens the target considered obvious
    (headroom to train).
    """
    per_rid = load_verify_entropy_dump(dump_path)
    if not per_rid:
        print(
            f"\n[verify-entropy] nothing to read at {dump_path!r} (or its .pid* "
            "siblings); was the server started with SGLANG_DFLASH_ENTROPY_DUMP "
            "and the dflash-verify-entropy patch applied?"
        )
        return None, []

    print(f"\n{f' {benchmark_name}: verify-time target entropy (nats) ':=^78}")
    pools: Dict[str, List[float]] = {
        name: [] for name in ("accepted", "rejection", "discarded")
    }
    by_position = collections.defaultdict(list)
    per_sample: List[Dict[str, Any]] = []
    matched = 0
    for stat in stats:
        bucket = per_rid.get(str(stat.get("rid")))
        if bucket is None:
            continue
        matched += 1
        record = {
            "index": stat["index"],
            "rid": stat["rid"],
            "accept_length": stat.get("accept_length"),
            "verify_steps": bucket["steps"],
            "mean_accept_len": float(np.mean(bucket["accept_lens"])),
        }
        for name, values in pools.items():
            values.extend(bucket[name])
            record[f"{name}_entropy_mean"] = (
                float(np.mean(bucket[name])) if bucket[name] else None
            )
        for position, values in bucket["by_block_position"].items():
            by_position[position].extend(values)
        per_sample.append(record)

    print(f"questions joined by rid:   {matched} / {len(stats)}")
    print(f"verify steps recorded:     {sum(b['steps'] for b in per_rid.values())}")
    print(f"{'position class':<14}{'tokens':>10}{'mean H':>10}{'median':>10}{'p90':>10}")
    summary: Dict[str, Any] = {"unit": "nats", "num_questions": matched}
    for name, values in pools.items():
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        print(
            f"{name:<14}{len(array):>10}{array.mean():>10.4f}"
            f"{np.median(array):>10.4f}{np.percentile(array, 90):>10.4f}"
        )
        summary[name] = {
            "num_tokens": int(len(array)),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
        }

    if pools["accepted"] and pools["rejection"]:
        gap = np.mean(pools["rejection"]) - np.mean(pools["accepted"])
        summary["rejection_minus_accepted"] = float(gap)
        print(
            f"\nrejection - accepted:      {gap:+.4f} nats  "
            + (
                "(rejections land on genuinely uncertain tokens)"
                if gap > 0.2
                else "(rejections happen on tokens the target was sure about "
                "-> the draft, not the task, is the bottleneck)"
            )
        )

    if by_position:
        positions = sorted(by_position)
        profile = [float(np.mean(by_position[p])) for p in positions]
        summary["accepted_entropy_by_block_position"] = profile
        print("\nmean entropy of accepted positions, by index within the block:")
        print("  pos " + " ".join(f"{p:>6}" for p in positions[:16]))
        print("  H   " + " ".join(f"{v:6.3f}" for v in profile[:16]))
    print("=" * 78)
    return summary, per_sample


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
        print(f"Spec verify ct:   {int(accept_summary['verify_count'])}")
    if metrics.accuracy is not None:
        print(
            f"Accuracy:         {metrics.accuracy:.4f}   "
            f"({metrics.num_valid_predictions}/{metrics.num_questions} parsed)"
        )
    print(f"{'=' * 50}")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def parse_benchmark_list(
    items: Sequence[str],
) -> List[Tuple[str, Optional[int], Optional[List[str]]]]:
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
        if name not in MM_BENCHMARKS.benchmarks:
            raise KeyError(
                f"Unknown multimodal benchmark {name!r}. Available: "
                f"{', '.join(sorted(MM_BENCHMARKS.benchmarks))}. Text-only "
                "benchmarks are run by bench_text.py."
            )
        parsed.append((name, int(num_prompts) if num_prompts else None, subset))
    return parsed


def instantiate(
    name: str,
    num_samples: Optional[int],
    subset: Optional[List[str]],
    single_image_only: bool,
):
    """
    Build one benchmarker, passing only the arguments its signature declares.

    `single_image_only` is MMMU's; the chat API can send several images per
    request, so it defaults to False here and only MMMU is asked about it.
    """
    cls = MM_BENCHMARKS.get(name)
    kwargs: Dict[str, Any] = {"num_samples": num_samples}
    parameters = inspect.signature(cls.__init__).parameters
    if subset is not None:
        kwargs["subset"] = subset
    if "single_image_only" in parameters:
        kwargs["single_image_only"] = single_image_only
    return cls(**kwargs)


def main() -> None:
    args = parse_args()

    if args.disable_thinking:
        if args.reasoning not in (None, "off"):
            raise ValueError(
                f"--disable-thinking conflicts with --reasoning {args.reasoning}"
            )
        args.reasoning = "off"

    if args.token_entropy and args.temperature > 1e-5:
        print(
            f"[warning] --temperature {args.temperature} scales the logprobs the "
            "server returns, so the entropy will be of the sampling "
            "distribution, not the model's. Use --temperature 0.0 for the raw one.",
            flush=True,
        )

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
        # SGLang disables top_k with -1; sending 0 is rejected
        if name == "top_k" and value <= 0:
            continue
        sampling_params[name] = value

    metadata: Dict[str, Any] = {
        "model": args.model,
        "base_urls": base_urls,
        "concurrency": concurrency,
        "block_size": args.block_size,
        "sampling_params": sampling_params,
        "max_tokens": args.max_tokens,
        "reasoning": args.reasoning,
        "warmup": warmup_count,
        "single_image_only": args.single_image_only,
        "transport": "openai-chat-completions",
    }

    # one file per --name, written again after every benchmark. A rerun keeps
    # the benchmarks it already holds and only measures the missing ones, so a
    # suite that dies halfway (or a benchmark added to --benchmark-list later)
    # costs only what has not been run yet.
    result_file = results_path(args.output_dir, args.name)
    results, done = load_results(result_file, metadata)
    if done:
        print(
            f"Resuming {result_file}: {len(done)} benchmark(s) already recorded "
            f"({', '.join(sorted(done))})"
        )

    for benchmark_name, num_prompts, subset in benchmark_list:
        if benchmark_name in done:
            recorded = results[benchmark_name]
            recorded_questions = (
                recorded[-1].get("num_questions")
                if isinstance(recorded, list) and isinstance(recorded[-1], dict)
                else None
            )
            print(
                f"Skipping {benchmark_name}: already in {result_file}"
                + (
                    f" ({recorded_questions} questions)"
                    if recorded_questions is not None
                    else ""
                )
                + ". Delete its key in that file to measure it again."
            )
            continue

        print(
            f"Running benchmark {benchmark_name} with {num_prompts} prompts, "
            f"concurrency {concurrency} x {len(base_urls)} server(s), "
            f"reasoning {args.reasoning}, subset {subset}"
        )

        # load the evaluated slice plus the warmup questions that follow it, so
        # that the evaluated questions stay the FIRST num_prompts of the dataset
        load_count = None if num_prompts is None else num_prompts + warmup_count
        benchmarker = instantiate(
            benchmark_name, load_count, subset, args.single_image_only
        )
        questions, labels = benchmarker.load_data()
        if not questions:
            print("No valid questions found. Please check the dataset format.")
            continue

        eval_count = (
            len(questions) if num_prompts is None else min(num_prompts, len(questions))
        )
        # kept so that the per-sample report can name the question behind an
        # accept length, as MMBenchmarker.run() does
        benchmarker.questions = questions[:eval_count]

        images = encode_images(questions)
        contents = [question_content(q, images) for q in questions]
        eval_contents = contents[:eval_count]
        warmup_contents = contents[eval_count : eval_count + warmup_count]
        if warmup_count and len(warmup_contents) < warmup_count:
            # too few questions to spare any: wrap around
            warmup_contents += [
                contents[i % len(contents)]
                for i in range(warmup_count - len(warmup_contents))
            ]

        max_new_tokens = args.max_tokens or benchmarker.get_max_new_tokens()
        request_kwargs = dict(
            model=args.model,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params,
            reasoning=args.reasoning,
            top_logprobs=args.token_entropy or None,
            timeout_s=args.timeout_s,
        )

        if warmup_contents:
            print(f"[warmup] {len(warmup_contents)} requests ...")
            run_requests(
                warmup_contents,
                base_urls,
                concurrency,
                "Warmup",
                **{**request_kwargs, "max_new_tokens": min(64, max_new_tokens)},
            )
            if not args.no_flush_cache:
                flush_cache(base_urls)

        print(
            f"Running benchmark: {len(eval_contents)} prompts, "
            f"max_new_tokens={max_new_tokens} ..."
        )
        start = time.perf_counter()
        stats = run_requests(
            eval_contents, base_urls, concurrency, "Benchmarking", **request_kwargs
        )
        latency = time.perf_counter() - start

        # score with the benchmark's own extractor, on the visible answer: with
        # a reasoning parser installed the thinking text is in reasoning_content
        # and is not what the answer should be read out of
        eval_labels = list(labels[:eval_count]) if labels else labels
        generations = [stat["generation"] for stat in stats]
        predictions = [
            benchmarker.extract_answer(
                generation,
                (
                    eval_labels[index]
                    if eval_labels and index < len(eval_labels)
                    else None
                ),
            )
            if isinstance(generation, str)
            else generation
            for index, generation in enumerate(generations)
        ]
        # what MMBenchmarker._summarize() publishes, so that the benchmark's own
        # reporting and dump_generations() keep working
        benchmarker.generations = generations
        benchmarker.predictions = predictions
        benchmarker.labels = eval_labels

        accuracy = None
        if eval_labels:
            accuracy = benchmarker.compute_accuracy(predictions, eval_labels)
            valid_count = sum(1 for p in predictions if p is not None)
            if accuracy is not None and valid_count < len(predictions):
                print(
                    f"Warning: {len(predictions) - valid_count} predictions could "
                    "not be extracted."
                )

        # the SGL-state interface the shared scoring helpers read
        states = [
            MetaInfoState(stat["meta_info"], stat["generation"]) for stat in stats
        ]
        benchmarker.per_sample_stats = [
            {key: value for key, value in stat.items() if key != "meta_info"}
            for stat in stats
        ]
        entropy_stats, entropy_series = per_sample_entropy_stats(states)
        benchmarker.per_sample_entropy = entropy_stats
        benchmarker.token_entropy_series = entropy_series

        num_output_tokens = sum(stat["completion_tokens"] for stat in stats)
        accept_summary = report_accept_length(
            stats,
            benchmarker,
            benchmark_name,
            args.accept_length_report,
            args.report_max_chars,
        )
        entropy_summary = report_token_entropy(
            entropy_stats, entropy_series, benchmark_name
        )
        verify_entropy_summary, per_sample_verify_entropy = (None, [])
        if args.verify_entropy_dump:
            verify_entropy_summary, per_sample_verify_entropy = report_verify_entropy(
                stats, benchmark_name, args.verify_entropy_dump
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
        # the benchmark's own breakdown, whose accept length stays token-weighted
        # because it goes through compute_metrics
        metrics.categorical_performance = benchmarker.compute_categorical_performance(
            states, latency, "answer"
        )
        length_stats = length_summary(stats)
        throughput_stats = throughput_summary(stats, latency)
        print_summary(
            benchmark_name,
            metrics,
            len(eval_contents),
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
                "_generations.jsonl",
            )
            written = getattr(benchmarker, "dump_generations", lambda path: 0)(
                dump_path
            )
            if written:
                print(f"Saved {written} generations to {dump_path}")

        results.setdefault(benchmark_name, []).append(
            dict(
                num_samples=num_prompts,
                num_questions=len(stats),
                max_tokens=max_new_tokens,
                # what the benchmark has to say about the questions it actually
                # ran, e.g. the multi-image ones MMMU filtered
                dataset_info=getattr(benchmarker, "describe_run", lambda: None)(),
                metrics=[asdict(metrics)],
                accept_length_summary=accept_summary,
                length_summary=length_stats,
                throughput_summary=throughput_stats,
                per_sample_stats=benchmarker.per_sample_stats,
                # target-model next-token entropy, None unless --token-entropy
                token_entropy_summary=entropy_summary,
                per_sample_entropy=entropy_stats,
                # verify-time entropy, None unless --verify-entropy-dump
                verify_entropy_summary=verify_entropy_summary,
                per_sample_verify_entropy=per_sample_verify_entropy,
            )
        )

        # written here rather than after the loop, so that a suite killed in a
        # later benchmark keeps everything measured up to this point
        save_results(result_file, results)
        done.add(benchmark_name)
        print(f"Saved {benchmark_name} to {result_file}")

    if os.path.exists(result_file):
        print(f"Results saved to {result_file}")
    else:
        print("No benchmark produced results, nothing was written.")


if __name__ == "__main__":
    main()
