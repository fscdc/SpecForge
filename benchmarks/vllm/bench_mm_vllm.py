#!/usr/bin/env python3
"""
bench_mm.py run against a vLLM server instead of an SGLang one.

Everything bench_mm.py does is kept: the same flags, questions, prompts, warmup,
thread-pool client, scoring and results file. Only the places where it talks to
SGLang-specific parts of the server are swapped, inside this process (bench_mm.py
itself is not modified, and the SGLang runs are unaffected):

* **Responses.** vLLM has no ``meta_info``. The server is started with
  ``--enable-per-request-metrics`` (and, when speculating,
  ``--per-request-spec-decode-metrics summary``), and each response's ``metrics``
  object fills the same per-request fields: ``spec_verify_ct`` is vLLM's
  ``num_spec_steps``, ``accept_length`` is ``completion_tokens / spec_verify_ct``
  (SGLang's definition, so the two frameworks compare; vLLM's own
  ``mean_acceptance_length``, which leaves out the token the prefill produced, is kept
  beside it), and ``decode_latency`` is ``generation_time_ms`` (first to last output
  token, server clock). ``e2e_latency`` is the client's round trip and
  ``first_token_latency`` the round trip minus the decode latency, so every request's
  decode window lies on the client timeline the way bench_mm.py lays them out.
* **Phase accounting.** Read from ``--phase-url``, served by the plugin in
  ``benchmarks/vllm/phase_plugin`` (the counterpart of the SGLang phase-accounting
  patch), instead of SGLang's ``/server_info``. The summary keeps SGLang's keys and
  adds vLLM's steps that mixed a prefill chunk with decodes (``mixed_*``) and
  ``busy_generation_throughput``, every generated token over all busy time.
* **Cache flush.** After the warmup, the prefix, multimodal-processor and encoder
  caches are reset through vLLM's development endpoints (server started with
  ``VLLM_SERVER_DEV_MODE=1``), instead of SGLang's ``/flush_cache``.
* **Results metadata.** The file also records ``framework`` (vLLM and its version).
* **Rejected requests.** vLLM logs only "Exception caught" for a request it rejects, so
  the reason is taken from the response and printed, and the request is sent again up
  to ``MAX_RETRIES`` times (timed from the attempt that succeeds, counted per question
  as ``retries``); one that keeps failing fails the run, as in bench_mm.py.

Run it from the repository root with the SpecForge environment's python (bench_mm's
helpers import sglang); the vLLM server runs from its own environment, see
scripts/benchmark_vllm_concurrency.sh.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import requests

# bench_mm.py and the packages it imports live one directory up
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench_mm  # noqa: E402

PHASE_URLS: List[str] = []
FRAMEWORK: Dict[str, Any] = {"name": "vllm"}
#: a rejected request is sent again this many times, with the server's reason printed;
#: the attempts are recorded per question as `retries`
MAX_RETRIES = 2


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
    """One question through vLLM, returned with bench_mm.run_sample's keys."""
    if top_logprobs:
        raise ValueError("--token-entropy is not supported against a vLLM server")
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_new_tokens,
        "chat_template_kwargs": bench_mm.reasoning_kwargs(reasoning),
    }
    body.update(sampling_params)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    retries = 0
    while True:
        # timed from the attempt that succeeds
        send_time = time.perf_counter()
        response = requests.post(url, json=body, timeout=timeout_s)
        if response.status_code < 400:
            break
        # vLLM's log keeps only "Exception caught" for a rejected request; the reason is
        # in the body. A rejection that repeats is real and fails the run as bench_mm would.
        detail = f"HTTP {response.status_code} for question {index}: {response.text[:600]}"
        if retries >= MAX_RETRIES:
            raise RuntimeError(detail)
        retries += 1
        print(f"[warning] {detail} -- retry {retries}/{MAX_RETRIES}", flush=True)
        time.sleep(1.0)
    payload = response.json()
    recv_time = time.perf_counter()

    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = payload.get("usage") or {}
    metrics = payload.get("metrics") or {}
    spec = metrics.get("speculative_decoding")

    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    cached_tokens = int(
        ((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
    )
    e2e_latency = recv_time - send_time
    generation_ms = metrics.get("generation_time_ms")
    decode_latency = generation_ms / 1000.0 if generation_ms is not None else None
    first_token_latency = (
        e2e_latency - decode_latency if decode_latency is not None else None
    )

    is_speculative = spec is not None
    spec_verify_ct = int(spec.get("num_spec_steps", 0) or 0) if is_speculative else 0
    accept_length = None
    if is_speculative:
        # a request that finished in its prefill never ran a verify step
        accept_length = completion_tokens / spec_verify_ct if spec_verify_ct else 1.0

    # the SGL-state fields the shared scoring helpers read (compute_metrics & co.)
    meta: Dict[str, Any] = {
        "id": payload.get("id"),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "e2e_latency": e2e_latency,
        "first_token_latency": first_token_latency,
        "decode_latency": decode_latency,
        "finish_reason": {"type": choice.get("finish_reason")},
        "vllm_metrics": metrics,
    }
    if is_speculative:
        meta["spec_verify_ct"] = spec_verify_ct
        meta["spec_accept_length"] = accept_length
        meta["spec_num_correct_drafts"] = spec.get("num_accepted_draft_tokens")

    return {
        "index": index,
        "rid": payload.get("id"),
        "accept_length": accept_length,
        "accept_length_token_weighted": accept_length,
        "completion_tokens": completion_tokens,
        "spec_verify_ct": spec_verify_ct if is_speculative else None,
        # vLLM's own figure: tokens per verify step, the prefill's token left out
        "vllm_mean_acceptance_length": (
            spec.get("mean_acceptance_length") if is_speculative else None
        ),
        "num_accepted_draft_tokens": (
            spec.get("num_accepted_draft_tokens") if is_speculative else None
        ),
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
        "e2e_latency": e2e_latency,
        "first_token_latency": first_token_latency,
        "decode_latency": decode_latency,
        "server_ttft": (
            metrics["time_to_first_token_ms"] / 1000.0
            if metrics.get("time_to_first_token_ms") is not None
            else None
        ),
        "server_queue_time": (
            metrics["queue_time_ms"] / 1000.0
            if metrics.get("queue_time_ms") is not None
            else None
        ),
        "send_time": send_time,
        "recv_time": recv_time,
        "retries": retries,
        "finish_reason": choice.get("finish_reason"),
        "generation": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
        "meta_info": meta,
    }


def server_phase_accounting(
    base_urls: Sequence[str],
) -> Optional[List[Dict[str, Any]]]:
    """Every server's lifetime phase counters from the plugin, or None."""
    if not PHASE_URLS:
        return None
    if len(PHASE_URLS) != len(base_urls):
        raise ValueError("give one --phase-url per --base-url, in the same order")
    snapshots = []
    for url in PHASE_URLS:
        try:
            response = requests.get(url.rstrip("/") + "/phase_accounting", timeout=60)
            response.raise_for_status()
            snapshots.append(response.json())
        except (requests.RequestException, ValueError) as error:
            print(f"[warning] no phase accounting from {url}: {error}")
            return None
    return snapshots


def flush_cache(base_urls: Sequence[str]) -> None:
    """Reset the prefix, multimodal-processor and encoder caches the warmup filled."""
    for base_url in base_urls:
        base = base_url.rstrip("/")
        for attempt in range(60):
            try:
                response = requests.post(base + "/reset_prefix_cache", timeout=60)
                response.raise_for_status()
                if response.json().get("success", False):
                    break
            except (requests.RequestException, ValueError) as error:
                print(f"[warning] could not reset the prefix cache of {base}: {error}")
                break
            # blocks still held by a request that is finishing: try again
            time.sleep(0.5)
        else:
            print(f"[warning] the prefix cache of {base} never reported a reset")
        for endpoint in ("/reset_mm_cache", "/reset_encoder_cache"):
            try:
                requests.post(base + endpoint, timeout=60).raise_for_status()
            except requests.RequestException as error:
                print(f"[warning] {endpoint} failed on {base}: {error}")


_load_results = bench_mm.load_results
_phase_accounting_summary = bench_mm.phase_accounting_summary


def load_results(path: str, metadata: Dict[str, Any]):
    return _load_results(path, {**metadata, "framework": FRAMEWORK})


def phase_accounting_summary(
    before: Optional[List[Dict[str, Any]]], after: Optional[List[Dict[str, Any]]]
) -> Optional[Dict[str, Any]]:
    """The SGLang summary plus what only vLLM has: steps that mixed a prefill with decodes.

    `busy_generation_throughput` is every generated token (first tokens and the decodes
    of mixed steps included) over all the time the server was busy, prefill included:
    the throughput that does not depend on how a step is classified.
    """
    summary = _phase_accounting_summary(before, after)
    if summary is None:
        return None
    for key in ("mixed_steps", "mixed_decode_reqs", "mixed_decode_tokens", "generated_tokens"):
        summary[key] = int(
            sum((new.get(key) or 0) - (old.get(key) or 0) for old, new in zip(before, after))
        )
    if summary.get("busy_time"):
        summary["busy_generation_throughput"] = summary["generated_tokens"] / summary["busy_time"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--phase-url",
        nargs="+",
        default=[],
        help=(
            "Where each server's phase-accounting plugin listens "
            "(http://127.0.0.1:$SPECFORGE_PHASE_PORT), one per --base-url."
        ),
    )
    known, rest = parser.parse_known_args()
    if "-h" in rest or "--help" in rest:
        print(__doc__)
        print("vLLM-only flag: --phase-url URL [URL ...]\n\nbench_mm.py flags:")
    PHASE_URLS.extend(known.phase_url)

    base_urls = []
    for i, arg in enumerate(rest):
        if arg == "--base-url":
            base_urls = [a for a in rest[i + 1 :] if not a.startswith("--")]
            break
    try:
        version = requests.get(
            (base_urls or ["http://127.0.0.1:30000"])[0].rstrip("/") + "/version",
            timeout=30,
        ).json()
        FRAMEWORK["version"] = version.get("version")
    except (requests.RequestException, ValueError):
        FRAMEWORK["version"] = None
    FRAMEWORK["server_args"] = os.environ.get("VLLM_SERVER_ARGS")

    bench_mm.run_sample = run_sample
    bench_mm.server_phase_accounting = server_phase_accounting
    bench_mm.flush_cache = flush_cache
    bench_mm.load_results = load_results
    bench_mm.phase_accounting_summary = phase_accounting_summary
    sys.argv = [sys.argv[0]] + rest
    bench_mm.main()


if __name__ == "__main__":
    main()
