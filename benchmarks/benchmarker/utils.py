"""
Utility functions for benchmark scripts.
"""

import collections
import glob
import json
import math
import os
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import sglang as sgl


@dataclass
class BenchmarkMetrics:
    """Container for benchmark performance metrics."""

    latency: float
    output_throughput: float
    accept_length: float
    accuracy: Optional[float] = None
    num_questions: int = 0
    num_valid_predictions: int = 0
    categorical_performance: Optional[Dict[str, "BenchmarkMetrics"]] = None


def compute_metrics(
    states: List[Any],
    latency: float,
    answer_key: str = "answer",
    additional_answer_keys: Optional[List[str]] = None,
) -> BenchmarkMetrics:
    """
    Compute performance metrics from SGLang states.

    Args:
        states: List of SGLang state objects from run_batch
        latency: Total latency in seconds
        answer_key: Primary key for answer in state meta info
        additional_answer_keys: Additional keys to include in token count (e.g., ["answer_1", "answer_2"])

    Returns:
        BenchmarkMetrics object with computed metrics
    """
    # Compute output tokens
    num_output_tokens = 0
    if additional_answer_keys:
        for key in [answer_key] + additional_answer_keys:
            num_output_tokens += sum(
                s.get_meta_info(key)["completion_tokens"] for s in states
            )
    else:
        num_output_tokens = sum(
            s.get_meta_info(answer_key)["completion_tokens"] for s in states
        )

    output_throughput = num_output_tokens / latency if latency > 0 else 0.0

    # Compute accept length (speculative decoding metric)
    has_verify = "spec_verify_ct" in states[0].get_meta_info(answer_key)
    if has_verify:
        num_verify_tokens = 0
        if additional_answer_keys:
            for key in [answer_key] + additional_answer_keys:
                num_verify_tokens += sum(
                    s.get_meta_info(key).get("spec_verify_ct", 0) for s in states
                )
        else:
            num_verify_tokens = sum(
                s.get_meta_info(answer_key).get("spec_verify_ct", 0) for s in states
            )

        if num_verify_tokens == 0:
            accept_length = 1.0
        else:
            accept_length = num_output_tokens / num_verify_tokens
    else:
        accept_length = 1.0

    return BenchmarkMetrics(
        latency=latency,
        output_throughput=output_throughput,
        accept_length=accept_length,
        num_questions=len(states),
    )


def per_sample_spec_stats(
    states: List[Any],
    answer_key: str = "answer",
    additional_answer_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    The per-request view of what `compute_metrics` averages over the whole run.

    `accept_length` is `completion_tokens / spec_verify_ct`, i.e. how many tokens
    one verification forward pass produced for THIS request, using the same
    definition as the aggregate metric so the two are comparable. It is None when
    the server answered without speculative decoding (a baseline run has no
    `spec_verify_ct`), so that a missing measurement is never averaged in as 1.0.

    Multi-turn answers are summed on the generation side and maxed on the prompt
    side: turn 2 re-sends turn 1, so summing prompt_tokens would count it twice.

    Returns:
        One dict per state, in the order the questions were asked.
    """
    keys = [answer_key] + list(additional_answer_keys or [])
    stats: List[Dict[str, Any]] = []
    for index, state in enumerate(states):
        completion_tokens = 0
        spec_verify_ct = 0
        prompt_tokens = 0
        cached_tokens = 0
        finish_reason = None
        is_speculative = False
        for key in keys:
            meta = state.get_meta_info(key) or {}
            completion_tokens += int(meta.get("completion_tokens", 0) or 0)
            prompt_tokens = max(prompt_tokens, int(meta.get("prompt_tokens", 0) or 0))
            cached_tokens = max(cached_tokens, int(meta.get("cached_tokens", 0) or 0))
            if "spec_verify_ct" in meta:
                is_speculative = True
                spec_verify_ct += int(meta.get("spec_verify_ct", 0) or 0)
            reason = meta.get("finish_reason")
            if isinstance(reason, dict):
                reason = reason.get("type")
            finish_reason = reason or finish_reason

        accept_length = None
        if is_speculative:
            # a request that finished during prefill never ran a verify step
            accept_length = (
                completion_tokens / spec_verify_ct if spec_verify_ct else 1.0
            )

        stats.append(
            {
                "index": index,
                # the server's request id, the join key for anything the server
                # recorded on its own (e.g. the DFlash verify-entropy dump)
                "rid": (state.get_meta_info(answer_key) or {}).get("id"),
                "accept_length": accept_length,
                "completion_tokens": completion_tokens,
                "spec_verify_ct": spec_verify_ct if is_speculative else None,
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cached_tokens": cached_tokens,
                "finish_reason": finish_reason,
            }
        )
    return stats


def length_summary(stats: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prompt, completion and total token counts over a run's requests.

    Read from the per-sample records both drivers build, whose counts are the
    server's own ``usage`` numbers rather than a re-tokenization here.

    ``total`` is prompt + completion. For a multi-turn question the prompt side
    is the longest turn's -- turn 2 re-sends turn 1, so the maximum is the final
    context -- which makes the sum the length of the whole conversation as the
    model last saw it, not the sum of what every turn sent.
    """
    if not stats:
        return None

    def block(values: List[int]) -> Dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
        }

    prompt = [int(stat.get("prompt_tokens") or 0) for stat in stats]
    completion = [int(stat.get("completion_tokens") or 0) for stat in stats]
    return {
        "requests": len(stats),
        "prompt_tokens": block(prompt),
        "completion_tokens": block(completion),
        "total_tokens": block([p + c for p, c in zip(prompt, completion)]),
    }


def _finite(stats: List[Dict[str, Any]], key: str) -> Optional[List[float]]:
    """The key's value on every record, or None if any record lacks it.

    All-or-nothing on purpose: a partial set of server timings would make the
    sums below cover a different request set than the token counts do.
    """
    values = []
    for stat in stats:
        value = stat.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        values.append(float(value))
    return values


def throughput_summary(
    stats: List[Dict[str, Any]], latency: float
) -> Optional[Dict[str, Any]]:
    """Every throughput the run can support, wall-clock and server-side.

    The wall-clock number is the only one available without the server timing
    patch (patches/sglang/*/request-timing-split.patch): it divides the tokens
    the run generated by how long the whole run took, so on a prompt-heavy
    workload it is mostly a prefill measurement and a draft model cannot move
    it. The patch adds `first_token_latency` and `decode_latency` per request,
    which split that into the phase the prompt pays for and the phase
    speculative decoding actually accelerates.

    Everything is token-weighted (sum over sum), not a mean of per-request
    ratios, so one short request cannot dominate. The decode phase counts
    `completion_tokens - 1` tokens: the first one is what TTFT ends on.
    """
    if not stats:
        return None

    completion = sum(int(stat.get("completion_tokens") or 0) for stat in stats)
    prompt = sum(int(stat.get("prompt_tokens") or 0) for stat in stats)
    cached = sum(int(stat.get("cached_tokens") or 0) for stat in stats)
    summary: Dict[str, Any] = {
        "requests": len(stats),
        "wall_latency": latency,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "wall_output_throughput": completion / latency if latency > 0 else 0.0,
    }

    e2e = _finite(stats, "e2e_latency")
    if e2e and sum(e2e) > 0:
        summary["e2e_latency_sum"] = sum(e2e)
        summary["e2e_latency_mean"] = statistics.fmean(e2e)
        summary["e2e_output_throughput"] = completion / sum(e2e)

    ttft = _finite(stats, "first_token_latency")
    if ttft and sum(ttft) > 0:
        summary["ttft_sum"] = sum(ttft)
        summary["ttft_mean"] = statistics.fmean(ttft)
        summary["ttft_median"] = statistics.median(ttft)
        summary["prefill_throughput"] = prompt / sum(ttft)
        if prompt:
            summary["cached_prompt_share"] = cached / prompt

    decode = _finite(stats, "decode_latency")
    if decode and sum(decode) > 0:
        # the token TTFT ends on is produced by the prefill, not the decode loop
        decoded = sum(
            max(int(stat.get("completion_tokens") or 0) - 1, 0) for stat in stats
        )
        summary["decode_tokens"] = decoded
        summary["decode_latency_sum"] = sum(decode)
        summary["decode_output_throughput"] = decoded / sum(decode)

    if "ttft_sum" in summary and "e2e_latency_sum" in summary:
        summary["prefill_share"] = summary["ttft_sum"] / summary["e2e_latency_sum"]
    return summary


def format_throughput_summary(summary: Optional[Dict[str, Any]]) -> List[str]:
    """The throughput lines `print_summary` shows, deepest breakdown last."""
    if not summary:
        return []
    lines = [
        f"{'Throughput:':<18}{summary['wall_output_throughput']:>10,.2f} tok/s"
        f"   (output / wall clock)"
    ]
    if "e2e_output_throughput" in summary:
        lines.append(
            f"{'  end-to-end:':<18}{summary['e2e_output_throughput']:>10,.2f} tok/s"
            f"   (output / sum of per-request e2e)"
        )
    if "prefill_throughput" in summary:
        lines.append(
            f"{'  prefill:':<18}{summary['prefill_throughput']:>10,.2f} tok/s"
            f"   (prompt / sum of TTFT)"
        )
    if "decode_output_throughput" in summary:
        lines.append(
            f"{'  decode:':<18}{summary['decode_output_throughput']:>10,.2f} tok/s"
            f"   (output-1 / sum of decode latency)"
        )
    if "ttft_mean" in summary:
        lines.append(
            f"{'TTFT:':<18}{summary['ttft_mean']:>10,.3f}s"
            f"   median {summary['ttft_median']:,.3f}s"
        )
    if "prefill_share" in summary:
        lines.append(
            f"{'Prefill share:':<18}{summary['prefill_share']:>10.1%}"
            f"   of end-to-end (a draft model only moves the rest)"
        )
    if summary.get("cached_prompt_share"):
        lines.append(
            f"{'Cached prompt:':<18}{summary['cached_prompt_share']:>10.1%}"
            f"   of prompt tokens came from the prefix cache"
        )
    if "e2e_output_throughput" not in summary:
        lines.append(
            "  (per-phase throughput needs "
            "patches/sglang/v0.5.14/request-timing-split.patch on the server)"
        )
    return lines


def format_length_summary(summary: Optional[Dict[str, Any]]) -> List[str]:
    """The three lines `print_summary` shows for a length summary."""
    if not summary:
        return []
    lines = []
    for key, label in (
        ("prompt_tokens", "Prompt tokens"),
        ("completion_tokens", "Answer tokens"),
        ("total_tokens", "Total tokens"),
    ):
        block = summary[key]
        lines.append(
            f"{label + ':':<18}mean {block['mean']:,.1f}   "
            f"median {block['median']:,.1f}   "
            f"min {block['min']:,}   max {block['max']:,}"
        )
    return lines


def token_entropy_series(top_logprobs: Any) -> Dict[str, List[float]]:
    """Per-token entropy of the target's next-token distribution, in nats.

    ``top_logprobs`` is SGLang's ``meta_info["output_top_logprobs"]``: one list of
    ``(logprob, token_id, text)`` per generated position, holding only the top ``k``
    entries. The exact entropy needs the whole vocabulary, so what is computed here
    is the top-k sum with the leftover probability mass folded into a single bucket::

        H = -sum_{i in top-k} p_i*log(p_i)  +  m*log(1/m),   m = 1 - sum_i p_i

    That is a lower bound on the true entropy — the tail can only be more spread out
    than one bucket — and it is tight whenever ``m`` is small, which is why
    ``tail_mass`` is returned next to it.

    Only valid when the logprobs are the model's raw distribution. With
    ``temperature=0`` SGLang normalizes the request to ``top_k=1`` and takes its
    all-greedy path, which returns ``log_softmax(logits)`` with no temperature
    scaling and no top-k/top-p filtering, so the numbers mean what they say.
    """
    entropy: List[float] = []
    top1: List[float] = []
    tail: List[float] = []
    for position in top_logprobs or []:
        # (logprob, token_id, text); a truncated/streamed row may be empty
        logprobs = [float(entry[0]) for entry in position if entry is not None]
        if not logprobs:
            continue
        probs = [math.exp(value) for value in logprobs]
        total = math.fsum(probs)
        # -sum p*log(p), reusing the logprobs instead of recomputing the log
        head = -math.fsum(p * lp for p, lp in zip(probs, logprobs))
        rest = max(0.0, 1.0 - total)
        # entropy is non-negative by definition; the clamp keeps a certain token
        # from being reported as -0.0
        entropy.append(max(0.0, head - rest * math.log(rest) if rest > 1e-12 else head))
        top1.append(max(probs))
        tail.append(rest)
    return {"entropy": entropy, "top1_prob": top1, "tail_mass": tail}


def _quantiles(values: List[float], probs=(10, 50, 90)) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {f"p{q}": float(np.percentile(array, q)) for q in probs}


def per_sample_entropy_stats(
    states: List[Any],
    answer_key: str = "answer",
    additional_answer_keys: Optional[List[str]] = None,
    deciles: int = 10,
) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
    """Per-request target-model entropy stats, plus the raw per-token series.

    Returns ``(stats, series)``. ``stats`` is small enough to live in the results
    file; ``series`` is the full per-token entropy of each request, which the
    caller can dump next to the generations for a position-resolved analysis.
    A request whose response carried no ``output_top_logprobs`` yields a record
    with ``n_tokens=0`` and ``None`` statistics rather than being dropped, so the
    list stays aligned with ``states``.
    """
    keys = [answer_key] + list(additional_answer_keys or [])
    stats: List[Dict[str, Any]] = []
    series: List[List[float]] = []
    for index, state in enumerate(states):
        entropy: List[float] = []
        top1: List[float] = []
        tail: List[float] = []
        for key in keys:
            meta = state.get_meta_info(key) or {}
            measured = token_entropy_series(meta.get("output_top_logprobs"))
            entropy.extend(measured["entropy"])
            top1.extend(measured["top1_prob"])
            tail.extend(measured["tail_mass"])
        series.append(entropy)
        if not entropy:
            stats.append({"index": index, "n_tokens": 0, "entropy_mean": None})
            continue
        # relative position profile: does the model get less certain further in?
        buckets = None
        if len(entropy) >= deciles:
            chunks = np.array_split(np.asarray(entropy), deciles)
            buckets = [float(np.mean(chunk)) for chunk in chunks]
        stats.append(
            {
                "index": index,
                "n_tokens": len(entropy),
                "entropy_mean": float(np.mean(entropy)),
                **{f"entropy_{k}": v for k, v in _quantiles(entropy).items()},
                "top1_prob_mean": float(np.mean(top1)),
                "tail_mass_mean": float(np.mean(tail)),
                # share of near-deterministic vs genuinely uncertain positions
                "frac_entropy_below_0.1": float(np.mean(np.asarray(entropy) < 0.1)),
                "frac_entropy_above_1.0": float(np.mean(np.asarray(entropy) > 1.0)),
                "entropy_by_decile": buckets,
            }
        )
    return stats, series


def load_verify_entropy_dump(path: str) -> Dict[str, Dict[str, Any]]:
    """Aggregate the DFlash verify-entropy dump, per request id.

    The dump is written by the server (patches/sglang/*/dflash-verify-entropy.patch)
    when ``SGLANG_DFLASH_ENTROPY_DUMP`` is set: one JSON object per request per
    verify step, holding the target's entropy at every one of the ``block_size``
    positions it just verified, plus how many draft tokens it accepted.

    Positions are classified against ``accept_len``, the number of drafted
    tokens the target confirmed:

    ``accepted``   indices ``[0, accept_len)`` — the draft was right here.
    ``rejection``  index ``accept_len`` — where the block stopped; the target's
                   own token is taken from this distribution. This is the one
                   that answers "is the draft losing because the target was
                   uncertain, or because the draft is simply weak?".
    ``discarded``  indices after that, conditioned on draft tokens the target
                   rejected, so they describe no real decoding step and are kept
                   apart from everything else.

    Reads ``path`` plus any per-process ``<base>.pid*<ext>`` siblings the server
    wrote (one per tensor-parallel rank).
    """
    base, ext = os.path.splitext(path)
    files = sorted(set(glob.glob(path) + glob.glob(f"{base}.pid*{ext or '.jsonl'}")))
    if not files:
        return {}

    per_rid: Dict[str, Dict[str, Any]] = {}
    for name in files:
        with open(name) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entropy = row.get("entropy") or []
                accept_len = int(row.get("accept_len", 0))
                if not entropy:
                    continue
                bucket = per_rid.setdefault(
                    str(row.get("rid")),
                    {
                        "steps": 0,
                        "accept_lens": [],
                        "accepted": [],
                        "rejection": [],
                        "discarded": [],
                        # entropy by index within the block, accepted positions
                        # only, so a per-position profile is not polluted by the
                        # distributions that were thrown away
                        "by_block_position": collections.defaultdict(list),
                    },
                )
                bucket["steps"] += 1
                bucket["accept_lens"].append(accept_len)
                for position, value in enumerate(entropy):
                    if position < accept_len:
                        bucket["accepted"].append(value)
                        bucket["by_block_position"][position].append(value)
                    elif position == accept_len:
                        bucket["rejection"].append(value)
                    else:
                        bucket["discarded"].append(value)
    return per_rid


def results_path(output_dir: str, name: Optional[str]) -> str:
    """
    Where a run writes its results.

    Deliberately without a timestamp: a rerun with the same --name lands on the
    same file, so it can pick up the benchmarks the previous run finished and
    only run what is missing.
    """
    return os.path.join(output_dir, f"{name + '_' if name else ''}results.jsonl")


def load_results(
    path: str, metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], Set[str]]:
    """
    The results accumulated so far, ready for this run to add to.

    The file holds one key per finished benchmark alongside the run's
    configuration, so the benchmarks already in it are exactly the keys that
    `metadata` does not claim. Their entries are returned untouched and named
    in the second return value; the configuration is replaced by this run's,
    which is the one the remaining benchmarks will actually use.

    A file that cannot be parsed raises rather than being overwritten -- it
    still holds hours of measurements.
    """
    results = dict(metadata)
    if not os.path.exists(path):
        return results, set()

    try:
        with open(path) as handle:
            stored = json.load(handle)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"{path} exists but could not be read back ({error}). Move it aside "
            "or delete it -- continuing would overwrite the benchmarks it holds."
        ) from error
    if not isinstance(stored, dict):
        raise RuntimeError(
            f"{path} holds a {type(stored).__name__}, not the object this "
            "script writes. Move it aside or delete it."
        )

    # an empty list means the benchmark was never recorded, so it is not done
    done = {key for key in stored if key not in metadata and stored[key]}
    for key in done:
        results[key] = stored[key]
    return results, done


def save_results(path: str, results: Dict[str, Any]) -> None:
    """
    Write the results file in one step.

    Written to a sibling and renamed, because this is now called after every
    benchmark: a run interrupted mid-write leaves the previous file intact
    instead of a truncated one the next run would refuse to read.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as handle:
        json.dump(results, handle, indent=4, default=str)
    os.replace(temporary, path)


def print_results(
    metrics_list: List[BenchmarkMetrics],
    benchmark_name: str,
    show_accuracy: bool = False,
):
    """
    Print benchmark results in a formatted way.

    Args:
        metrics_list: List of BenchmarkMetrics from multiple runs
        benchmark_name: Name of the benchmark
        show_accuracy: Whether to show accuracy metrics
    """
    avg_latency = np.mean([m.latency for m in metrics_list])
    avg_throughput = np.mean([m.output_throughput for m in metrics_list])
    avg_accept_length = np.mean([m.accept_length for m in metrics_list])

    print(f"\n{'='*50}")
    print(f"{benchmark_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Number of questions: {metrics_list[0].num_questions}")
    if show_accuracy:
        if metrics_list[0].accuracy is not None:
            avg_accuracy = np.mean(
                [m.accuracy for m in metrics_list if m.accuracy is not None]
            )
            print(f"Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
        else:
            print(f"Average Accuracy: None")
    print(f"Average Latency: {avg_latency:.3f} s")
    print(f"Average Output throughput: {avg_throughput:.3f} token/s")
    print(f"Average Accept length: {avg_accept_length:.3f}")
    print(f"{'='*50}\n")


def create_simple_sgl_function(
    function_name: str = "get_answer",
    answer_key: str = "answer",
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
    stop: Optional[List[str]] = None,
    user_prefix: Optional[str] = None,
) -> Callable:
    """
    Create a simple SGL function for single-turn Q&A.

    Args:
        function_name: Name of the function
        answer_key: Key for storing the answer
        system_prompt: Optional system prompt
        max_tokens: Maximum tokens to generate
        stop: Optional stop sequences
        user_prefix: Optional suffix to append to user message (appended after question)

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, question):
        if system_prompt:
            s += sgl.system(system_prompt)
        user_content = question
        if user_prefix:
            user_content = question + user_prefix
        s += sgl.user(user_content)
        gen_kwargs = {"max_tokens": max_tokens}
        if stop:
            gen_kwargs["stop"] = stop
        s += sgl.assistant(sgl.gen(answer_key, **gen_kwargs))

    sgl_func.__name__ = function_name
    return sgl_func


def create_few_shot_sgl_function(
    few_shot_examples: str,
    function_name: str = "few_shot_answer",
    answer_key: str = "answer",
    max_tokens: int = 512,
    stop: Optional[List[str]] = None,
) -> Callable:
    """
    Create an SGL function for few-shot learning.

    Args:
        few_shot_examples: String containing few-shot examples
        function_name: Name of the function
        answer_key: Key for storing the answer
        max_tokens: Maximum tokens to generate
        stop: Optional stop sequences

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, question):
        s += few_shot_examples + question
        gen_kwargs = {"max_tokens": max_tokens}
        if stop:
            gen_kwargs["stop"] = stop
        s += sgl.gen(answer_key, **gen_kwargs)

    sgl_func.__name__ = function_name
    return sgl_func


def create_multi_turn_sgl_function(
    function_name: str = "multi_turn_answer",
    system_prompt: Optional[str] = None,
    num_turns: int = 2,
    max_tokens: int = 2048,
) -> Callable:
    """
    Create an SGL function for multi-turn conversations (e.g., MT-Bench with 2 turns).

    Args:
        function_name: Name of the function
        system_prompt: Optional system prompt
        num_turns: Number of conversation turns (default: 2)
        max_tokens: Maximum tokens to generate per turn

    Returns:
        SGL function decorated with @sgl.function
    """
    if num_turns == 2:
        # Most common case: 2-turn conversation
        @sgl.function
        def sgl_func(s, question_1, question_2):
            if system_prompt:
                s += sgl.system(system_prompt)
            s += sgl.user(question_1)
            s += sgl.assistant(sgl.gen("answer_1", max_tokens=max_tokens))
            s += sgl.user(question_2)
            s += sgl.assistant(sgl.gen("answer_2", max_tokens=max_tokens))

    else:
        # Generic case: create function with dynamic number of turns
        # Note: This requires the caller to pass arguments as a dict
        @sgl.function
        def sgl_func(s, **kwargs):
            if system_prompt:
                s += sgl.system(system_prompt)
            for i in range(num_turns):
                question_key = f"question_{i+1}"
                answer_key = f"answer_{i+1}"
                if question_key in kwargs:
                    s += sgl.user(kwargs[question_key])
                    s += sgl.assistant(sgl.gen(answer_key, max_tokens=max_tokens))

    sgl_func.__name__ = function_name
    return sgl_func


def create_image_sgl_function(
    function_name: str = "get_image_answer",
    answer_key: str = "answer",
    max_tokens: int = 2048,
) -> Callable:
    """
    Create an SGL function for image-based Q&A.

    Args:
        function_name: Name of the function
        answer_key: Key for storing the answer
        max_tokens: Maximum tokens to generate

    Returns:
        SGL function decorated with @sgl.function
    """

    @sgl.function
    def sgl_func(s, image_path, question, **kwargs):
        """
        The body of the SGL function: constructs a multimodal conversation flow.

        - First, it inputs an image + text question as 'user'.
        - Then, it generates an answer as 'assistant', binding the response to the specified `answer_key`.

        Note: sgl.image() automatically encodes the image into a format supported by the model for multimodal input.
        """
        # User input: Image + Text question
        s += sgl.user(sgl.image(image_path) + question)
        s += sgl.assistant(sgl.gen(answer_key, max_tokens=max_tokens))

    sgl_func.__name__ = function_name
    return sgl_func
