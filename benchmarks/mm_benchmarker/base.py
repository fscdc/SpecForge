"""
Base class for the multimodal benchmarks.

It adds one thing to `benchmarker.base.Benchmarker`: `run()` accepts several
ports, so that a data-parallel deployment (one SGLang server per GPU, each on its
own port) can be driven by a single benchmark command. The questions are sharded
round-robin over the servers and run concurrently, and the metrics are computed
over the merged states, so the reported throughput is the aggregate of all the
servers while the accept length stays a per-token average.

`batch_size` keeps its meaning of "concurrent requests per server": every shard
is run with that many threads, so N servers see N * batch_size requests in
flight in total.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from benchmarker.base import Benchmarker
from benchmarker.utils import (
    compute_metrics,
    per_sample_entropy_stats,
    per_sample_spec_stats,
)
from sglang import set_default_backend
from sglang.global_config import global_config
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.lang.ir import SglSamplingParams
from sglang.utils import normalize_base_url

# Sampling parameters the server accepts but the SGL frontend drops: they are
# neither arguments of run_batch() nor fields of SglSamplingParams, so they never
# reach to_srt_kwargs(). install_extra_sampling_params() puts them back.
EXTRA_SAMPLING_PARAMS = ("repetition_penalty",)


def install_extra_sampling_params(values: Dict[str, Any]) -> None:
    """
    Carry the given sampling parameters into every request of this process.

    They are set on the SglSamplingParams class rather than passed through
    run_batch(), which rejects arguments it does not declare, and added to the
    payload by wrapping to_srt_kwargs().
    """
    if not values:
        return

    for name, value in values.items():
        setattr(SglSamplingParams, name, value)

    if getattr(SglSamplingParams, "_carries_extra_params", False):
        return

    original_to_srt_kwargs = SglSamplingParams.to_srt_kwargs

    def to_srt_kwargs(self):
        kwargs = original_to_srt_kwargs(self)
        for name in EXTRA_SAMPLING_PARAMS:
            value = getattr(self, name, None)
            if value is not None:
                kwargs[name] = value
        return kwargs

    SglSamplingParams.to_srt_kwargs = to_srt_kwargs
    SglSamplingParams._carries_extra_params = True


class ShardedSGLFunction:
    """
    An SGL function whose batch is spread over several backends.

    `run_batch` splits the arguments round-robin, runs one thread per backend and
    reassembles the states in the original order, so that the caller cannot tell
    the difference from a single-server run.
    """

    def __init__(self, sgl_function: Callable, backends: List[Any]):
        self.sgl_function = sgl_function
        self.backends = backends

    def run_batch(self, batch_kwargs: Sequence[Any], **kwargs) -> List[Any]:
        num_shards = len(self.backends)
        shards = {idx: list(batch_kwargs[idx::num_shards]) for idx in range(num_shards)}
        # with fewer questions than servers, some endpoints get nothing to do
        shards = {idx: shard for idx, shard in shards.items() if shard}

        results: Dict[int, List[Any]] = {}
        with ThreadPoolExecutor(max_workers=len(shards)) as pool:
            futures = {
                pool.submit(self._run_shard, idx, shard, kwargs): idx
                for idx, shard in shards.items()
            }
            for future in as_completed(futures):
                # a failing shard propagates here, the others are awaited by the
                # context manager before it does
                results[futures[future]] = future.result()

        states: List[Any] = [None] * len(batch_kwargs)
        for idx, shard_states in results.items():
            states[idx::num_shards] = shard_states
        return states

    def _run_shard(self, idx: int, shard: List[Any], kwargs: Dict[str, Any]):
        # one progress bar per shard would interleave into an unreadable mess, so
        # only the first shard draws one and it stands for the whole run
        kwargs = {
            **kwargs,
            "progress_bar": kwargs.get("progress_bar", False) and idx == 0,
        }
        return self.sgl_function.run_batch(shard, backend=self.backends[idx], **kwargs)


class MMBenchmarker(Benchmarker):
    """
    Base class for the multimodal benchmarks, see `benchmarker.base.Benchmarker`
    for the methods a subclass has to implement. A subclass that wants a maximum
    generation length of its own overrides `default_max_new_tokens()` rather than
    `get_max_new_tokens()`, so that the caller keeps the last word.
    """

    def __init__(
        self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None
    ):
        super().__init__(num_samples, subset)
        # set by run(), from the command line
        self.max_new_tokens: Optional[int] = None
        self.assistant_prefix: Optional[str] = None

    def default_max_new_tokens(self) -> int:
        """The generation length the benchmark asks for when none is imposed."""
        return super().get_max_new_tokens()

    def get_max_new_tokens(self) -> int:
        """The requested generation length, or the benchmark default."""
        if self.max_new_tokens is not None:
            return self.max_new_tokens
        return self.default_max_new_tokens()

    def run(
        self,
        host: str,
        port: Union[int, Sequence[int]],
        batch_size: int,
        max_new_tokens: int = None,
        num_runs: int = 1,
        sampling_params: Optional[Dict[str, Any]] = None,
        assistant_prefix: Optional[str] = None,
        chat_template_name: Optional[str] = None,
        top_logprobs_num: Optional[int] = None,
    ):
        """
        Run the benchmark evaluation.

        Mirrors `Benchmarker.run()`, except that `port` may be a list of ports:
        the questions are then sharded over one endpoint per port and run
        concurrently against those servers.

        Args:
            host (str): The host of the SGLang server(s)
            port (int | list[int]): The port of the SGLang server, or the ports of
                the data-parallel replicas to spread the questions over
            batch_size (int): The number of prompts to process in parallel, per server
            max_new_tokens (int): Maximum number of new tokens to generate, the
                benchmark's own default when not given
            num_runs (int): The number of times to run this benchmark, default is 1. You can set it to a larger number if you want to get more stable results.
            sampling_params (dict): Sampling arguments passed to every request,
                e.g. temperature / top_p / top_k. Defaults to greedy decoding.
            assistant_prefix (str): Text to prefill the assistant turn with, e.g.
                `NO_THINKING_PREFIX` to keep a reasoning model out of its
                thinking mode
            chat_template_name (str): The frontend chat template used to build the
                prompts, e.g. "qwen2-vl". SGLang guesses it from the model path
                when not given, and falls back to a generic USER:/ASSISTANT:
                template it cannot, which is wrong for most VL models.
            top_logprobs_num (int): Ask the server for this many top logprobs per
                generated token, which is what the target-model entropy is
                computed from. None (the default) leaves logprobs off. DFlash
                rejects `return_logprob`, so this only works on a target-only run.
        """
        ports = [port] if isinstance(port, int) else list(port)
        assert len(ports) > 0, "at least one port is required"

        # applied by create_sgl_function() below, so they have to be in place first
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if assistant_prefix is not None:
            self.assistant_prefix = assistant_prefix

        # Initialize one backend per server. This is what select_sglang_backend()
        # does for the "srt-no-parallel" backend, plus the chat template override.
        global_config.enable_parallel_encoding = False
        backends = [
            RuntimeEndpoint(
                normalize_base_url(host, p), chat_template_name=chat_template_name
            )
            for p in ports
        ]
        # programs that do not carry a backend fall back on the global one
        set_default_backend(backends[0])

        # Load data
        questions, labels = self.load_data()
        if len(questions) == 0:
            print("No valid questions found. Please check the dataset format.")
            return
        # kept so that a per-sample report can say which question produced a
        # given accept length, not just its index
        self.questions = questions

        # Create SGL function
        sgl_function = self.create_sgl_function()
        if len(backends) > 1:
            print(
                f"Sharding {len(questions)} questions over {len(backends)} servers "
                f"on ports {', '.join(str(p) for p in ports)}"
            )
            sgl_function = ShardedSGLFunction(sgl_function, backends)

        # Run evaluation loops
        metrics_list = []
        answer_keys = self.get_answer_keys()
        # greedy unless the caller asked for something else
        sampling_params = {"temperature": 0, **(sampling_params or {})}
        # the ones run_batch() does not declare travel with the class instead
        install_extra_sampling_params(
            {
                name: sampling_params.pop(name)
                for name in EXTRA_SAMPLING_PARAMS
                if name in sampling_params
            }
        )

        # run_batch declares these, and _resolve_sampling_params lets every gen()
        # inherit them, so no per-benchmark SGL function has to be touched
        logprob_params = {}
        if top_logprobs_num:
            logprob_params = {
                "return_logprob": True,
                "top_logprobs_num": int(top_logprobs_num),
                # the generated tokens are what the entropy is about; leaving the
                # start len unset keeps the prompt's logprobs out of the response
                "return_text_in_logprobs": False,
            }

        for _ in range(num_runs):
            tic = time.perf_counter()
            states = sgl_function.run_batch(
                questions,
                max_new_tokens=self.get_max_new_tokens(),
                num_threads=batch_size,
                progress_bar=True,
                **logprob_params,
                **sampling_params,
            )
            latency = time.perf_counter() - tic
            metrics_list.append(self._summarize(states, labels, latency, answer_keys))

        return metrics_list

    def _summarize(
        self,
        states: List[Any],
        labels: List[Any],
        latency: float,
        answer_keys: Optional[List[str]],
    ):
        """Turn the states of one run into metrics, as `Benchmarker.run()` does."""
        self._raise_on_failed_requests(states)

        # Extract predictions
        predictions = []
        generations = []
        primary_answer_key = answer_keys[0] if answer_keys else "answer"
        for i in range(len(states)):
            # Access answer from state object (states[i] supports dict-like access)
            output = self._get_answer(states[i], i, primary_answer_key)
            generations.append(output)
            if isinstance(output, str):
                extracted = self.extract_answer(
                    output,
                    (labels[i] if labels and i < len(labels) else None),
                )
            else:
                extracted = output
            predictions.append(extracted)

        # kept for --save-generations, the only way to tell a model that answers
        # wrongly from one whose answer was not extracted properly
        self.generations = generations
        self.predictions = predictions
        self.labels = labels

        # Compute accuracy if applicable
        accuracy = None
        has_labels_list = labels and len(labels) > 0
        if has_labels_list:
            accuracy = self.compute_accuracy(predictions, labels)
            if accuracy is not None:
                valid_count = sum(1 for p in predictions if p is not None)
                if valid_count < len(predictions):
                    print(
                        f"Warning: {len(predictions) - valid_count} predictions could not be extracted."
                    )

        # Compute performance metrics
        additional_answer_keys = (
            answer_keys[1:] if answer_keys and len(answer_keys) > 1 else None
        )
        metrics = compute_metrics(
            states,
            latency,
            answer_key=primary_answer_key,
            additional_answer_keys=additional_answer_keys,
        )
        # the same numbers before they are averaged away: the run-level accept
        # length hides which questions the draft did well or badly on
        self.per_sample_stats = per_sample_spec_stats(
            states,
            answer_key=primary_answer_key,
            additional_answer_keys=additional_answer_keys,
        )
        # target-model next-token entropy; empty unless the run asked for logprobs
        self.per_sample_entropy, self.token_entropy_series = per_sample_entropy_stats(
            states,
            answer_key=primary_answer_key,
            additional_answer_keys=additional_answer_keys,
        )
        if has_labels_list:
            metrics.accuracy = accuracy
            if accuracy is not None:
                metrics.num_valid_predictions = sum(
                    1 for p in predictions if p is not None
                )

        # optional per-category breakdown, computed after compute_accuracy() so a
        # benchmark can reuse whatever it scored there
        metrics.categorical_performance = self.compute_categorical_performance(
            states, latency, primary_answer_key
        )
        return metrics

    @staticmethod
    def _raise_on_failed_requests(states: List[Any]) -> None:
        """
        Stop on a request the server rejected.

        A program that raised keeps its variables unset and reports no tokens, so
        it would otherwise surface as a bare KeyError on the answer variable, and
        the throughput of the run would be computed over the requests that did
        succeed. SGLang only warns about the underlying error, hence repeating it
        here.
        """
        failures = [
            (index, state.error())
            for index, state in enumerate(states)
            if getattr(state, "error", lambda: None)() is not None
        ]
        if not failures:
            return
        index, error = failures[0]
        raise RuntimeError(
            f"{len(failures)} of {len(states)} requests failed, so the metrics of "
            f"this run would be meaningless. First failure (question {index}): "
            f"{type(error).__name__}: {error}"
        )

    @staticmethod
    def _get_answer(state: Any, index: int, answer_key: str) -> Any:
        """The generated answer, with a readable error when there is none."""
        try:
            return state[answer_key]
        except KeyError:
            raise RuntimeError(
                f"request {index} produced no '{answer_key}' variable, i.e. its SGL "
                "program did not run to completion. Look for a preceding "
                "'Error in stream_executor' warning for the underlying cause."
            ) from None

    def describe_run(self) -> Optional[Dict[str, Any]]:
        """
        Benchmark-specific notes about what was actually evaluated, e.g. the
        questions a benchmark had to skip and why. Stored next to the metrics in
        the results file. None (the default) when there is nothing to report.
        """
        return None

    def dump_generations(self, path: str) -> int:
        """
        Write what the model actually generated in the last run to a JSONL file.

        One object per question, with the raw generation, the answer that was
        extracted from it and the reference. This is what tells a model that got
        the question wrong apart from an answer the parsing failed to read.

        The prompt (image path included) and the per-sample decoding stats
        (accept length, verify count, finish reason) travel in the same record,
        so this file alone is enough to ask what a low accept length correlates
        with.
        """
        generations = getattr(self, "generations", None)
        if not generations:
            return 0

        labels = getattr(self, "labels", None) or []
        predictions = getattr(self, "predictions", None) or []
        # a benchmark that scores by index rather than through extract_answer can
        # expose what it actually compared, and whether it counted
        parsed = getattr(self, "parsed_answers", None) or []
        hits = getattr(self, "hits", None) or []
        questions = getattr(self, "questions", None) or []
        stats = getattr(self, "per_sample_stats", None) or []
        entropy_stats = getattr(self, "per_sample_entropy", None) or []
        entropy_series = getattr(self, "token_entropy_series", None) or []
        # per-index side channels a benchmark may keep, e.g. the MMStar taxonomy.
        # Stored under the singular name, which is what one record is about.
        side_lists = {
            key: values
            for attribute, key in (
                ("categories", "category"),
                ("l2_categories", "l2_category"),
            )
            for values in [getattr(self, attribute, None)]
            if isinstance(values, list) and len(values) == len(generations)
        }

        with open(path, "w") as handle:
            for index, generation in enumerate(generations):
                record = {
                    "index": index,
                    "generation": generation,
                    "prediction": (
                        predictions[index] if index < len(predictions) else None
                    ),
                    "label": labels[index] if index < len(labels) else None,
                }
                if index < len(parsed):
                    record["parsed"] = parsed[index]
                if index < len(hits):
                    record["score"] = hits[index]
                if index < len(questions):
                    question = questions[index]
                    if isinstance(question, dict):
                        record.update(
                            {
                                key: value
                                for key, value in question.items()
                                if key not in record
                            }
                        )
                    else:
                        record["question"] = question
                for name, values in side_lists.items():
                    record[name] = values[index]
                if index < len(stats):
                    record.update(
                        {
                            key: value
                            for key, value in stats[index].items()
                            if key != "index"
                        }
                    )
                if index < len(entropy_stats) and entropy_stats[index].get("n_tokens"):
                    record.update(
                        {
                            key: value
                            for key, value in entropy_stats[index].items()
                            if key != "index"
                        }
                    )
                    # the raw series is what a position-resolved study needs, and
                    # this file is the only place big enough to keep it
                    if index < len(entropy_series):
                        record["token_entropy"] = [
                            round(value, 4) for value in entropy_series[index]
                        ]
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return len(generations)

    def compute_categorical_performance(
        self, states: List[Any], latency: float, answer_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Break the run down per category, e.g. per MMMU subset.

        Returns a mapping of category name to `BenchmarkMetrics`, which lands in
        the results file next to the overall metrics. None (the default) when the
        benchmark has nothing to break down.
        """
        return None
