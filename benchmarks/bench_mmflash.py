#!/usr/bin/env python3
import argparse
import json
import os
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import requests
from mm_benchmarker import MM_BENCHMARKS
from mm_benchmarker.utils import NO_THINKING_PREFIX
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import kill_process_tree, popen_launch_server
from sglang.utils import wait_for_server

# tokens that mean "let SGLang decide" in a --config-list field
AUTO_TOKENS = {"", "auto", "none"}


def parse_args():
    parser = argparse.ArgumentParser()
    sglang_group = parser.add_argument_group("sglang")
    ServerArgs.add_cli_args(sglang_group)

    # make the follow args a group
    benchmark_group = parser.add_argument_group("benchmark")
    benchmark_group.add_argument(
        "--skip-launch-server", action="store_true", default=False
    )
    benchmark_group.add_argument("--timeout-for-server-launch", type=int, default=600)
    benchmark_group.add_argument("--output-dir", type=str, default="./results")
    benchmark_group.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Ports of the data-parallel replicas to spread the questions over, one "
            "server per GPU. Requires --skip-launch-server, since this script only "
            "launches a single server. Defaults to --port alone."
        ),
    )
    benchmark_group.add_argument(
        "--config-list",
        type=str,
        nargs="+",
        default=["1,0", "1,16"],
        help=(
            "The list of DFlash configurations to test, the format is "
            "<batch-size>,<block-size>[,<draft-window-size>]. A block size of 0 runs the "
            "target model alone (baseline), 'auto' lets SGLang infer it from the draft "
            "model config. The draft window size is optional and defaults to full context."
        ),
    )
    benchmark_group.add_argument(
        "--name",
        type=str,
        default=None,
        help="name of this benchmark run, if provided, will be added to the output file name",
    )
    benchmark_group.add_argument(
        "--save-generations",
        action="store_true",
        default=False,
        help=(
            "Write what the model generated for every question next to the results, "
            "as <benchmark>_generations_<timestamp>.jsonl. The only way to tell a "
            "wrong answer from one the scoring failed to parse."
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
    sampling_group.add_argument("--top-k", type=int, default=None)
    sampling_group.add_argument("--min-p", type=float, default=None)
    sampling_group.add_argument(
        "--presence-penalty",
        type=float,
        default=None,
        help="Penalize tokens that already appeared, 0 disables it.",
    )
    sampling_group.add_argument(
        "--frequency-penalty",
        type=float,
        default=None,
        help="Penalize tokens proportionally to how often they appeared, 0 disables it.",
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help=(
            "Divide the logits of tokens already seen, in (0, 2], 1.0 disables it. "
            "The SGL frontend drops this one, so it is added to the request payload "
            "separately."
        ),
    )
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of tokens to generate per question. Defaults to what "
            "the benchmark asks for (512 for chartqa, 2048 for mmstar)."
        ),
    )
    sampling_group.add_argument(
        "--disable-thinking",
        action="store_true",
        default=False,
        help=(
            "Prefill the assistant turn with an empty <think></think> block, which "
            "is how the Qwen3-family templates turn reasoning off. The SGL frontend "
            "builds the prompt itself and never runs the HF chat template, so "
            "enable_thinking=False cannot be passed to the server instead."
        ),
    )
    sampling_group.add_argument(
        "--chat-template-name",
        type=str,
        default=None,
        help=(
            "Frontend chat template used to build the prompts, e.g. 'qwen2-vl'. "
            "SGLang guesses it from the model path, and falls back to a generic "
            "USER:/ASSISTANT: template with an <image> placeholder when the name "
            "matches nothing, which is wrong for most VL models."
        ),
    )
    benchmark_group.add_argument(
        "--benchmark-list",
        type=str,
        nargs="+",
        default=["chartqa:200"],
        help=(
            "The list of multimodal benchmarks to run. The format is "
            "<benchmark-name>:<num-prompts>:<subset>,<subset>. We support the following "
            f"benchmarks: {', '.join(MM_BENCHMARKS.benchmarks.keys()) or '<none registered yet>'}"
        ),
    )
    return parser.parse_args()


def parse_optional_int(value: str, field_name: str) -> Optional[int]:
    """
    Parse a --config-list field, mapping the 'auto' tokens to None.
    """
    value = value.strip().lower()
    if value in AUTO_TOKENS:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"Invalid {field_name} '{value}': expected an integer or 'auto'"
        )


def parse_config(config: str) -> Tuple[int, Optional[int], Optional[int]]:
    """
    Parse one --config-list entry into (batch_size, block_size, draft_window_size).

    block_size is 0 for a baseline run without speculative decoding and None when
    SGLang should infer it from the draft model config. draft_window_size is None
    when the draft model should attend to the full context.
    """
    fields = config.split(",")
    if len(fields) not in (2, 3):
        raise ValueError(
            f"Invalid config '{config}': expected <batch-size>,<block-size>[,<draft-window-size>]"
        )

    batch_size = int(fields[0])
    block_size = parse_optional_int(fields[1], "block size")
    draft_window_size = (
        parse_optional_int(fields[2], "draft window size") if len(fields) == 3 else None
    )
    # a window of 0 is the same as leaving it unset
    if draft_window_size == 0:
        draft_window_size = None

    if block_size == 0 and draft_window_size is not None:
        raise ValueError(
            f"Invalid config '{config}': a draft window size is meaningless for a "
            "baseline run (block size 0)"
        )
    if block_size and draft_window_size and draft_window_size < block_size:
        raise ValueError(
            f"Invalid config '{config}': the draft window size ({draft_window_size}) "
            f"must be >= the block size ({block_size})"
        )
    return batch_size, block_size, draft_window_size


def launch_sglang_server(
    server_args: ServerArgs,
    base_url: str,
    batch_size: int,
    block_size: Optional[int],
    draft_window_size: Optional[int],
    timeout: int,
):
    """
    This function launches the SGLang server with the given server arguments.

    DFlash is enabled unless block_size is 0, in which case the target model is
    launched alone so that it can be used as the baseline of the speedup.
    """
    sglang_args: List[str] = []
    if block_size != 0:
        assert (
            server_args.speculative_draft_model_path
        ), "--speculative-draft-model-path is required to run DFlash"
        sglang_args.extend(
            [
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                server_args.speculative_draft_model_path,
            ]
        )
        if block_size is not None:
            # --speculative-dflash-block-size is an alias of this flag
            sglang_args.extend(["--speculative-num-draft-tokens", str(block_size)])
        if draft_window_size is not None:
            sglang_args.extend(
                ["--speculative-draft-window-size", str(draft_window_size)]
            )

    sglang_args.extend(
        [
            "--cuda-graph-max-bs",
            str(batch_size),
            "--tp-size",
            str(server_args.tp_size),
            "--max-running-requests",
            str(batch_size),
        ]
    )

    if server_args.mem_fraction_static:
        sglang_args.extend(
            ["--mem-fraction-static", str(server_args.mem_fraction_static)]
        )

    if server_args.trust_remote_code:
        sglang_args.extend(["--trust-remote-code"])

    if server_args.disable_radix_cache:
        sglang_args.extend(["--disable-radix-cache"])

    if server_args.ep_size:
        sglang_args.extend(["--ep-size", str(server_args.ep_size)])

    if server_args.attention_backend:
        sglang_args.extend(["--attention-backend", server_args.attention_backend])

    if server_args.quantization:
        sglang_args.extend(["--quantization", server_args.quantization])

    if server_args.dtype:
        sglang_args.extend(["--dtype", server_args.dtype])

    # VL models often need an explicit chat template to keep the image placeholders
    if server_args.chat_template:
        sglang_args.extend(["--chat-template", server_args.chat_template])

    process = popen_launch_server(
        server_args.model_path,
        base_url,
        timeout=timeout,
        other_args=sglang_args,
        env={
            "SGLANG_RECORD_STEP_TIME": "1",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
            **os.environ,
        },
    )
    return process


def send_flush_cache_request(base_urls: List[str]):
    for base_url in base_urls:
        requests.post(base_url + "/flush_cache")


def main():
    args = parse_args()
    server_args: ServerArgs = ServerArgs.from_cli_args(args)
    configs = [parse_config(config) for config in args.config_list]

    # split the arg into list of (bench_name, num_prompts, subset)
    benchmark_list = []
    for item in args.benchmark_list:
        splits = item.split(":")
        if len(splits) == 1:
            bench_name = splits[0]
            num_prompts = None
            subset = None
        elif len(splits) == 2:
            bench_name, num_prompts = splits
            subset = None
        elif len(splits) == 3:
            bench_name, num_prompts, subset = splits
            subset = subset.split(",")
        else:
            raise ValueError(f"Invalid benchmark list format: {item}")
        benchmark_list.append((bench_name, num_prompts, subset))
    assert len(benchmark_list) != 0, (
        "the number of benchmark list is 0, register a multimodal benchmark in "
        "benchmarks/mm_benchmarker and pass it as --benchmark-list <name>:<num-prompts>"
    )
    # resolve every benchmark up front so that a typo does not surface after we
    # already paid for a server launch
    for bench_name, _, _ in benchmark_list:
        MM_BENCHMARKS.get(bench_name)
    # same for the chat template, which SGLang would otherwise reject with a bare
    # KeyError once the first request is built
    if args.chat_template_name is not None:
        from sglang.lang.chat_template import chat_template_registry

        if args.chat_template_name not in chat_template_registry:
            raise ValueError(
                f"Unknown chat template '{args.chat_template_name}'. Available: "
                f"{', '.join(sorted(chat_template_registry))}"
            )

    # one endpoint per data-parallel replica, the questions are spread over them
    ports = args.ports or [args.port]
    if len(ports) > 1 and not args.skip_launch_server:
        raise ValueError(
            "--ports needs --skip-launch-server: this script launches a single "
            "server, so the data-parallel replicas have to be launched separately, "
            "one per GPU (see scripts/benchmark.sh)"
        )
    base_urls = [f"http://localhost:{port}" for port in ports]
    # the server this script launches itself is always the first one
    base_url = base_urls[0]

    # every sampling argument that was actually given, greedy decoding otherwise
    sampling_params = {"temperature": args.temperature}
    for name in (
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    ):
        value = getattr(args, name)
        if value is not None:
            sampling_params[name] = value

    results = {}
    results["model"] = server_args.model_path
    results["draft_model"] = server_args.speculative_draft_model_path
    results["ports"] = ports
    results["sampling_params"] = sampling_params
    results["max_tokens"] = args.max_tokens
    results["thinking_disabled"] = args.disable_thinking

    def run_benchmarks(
        batch_size: int, block_size: Optional[int], draft_window_size: Optional[int]
    ):
        for benchmark_name, num_prompts, subset in benchmark_list:
            print(
                f"Running benchmark {benchmark_name} with {num_prompts} prompts, batch size {batch_size}, block size {block_size}, draft window size {draft_window_size}, subset {subset}"
            )
            benchmarker_cls = MM_BENCHMARKS.get(benchmark_name)
            num_prompts = int(num_prompts) if num_prompts is not None else None
            if subset is None:
                benchmarker = benchmarker_cls(num_samples=num_prompts)
            else:
                benchmarker = benchmarker_cls(num_samples=num_prompts, subset=subset)
            metrics_list = benchmarker.run(
                host=args.host,
                port=ports,
                batch_size=batch_size,
                max_new_tokens=args.max_tokens,
                sampling_params=sampling_params,
                assistant_prefix=NO_THINKING_PREFIX if args.disable_thinking else None,
                chat_template_name=args.chat_template_name,
            )
            send_flush_cache_request(base_urls)

            if args.save_generations:
                os.makedirs(args.output_dir, exist_ok=True)
                dump_path = os.path.join(
                    args.output_dir,
                    f"{args.name + '_' if args.name else ''}{benchmark_name}"
                    f"_generations_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
                )
                written = getattr(benchmarker, "dump_generations", lambda path: 0)(
                    dump_path
                )
                if written:
                    print(f"Saved {written} generations to {dump_path}")

            if benchmark_name not in results:
                results[benchmark_name] = []
            # what the benchmark has to say about the questions it actually ran,
            # e.g. the multi-image ones MMMU had to skip
            dataset_info = getattr(benchmarker, "describe_run", lambda: None)()
            results[benchmark_name].append(
                dict(
                    batch_size=batch_size,
                    block_size=block_size,
                    draft_window_size=draft_window_size,
                    dataset_info=dataset_info,
                    metrics=[asdict(metric) for metric in metrics_list],
                    num_samples=num_prompts,
                )
            )

    if args.skip_launch_server:
        # the server is already running, so the first config only tells us which
        # batch size to send with and how to label the results: pass it through
        # so that a baseline run (block size 0) is not reported as an unknown one
        batch_size, block_size, draft_window_size = (
            configs[0] if len(configs) > 0 else (8, None, None)
        )
        run_benchmarks(batch_size, block_size, draft_window_size)
    else:
        # we iterate over each config from args
        for batch_size, block_size, draft_window_size in configs:
            process = launch_sglang_server(
                server_args,
                base_url,
                batch_size,
                block_size,
                draft_window_size,
                args.timeout_for_server_launch,
            )
            wait_for_server(base_url)
            run_benchmarks(batch_size, block_size, draft_window_size)
            kill_process_tree(process.pid)
            process.wait()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        args.output_dir,
        f"{args.name + '_' if args.name else ''}results_{timestamp}.jsonl",
    )
    with open(result_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {result_file}")


if __name__ == "__main__":
    main()
