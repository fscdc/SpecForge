#!/usr/bin/env python3
"""Serving-behaviour probes against a running Gemma 4 SGLang server (GPU gate G2).

Two questions only a live server answers:

1. Do grayscale JPEGs still kill it? Stock SGLang decodes JPEGs with nvJPEG,
   which keeps a grayscale image at one channel; Gemma 4's vision projection
   then fails and the scheduler dies with every request in flight (86 crashes
   during the Gemma regen, and a capture-server crash ends a managed_local run
   for good). The gemma4-spec patch decodes with PIL instead. The probe sends
   every non-RGB JPEG row of the regen's ``_error`` file and counts the
   requests that fail and the crashes, i.e. failures after which
   ``/health_generate`` stays down. After a crash it waits for the server to
   come back (scripts/gates/gemma4_probe_hpc.sh relaunches it) and goes on.
2. Does E4B open an empty thought channel ``<|channel>thought\\n<channel|>``
   before its answer when thinking is off? The regen data cannot tell: SGLang's
   gemma4 reasoning parser strips it. The probe sends ~100 regen prompts to
   ``/generate`` with ``skip_special_tokens=false`` and counts the outputs that
   start with those ids; a non-zero count decides change C5.

Prompts are the trainer's chat rendering (``to_chat_messages`` + the
processor's template, thinking off, generation prompt on) plus the image path
as ``image_data``, shaped for the server (see ``prompt_request``): a server
with a tokenizer gets the rendered text and expands the image placeholder
itself, as for a chat-completions request; a ``--skip-tokenizer-init`` server
gets the HF processor's expanded ``input_ids``, as for a capture request. The
prompt ids the server reports back are checked against the processor's. Only
``output_ids`` are read; a ``--reasoning-parser`` does not touch
``/generate`` either.

    python scripts/gates/gemma4_serving_probe.py --server-url http://127.0.0.1:30000 \\
        --output-dir results/gemma4_probe/serving

writes ``serving_probe.json`` and ``summary.md``. Exit status 1 when a grayscale
row was not answered (failed, crashed the server or could not be sent), or when
the thought-channel probe cannot decide C5 (no empty channel and under 90% of
its prompts answered); 2 when the server is not up.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]

DEFAULT_TARGET = (
    "/scratch/fengsicheng/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
    "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
DEFAULT_DATA = (
    "/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/regen_data/"
    "gemma4-E4B-it_llava-ov15-1M-prompted_regen_first_turn.jsonl"
)
#: '<|channel>' 'thought' '\n' '<channel|>' in the E4B tokenizer
EMPTY_THOUGHT_IDS = (100, 45518, 107, 101)
EMPTY_THOUGHT_TEXT = "<|channel>thought\n<channel|>"
#: without an empty channel, C5 is ruled out only on this share of answered prompts
MIN_ANSWERED_SHARE = 0.9


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model-path", default=DEFAULT_TARGET, help="the served snapshot (processor + template)")
    parser.add_argument("--data-path", default=DEFAULT_DATA, help="regen jsonl for the thought-channel prompts")
    parser.add_argument("--error-path", default=None, help="default: <data-path stem>_error.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--prompt-stride", type=int, default=9973, help="take every Nth regen line")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-grayscale-rows", type=int, default=None, help="default: all of them")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument(
        "--crash-detect-s",
        type=float,
        default=60.0,
        help="a failed request counts as a crash when /health_generate stays down this long",
    )
    parser.add_argument("--restart-wait-s", type=float, default=1200.0, help="wait this long for a relaunch")
    parser.add_argument("--skip-grayscale", action="store_true")
    parser.add_argument("--skip-thought-channel", action="store_true")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
def _get(url: str, timeout: float = 10.0):
    import requests

    try:
        return requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None


def healthy(server_url: str, timeout: float = 30.0) -> bool:
    """``/health_generate`` runs a one-token generation through the scheduler."""
    response = _get(f"{server_url}/health_generate", timeout=timeout)
    return response is not None and response.status_code == 200


def wait_healthy(server_url: str, seconds: float, interval: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if healthy(server_url):
            return True
        time.sleep(interval)
    return False


def server_info(server_url: str) -> Dict[str, Any]:
    for path in ("/get_server_info", "/server_info"):
        response = _get(f"{server_url}{path}")
        if response is not None and response.status_code == 200:
            try:
                info = response.json()
            except ValueError:
                continue
            keys = (
                "model_path",
                "attention_backend",
                "mm_attention_backend",
                "skip_tokenizer_init",
                "reasoning_parser",
                "speculative_algorithm",
                "version",
            )
            return {key: info.get(key) for key in keys if key in info}
    return {}


def generate(server_url: str, body: Dict[str, Any], timeout: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    import requests

    try:
        response = requests.post(f"{server_url}/generate", json=body, timeout=timeout)
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:300]}"
    try:
        out = response.json()
    except ValueError:
        return None, "response is not JSON"
    if isinstance(out, list):
        out = out[0] if out else {}
    if "output_ids" not in out:
        return None, f"no output_ids in the response: {str(out)[:300]}"
    return out, None


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
def load_processor(model_path: str):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_path)


def prompt_request(
    processor, record: Dict[str, Any], max_new_tokens: int, as_text: bool = True
) -> Tuple[Dict[str, Any], List[int]]:
    """``/generate`` body for one regen row (its turns before the answer, thinking
    off), and the ids capture sends for it: the HF processor's, each image
    placeholder expanded to ``<|image>``, the soft tokens and ``<image|>``.

    A server with a tokenizer runs SGLang's Gemma 4 processor over the prompt,
    which expands the placeholders itself, so ``as_text`` sends the rendered
    text (one ``<|image|>`` per image, what the chat-completions endpoint passes
    on). Expanded ids would be expanded again (StopIteration, HTTP 500 on every
    image row), and unexpanded ids come out without ``<|image>`` and
    ``<image|>``. Only a ``--skip-tokenizer-init`` server, like the capture
    server, collapses expanded ids first; it gets them with ``as_text=False``.
    """
    from PIL import Image

    from specforge.data.mm_preprocessing import to_chat_messages

    turns = list(record["conversations"])
    while turns and turns[-1].get("role") == "assistant":
        turns.pop()
    if not turns:
        raise ValueError("row has no prompt turn")
    text = processor.apply_chat_template(
        to_chat_messages(turns), tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    image = record.get("image")
    if image:
        with Image.open(image) as handle:
            rgb = handle.convert("RGB")
        input_ids = processor(text=[text], images=[rgb], return_tensors="pt")["input_ids"][0].tolist()
    else:
        input_ids = processor(text=[text], return_tensors="pt")["input_ids"][0].tolist()
    body: Dict[str, Any] = {"text": text} if as_text else {"input_ids": input_ids}
    body["sampling_params"] = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.0,
        "skip_special_tokens": False,
    }
    body["return_prompt_token_ids"] = True
    if image:
        body["image_data"] = [image]
    return body, input_ids


def find_non_rgb_jpegs(error_path: str, limit: Optional[int]) -> Tuple[List[Dict[str, Any]], Counter]:
    """Rows of the regen error file whose image is a JPEG that is not RGB."""
    from PIL import Image

    rows: List[Dict[str, Any]] = []
    modes: Counter = Counter()
    with open(error_path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            image = record.get("image")
            if not isinstance(image, str):
                continue
            try:
                with open(image, "rb") as stream:
                    magic = stream.read(2)
                if magic != b"\xff\xd8":
                    continue
                with Image.open(image) as decoded:
                    mode = decoded.mode
            except OSError:
                modes["unreadable"] += 1
                continue
            modes[mode] += 1
            if mode != "RGB":
                rows.append(record)
    return (rows if limit is None else rows[:limit]), modes


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def grayscale_probe(args, processor, as_text: bool) -> Dict[str, Any]:
    error_path = args.error_path or str(Path(args.data_path).with_name(Path(args.data_path).stem + "_error.jsonl"))
    rows, modes = find_non_rgb_jpegs(error_path, args.max_grayscale_rows)
    result: Dict[str, Any] = {
        "error_path": error_path,
        "jpeg_modes": dict(modes),
        "candidates": len(rows),
        "ok": 0,
        "failed": 0,
        "not_sent": 0,
        "crashes": 0,
        "aborted": False,
        "rows": [],
    }
    print(f"[serving] grayscale: {len(rows)} non-RGB JPEG rows in {error_path} (JPEG modes {dict(modes)})", flush=True)
    for record in rows:
        entry: Dict[str, Any] = {"id": record.get("id"), "image": record.get("image")}
        try:
            body, _ = prompt_request(processor, record, max_new_tokens=1, as_text=as_text)
        except (OSError, ValueError) as exc:
            entry["outcome"] = f"not sent: {exc}"
            result["not_sent"] += 1
            result["rows"].append(entry)
            continue
        started = time.monotonic()
        out, error = generate(args.server_url, body, args.request_timeout)
        entry["latency_s"] = round(time.monotonic() - started, 2)
        if error is None:
            entry["outcome"] = "ok"
            result["ok"] += 1
            result["rows"].append(entry)
            continue
        entry["error"] = error
        result["failed"] += 1
        if wait_healthy(args.server_url, args.crash_detect_s):
            entry["outcome"] = "request failed, server alive"
        else:
            entry["outcome"] = "server crashed"
            result["crashes"] += 1
            print(f"[serving] grayscale: {record.get('id')} took the server down; waiting for a relaunch", flush=True)
            if not wait_healthy(args.server_url, args.restart_wait_s):
                entry["outcome"] += "; not back"
                result["aborted"] = True
                result["rows"].append(entry)
                break
        result["rows"].append(entry)
    # every row must be answered: a row that was never sent proves nothing
    result["passed"] = result["candidates"] > 0 and result["ok"] == result["candidates"] and not result["aborted"]
    return result


def thought_probe(args, processor, as_text: bool) -> Dict[str, Any]:
    tokenizer = processor.tokenizer
    thought_ids = tuple(tokenizer.encode(EMPTY_THOUGHT_TEXT, add_special_tokens=False))
    result: Dict[str, Any] = {
        "thought_ids": list(thought_ids),
        "documented_ids": list(EMPTY_THOUGHT_IDS),
        "sent": 0,
        "failed": 0,
        "answered": 0,
        "empty_thought": 0,
        "opens_channel": 0,
        "regen_prefix_match": 0,
        "prompt_ids_reported": 0,
        "prompt_ids_match": 0,
        "by_kind": {},
        "first_tokens": {},
        "rows": [],
    }
    first_tokens: Counter = Counter()
    by_kind: Dict[str, Counter] = {"image": Counter(), "text": Counter()}
    stride = max(1, args.prompt_stride)
    with open(args.data_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if result["sent"] >= args.num_prompts:
                break
            if line_number % stride:
                continue
            record = json.loads(line)
            if isinstance(record.get("image"), (list, tuple)):
                continue
            answer = record["conversations"][-1]
            if answer.get("role") != "assistant" or not isinstance(answer.get("content"), str):
                continue
            try:
                body, expected_ids = prompt_request(processor, record, args.max_new_tokens, as_text=as_text)
            except (OSError, ValueError):
                continue
            kind = "image" if record.get("image") else "text"
            out, error = generate(args.server_url, body, args.request_timeout)
            result["sent"] += 1
            by_kind[kind]["sent"] += 1
            if error is not None:
                result["failed"] += 1
                by_kind[kind]["failed"] += 1
                result["rows"].append({"id": record.get("id"), "kind": kind, "error": error})
                continue
            output_ids = [int(token) for token in out["output_ids"]]
            # the prompt the server ran, against the ids capture would send
            prompt_ids = out.get("prompt_token_ids")
            same_prompt = None if prompt_ids is None else [int(token) for token in prompt_ids] == expected_ids
            if same_prompt is not None:
                result["prompt_ids_reported"] += 1
                result["prompt_ids_match"] += int(same_prompt)
                by_kind[kind]["prompt_ids_match"] += int(same_prompt)
            empty = tuple(output_ids[: len(thought_ids)]) == thought_ids
            opens = bool(output_ids) and output_ids[0] == thought_ids[0]
            offset = len(thought_ids) if empty else 0
            answer_ids = tokenizer.encode(answer["content"], add_special_tokens=False)
            width = min(4, len(answer_ids), len(output_ids) - offset)
            matches = width > 0 and output_ids[offset : offset + width] == answer_ids[:width]
            result["empty_thought"] += int(empty)
            result["opens_channel"] += int(opens)
            result["regen_prefix_match"] += int(matches)
            by_kind[kind]["empty_thought"] += int(empty)
            by_kind[kind]["opens_channel"] += int(opens)
            if output_ids:
                first_tokens[output_ids[0]] += 1
            result["rows"].append(
                {
                    "id": record.get("id"),
                    "kind": kind,
                    "output_ids": output_ids,
                    "empty_thought": empty,
                    "regen_prefix_match": matches,
                    "prompt_ids_match": same_prompt,
                    "finish_reason": (out.get("meta_info") or {}).get("finish_reason"),
                }
            )
    result["by_kind"] = {kind: dict(counts) for kind, counts in by_kind.items()}
    result["first_tokens"] = {
        f"{token} {tokenizer.convert_ids_to_tokens(token)!r}": count for token, count in first_tokens.most_common(8)
    }
    result["answered"] = result["sent"] - result["failed"]
    answered = max(result["answered"], 1)
    result["empty_thought_rate"] = result["empty_thought"] / answered
    result["regen_prefix_match_rate"] = result["regen_prefix_match"] / answered
    return result


def c5_decision(thought: Dict[str, Any]) -> Tuple[bool, str]:
    """(decided, wording). One empty channel decides C5; ruling it out takes answers."""
    if thought["empty_thought"]:
        return True, "needed -- the target opens an empty thought channel the drafts never train on"
    answered = thought["sent"] - thought["failed"]
    if answered == 0 or answered < MIN_ANSWERED_SHARE * thought["sent"]:
        return False, f"undetermined -- only {answered} of {thought['sent']} prompts were answered"
    return True, "not needed -- no answer opened an empty thought channel"


def render_summary(report: Dict[str, Any]) -> str:
    lines = ["# Gemma 4 serving probe", ""]
    lines.append(f"server {report['server_url']}: {json.dumps(report['server_info'])}")
    lines.append(
        "requests: "
        + (
            "rendered chat text + image_data (the server expands the image)"
            if report["request_format"] == "text"
            else "processor-expanded input_ids + image_data (--skip-tokenizer-init server)"
        )
    )
    lines.append("")
    gray = report.get("grayscale")
    if gray:
        verdict = "PASS" if gray["passed"] else "FAIL"
        lines.append("## Grayscale JPEGs")
        lines.append(
            f"- {verdict}: {gray['candidates']} non-RGB JPEG rows from {gray['error_path']}: {gray['ok']} ok, "
            f"{gray['failed']} failed, {gray['not_sent']} not sent, {gray['crashes']} server crash(es)"
            + (", stopped early: the server did not come back" if gray["aborted"] else "")
        )
        lines.append(f"- JPEG modes in the error file: {gray['jpeg_modes']}")
        lines.append("")
    thought = report.get("thought_channel")
    if thought:
        lines.append("## Empty thought channel with thinking off")
        lines.append(
            f"- {thought['empty_thought']} of {thought['answered']} answers start with "
            f"{thought['thought_ids']} ({thought['empty_thought_rate']:.1%}); {thought['opens_channel']} open "
            f"a channel at all; {thought['failed']} of {thought['sent']} requests failed"
        )
        lines.append(f"- per kind: {thought['by_kind']}")
        lines.append(f"- most common first tokens: {thought['first_tokens']}")
        lines.append(
            f"- first answer tokens equal to the regen answer (after any empty channel): "
            f"{thought['regen_prefix_match_rate']:.1%}"
        )
        lines.append(
            f"- prompts the server ran equal to the processor's ids: {thought['prompt_ids_match']} of "
            f"{thought['prompt_ids_reported']} reported"
        )
        if thought["thought_ids"] != thought["documented_ids"]:
            lines.append(f"- NOTE: the tokenizer's ids differ from the documented {thought['documented_ids']}")
        lines.append("- decision C5: " + c5_decision(thought)[1])
        lines.append("")
    elif report.get("thought_channel_error"):
        lines.append("## Empty thought channel with thinking off")
        lines.append(f"- not run: {report['thought_channel_error']}")
        lines.append("- decision C5: undetermined")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    args.server_url = args.server_url.rstrip("/")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not healthy(args.server_url):
        print(f"[serving] ERROR: {args.server_url}/health_generate does not answer", file=sys.stderr)
        return 2
    info = server_info(args.server_url)
    as_text = not info.get("skip_tokenizer_init")
    processor = load_processor(args.model_path)
    report: Dict[str, Any] = {
        "server_url": args.server_url,
        "server_info": info,
        "request_format": "text" if as_text else "input_ids",
        "model_path": args.model_path,
        "host": platform.node(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if not args.skip_grayscale:
        report["grayscale"] = grayscale_probe(args, processor, as_text)
    if not args.skip_thought_channel:
        if healthy(args.server_url):
            report["thought_channel"] = thought_probe(args, processor, as_text)
        else:
            report["thought_channel_error"] = "server down after the grayscale probe"
    with open(out_dir / "serving_probe.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    summary = render_summary(report)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[serving] wrote {out_dir / 'serving_probe.json'} and {out_dir / 'summary.md'}")
    gray = report.get("grayscale")
    if gray is not None and not gray["passed"]:
        return 1
    thought = report.get("thought_channel")
    if report.get("thought_channel_error") or (thought is not None and not c5_decision(thought)[0]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
