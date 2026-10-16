#!/usr/bin/env python3
"""Real-weight capture-parity probe for a Gemma 4 target (GPU gate G1).

Training never sees the target model itself, only what the SGLang capture
server writes for it. This probe starts that server on the real checkpoint,
captures a few regen rows exactly the way training does (pre-expanded
``input_ids`` plus ``image_data`` on ``/generate`` with ``spec_capture``, read
back from Mooncake), and compares them with an independent HF forward of the
same rows:

* per tap, cosine and relative error against HF ``hidden_states[k+1]`` (the
  output of decoder layer ``k``), and against the residual inside layer
  ``k+1`` just before its MLP. The second is what a capture holds once the
  next layer's fused add-RMSNorm has overwritten it in place: the stock SGLang
  behaviour that the gemma4-spec patch removes;
* per tap, the RMS of the captured features (input to an EAGLE3 ``fc_norm``
  decision);
* top-1 agreement of ``softcap(lm_head(last_hidden))`` with HF's logits, the
  share of positions whose raw max logit exceeds 20 / 30, and
  KL(softcapped || raw) on the loss positions, i.e. how much the softcap moves
  the EAGLE3 soft targets;
* the gap between features captured with the vision tower on fa3 and on sdpa.

Image rows are reported per region (the prompt before the image, the image
tokens, everything after) and against HF run with three attention masks:
``causal`` (what the E4B config asks HF for, and what the patched SGLang
runs), ``bidir`` (each image attends to itself in both directions, which is
what stock SGLang's Gemma 4 sets up on triton, whatever the config)
and ``sglang_tiled`` (that mask as SGLang's triton extend kernel applies it: a
query reaches later image keys only inside its own BLOCK_M query tile).

Server modes. A run is ``<mode>:<mm attention backend>``. ``stock`` and
``patched`` request the SGLang behaviour without and with the gemma4-spec
patch whatever the installed tree carries: when the tree differs from the
request, a ``sitecustomize`` shim on the server's PYTHONPATH rewrites
``gemma4_causal`` (the captured copy, the copied input embeddings) and
``gemma4_mm`` (the per-image bidirectional mask, which the patch applies only
when the config sets ``use_bidirectional_attention="vision"``, so E4B image
tokens are causal) and sets ``Gemma4SGLangProcessor.gpu_image_decode`` at
import time, in every process the server spawns. Nothing on disk changes.
``installed`` runs the tree as is.
Image rows are restricted to RGB images, because the stock tree crashes on a
grayscale JPEG (that is gemma4_serving_probe.py's job).

One GPU, servers one at a time, then HF:

    python scripts/gates/gemma4_capture_parity.py --output-dir results/gemma4_probe/parity

writes ``report.json`` and ``summary.md`` there. CPU checks: ``--dry-run``
selects the rows and prints the server commands; ``--self-check`` builds each
run's captures from the HF forward instead of SGLang (``patched`` = layer
outputs, ``stock`` = mid-block residuals, fa3 = sdpa plus noise on the image
features) and runs the whole analysis, e.g. on a tiny random checkpoint.
``--write-sglang-shim DIR`` only writes the shim (scripts/gates/gemma4_probe_hpc.sh
uses it to emulate the patch for a serving server).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]

DEFAULT_TARGET = (
    "/scratch/fengsicheng/.cache/huggingface/hub/models--google--gemma-4-E4B-it/"
    "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
)
DEFAULT_DATA = (
    "/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/regen_data/"
    "gemma4-E4B-it_llava-ov15-1M-prompted_regen_first_turn.jsonl"
)
#: EAGLE3 taps 1/20/38 and DFlash/MMFlash taps 1/10/20/30/39.
DEFAULT_TAPS = (1, 10, 20, 30, 38, 39)
DEFAULT_RUNS = "stock:sdpa,patched:sdpa,patched:fa3"
MASK_VARIANTS = ("causal", "bidir", "sglang_tiled")
REGIONS = ("text", "image_prefix", "image_tokens", "image_post")

#: The gemma4-spec patch (patches/sglang/v0.5.14/gemma4-spec.patch) and the one
#: string per patched file that only its patched version contains.
CAPTURE_FILE = "sglang/srt/models/gemma4_causal.py"
MASK_FILE = "sglang/srt/models/gemma4_mm.py"
DECODE_FILE = "sglang/srt/multimodal/processors/gemma4.py"
CAPTURE_SENTINEL = "aux_hidden_states.append(hidden_states.clone())"
MASK_SENTINEL = "and vision_bidirectional"
DECODE_SENTINEL = "gpu_image_decode = False"
SHIM_MODE_ENV = "SPECFORGE_GEMMA4_SGLANG_MODE"
SHIM_LOG_TAG = "[gemma4-sglang-shim]"

# Written as sitecustomize.py into a directory that is put first on the
# server's PYTHONPATH. Python imports sitecustomize at startup, so the finder
# is in place before SGLang imports its models -- also in the scheduler, which
# SGLang starts with the spawn method (a fresh interpreter, same environment).
_SHIM_SOURCE = r'''"""Gemma 4 SGLang behaviour switch (scripts/gates/gemma4_capture_parity.py).

SPECFORGE_GEMMA4_SGLANG_MODE=stock loads gemma4_causal, gemma4_mm and the
Gemma 4 image processor as the stock tree has them; =patched as the gemma4-spec
patch has them (patches/sglang/v0.5.14/gemma4-spec.patch; on an unpatched tree
this also copies the capture taken after the layer loop, which changes
nothing). The
rewrite happens on the module source at import time; the installed files are
not modified. Unset or any other value: no effect.
"""

import importlib.abc
import importlib.machinery
import os
import sys

_MODE = os.environ.get("SPECFORGE_GEMMA4_SGLANG_MODE", "")
_CAUSAL = "sglang.srt.models.gemma4_causal"
_MM = "sglang.srt.models.gemma4_mm"
_PROCESSOR = "sglang.srt.multimodal.processors.gemma4"
_PATCHED_CAPTURE = "aux_hidden_states.append(hidden_states.clone())"
_STOCK_CAPTURE = "aux_hidden_states.append(hidden_states)"
_PATCHED_EMBEDS = (
    "hidden_states = input_embeds.clone() if input_ids is None else input_embeds\n"
)
_STOCK_EMBEDS = "hidden_states = input_embeds\n"
# the image-mask condition of Gemma4ForConditionalGeneration.forward
_PATCHED_MASK_GUARD = "            and vision_bidirectional\n"
_STOCK_MASK_CONDITION = "            and forward_batch.contains_image_inputs()\n        ):\n"
_PATCHED_MASK_CONDITION = (
    "            and forward_batch.contains_image_inputs()\n"
    "            and getattr(self.config.text_config, \"use_bidirectional_attention\", None)"
    " == \"vision\"\n"
    "        ):\n"
)


def _say(message):
    sys.stderr.write(
        "[gemma4-sglang-shim] pid=%d mode=%s %s\n" % (os.getpid(), _MODE, message)
    )
    sys.stderr.flush()


def _rewrite(source):
    if _MODE == "stock":
        pairs = ((_PATCHED_CAPTURE, _STOCK_CAPTURE), (_PATCHED_EMBEDS, _STOCK_EMBEDS))
    else:
        pairs = ((_STOCK_CAPTURE, _PATCHED_CAPTURE), (_STOCK_EMBEDS, _PATCHED_EMBEDS))
    counts = []
    for old, new in pairs:
        counts.append(source.count(old))
        source = source.replace(old, new)
    return source, counts


class _RewritingLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode("utf-8")
        source, (captures, embeds) = _rewrite(source)
        _say(
            "%s: rewrote %d capture line(s), %d input-embedding line(s); "
            "captures cloned=%s" % (fullname, captures, embeds, _PATCHED_CAPTURE in source)
        )
        return compile(source, self.path, "exec", dont_inherit=True)


class _MaskLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        source = self.get_data(self.path).decode("utf-8")
        if _MODE == "stock":
            count = source.count(_PATCHED_MASK_GUARD)
            source = source.replace(_PATCHED_MASK_GUARD, "")
        elif _PATCHED_MASK_GUARD in source:
            count = 0
        else:
            count = source.count(_STOCK_MASK_CONDITION)
            source = source.replace(_STOCK_MASK_CONDITION, _PATCHED_MASK_CONDITION)
        _say(
            "%s: rewrote %d image-mask condition(s); mask gated on the config=%s"
            % (fullname, count, _MODE != "stock")
        )
        return compile(source, self.path, "exec", dont_inherit=True)


class _ProcessorLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        super().exec_module(module)
        processor = getattr(module, "Gemma4SGLangProcessor", None)
        if processor is not None:
            processor.gpu_image_decode = _MODE == "stock"
            _say("%s: gpu_image_decode=%s" % (module.__name__, processor.gpu_image_decode))


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (_CAUSAL, _MM, _PROCESSOR):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return spec
        loader = {_CAUSAL: _RewritingLoader, _MM: _MaskLoader}.get(fullname, _ProcessorLoader)
        spec.loader = loader(fullname, spec.origin)
        return spec


def _chain_next_sitecustomize():
    # this file shadows any sitecustomize later on sys.path; run that one too
    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if not entry or os.path.abspath(entry) == here:
            continue
        candidate = os.path.join(entry, "sitecustomize.py")
        if os.path.isfile(candidate):
            with open(candidate, encoding="utf-8") as handle:
                code = compile(handle.read(), candidate, "exec")
            exec(code, {"__name__": "sitecustomize", "__file__": candidate})
            return


if _MODE in ("stock", "patched"):
    sys.meta_path.insert(0, _Finder())
_chain_next_sitecustomize()
'''


class RunSpec(NamedTuple):
    """One server configuration: the gemma4-spec behaviour and the vision backend."""

    mode: str  # stock | patched | installed
    mm_backend: str

    @property
    def name(self) -> str:
        return f"{self.mode}:{self.mm_backend}"


# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--target-model-path", default=DEFAULT_TARGET)
    parser.add_argument("--data-path", default=DEFAULT_DATA, help="regen jsonl to sample rows from")
    parser.add_argument("--output-dir", help="where report.json, summary.md and logs/ go")
    parser.add_argument("--num-image-rows", type=int, default=8)
    parser.add_argument("--num-text-rows", type=int, default=8)
    parser.add_argument(
        "--row-stride",
        type=int,
        default=20011,
        help="look at every Nth line only; the regen's first lines are all one subset",
    )
    parser.add_argument(
        "--max-rows-per-subset",
        type=int,
        default=2,
        help="rows per id prefix (the source subset) before the quota is filled from any",
    )
    parser.add_argument("--max-length", type=int, default=4096, help="training data.max_length")
    parser.add_argument("--taps", default=",".join(str(t) for t in DEFAULT_TAPS))
    parser.add_argument(
        "--runs",
        default=DEFAULT_RUNS,
        help="comma list of <stock|patched|installed>:<mm attention backend>",
    )
    parser.add_argument("--hf-masks", default=",".join(MASK_VARIANTS), help="HF mask variants for image rows")
    parser.add_argument(
        "--sglang-block-m",
        default="auto",
        help="BLOCK_M of SGLang's triton extend kernel for 'sglang_tiled', e.g. "
        "sliding=64,full=32; auto asks SGLang (needs the GPU)",
    )
    parser.add_argument("--port", type=int, default=31990)
    parser.add_argument("--mooncake-rpc-port", type=int, default=46551)
    parser.add_argument("--mooncake-http-port", type=int, default=46880)
    parser.add_argument("--mooncake-metrics-port", type=int, default=46903)
    parser.add_argument("--mem-fraction-static", type=float, default=0.40)
    parser.add_argument("--server-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--server-extra-args", default="", help="appended to every capture server's argv")
    parser.add_argument("--hf-device", default="cuda")
    parser.add_argument("--hf-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--save-tensors", action="store_true", help="also write captures/<run>.pt")
    parser.add_argument("--dry-run", action="store_true", help="select rows, print the plan, start nothing")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="no SGLang: build every run's captures from the HF forward (CPU-friendly)",
    )
    parser.add_argument("--write-sglang-shim", metavar="DIR", help="write DIR/sitecustomize.py and exit")
    args = parser.parse_args(argv)
    if not args.write_sglang_shim and not args.output_dir:
        parser.error("--output-dir is required")
    return args


def parse_taps(text: str) -> List[int]:
    taps = sorted({int(value) for value in text.replace(" ", "").split(",") if value})
    if not taps or taps[0] < 0:
        raise SystemExit(f"--taps must be non-negative layer ids, got {text!r}")
    return taps


def parse_runs(text: str) -> List[RunSpec]:
    runs = []
    for item in text.replace(" ", "").split(","):
        if not item:
            continue
        mode, _, backend = item.partition(":")
        if mode not in ("stock", "patched", "installed") or not backend:
            raise SystemExit(f"--runs entries are <stock|patched|installed>:<backend>, got {item!r}")
        runs.append(RunSpec(mode, backend))
    if len({run.name for run in runs}) != len(runs):
        raise SystemExit(f"--runs repeats an entry: {text!r}")
    return runs


def parse_block_m(text: str) -> Optional[Dict[str, int]]:
    if text == "auto":
        return None
    values = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        if key not in ("sliding", "full"):
            raise SystemExit(f"--sglang-block-m takes sliding=N,full=N, got {text!r}")
        values[key] = int(value)
    if set(values) != {"sliding", "full"}:
        raise SystemExit(f"--sglang-block-m needs both sliding and full, got {text!r}")
    return values


def write_sglang_shim(directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "sitecustomize.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_SHIM_SOURCE)
    return path


# ---------------------------------------------------------------------------
# installed SGLang
# ---------------------------------------------------------------------------
def installed_patch_state() -> Dict[str, Any]:
    """Which gemma4-spec sentinels the importable sglang carries (no import)."""
    spec = importlib.util.find_spec("sglang")
    state: Dict[str, Any] = {
        "sglang_root": None,
        "capture_patch": False,
        "mask_patch": False,
        "decode_patch": False,
    }
    if spec is None or not spec.origin:
        return state
    root = Path(spec.origin).resolve().parent.parent
    state["sglang_root"] = str(root)
    for key, relative, sentinel in (
        ("capture_patch", CAPTURE_FILE, CAPTURE_SENTINEL),
        ("mask_patch", MASK_FILE, MASK_SENTINEL),
        ("decode_patch", DECODE_FILE, DECODE_SENTINEL),
    ):
        try:
            state[key] = sentinel in (root / relative).read_text(encoding="utf-8")
        except OSError:
            state[key] = False
    try:
        from importlib.metadata import version

        state["sglang_version"] = version("sglang")
    except Exception:  # pragma: no cover - metadata missing
        state["sglang_version"] = None
    return state


def shim_for(run: RunSpec, installed: Dict[str, Any]) -> Optional[str]:
    """The shim mode a run needs on this tree, or None when the tree already matches."""
    if run.mode == "installed":
        return None
    parts = [installed[key] for key in ("capture_patch", "mask_patch", "decode_patch")]
    patched = all(parts)
    unpatched = not any(parts)
    if run.mode == "patched":
        return None if patched else "patched"
    return None if unpatched else "stock"


def resolve_block_m(text: str, sliding_head_dim: int, full_head_dim: int) -> Tuple[Optional[Dict[str, int]], str]:
    explicit = parse_block_m(text)
    if explicit is not None:
        return explicit, "--sglang-block-m"
    import torch

    if not torch.cuda.is_available():  # the sizes depend on the GPU generation
        return None, "unavailable without a GPU; pass --sglang-block-m"
    try:
        from sglang.srt.layers.attention.triton_ops.extend_attention import (
            _get_block_sizes_for_extend_attention,
        )

        return (
            {
                "sliding": int(_get_block_sizes_for_extend_attention(sliding_head_dim, sliding_head_dim)[3]),
                "full": int(_get_block_sizes_for_extend_attention(full_head_dim, full_head_dim)[3]),
            },
            "sglang _get_block_sizes_for_extend_attention",
        )
    except Exception as exc:  # no GPU, or the helper moved
        return None, f"unavailable ({type(exc).__name__}: {exc})"


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------
def image_runs(input_ids: Sequence[int], image_token_id: int) -> List[Tuple[int, int]]:
    """Inclusive (start, end) of every run of image soft tokens."""
    runs: List[Tuple[int, int]] = []
    start = None
    for index, token in enumerate(input_ids):
        if token == image_token_id:
            if start is None:
                start = index
        elif start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(input_ids) - 1))
    return runs


def _image_mode(path: str) -> Optional[str]:
    from PIL import Image

    try:
        with Image.open(path) as image:
            return image.mode
    except (OSError, ValueError):
        return None


def select_rows(args, processor, header_ids, end_ids, image_token_id) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Every ``--row-stride``-th line, until the image and text quotas are met.

    At most ``--max-rows-per-subset`` rows come from one source subset (the id
    prefix) while other subsets can still fill the quota. Rows are encoded with
    the trainer's own ``encode_mm_record``, so their ids, loss mask and length
    limit are exactly what capture sends. Image rows must decode as RGB (see
    the module docstring).
    """
    from specforge.data.mm_preprocessing import encode_mm_record

    want = {"image": args.num_image_rows, "text": args.num_text_rows}
    have: Counter = Counter()
    per_subset: Counter = Counter()
    skipped: Counter = Counter()
    rows: List[Dict[str, Any]] = []
    seen_lines = set()
    stride = max(1, args.row_stride)
    offsets = (0, stride // 2, stride // 4, (3 * stride) // 4)
    passes = [(offset, args.max_rows_per_subset) for offset in offsets]
    passes += [(offset, None) for offset in offsets]
    for offset, subset_cap in passes:
        if all(have[kind] >= want[kind] for kind in want):
            break
        with open(args.data_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                if line_number % stride != offset or line_number in seen_lines:
                    continue
                if all(have[kind] >= want[kind] for kind in want):
                    break
                record = json.loads(line)
                subset = str(record.get("id", "")).split("#", 1)[0]
                if subset_cap is not None and per_subset[subset] >= subset_cap:
                    continue
                seen_lines.add(line_number)
                image = record.get("image")
                if isinstance(image, (list, tuple)):
                    skipped["multi_image"] += 1
                    continue
                kind = "image" if image else "text"
                if have[kind] >= want[kind]:
                    continue
                if kind == "image":
                    mode = _image_mode(image)
                    if mode != "RGB":
                        skipped[f"image_mode_{mode}"] += 1
                        continue
                try:
                    payload = encode_mm_record(
                        record,
                        processor,
                        image_root="",
                        max_length=args.max_length,
                        train_only_last_turn=False,
                        header_ids=header_ids,
                        end_ids=end_ids,
                    )
                except (OSError, ValueError):
                    skipped["encode_error"] += 1
                    continue
                if payload is None:
                    skipped["too_long_or_no_loss"] += 1
                    continue
                runs = image_runs(payload["input_ids"], image_token_id)
                if kind == "image" and not runs:
                    skipped["image_without_image_tokens"] += 1
                    continue
                rows.append(
                    {
                        "index": len(rows),
                        "line": line_number,
                        "id": record.get("id"),
                        "kind": kind,
                        "image": payload["image"],
                        "input_ids": payload["input_ids"],
                        "loss_mask": payload["loss_mask"],
                        "image_runs": runs,
                        "conversations": record["conversations"],
                    }
                )
                have[kind] += 1
                per_subset[subset] += 1
    return rows, dict(skipped)


def region_masks(row: Dict[str, Any]) -> Dict[str, Any]:
    """Boolean position masks of the regions that exist in this row."""
    import torch

    length = len(row["input_ids"])
    if row["kind"] == "text":
        return {"text": torch.ones(length, dtype=torch.bool)}
    first = row["image_runs"][0][0]
    last = row["image_runs"][-1][1]
    image = torch.zeros(length, dtype=torch.bool)
    for start, end in row["image_runs"]:
        image[start : end + 1] = True
    positions = torch.arange(length)
    return {
        "image_prefix": positions < first,
        "image_tokens": image,
        "image_post": positions > last,
    }


# ---------------------------------------------------------------------------
# processes
# ---------------------------------------------------------------------------
def _http_ok(url: str, timeout: float = 5.0, any_status: bool = False) -> bool:
    import requests

    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException:
        return False
    return any_status or response.status_code == 200


def _tcp_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _wait_for(predicate, timeout: float, process=None, interval: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(interval)
    return False


def _start(argv: Sequence[str], env: Dict[str, str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            list(argv), stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    finally:
        log.close()  # the child holds its own descriptor
    return process


def _stop(process: Optional[subprocess.Popen], grace: float = 60.0) -> None:
    """Terminate a process group started by ``_start`` (SGLang spawns children)."""
    if process is None:
        return
    for sig, wait_s in ((signal.SIGTERM, grace), (signal.SIGKILL, 30.0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait_s)
            break
        except subprocess.TimeoutExpired:
            continue
    try:  # descendants of a leader that already exited
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - unkillable
        pass


def mooncake_env(args) -> Dict[str, str]:
    return {
        "MOONCAKE_MASTER_SERVER_ADDR": f"127.0.0.1:{args.mooncake_rpc_port}",
        "MOONCAKE_METADATA_SERVER": f"http://127.0.0.1:{args.mooncake_http_port}/metadata",
        "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
        "MOONCAKE_PROTOCOL": "tcp",
    }


def start_mooncake(args, log_dir: Path) -> subprocess.Popen:
    binary = shutil.which("mooncake_master")
    if binary is None:
        raise RuntimeError("mooncake_master is not on PATH (needed for spec capture)")
    for port in (args.mooncake_rpc_port, args.mooncake_http_port, args.mooncake_metrics_port):
        if _tcp_open(port):
            raise RuntimeError(f"port {port} is already in use; pick other --mooncake-*-port values")
    process = _start(
        [
            binary,
            "--enable_http_metadata_server=true",
            "--http_metadata_server_host=127.0.0.1",
            f"--rpc_port={args.mooncake_rpc_port}",
            f"--http_metadata_server_port={args.mooncake_http_port}",
            f"--metrics_port={args.mooncake_metrics_port}",
        ],
        {**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        log_dir / "mooncake.log",
    )
    health = f"http://127.0.0.1:{args.mooncake_http_port}/metadata?key=specforge-health-check"
    ready = _wait_for(
        lambda: _tcp_open(args.mooncake_rpc_port) and _http_ok(health, any_status=True),
        timeout=120,
        process=process,
        interval=1.0,
    )
    if not ready:
        _stop(process)
        raise RuntimeError(f"mooncake_master did not come up; see {log_dir / 'mooncake.log'}")
    return process


def server_argv(args, run: RunSpec, taps: Sequence[int]) -> List[str]:
    """The managed capture server's argv (specforge/launch_plan.py) for one run."""
    argv = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.target_model_path,
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--skip-tokenizer-init",
        "--tp-size",
        "1",
        "--chunked-prefill-size",
        "-1",
        "--enable-spec-capture",
        "--spec-capture-method",
        "dflash",
        "--spec-capture-aux-layer-ids",
        *[str(tap) for tap in taps],
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--context-length",
        str(args.max_length + 7),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--attention-backend",
        "triton",
        "--mm-attention-backend",
        run.mm_backend,
    ]
    return argv + shlex.split(args.server_extra_args)


def server_env(args, shim: Optional[str], shim_dir: Path) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(mooncake_env(args))
    env.update(
        {
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": str(4 << 30),
            "MOONCAKE_LOCAL_BUFFER_SIZE": str(1 << 30),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop(SHIM_MODE_ENV, None)
    if shim is not None:
        env[SHIM_MODE_ENV] = shim
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(shim_dir), env.get("PYTHONPATH", "")) if part
        )
    return env


def _shim_lines(log_path: Path) -> List[str]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return sorted({line.split(" pid=", 1)[-1] for line in text.splitlines() if SHIM_LOG_TAG in line})


def _eagle3_image_schema():
    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.inference.adapters.server_capture import ServerCaptureSchema

    layout = builtin_algorithm_registry().resolve("eagle3").providers.server_streaming_for("image").layout
    return ServerCaptureSchema(
        aux_feature=layout.aux_feature,
        last_hidden_feature=layout.last_hidden_feature,
        passthrough=layout.passthrough,
        attention_mask_feature=layout.attention_mask_feature,
    )


def capture_run(args, run: RunSpec, shim: Optional[str], rows, taps, hidden_size, out_dir: Path, shim_dir: Path):
    """Serve the target in one configuration and capture every row through Mooncake."""
    from specforge.algorithms.common.mm_server_input import ImageServerInputAdapter
    from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
    from specforge.inference.capture import CaptureConfig
    from specforge.runtime.contracts import PromptTask, SampleRef
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    safe = run.name.replace(":", "-")
    log_path = out_dir / "logs" / f"server-{safe}.log"
    info: Dict[str, Any] = {
        "mode": run.mode,
        "mm_backend": run.mm_backend,
        "shim": shim,
        "server_log": str(log_path),
        "argv": server_argv(args, run, taps),
        "status": "failed",
        "errors": [],
    }
    captures: Dict[int, Dict[str, Any]] = {}
    if _tcp_open(args.port):
        info["errors"].append(f"port {args.port} is already in use")
        return info, captures
    started = time.monotonic()
    process = _start(info["argv"], server_env(args, shim, shim_dir), log_path)
    try:
        url = f"http://127.0.0.1:{args.port}"
        if not _wait_for(lambda: _http_ok(f"{url}/health"), args.server_timeout, process):
            info["errors"].append(f"server did not become healthy (exit code {process.poll()}); see {log_path}")
            return info, captures
        info["startup_s"] = round(time.monotonic() - started, 1)
        store_id = f"g4parity-{os.getpid()}-{safe}"
        try:
            store = MooncakeFeatureStore(
                store_id=store_id,
                setup_kwargs={
                    "local_hostname": "127.0.0.1",
                    "metadata_server": mooncake_env(args)["MOONCAKE_METADATA_SERVER"],
                    "master_server_addr": mooncake_env(args)["MOONCAKE_MASTER_SERVER_ADDR"],
                    "global_segment_size": 0,  # server-owned objects, as for training consumers
                    "local_buffer_size": 512 << 20,
                    "protocol": "tcp",
                    "rdma_devices": "",
                },
            )
        except Exception as exc:  # the other runs may still work
            info["errors"].append(f"mooncake client: {type(exc).__name__}: {exc}")
            return info, captures
        adapter = SGLangServerCaptureAdapter(
            url,
            store,
            run_id=store_id,
            algorithm="eagle3",
            schema=_eagle3_image_schema(),
            request_input_adapter=ImageServerInputAdapter(None),
            timeout_s=args.request_timeout,
        )
        contract = CaptureConfig.from_strategy(
            required_features={"input_ids", "attention_mask", "loss_mask", "hidden_state", "target"},
            aux_hidden_state_layer_ids=tuple(taps),
            target_repr="hidden_state",
            target_hidden_size=hidden_size,
        )
        capture_started = time.monotonic()
        for row in rows:
            task = PromptTask(
                task_id=f"row{row['index']}",
                run_id=store_id,
                source_id="gemma4-capture-parity",
                payload={"input_ids": row["input_ids"], "loss_mask": row["loss_mask"], "image": row["image"]},
                max_length=len(row["input_ids"]),
            )
            try:
                (ref,) = adapter.produce_refs([task], capture=contract)
            except Exception as exc:  # one row must not end the run
                info["errors"].append(f"row {row['index']} ({row['id']}): {type(exc).__name__}: {exc}")
                if process.poll() is not None:
                    info["errors"].append(f"server exited with {process.poll()}; see {log_path}")
                    break
                continue
            if not isinstance(ref, SampleRef):
                info["errors"].append(f"row {row['index']} ({row['id']}): {ref.reason}")
                continue
            try:
                out, handle = store.get(ref)
            except Exception as exc:
                info["errors"].append(f"row {row['index']} fetch: {type(exc).__name__}: {exc}")
                continue
            if out["input_ids"][0].tolist() != row["input_ids"]:
                info["errors"].append(f"row {row['index']}: captured input_ids differ from the request")
            captures[row["index"]] = {
                "aux": out["hidden_state"][0].clone(),
                "last": out["target"][0].clone(),
            }
            store.release(handle)
        info["capture_s"] = round(time.monotonic() - capture_started, 1)
        try:
            store.drain_pending_removals(max_attempts=60, retry_interval_s=0.5)
        except RuntimeError as exc:  # leftovers only cost Mooncake memory
            info["errors"].append(f"drain: {exc}")
        info["status"] = "ok" if len(captures) == len(rows) else "partial"
    finally:
        _stop(process)
        time.sleep(5)  # let the CUDA contexts go before the next server starts
        info["shim_log"] = _shim_lines(log_path)
    info["captured_rows"] = len(captures)
    return info, captures


# ---------------------------------------------------------------------------
# HF reference
# ---------------------------------------------------------------------------
def hf_inputs(processor, row: Dict[str, Any]):
    """The processor batch for a row, rendered exactly as ``encode_mm_record`` does."""
    from PIL import Image

    from specforge.data.mm_preprocessing import to_chat_messages

    text = processor.apply_chat_template(
        to_chat_messages(row["conversations"]), tokenize=False, add_generation_prompt=False
    )
    if row["image"] is None:
        return processor(text=[text], return_tensors="pt")
    with Image.open(row["image"]) as handle:
        image = handle.convert("RGB")
    return processor(text=[text], images=[image], return_tensors="pt")


def build_masks(length: int, runs, variant: str, window: int, block_m, dtype, device):
    """Additive per-layer-type masks for HF (``attention_mask`` as a dict).

    ``bidir``: causal, plus every image run attends to itself both ways, on the
    full and the sliding layers alike, which is the mask SGLang's Gemma 4 hands
    to the triton backend. ``sglang_tiled``: the same mask as that backend's
    extend kernel applies it -- its causal loop over key tiles stops at the end
    of the query's own BLOCK_M tile, so a query sees later image keys only
    inside that tile. Sliding layers additionally keep the window.
    """
    import torch

    positions = torch.arange(length, device=device)
    query = positions[:, None]
    key = positions[None, :]
    causal = key <= query
    group = torch.full((length,), -1, dtype=torch.long, device=device)
    for index, (start, end) in enumerate(runs):
        group[start : end + 1] = index
    same_image = (group[:, None] == group[None, :]) & (group[:, None] >= 0)
    masks = {}
    for layer_type, tile_key in (("full_attention", "full"), ("sliding_attention", "sliding")):
        if variant == "bidir":
            allowed = causal | same_image
        elif variant == "sglang_tiled":
            tile = int(block_m[tile_key])
            allowed = causal | (same_image & (key < (query // tile + 1) * tile))
        else:
            raise ValueError(f"no custom mask for variant {variant!r}")
        if layer_type == "sliding_attention":
            allowed = allowed & ((query - key) < window)
        mask = torch.zeros((length, length), dtype=dtype, device=device)
        masks[layer_type] = mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]
    return masks


def hf_forward(model, inputs, row, variant, taps, text_config, block_m, device):
    """One HF forward: tap outputs, next-layer mid-block residuals, last hidden, logits."""
    import torch

    language_model = model.model.language_model
    num_layers = int(text_config.num_hidden_layers)
    grabbed: Dict[str, Any] = {}
    hooks = []
    for tap in taps:
        if tap + 1 < num_layers:
            def _midblock(_module, hook_args, tap=tap):
                grabbed[f"mid{tap}"] = hook_args[0].detach()

            hooks.append(language_model.layers[tap + 1].pre_feedforward_layernorm.register_forward_pre_hook(_midblock))

    def _prenorm(_module, hook_args):
        grabbed["prenorm"] = hook_args[0].detach()

    hooks.append(language_model.norm.register_forward_pre_hook(_prenorm))
    kwargs = {key: value.to(device) for key, value in inputs.items() if key != "attention_mask"}
    if variant == "causal":
        kwargs["attention_mask"] = inputs["attention_mask"].to(device)
    else:
        kwargs["attention_mask"] = build_masks(
            len(row["input_ids"]),
            row["image_runs"],
            variant,
            int(text_config.sliding_window),
            block_m,
            model.dtype,
            device,
        )
    try:
        with torch.no_grad():
            out = model(**kwargs, output_hidden_states=True, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    hidden = out.hidden_states
    layer, mid = [], []
    for tap in taps:
        # hidden_states[k + 1] is the output of layer k, except that the last
        # entry is post-norm; SGLang captures the last layer before the norm.
        output = hidden[tap + 1] if tap + 1 < num_layers else grabbed["prenorm"]
        layer.append(output[0])
        mid.append(grabbed.get(f"mid{tap}", output)[0])
    return {
        "layer": torch.stack(layer, dim=1),
        "mid": torch.stack(mid, dim=1),
        "last": hidden[-1][0],
        "logits": out.logits[0],
    }


def load_hf(args):
    import torch
    from transformers import AutoModelForImageTextToText

    dtype = getattr(torch, args.hf_dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.target_model_path, dtype=dtype, attn_implementation="sdpa"
    )
    return model.to(args.hf_device).eval()


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
class Series:
    """Per-position values collected under nested keys, summarized at the end."""

    def __init__(self) -> None:
        self.values: Dict[Tuple, List[Any]] = defaultdict(list)

    def add(self, key: Tuple, values) -> None:
        if values.numel():
            self.values[key].append(values.detach().float().cpu().flatten())

    def summary(self, key: Tuple) -> Optional[Dict[str, float]]:
        import torch

        if key not in self.values:
            return None
        data = torch.cat(self.values[key])
        return {
            "mean": float(data.mean()),
            "p05": float(torch.quantile(data, 0.05)) if data.numel() > 1 else float(data[0]),
            "p95": float(torch.quantile(data, 0.95)) if data.numel() > 1 else float(data[0]),
            "min": float(data.min()),
            "max": float(data.max()),
            "n": int(data.numel()),
        }

    def keys(self):
        return list(self.values)


def _rows_cos(a, b):
    import torch

    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1, eps=1e-12)


def _rows_rel_err(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)


def _rows_rms(a):
    return a.float().pow(2).mean(dim=-1).sqrt()


def softcap(logits, cap: Optional[float]):
    import torch

    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


def head_logits(hidden, weight, cap, chunk: int = 256):
    """``softcap(hidden @ weight.T)`` in fp32, chunked over positions; yields (start, raw, capped).

    ``weight`` is the fp32 tied embedding (converted once by the caller).
    """
    for start in range(0, hidden.shape[0], chunk):
        raw = hidden[start : start + chunk].float() @ weight.T
        yield start, raw, softcap(raw, cap)


def analyse_row(row, refs, captures, runs, taps, weight, cap, stats: Dict[str, Series], device) -> None:
    """Compare every run's capture of one row with the HF references (``refs[variant]``)."""
    import torch

    regions = {name: mask.to(device) for name, mask in region_masks(row).items()}
    loss = torch.tensor(row["loss_mask"], dtype=torch.bool, device=device)
    variants = list(refs)
    num_taps = len(taps)

    for run in runs:
        capture = captures.get(run.name, {}).get(row["index"])
        if capture is None:
            continue
        aux = capture["aux"].to(device).float().view(len(row["input_ids"]), num_taps, -1)
        last = capture["last"].to(device).float()
        for variant in variants:
            ref = refs[variant]
            for region, mask in regions.items():
                if not bool(mask.any()):
                    continue
                for index, tap in enumerate(taps):
                    for ref_name, ref_tensor in (("layer_output", ref["layer"]), ("next_midblock", ref["mid"])):
                        a = aux[mask, index]
                        b = ref_tensor[mask, index]
                        stats["cos"].add((run.name, region, variant, ref_name, tap), _rows_cos(a, b))
                        stats["rel_err"].add((run.name, region, variant, ref_name, tap), _rows_rel_err(a, b))
                stats["cos"].add(
                    (run.name, region, variant, "layer_output", "last"),
                    _rows_cos(last[mask], ref["last"][mask]),
                )
                if variant == variants[0]:
                    for index, tap in enumerate(taps):
                        stats["rms"].add((run.name, region, tap), _rows_rms(aux[mask, index]))
            # top-1 agreement of the captured head with HF's (softcapped) logits
            hf_top1 = ref["logits"].argmax(dim=-1)
            agree = torch.empty(len(row["input_ids"]), dtype=torch.bool, device=device)
            for start, _raw, capped in head_logits(last, weight, cap):
                stop = start + capped.shape[0]
                agree[start:stop] = capped.argmax(dim=-1) == hf_top1[start:stop]
            for region, mask in regions.items():
                stats["top1"].add((run.name, region, variant, "all"), agree[mask])
                stats["top1"].add((run.name, region, variant, "loss"), agree[mask & loss])

    # HF alone: magnitudes and how much the softcap matters (first variant)
    ref = refs[variants[0]]
    for region, mask in regions.items():
        for index, tap in enumerate(taps):
            stats["rms"].add(("hf", region, tap), _rows_rms(ref["layer"][mask, index]))
    raw_max = torch.empty(len(row["input_ids"]), device=device)
    kl = torch.empty(len(row["input_ids"]), device=device)
    for start, raw, capped in head_logits(ref["last"], weight, cap):
        stop = start + raw.shape[0]
        raw_max[start:stop] = raw.max(dim=-1).values
        log_p = torch.log_softmax(capped, dim=-1)
        log_q = torch.log_softmax(raw, dim=-1)
        kl[start:stop] = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    for region, mask in regions.items():
        for subset, subset_mask in (("all", mask), ("loss", mask & loss)):
            stats["raw_max"].add((region, subset), raw_max[subset_mask])
        stats["kl_cap_raw"].add((region, "loss"), kl[mask & loss])

    # capture against capture: the vision backend, and stock against patched
    for position, first in enumerate(runs):
        for second in runs[position + 1 :]:
            same_mode = first.mode == second.mode and first.mm_backend != second.mm_backend
            same_backend = first.mm_backend == second.mm_backend and first.mode != second.mode
            if not (same_mode or same_backend):
                continue
            a = captures.get(first.name, {}).get(row["index"])
            b = captures.get(second.name, {}).get(row["index"])
            if a is None or b is None:
                continue
            pair = f"{first.name} vs {second.name}"
            a_aux = a["aux"].to(device).float().view(len(row["input_ids"]), num_taps, -1)
            b_aux = b["aux"].to(device).float().view(len(row["input_ids"]), num_taps, -1)
            for region, mask in regions.items():
                for index, tap in enumerate(taps):
                    stats["pair_cos"].add((pair, region, tap), _rows_cos(a_aux[mask, index], b_aux[mask, index]))
                    stats["pair_rel_err"].add(
                        (pair, region, tap), _rows_rel_err(a_aux[mask, index], b_aux[mask, index])
                    )


def simulate_captures(model, inputs_by_row, rows, runs, taps, text_config, block_m, device):
    """``--self-check``: captures built from HF instead of SGLang.

    patched = HF layer outputs, stock = next-layer mid-block residuals (what an
    overwritten capture holds); image rows use the mask that run's SGLang
    applies (stock: the SGLang-like per-image mask; patched: that mask only
    when the config sets use_bidirectional_attention="vision", else causal); a
    non-sdpa vision backend adds 1% noise to the image positions of every tap.
    """
    import torch

    generator = torch.Generator().manual_seed(0)
    captures: Dict[str, Dict[int, Dict[str, Any]]] = {run.name: {} for run in runs}
    vision_bidirectional = getattr(text_config, "use_bidirectional_attention", None) == "vision"
    for row in rows:
        refs: Dict[str, Dict[str, Any]] = {}
        for run in runs:
            variant = "causal"
            if row["kind"] == "image" and (run.mode == "stock" or vision_bidirectional):
                variant = "sglang_tiled" if block_m else "bidir"
            if variant not in refs:
                refs[variant] = hf_forward(
                    model, inputs_by_row[row["index"]], row, variant, taps, text_config, block_m, device
                )
            ref = refs[variant]
            source = ref["mid"] if run.mode == "stock" else ref["layer"]
            aux = source.detach().float().cpu().clone()
            if run.mm_backend != "sdpa" and row["kind"] == "image":
                image = region_masks(row)["image_tokens"]
                noise = torch.randn(aux[image].shape, generator=generator)
                aux[image] = aux[image] * (1 + 0.01 * noise)
            captures[run.name][row["index"]] = {
                "aux": aux.reshape(aux.shape[0], -1).to(torch.bfloat16),
                "last": ref["last"].detach().cpu().to(torch.bfloat16),
            }
    return captures


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _nest(series: Series, depth_keys: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in series.keys():
        node = out
        for part in key[:depth_keys - 1]:
            node = node.setdefault(str(part), {})
        node[str(key[depth_keys - 1])] = series.summary(key)
    return out


def build_checks(report: Dict[str, Any], runs: Sequence[RunSpec], taps: Sequence[int]) -> List[Dict[str, Any]]:
    """Pass/fail lines with the plan's thresholds (G1)."""
    checks = []
    parity = report["capture_parity"]["cos"]
    for run in runs:
        text = parity.get(run.name, {}).get("text", {}).get("causal", {})
        layer = text.get("layer_output", {})
        mid = text.get("next_midblock", {})
        per_tap = {str(tap): layer.get(str(tap), {}).get("mean") for tap in taps}
        if any(value is None for value in per_tap.values()):
            checks.append({"name": f"{run.name}: text-row capture parity", "passed": False, "detail": "no capture"})
            continue
        worst = min(per_tap.values())
        if run.mode in ("patched", "installed"):
            checks.append(
                {
                    "name": f"{run.name}: captured tap == HF hidden_states[k+1] on text rows (cosine)",
                    "value": worst,
                    "threshold": 0.99,
                    "passed": worst >= 0.99,
                    "detail": per_tap,
                }
            )
        mid_tap = {str(tap): mid.get(str(tap), {}).get("mean") for tap in taps}
        closer_to_mid = [
            str(tap)
            for tap in taps
            if mid_tap[str(tap)] is not None and mid_tap[str(tap)] > per_tap[str(tap)] + 1e-4
        ]
        checks.append(
            {
                "name": f"{run.name}: taps closer to the next layer's mid-block residual than to the layer output",
                "value": closer_to_mid,
                "informational": True,
                "detail": "stock SGLang is expected to list every tap below the last layer; patched none",
            }
        )
        top1 = report["logits"]["top1"].get(run.name, {}).get("text", {}).get("causal", {}).get("all")
        if top1 is not None and run.mode in ("patched", "installed"):
            checks.append(
                {
                    "name": f"{run.name}: top-1 of softcap(lm_head(last_hidden)) == HF on text rows",
                    "value": top1["mean"],
                    "threshold": 0.98,
                    "passed": top1["mean"] >= 0.98,
                }
            )
    return checks


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_summary(report: Dict[str, Any], runs: Sequence[RunSpec], taps: Sequence[int]) -> str:
    lines = ["# Gemma 4 capture parity", ""]
    meta = report["meta"]
    lines.append(
        f"target `{meta['target_model_path']}`; {meta['rows']['image']} image + {meta['rows']['text']} "
        f"text rows; taps {list(taps)}; installed sglang {meta['installed'].get('sglang_version')} "
        f"(capture patch {meta['installed']['capture_patch']}, mask patch {meta['installed'].get('mask_patch')}, "
        f"decode patch {meta['installed']['decode_patch']}); "
        f"sglang_tiled BLOCK_M {meta['block_m']} ({meta['block_m_source']})."
    )
    lines.append("")
    lines.append("## Runs")
    for run in runs:
        info = report["runs"].get(run.name, {})
        lines.append(
            f"- `{run.name}`: {info.get('status')} ({info.get('captured_rows', 0)} rows), shim "
            f"{info.get('shim') or 'none'}; " + ("; ".join(info.get("errors", [])[:3]) or "no errors")
        )
        for shim_line in info.get("shim_log", [])[:4]:
            lines.append(f"  - shim: {shim_line}")
    lines.append("")
    cos = report["capture_parity"]["cos"]
    for ref_name, title in (
        ("layer_output", "cosine vs HF hidden_states[k+1] (layer output)"),
        ("next_midblock", "cosine vs the next layer's residual before its MLP (overwritten capture)"),
    ):
        lines.append(f"## Text rows: {title}, mean / p05")
        header = "| run | " + " | ".join(f"tap {tap}" for tap in taps) + " | last hidden |"
        lines += [header, "|" + " --- |" * (len(taps) + 2)]
        for run in runs:
            node = cos.get(run.name, {}).get("text", {}).get("causal", {}).get(ref_name, {})
            cells = []
            for tap in taps:
                item = node.get(str(tap))
                cells.append("-" if item is None else f"{item['mean']:.4f} / {item['p05']:.4f}")
            last = cos.get(run.name, {}).get("text", {}).get("causal", {}).get("layer_output", {}).get("last")
            cells.append("-" if last is None or ref_name != "layer_output" else f"{last['mean']:.4f}")
            lines.append(f"| {run.name} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Image rows: mean cosine over taps vs HF layer output, per region and HF mask")
    lines += ["| run | region | " + " | ".join(MASK_VARIANTS) + " |", "| --- | --- |" + " --- |" * len(MASK_VARIANTS)]
    for run in runs:
        for region in REGIONS[1:]:
            cells = []
            for variant in MASK_VARIANTS:
                node = cos.get(run.name, {}).get(region, {}).get(variant, {}).get("layer_output", {})
                values = [node[str(tap)]["mean"] for tap in taps if node.get(str(tap))]
                cells.append(_fmt(sum(values) / len(values)) if values else "-")
            lines.append(f"| {run.name} | {region} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Per-tap RMS (mean over positions)")
    lines += [
        "| source | region | " + " | ".join(f"tap {tap}" for tap in taps) + " |",
        "| --- | --- |" + " --- |" * len(taps),
    ]
    rms = report["tap_rms"]
    for source in ["hf"] + [run.name for run in runs]:
        for region in REGIONS:
            node = rms.get(source, {}).get(region)
            if not node:
                continue
            cells = [_fmt((node.get(str(tap)) or {}).get("mean"), 3) for tap in taps]
            lines.append(f"| {source} | {region} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Logits")
    top1 = report["logits"]["top1"]
    for run in runs:
        for region in REGIONS:
            node = top1.get(run.name, {}).get(region, {})
            parts = []
            for variant, values in node.items():
                parts.append(
                    f"{variant}: {_fmt((values.get('all') or {}).get('mean'))} all / "
                    f"{_fmt((values.get('loss') or {}).get('mean'))} loss"
                )
            if parts:
                lines.append(f"- top-1 agreement `{run.name}` {region}: " + "; ".join(parts))
    raw = report["logits"]["raw_max"]
    for region in REGIONS:
        node = raw.get(region)
        if not node:
            continue
        for subset in ("all", "loss"):
            item = report["logits"]["raw_max_exceeds"].get(region, {}).get(subset)
            if item:
                lines.append(
                    f"- HF raw max logit, {region} {subset} positions: > 20 on {item['gt20']:.3f}, "
                    f"> 30 on {item['gt30']:.3f} (n={item['n']})"
                )
        kl = report["logits"]["kl_cap_raw"].get(region, {}).get("loss")
        if kl:
            lines.append(
                f"- KL(softcapped || raw) on {region} loss positions: mean {kl['mean']:.4f}, p95 {kl['p95']:.4f}, "
                f"max {kl['max']:.4f} nats"
            )
    lines.append("")
    pairs = report["capture_pairs"]["cos"]
    if pairs:
        lines.append("## Capture against capture (cosine, mean / p05)")
        lines += [
            "| pair | region | " + " | ".join(f"tap {tap}" for tap in taps) + " |",
            "| --- | --- |" + " --- |" * len(taps),
        ]
        for pair, regions in pairs.items():
            for region in REGIONS:
                node = regions.get(region)
                if not node:
                    continue
                cells = [
                    "-" if not node.get(str(tap)) else f"{node[str(tap)]['mean']:.4f} / {node[str(tap)]['p05']:.4f}"
                    for tap in taps
                ]
                lines.append(f"| {pair} | {region} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("## Checks")
    for check in report["checks"]:
        if check.get("informational"):
            lines.append(f"- INFO {check['name']}: {check['value']}")
        else:
            lines.append(
                f"- {'PASS' if check['passed'] else 'FAIL'} {check['name']}: "
                f"{_fmt(check.get('value'))} (threshold {check.get('threshold')})"
            )
    lines.append("")
    return "\n".join(lines)


def assemble_report(meta, rows, run_info, stats, runs, taps) -> Dict[str, Any]:
    raw_exceeds: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, values in stats["raw_max"].values.items():
        import torch

        data = torch.cat(values)
        raw_exceeds.setdefault(key[0], {})[key[1]] = {
            "gt20": float((data > 20).float().mean()),
            "gt30": float((data > 30).float().mean()),
            "n": int(data.numel()),
        }
    report = {
        "meta": meta,
        "rows": [
            {
                "index": row["index"],
                "line": row["line"],
                "id": row["id"],
                "kind": row["kind"],
                "image": row["image"],
                "n_tokens": len(row["input_ids"]),
                "n_loss": int(sum(row["loss_mask"])),
                "image_runs": row["image_runs"],
            }
            for row in rows
        ],
        "runs": run_info,
        "capture_parity": {
            "cos": _nest(stats["cos"], 5),
            "rel_err": _nest(stats["rel_err"], 5),
        },
        "tap_rms": _nest(stats["rms"], 3),
        "logits": {
            "top1": _nest(stats["top1"], 4),
            "raw_max": _nest(stats["raw_max"], 2),
            "raw_max_exceeds": raw_exceeds,
            "kl_cap_raw": _nest(stats["kl_cap_raw"], 2),
        },
        "capture_pairs": {
            "cos": _nest(stats["pair_cos"], 3),
            "rel_err": _nest(stats["pair_rel_err"], 3),
        },
    }
    report["checks"] = build_checks(report, runs, taps)
    return report


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.write_sglang_shim:
        print(write_sglang_shim(args.write_sglang_shim))
        return 0
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    import torch
    from transformers import AutoConfig

    from specforge.data.mm_preprocessing import _assistant_header_ids, _end_token_ids, load_mm_processor

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(exist_ok=True)
    taps = parse_taps(args.taps)
    runs = parse_runs(args.runs)
    variants = [name for name in args.hf_masks.replace(" ", "").split(",") if name]
    unknown = sorted(set(variants) - set(MASK_VARIANTS))
    if unknown or "causal" not in variants:
        raise SystemExit(f"--hf-masks takes {MASK_VARIANTS} and must include causal, got {variants}")

    config = AutoConfig.from_pretrained(args.target_model_path)
    text_config = config.get_text_config()
    num_layers = int(text_config.num_hidden_layers)
    if taps[-1] >= num_layers:
        raise SystemExit(f"tap {taps[-1]} does not exist: the target has {num_layers} layers")
    hidden_size = int(text_config.hidden_size)
    cap = getattr(text_config, "final_logit_softcapping", None) or None
    sliding_head_dim = int(text_config.head_dim)
    full_head_dim = int(getattr(text_config, "global_head_dim", None) or sliding_head_dim)

    processor = load_mm_processor(
        argparse.Namespace(
            model=argparse.Namespace(
                target_model_path=args.target_model_path,
                cache_dir=None,
                trust_remote_code=True,
                tokenizer_pad_token_id=None,
            ),
            data=argparse.Namespace(image_max_tokens=0),
        )
    )
    header_ids = _assistant_header_ids(processor)
    end_ids = _end_token_ids(processor)
    started = time.monotonic()
    rows, skipped = select_rows(args, processor, header_ids, end_ids, int(config.image_token_id))
    counts = Counter(row["kind"] for row in rows)
    print(
        f"[parity] {len(rows)} rows ({counts['image']} image, {counts['text']} text) from {args.data_path} "
        f"in {time.monotonic() - started:.0f}s; skipped {skipped}; end ids {sorted(end_ids)}",
        flush=True,
    )
    with open(out_dir / "rows.json", "w", encoding="utf-8") as handle:
        json.dump(rows, handle)

    installed = installed_patch_state()
    shims = {run.name: shim_for(run, installed) for run in runs}
    shim_dir = out_dir / "sglang_shim"
    block_m, block_m_source = resolve_block_m(args.sglang_block_m, sliding_head_dim, full_head_dim)
    if "sglang_tiled" in variants and block_m is None:
        print(f"[parity] sglang_tiled mask skipped: BLOCK_M {block_m_source}", flush=True)
        variants.remove("sglang_tiled")

    meta = {
        "target_model_path": args.target_model_path,
        "data_path": args.data_path,
        "taps": taps,
        "runs": [run.name for run in runs],
        "hf_masks": variants,
        "block_m": block_m,
        "block_m_source": block_m_source,
        "installed": installed,
        "shims": shims,
        "rows": {"image": counts["image"], "text": counts["text"], "skipped": skipped},
        "max_length": args.max_length,
        "self_check": args.self_check,
        "end_ids": sorted(int(value) for value in end_ids),
        "header_ids": list(header_ids),
        "softcap": cap,
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "host": platform.node(),
    }
    if args.dry_run:
        plan = {
            "meta": meta,
            "servers": {
                run.name: {
                    "argv": server_argv(args, run, taps),
                    "shim": shims[run.name],
                    "env": {key: value for key, value in server_env(args, shims[run.name], shim_dir).items()
                            if key.startswith(("MOONCAKE_", "SPECFORGE_")) or key == "PYTHONPATH"},
                }
                for run in runs
            },
        }
        with open(out_dir / "plan.json", "w", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
        for run in runs:
            print(f"[parity] {run.name}: shim={shims[run.name]}\n  {shlex.join(server_argv(args, run, taps))}")
        print(f"[parity] dry run: wrote {out_dir / 'plan.json'} and {out_dir / 'rows.json'}")
        return 0

    device = torch.device(args.hf_device)
    run_info: Dict[str, Any] = {}
    captures: Dict[str, Dict[int, Dict[str, Any]]] = {}
    inputs_by_row = {row["index"]: hf_inputs(processor, row) for row in rows}
    for row in rows:
        if inputs_by_row[row["index"]]["input_ids"][0].tolist() != row["input_ids"]:
            raise SystemExit(f"row {row['id']}: the HF processor batch does not reproduce encode_mm_record's ids")

    model = None
    if args.self_check:
        model = load_hf(args)
        captures = simulate_captures(model, inputs_by_row, rows, runs, taps, text_config, block_m, device)
        for run in runs:
            run_info[run.name] = {"mode": run.mode, "mm_backend": run.mm_backend, "shim": "self-check",
                                  "status": "ok", "captured_rows": len(captures[run.name]), "errors": []}
    else:
        if any(shims.values()):
            write_sglang_shim(str(shim_dir))
        mooncake = start_mooncake(args, out_dir / "logs")
        try:
            for run in runs:
                print(f"[parity] {run.name}: starting the capture server (shim {shims[run.name]})", flush=True)
                info, run_captures = capture_run(args, run, shims[run.name], rows, taps, hidden_size, out_dir, shim_dir)
                run_info[run.name] = info
                captures[run.name] = run_captures
                print(
                    f"[parity] {run.name}: {info['status']}, {len(run_captures)} rows; "
                    f"errors {info['errors'][:3]}",
                    flush=True,
                )
                if args.save_tensors and run_captures:
                    torch.save(run_captures, out_dir / f"captures-{run.name.replace(':', '-')}.pt")
        finally:
            _stop(mooncake)
        model = load_hf(args)

    # the tied lm_head (raw embedding table), in fp32 once: 2.7 GB for E4B
    weight = model.get_input_embeddings().weight.detach().float()
    names = ("cos", "rel_err", "rms", "top1", "raw_max", "kl_cap_raw", "pair_cos", "pair_rel_err")
    stats = {name: Series() for name in names}
    for row in rows:
        row_variants = ["causal"] if row["kind"] == "text" else variants
        refs = {
            variant: hf_forward(model, inputs_by_row[row["index"]], row, variant, taps, text_config, block_m, device)
            for variant in row_variants
        }
        analyse_row(row, refs, captures, runs, taps, weight, cap, stats, device)
        del refs
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = assemble_report(meta, rows, run_info, stats, runs, taps)
    with open(out_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    summary = render_summary(report, runs, taps)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"[parity] wrote {out_dir / 'report.json'} and {out_dir / 'summary.md'}")
    failed = [check["name"] for check in report["checks"] if check.get("passed") is False]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
