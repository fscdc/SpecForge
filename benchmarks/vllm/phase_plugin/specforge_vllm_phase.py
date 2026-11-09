"""Phase accounting for a vLLM server, the counterpart of SGLang's phase-accounting patch.

A vLLM general plugin: install this directory into the vLLM environment once,

    <vllm env>/bin/python -m pip install -e benchmarks/vllm/phase_plugin

and it is active only in a server started with ``SPECFORGE_PHASE_PORT`` set; without
that variable it does nothing. The SGLang environments are not touched.

What it measures is what patches/sglang/v0.5.14/phase-accounting.patch measures, in
the same shape, so ``benchmarker.utils.phase_accounting_summary`` reads either:

* The wall time between two consecutive ``Scheduler.update_from_output`` calls is what
  one engine step costs end to end (target forward, drafting, sampling and scheduler
  overhead alike, with or without async scheduling). It is attributed to the phase of
  the step whose result is being processed: ``extend`` when any scheduled request was
  still inside its prompt, ``decode`` when every one was past it, ``other`` otherwise.
* vLLM puts prefill chunks and decodes into one step, which SGLang never does. Such a
  step counts as ``extend``, like SGLang's prefill batches, and what its decoding
  requests produced is kept apart under ``mixed_*``, so ``decode_tokens`` over
  ``time.decode`` stays the throughput of pure decode steps.
* Decode tokens are what each request got from the step: one sampled token plus the
  accepted drafts, as SGLang counts ``batch_size + num_correct_drafts``.
* The clock starts on the first step scheduled after an idle stretch and stops when
  the scheduler runs out of requests, so idle time between requests is never counted.

The lifetime counters are served as JSON at
``http://127.0.0.1:$SPECFORGE_PHASE_PORT/phase_accounting`` by a thread of the
EngineCore process; a client takes the difference of two reads around its run.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

ENV_PORT = "SPECFORGE_PHASE_PORT"

logger = logging.getLogger("specforge_vllm_phase")


class PhaseAccounting:
    """Lifetime counters, keyed as SGLang's ``phase_accounting()`` keys them."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.time = {"extend": 0.0, "decode": 0.0, "other": 0.0}
        self.steps = {"extend": 0, "decode": 0, "other": 0}
        self.decode_tokens = 0
        self.decode_bs_sum = 0
        self.prefill_tokens = 0
        self.prefill_seqs = 0
        self.decode_time_by_bs: Dict[int, float] = defaultdict(float)
        self.decode_steps_by_bs: Dict[int, int] = defaultdict(int)
        self.decode_tokens_by_bs: Dict[int, int] = defaultdict(int)
        # vLLM only: extend steps that also carried decoding requests
        self.mixed_steps = 0
        self.mixed_decode_reqs = 0
        self.mixed_decode_tokens = 0
        # every token any request received, first tokens included
        self.generated_tokens = 0
        self.last_tic: Optional[float] = None
        self.started = time.time()

    def mark_launch(self) -> None:
        """Start the clock on the first step after an idle stretch."""
        with self.lock:
            if self.last_tic is None:
                self.last_tic = time.perf_counter()

    def reset_clock(self) -> None:
        """The scheduler is idle: the next interval must not include the wait."""
        with self.lock:
            self.last_tic = None

    def account_step(
        self,
        prefill_reqs: int,
        prefill_tokens: int,
        new_seqs: int,
        decode_bs: int,
        decode_tokens: int,
        all_tokens: int,
    ) -> None:
        now = time.perf_counter()
        with self.lock:
            self.generated_tokens += all_tokens
            if self.last_tic is None:
                self.last_tic = now
                return
            elapsed = now - self.last_tic
            self.last_tic = now
            if prefill_reqs:
                self.time["extend"] += elapsed
                self.steps["extend"] += 1
                self.prefill_tokens += prefill_tokens
                self.prefill_seqs += new_seqs
                if decode_bs:
                    self.mixed_steps += 1
                    self.mixed_decode_reqs += decode_bs
                    self.mixed_decode_tokens += decode_tokens
            elif decode_bs:
                self.time["decode"] += elapsed
                self.steps["decode"] += 1
                self.decode_tokens += decode_tokens
                self.decode_bs_sum += decode_bs
                self.decode_time_by_bs[decode_bs] += elapsed
                self.decode_steps_by_bs[decode_bs] += 1
                self.decode_tokens_by_bs[decode_bs] += decode_tokens
            else:
                self.time["other"] += elapsed
                self.steps["other"] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "time": dict(self.time),
                "steps": dict(self.steps),
                "decode_tokens": self.decode_tokens,
                "decode_bs_sum": self.decode_bs_sum,
                "prefill_tokens": self.prefill_tokens,
                "prefill_seqs": self.prefill_seqs,
                "decode_time_by_bs": {str(k): v for k, v in self.decode_time_by_bs.items()},
                "decode_steps_by_bs": {str(k): v for k, v in self.decode_steps_by_bs.items()},
                "decode_tokens_by_bs": {str(k): v for k, v in self.decode_tokens_by_bs.items()},
                "mixed_steps": self.mixed_steps,
                "mixed_decode_reqs": self.mixed_decode_reqs,
                "mixed_decode_tokens": self.mixed_decode_tokens,
                "generated_tokens": self.generated_tokens,
                "idle": self.last_tic is None,
                "pid": os.getpid(),
                "started": self.started,
            }


ACCOUNTING = PhaseAccounting()
# scheduler output id -> how its requests split, from schedule() to update_from_output()
_PENDING: Dict[int, Dict[str, Any]] = {}
_server_lock = threading.Lock()
_server_started = False
_patched = False


def _ensure_server(port: int) -> None:
    """Serve the counters from this process, once; only the EngineCore gets here."""
    global _server_started
    if _server_started:
        return
    with _server_lock:
        if _server_started:
            return
        _server_started = True

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                if self.path.split("?")[0].rstrip("/") not in ("", "/phase_accounting"):
                    self.send_error(404)
                    return
                body = json.dumps(ACCOUNTING.snapshot()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: Any) -> None:
                pass

        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        except OSError as error:
            logger.warning("specforge phase accounting: cannot bind port %d: %s", port, error)
            return
        threading.Thread(target=server.serve_forever, name="specforge-phase", daemon=True).start()
        logger.warning(
            "specforge phase accounting: serving http://127.0.0.1:%d/phase_accounting (pid %d)",
            port,
            os.getpid(),
        )


def _split(scheduler: Any, scheduler_output: Any) -> Dict[str, Any]:
    """Which scheduled requests are still inside their prompt.

    Called right after schedule(), whose _update_after_schedule has already advanced
    every request's num_computed_tokens by what it scheduled, so the step starts at
    num_computed_tokens - scheduled. A request whose step starts inside its prompt is
    prefilling (a chunk of a long prompt included); one past it is decoding.
    """
    new_ids = {request.req_id for request in scheduler_output.scheduled_new_reqs}
    prefill_reqs = prefill_tokens = 0
    decode_ids: List[str] = []
    for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
        request = scheduler.requests.get(req_id)
        if req_id in new_ids:
            in_prompt = True
        elif request is not None:
            in_prompt = request.num_computed_tokens - scheduled < request.num_prompt_tokens
        else:
            in_prompt = False
        if in_prompt:
            prefill_reqs += 1
            prefill_tokens += scheduled
        else:
            decode_ids.append(req_id)
    return {
        "prefill_reqs": prefill_reqs,
        "prefill_tokens": prefill_tokens,
        "new_seqs": len(new_ids),
        "decode_ids": decode_ids,
    }


def _patch(port: int) -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original_init = Scheduler.__init__
    original_schedule = Scheduler.schedule
    original_update = Scheduler.update_from_output

    @functools.wraps(original_init)
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # only the EngineCore process builds a scheduler: serve from there
        _ensure_server(port)

    @functools.wraps(original_schedule)
    def schedule(self, *args: Any, **kwargs: Any):
        scheduler_output = original_schedule(self, *args, **kwargs)
        if scheduler_output.total_num_scheduled_tokens > 0:
            _PENDING[id(scheduler_output)] = _split(self, scheduler_output)
            ACCOUNTING.mark_launch()
        return scheduler_output

    @functools.wraps(original_update)
    def update_from_output(self, scheduler_output: Any, model_runner_output: Any):
        split = _PENDING.pop(id(scheduler_output), None)
        if split is not None:
            index = model_runner_output.req_id_to_index
            sampled = model_runner_output.sampled_token_ids

            def received(req_id: str) -> int:
                # what update_from_output will hand this request; nothing for one
                # that finished or was aborted while the step ran
                request = self.requests.get(req_id)
                position = index.get(req_id)
                if request is None or request.is_finished() or position is None or not sampled:
                    return 0
                return len(sampled[position])

            ACCOUNTING.account_step(
                prefill_reqs=split["prefill_reqs"],
                prefill_tokens=split["prefill_tokens"],
                new_seqs=split["new_seqs"],
                decode_bs=len(split["decode_ids"]),
                decode_tokens=sum(received(req_id) for req_id in split["decode_ids"]),
                all_tokens=sum(received(req_id) for req_id in scheduler_output.num_scheduled_tokens),
            )
        outputs = original_update(self, scheduler_output, model_runner_output)
        if not self.has_requests():
            ACCOUNTING.reset_clock()
        return outputs

    Scheduler.__init__ = __init__
    Scheduler.schedule = schedule
    Scheduler.update_from_output = update_from_output


def register() -> None:
    """Entry point of the ``vllm.general_plugins`` group; vLLM calls it in every process."""
    global _patched
    port = os.environ.get(ENV_PORT)
    if not port or _patched:
        return
    _patched = True
    try:
        _patch(int(port))
    except Exception as error:  # never take the server down over a measurement
        logger.warning("specforge phase accounting disabled: %s: %s", type(error).__name__, error)
