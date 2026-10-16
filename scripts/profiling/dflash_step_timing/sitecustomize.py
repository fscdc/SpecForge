"""Per-phase GPU time of DFlash (spec-v2) decode steps inside an SGLang server.

Loaded into every Python process through PYTHONPATH (it is a sitecustomize),
and inert unless DFLASH_TIMING_DIR is set. Nothing in the installed SGLang is
edited: once sglang.srt.speculative.dflash_worker_v2 has been imported, a
background thread wraps

  DFlashWorkerV2.forward_batch_generation             one decode step (extend and idle skipped)
  draft_model_runner.forward                          draft: the draft model's layers
  DFlashWorkerV2._greedy_sample_from_vocab_parallel_head
                                                      draft: token choice through the target lm_head
  DFlashWorkerV2._gather_draft_sparse_context         draft: sparse context gather (sparse runs only)
  target_worker.forward_batch_generation(is_verify=True)
                                                      verify: the target forward over the block
  DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc
                                                      writing the committed tokens into the draft KV

Each region is bracketed by CUDA events recorded on the current stream, i.e.
the time the GPU spent between those two points of the step. Events are read
back asynchronously (event.query()), so no host synchronisation is added to the
decode loop. "other" is the rest of the step: building the draft inputs, the
acceptance check, bookkeeping.

Totals are kept per label -- the first line of $DFLASH_TIMING_LABEL_FILE, read
every few steps, so a driver can name the benchmark being run -- and written
to $DFLASH_TIMING_DIR/<pid>.json.
"""

import os

_DIR = os.environ.get("DFLASH_TIMING_DIR")

if _DIR:
    import collections
    import functools
    import json
    import sys
    import threading
    import time

    os.makedirs(_DIR, exist_ok=True)
    _PID = os.getpid()
    _LABEL_FILE = os.environ.get("DFLASH_TIMING_LABEL_FILE", "")
    _MODULE = "sglang.srt.speculative.dflash_worker_v2"
    _REGIONS = ("draft_forward", "draft_sample", "sparse_gather", "verify_forward", "kv_append")

    _local = threading.local()
    _pending = collections.deque()
    _totals = {}  # label -> {"steps": n, "step_ms": x, region: ms, ...}
    _state = {"label": "unlabelled", "resolved": 0}

    def _event():
        import torch

        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def _label():
        if not _LABEL_FILE:
            return _state["label"]
        try:
            with open(_LABEL_FILE, encoding="utf-8") as handle:
                return handle.readline().strip() or "unlabelled"
        except OSError:
            return _state["label"]

    def _flush():
        path = os.path.join(_DIR, f"{_PID}.json")
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump({"pid": _PID, "argv": sys.argv[:3], "totals": _totals}, handle, indent=1)
        os.replace(path + ".tmp", path)

    def _resolve(block):
        """Fold every finished step into the totals; with block=True wait for all."""
        changed = False
        while _pending:
            label, marks = _pending[0]
            last = marks[-1][1]
            if not block and not last.query():
                break
            last.synchronize()
            _pending.popleft()
            start = marks[0][1]
            entry = _totals.setdefault(label, {"steps": 0, "step_ms": 0.0, **{r: 0.0 for r in _REGIONS}, **{r + "_calls": 0 for r in _REGIONS}})
            entry["steps"] += 1
            entry["step_ms"] += start.elapsed_time(last)
            opened = {}
            for name, ev in marks[1:-1]:
                region, edge = name.rsplit(":", 1)
                if edge == "start":
                    opened[region] = ev
                elif region in opened:
                    entry[region] += opened.pop(region).elapsed_time(ev)
                    entry[region + "_calls"] += 1
            _state["resolved"] += 1
            changed = True
        if changed and _state["resolved"] % 100 == 0:
            _flush()

    def _mark(name):
        marks = getattr(_local, "marks", None)
        if marks is not None:
            marks.append((name, _event()))

    def _region(region, fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if getattr(_local, "marks", None) is None:
                return fn(*args, **kwargs)
            _mark(region + ":start")
            try:
                return fn(*args, **kwargs)
            finally:
                _mark(region + ":end")

        wrapper._dflash_timed = True
        return wrapper

    def _wrap_instance(worker):
        """Wrap the draft runner's forward and the target's verify call of one worker.

        Done at its first decode step rather than in __init__, so a worker built
        before the class was patched is covered as well."""
        if getattr(worker, "_dflash_timing_wrapped", False):
            return
        runner = worker.draft_model_runner
        runner.forward = _region("draft_forward", runner.forward)
        target = worker.target_worker
        verify = target.forward_batch_generation

        @functools.wraps(verify)
        def timed_verify(*a, **kw):
            if kw.get("is_verify") and getattr(_local, "marks", None) is not None:
                _mark("verify_forward:start")
                try:
                    return verify(*a, **kw)
                finally:
                    _mark("verify_forward:end")
            return verify(*a, **kw)

        target.forward_batch_generation = timed_verify
        worker._dflash_timing_wrapped = True

    def _patch(module):
        cls = module.DFlashWorkerV2
        step = cls.forward_batch_generation

        @functools.wraps(step)
        def timed_step(self, model_worker_batch, *args, **kwargs):
            mode = getattr(model_worker_batch, "forward_mode", None)
            decode = (
                mode is not None
                and not mode.is_extend()
                and not getattr(model_worker_batch, "is_extend_in_batch", False)
                and not mode.is_idle()
            )
            if not decode:
                return step(self, model_worker_batch, *args, **kwargs)
            _wrap_instance(self)
            if _state["resolved"] % 20 == 0:
                _state["label"] = _label()
            _local.marks = [("step:start", _event())]
            try:
                return step(self, model_worker_batch, *args, **kwargs)
            finally:
                _local.marks.append(("step:end", _event()))
                _pending.append((_state["label"], _local.marks))
                _local.marks = None
                _resolve(block=False)

        cls.forward_batch_generation = timed_step
        for attr, region in (
            ("_greedy_sample_from_vocab_parallel_head", "draft_sample"),
            ("_gather_draft_sparse_context", "sparse_gather"),
            ("_append_target_hidden_to_draft_kv_by_loc", "kv_append"),
        ):
            if hasattr(cls, attr):
                setattr(cls, attr, _region(region, getattr(cls, attr)))
        with open(os.path.join(_DIR, f"{_PID}.patched"), "w", encoding="utf-8") as handle:
            handle.write(f"{time.time():.3f} patched {cls.__module__}.{cls.__name__}\n")

    def _patcher():
        # patch the class once its module has finished executing (a module still
        # running its body is already in sys.modules), before any worker exists
        while True:
            module = sys.modules.get(_MODULE)
            if module is not None and not getattr(getattr(module, "__spec__", None), "_initializing", False):
                if hasattr(module, "DFlashWorkerV2"):
                    try:
                        _patch(module)
                    except Exception as exc:  # pragma: no cover
                        with open(os.path.join(_DIR, f"{_PID}.error"), "w", encoding="utf-8") as handle:
                            handle.write(repr(exc))
                    return
            time.sleep(0.05)

    def _flusher():
        while True:
            time.sleep(10)
            try:
                if _totals:
                    _flush()
            except Exception:  # pragma: no cover
                pass

    threading.Thread(target=_patcher, name="dflash-timing-patcher", daemon=True).start()
    threading.Thread(target=_flusher, name="dflash-timing-flusher", daemon=True).start()
