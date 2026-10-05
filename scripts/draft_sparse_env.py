#!/usr/bin/env python
"""Print the serving flags that reproduce a draft's training-time sparse context.

A draft trained with ``training.draft_sparse`` is exported with
``dflash_config.draft_sparse`` in its config.json (see
``specforge/export/to_hf.py``). Serving it with any other context pattern -- or
the dense one -- breaks train/test alignment, so derive the flags from the
export instead of retyping them. In a job script:

    flags=$(python scripts/draft_sparse_env.py /path/to/export) || exit 1
    eval "$flags"
    bash scripts/benchmark_mmflash-video.sh   # serve ${DRAFT_MODEL} there

The output is ``export`` lines, so the benchmark scripts -- always run as
child processes -- see them:

  DRAFT_MODEL   the export directory (absolute), for --speculative-draft-model-path
  DRAFT_SPARSE  the SGLANG_DFLASH_DRAFT_SPARSE value (empty for a dense export)
  DRAFT_WINDOW  a --speculative-draft-window-size enabling it (empty for dense)
  NAME_SUFFIX   ``_<export dir name>`` unless already set: bench_mm resumes a
                results file by name, so without it a new checkpoint would land
                on (and be skipped against) the warm-start draft's results

Capture the output in a variable as above rather than writing
``eval "$(python ...)"``: a failure inside ``$( )`` does not fail the eval.
On any error this script prints ``false`` on stdout and exits non-zero, so
even the bare form stops a ``set -e`` script instead of serving dense.

    python scripts/draft_sparse_env.py /path/to/export --format sglang
    # -> the env var and the SGLang flags, for a hand-launched server

The sparse mode lives in a patch (patches/sglang/v0.5.14/
dflash-draft-sparse-context.patch). A stock SGLang ignores the env var and
would silently serve the plain window, so the installed worker is checked
unless --no-check-patch.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import sys

KEYS = ("sink", "text", "stride", "window")
PATCH_SENTINEL = "DFlashDraftSparseContext"


class SparseEnvError(Exception):
    """A reason the export cannot be served with its trained context."""


def read_sparse(export: str) -> tuple[dict | None, int, str]:
    """(draft_sparse or None, block size, export directory) of an export."""
    path = export
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.isfile(path):
        raise SparseEnvError(f"{export}: no config.json")
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    export_dir = os.path.dirname(os.path.abspath(path))
    method = config.get("dflash_config") or {}
    block_size = int(config.get("block_size") or method.get("block_size") or 16)
    sparse = method.get("draft_sparse")
    if sparse is None:
        return None, block_size, export_dir
    missing = [key for key in KEYS if key not in sparse]
    if missing:
        raise SparseEnvError(f"{path}: dflash_config.draft_sparse lacks {missing}")
    for key in KEYS:
        value = sparse[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SparseEnvError(
                f"{path}: draft_sparse.{key} must be an integer >= 0, got {value!r}"
            )
    if sparse["text"] not in (0, 1):
        raise SparseEnvError(f"{path}: draft_sparse.text must be 0 or 1")
    return sparse, block_size, export_dir


def installed_worker_has_patch() -> bool | None:
    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        return None
    worker = os.path.join(
        list(spec.submodule_search_locations)[0], "srt", "speculative", "dflash_worker_v2.py"
    )
    if not os.path.isfile(worker):
        return None
    with open(worker, encoding="utf-8") as handle:
        return PATCH_SENTINEL in handle.read()


def env_lines(sparse: dict | None, block_size: int, export_dir: str) -> list[str]:
    tag = "_" + os.path.basename(export_dir.rstrip("/"))
    if sparse is None:
        spec, window = "", ""
    else:
        spec = ",".join(f"{key}={int(sparse[key])}" for key in KEYS)
        # the compact draft cache must be on and cover a block; the env
        # string's own window decides the recent positions
        window = str(max(int(sparse["window"]), block_size))
    return [
        f"export DRAFT_MODEL={shlex.quote(export_dir)}",
        f"export DRAFT_SPARSE={shlex.quote(spec)}",
        f"export DRAFT_WINDOW={window}",
        # keep a NAME_SUFFIX the caller chose; otherwise name results after the export
        f'export NAME_SUFFIX="${{NAME_SUFFIX:-{tag}}}"',
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("export", help="exported draft directory (or its config.json)")
    parser.add_argument(
        "--format",
        choices=("env", "sglang"),
        default="env",
        help="env: export lines for the benchmark scripts (default); sglang: the "
        "env var and the --speculative-draft-window-size flag",
    )
    parser.add_argument(
        "--no-check-patch",
        action="store_true",
        help="do not check that the installed SGLang carries the sparse-context patch",
    )
    args = parser.parse_args(argv)

    try:
        sparse, block_size, export_dir = read_sparse(args.export)
        if sparse is not None and not args.no_check_patch:
            patched = installed_worker_has_patch()
            if patched is False:
                raise SparseEnvError(
                    "the installed SGLang lacks the draft sparse-context patch "
                    "(patches/sglang/v0.5.14/dflash-draft-sparse-context.patch); it "
                    "would ignore SGLANG_DFLASH_DRAFT_SPARSE and serve a plain window"
                )
            if patched is None:
                print("[draft_sparse_env] sglang not importable here; patch not checked", file=sys.stderr)
    except (SparseEnvError, OSError, ValueError) as error:
        print(f"[draft_sparse_env] ERROR: {error}", file=sys.stderr)
        # makes `eval "$(python scripts/draft_sparse_env.py ...)"` fail too
        print("false")
        return 1

    if sparse is None:
        print(f"[draft_sparse_env] {args.export}: dense draft (no draft_sparse)", file=sys.stderr)
    if args.format == "env":
        print("\n".join(env_lines(sparse, block_size, export_dir)))
    elif sparse is None:
        print("# dense draft: no SGLANG_DFLASH_DRAFT_SPARSE and no --speculative-draft-window-size")
    else:
        spec = ",".join(f"{key}={int(sparse[key])}" for key in KEYS)
        window = max(int(sparse["window"]), block_size)
        print(f"export SGLANG_DFLASH_DRAFT_SPARSE={shlex.quote(spec)}")
        print(f"--speculative-draft-window-size {window} --page-size 1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
