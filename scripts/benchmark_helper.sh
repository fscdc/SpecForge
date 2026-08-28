#!/usr/bin/env bash
# Prepare the installed SGLang for a benchmark run.
#
# Applies patches/sglang/v0.5.14/request-timing-split.patch, which makes every
# response carry `first_token_latency` and `decode_latency` next to the
# `e2e_latency` stock SGLang already returns. Without them a benchmark can only
# report `output tokens / wall clock`, and on a prompt of tens of thousands of
# visual tokens that number is ~90% prefill -- a draft model cannot move it, so
# a speculative-decoding run looks identical to its baseline.
#
# The patch edits tokenizer_manager.py, which launch_server imports at startup,
# so this has to run BEFORE any server is launched (and a server already
# running has to be restarted).
#
# Usage, from the benchmark scripts:
#     bash scripts/benchmark_helper.sh || exit 1
#     bash scripts/benchmark_helper.sh --unpatch     # restore a stock tree
#
# Idempotent, and safe to call on an already-patched tree.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PATCH_FILE="${REPO_ROOT}/patches/sglang/v0.5.14/request-timing-split.patch"
PATCH_TARGET="sglang/srt/managers/tokenizer_manager.py"
# A string only the patched file contains. `patch --reverse --dry-run` cannot be
# used to detect this: on an UNPATCHED file GNU patch prints "Unreversed patch
# detected! Ignoring -R" and dry-runs it forward instead, exiting 0 either way,
# so it reports "already applied" for both states.
PATCH_SENTINEL='meta_info["first_token_latency"]'

# The directory sglang is installed under, or empty when python3 is not the
# environment's python -- the usual cause, and worth its own message, since an
# empty path turns every patch command below into a confusing failure.
sglang_parent() {
    python3 -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))' 2> /dev/null
}

# prints: missing | applied | clean
patch_state() {
    local sgl_parent="$1"
    if [ ! -f "${sgl_parent}/${PATCH_TARGET}" ]; then
        echo missing
    elif grep -qF "${PATCH_SENTINEL}" "${sgl_parent}/${PATCH_TARGET}"; then
        echo applied
    else
        echo clean
    fi
}

require_sglang() {
    local sgl_parent
    sgl_parent="$(sglang_parent)"
    if [ -z "${sgl_parent}" ]; then
        echo "[helper] ERROR: python3 cannot import sglang" >&2
        echo "[helper] activate the environment first (conda activate specforge)" >&2
        return 1
    fi
    echo "${sgl_parent}"
}

do_apply() {
    local sgl_parent state
    sgl_parent="$(require_sglang)" || return 1
    if [ ! -f "${PATCH_FILE}" ]; then
        echo "[helper] ERROR: ${PATCH_FILE} not found" >&2
        return 1
    fi

    state="$(patch_state "${sgl_parent}")"
    case "${state}" in
        applied)
            echo "[helper] request-timing-split already applied at ${sgl_parent}"
            return 0
            ;;
        missing)
            echo "[helper] ERROR: ${sgl_parent}/${PATCH_TARGET} does not exist" >&2
            return 1
            ;;
    esac

    if ! patch -p2 --dry-run --batch -N -d "${sgl_parent}" < "${PATCH_FILE}" > /dev/null 2>&1; then
        echo "[helper] ERROR: the patch does not apply to this sglang" >&2
        echo "[helper] the tree is neither patched nor the version it targets" >&2
        return 1
    fi
    patch -p2 --batch -N -d "${sgl_parent}" < "${PATCH_FILE}" || return 1
    if [ "$(patch_state "${sgl_parent}")" != "applied" ]; then
        echo "[helper] ERROR: patch reported success but the sentinel is absent" >&2
        return 1
    fi
    echo "[helper] request-timing-split applied at ${sgl_parent}"
    echo "[helper] responses now carry first_token_latency and decode_latency"
}

do_unpatch() {
    local sgl_parent
    sgl_parent="$(require_sglang)" || return 1
    case "$(patch_state "${sgl_parent}")" in
        clean)
            echo "[helper] nothing to reverse; sglang is already stock"
            return 0
            ;;
        missing)
            echo "[helper] ERROR: ${sgl_parent}/${PATCH_TARGET} missing" >&2
            return 1
            ;;
    esac
    patch -p2 --reverse --batch -d "${sgl_parent}" < "${PATCH_FILE}" || return 1
    if [ "$(patch_state "${sgl_parent}")" = "applied" ]; then
        echo "[helper] ERROR: reverse reported success but the sentinel remains" >&2
        return 1
    fi
    echo "[helper] reversed; sglang is stock again"
}

case "${1:-}" in
    --unpatch) do_unpatch ;;
    "")        do_apply ;;
    *)
        echo "usage: bash scripts/benchmark_helper.sh [--unpatch]" >&2
        exit 2
        ;;
esac
