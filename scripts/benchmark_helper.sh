#!/usr/bin/env bash
# Prepares the installed SGLang for the multimodal benchmarks.
#
# Applies the patches under patches/sglang/v0.5.14/ that the benchmark harness
# reads back:
#   request-timing-split  every response carries first_token_latency and
#                         decode_latency in meta_info (tokenizer_manager.py)
#   phase-accounting      the scheduler attributes its wall time to prefill or
#                         decode per batch and exposes the counters under
#                         /server_info, which is what separates the two phases
#                         once more than one request is in flight
#                         (scheduler.py, metrics_reporter.py)
#
# Both edit modules that launch_server imports at startup, so run this before
# starting the server, in the same environment:
#
#     bash scripts/benchmark_helper.sh || exit 1
#     bash scripts/benchmark_helper.sh --unpatch     # restore a stock tree
#
# Idempotent, and safe to call on an already-patched tree.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/patches/sglang/v0.5.14"

# One entry per patch, in application order. TARGET is a file the patch edits
# and SENTINEL a string only the patched version of that file contains.
# `patch --reverse --dry-run` cannot be used to detect the state: on an
# UNPATCHED file GNU patch prints "Unreversed patch detected! Ignoring -R" and
# dry-runs it forward instead, exiting 0 either way, so it reports "already
# applied" for both states.
PATCH_NAMES=(request-timing-split phase-accounting)
PATCH_TARGETS=(
    "sglang/srt/managers/tokenizer_manager.py"
    "sglang/srt/managers/scheduler_components/metrics_reporter.py"
)
PATCH_SENTINELS=(
    'meta_info["first_token_latency"]'
    'def phase_account_step'
)

# The directory sglang is installed under, or empty when python3 is not the
# environment's python -- the usual cause, and worth its own message, since an
# empty path turns every patch command below into a confusing failure.
sglang_parent() {
    python3 -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))' 2> /dev/null
}

# prints: missing | applied | clean
patch_state() {
    local sgl_parent="$1" target="$2" sentinel="$3"
    if [ ! -f "${sgl_parent}/${target}" ]; then
        echo missing
    elif grep -qF "${sentinel}" "${sgl_parent}/${target}"; then
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

apply_one() {
    local sgl_parent="$1" name="$2" target="$3" sentinel="$4"
    local patch_file="${PATCH_DIR}/${name}.patch" state
    if [ ! -f "${patch_file}" ]; then
        echo "[helper] ERROR: ${patch_file} not found" >&2
        return 1
    fi

    state="$(patch_state "${sgl_parent}" "${target}" "${sentinel}")"
    case "${state}" in
        applied)
            echo "[helper] ${name} already applied at ${sgl_parent}"
            return 0
            ;;
        missing)
            echo "[helper] ERROR: ${sgl_parent}/${target} does not exist" >&2
            return 1
            ;;
    esac

    if ! patch -p2 --dry-run --batch -N -d "${sgl_parent}" < "${patch_file}" > /dev/null 2>&1; then
        echo "[helper] ERROR: ${name} does not apply to this sglang" >&2
        echo "[helper] the tree is neither patched nor the version it targets" >&2
        return 1
    fi
    patch -p2 --batch -N -d "${sgl_parent}" < "${patch_file}" || return 1
    if [ "$(patch_state "${sgl_parent}" "${target}" "${sentinel}")" != "applied" ]; then
        echo "[helper] ERROR: ${name} reported success but the sentinel is absent" >&2
        return 1
    fi
    echo "[helper] ${name} applied at ${sgl_parent}"
}

unpatch_one() {
    local sgl_parent="$1" name="$2" target="$3" sentinel="$4"
    local patch_file="${PATCH_DIR}/${name}.patch"
    case "$(patch_state "${sgl_parent}" "${target}" "${sentinel}")" in
        clean)
            echo "[helper] ${name} is not applied at ${sgl_parent}"
            return 0
            ;;
        missing)
            echo "[helper] ERROR: ${sgl_parent}/${target} does not exist" >&2
            return 1
            ;;
    esac
    patch -p2 --reverse --batch -d "${sgl_parent}" < "${patch_file}" || return 1
    if [ "$(patch_state "${sgl_parent}" "${target}" "${sentinel}")" = "applied" ]; then
        echo "[helper] ERROR: reverse patch reported success but the sentinel remains" >&2
        return 1
    fi
    echo "[helper] ${name} removed from ${sgl_parent}"
}

do_apply() {
    local sgl_parent i
    sgl_parent="$(require_sglang)" || return 1
    for i in "${!PATCH_NAMES[@]}"; do
        apply_one "${sgl_parent}" "${PATCH_NAMES[$i]}" "${PATCH_TARGETS[$i]}" "${PATCH_SENTINELS[$i]}" || return 1
    done
    echo "[helper] responses carry first_token_latency and decode_latency;"
    echo "[helper] /server_info carries phase_accounting"
}

do_unpatch() {
    local sgl_parent i
    sgl_parent="$(require_sglang)" || return 1
    # reverse order, in case a later patch ever touches an earlier one's file
    for (( i=${#PATCH_NAMES[@]}-1; i>=0; i-- )); do
        unpatch_one "${sgl_parent}" "${PATCH_NAMES[$i]}" "${PATCH_TARGETS[$i]}" "${PATCH_SENTINELS[$i]}" || return 1
    done
}

case "${1:-}" in
    "") do_apply ;;
    --unpatch) do_unpatch ;;
    *)
        echo "usage: bash scripts/benchmark_helper.sh [--unpatch]" >&2
        exit 2
        ;;
esac
