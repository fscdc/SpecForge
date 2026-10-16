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
#   qwen35-eagle3-capture EAGLE3 serving on the Qwen3.5 VL wrapper: its
#                         inherited aux-capture setter never reached the text
#                         stack, so --speculative-algorithm EAGLE3 died in
#                         CUDA-graph capture (qwen3_5.py)
#   gemma4-spec           Gemma 4 targets: aux hidden states are captured as
#                         copies, which the next layer's in-place fused
#                         add-RMSNorm no longer overwrites, the image tokens of
#                         E2B/E4B stay causal as in HF (gemma4_mm.py applies its
#                         per-image bidirectional mask only for
#                         use_bidirectional_attention="vision"), and JPEGs are
#                         decoded by PIL, so a grayscale JPEG no longer kills
#                         the server (gemma4_causal.py, gemma4_mm.py,
#                         processors/gemma4.py).
#                         Gemma 4 drafts are trained on these features, so
#                         serve them with it too; capture hosts apply it alone
#                         with scripts/apply_sglang_gemma4_patch.sh. Serving
#                         Gemma 4 also needs the environment variable
#                         SGLANG_GEMMA_OUT_OF_PLACE_POSITION_MUTATION=1, see
#                         specforge/inference/sglang_patch_inventory.md.
#                         SPECFORGE_ALLOW_UNPATCHED_GEMMA4=1 skips it and leaves
#                         the tree as it is, for a draft trained without it
#
# All of them edit modules that launch_server imports at startup, so run this before
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
# and SENTINEL a string only the patched version of that file contains. A patch
# that edits several files lists one TARGET per line and their SENTINELs in the
# same order, so a tree that carries only part of it is refused rather than
# taken for patched or for clean.
# `patch --reverse --dry-run` cannot be used to detect the state: on an
# UNPATCHED file GNU patch prints "Unreversed patch detected! Ignoring -R" and
# dry-runs it forward instead, exiting 0 either way, so it reports "already
# applied" for both states.
PATCH_NAMES=(request-timing-split phase-accounting qwen35-eagle3-capture gemma4-spec)
PATCH_TARGETS=(
    "sglang/srt/managers/tokenizer_manager.py"
    "sglang/srt/managers/scheduler_components/metrics_reporter.py"
    "sglang/srt/models/qwen3_5.py"
    $'sglang/srt/models/gemma4_causal.py\nsglang/srt/models/gemma4_mm.py\nsglang/srt/multimodal/processors/gemma4.py'
)
PATCH_SENTINELS=(
    'meta_info["first_token_latency"]'
    'def phase_account_step'
    'def set_eagle3_layers_to_capture'
    $'aux_hidden_states.append(hidden_states.clone())\nand vision_bidirectional\ngpu_image_decode = False'
)

# The directory sglang is installed under, or empty when python3 is not the
# environment's python -- the usual cause, and worth its own message, since an
# empty path turns every patch command below into a confusing failure.
sglang_parent() {
    python3 -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))' 2> /dev/null
}

# prints: missing | applied | clean | partial (some of the patch's files only)
patch_state() {
    local sgl_parent="$1" i n_applied=0
    local -a targets sentinels
    mapfile -t targets <<< "$2"
    mapfile -t sentinels <<< "$3"
    for i in "${!targets[@]}"; do
        if [ ! -f "${sgl_parent}/${targets[$i]}" ]; then
            echo missing
            return
        elif grep -qF "${sentinels[$i]}" "${sgl_parent}/${targets[$i]}"; then
            n_applied=$((n_applied + 1))
        fi
    done
    if [ "${n_applied}" -eq 0 ]; then
        echo clean
    elif [ "${n_applied}" -eq "${#targets[@]}" ]; then
        echo applied
    else
        echo partial
    fi
}

report_missing() {
    local sgl_parent="$1" target
    while IFS= read -r target; do
        if [ ! -f "${sgl_parent}/${target}" ]; then
            echo "[helper] ERROR: ${sgl_parent}/${target} does not exist" >&2
        fi
    done <<< "$2"
}

report_partial() {
    local sgl_parent="$1" name="$2"
    echo "[helper] ERROR: only part of ${name} is applied at ${sgl_parent}" >&2
    if [ "${name}" = gemma4-spec ]; then
        # an older revision of the patch, which edited fewer files
        echo "[helper] run scripts/apply_sglang_gemma4_patch.sh to update it, then re-run this script" >&2
        return
    fi
    echo "[helper] reinstall sglang, then re-run this script" >&2
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
            report_missing "${sgl_parent}" "${target}"
            return 1
            ;;
        partial)
            report_partial "${sgl_parent}" "${name}"
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
            report_missing "${sgl_parent}" "${target}"
            return 1
            ;;
        partial)
            # --batch would reverse the patched files and apply the rest forward
            report_partial "${sgl_parent}" "${name}"
            return 1
            ;;
    esac
    patch -p2 --reverse --batch -d "${sgl_parent}" < "${patch_file}" || return 1
    if [ "$(patch_state "${sgl_parent}" "${target}" "${sentinel}")" != "clean" ]; then
        echo "[helper] ERROR: reverse patch reported success but the sentinel remains" >&2
        return 1
    fi
    echo "[helper] ${name} removed from ${sgl_parent}"
}

do_apply() {
    local sgl_parent i
    sgl_parent="$(require_sglang)" || return 1
    for i in "${!PATCH_NAMES[@]}"; do
        # the training preflight's hatch, for a draft trained without
        # gemma4-spec: serve the tree as it is rather than patch it
        if [ "${PATCH_NAMES[$i]}" = gemma4-spec ] && [ "${SPECFORGE_ALLOW_UNPATCHED_GEMMA4:-0}" = 1 ]; then
            echo "[helper] gemma4-spec left as is at ${sgl_parent} (SPECFORGE_ALLOW_UNPATCHED_GEMMA4=1)"
            continue
        fi
        apply_one "${sgl_parent}" "${PATCH_NAMES[$i]}" "${PATCH_TARGETS[$i]}" "${PATCH_SENTINELS[$i]}" || return 1
    done
    echo "[helper] responses carry first_token_latency and decode_latency;"
    echo "[helper] /server_info carries phase_accounting; Qwen3.5 serves EAGLE3"
    if [ "${SPECFORGE_ALLOW_UNPATCHED_GEMMA4:-0}" != 1 ]; then
        echo "[helper] Gemma 4 captures true layer outputs, keeps E2B/E4B image tokens causal and decodes JPEGs with PIL"
    fi
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
