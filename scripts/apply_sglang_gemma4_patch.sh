#!/usr/bin/env bash
# Apply the Gemma 4 spec patch to the INSTALLED sglang (site-packages).
#
# patches/sglang/v0.5.14/gemma4-spec.patch, for google/gemma-4-* targets:
#   models/gemma4_causal.py          capture copies of the aux hidden states;
#                                    the next layer's fused add-RMSNorm adds its
#                                    attention branch to its input in place, so
#                                    a captured reference was not the layer
#                                    output. Caller-passed input_embeds (the
#                                    multimodal path) are copied for the same
#                                    reason: the EAGLE3 draft reads them back.
#   models/gemma4_mm.py              apply the per-image bidirectional mask only
#                                    when the config sets
#                                    use_bidirectional_attention="vision", as HF
#                                    does; E2B/E4B image tokens stay causal.
#   multimodal/processors/gemma4.py  decode JPEGs with PIL, not nvJPEG, so a
#                                    grayscale JPEG becomes RGB instead of
#                                    crashing the server
# Drafts are trained on the patched features, so the capture host and every
# serving host must carry the same patch.
#
# Same layout as apply_sglang_spec_capture_patch.sh: the patch is authored
# against the sglang source tree (a/python/sglang/...), so strip TWO components
# and apply from the site-packages parent. The applied patch text is recorded
# next to the tree, so a revised patch reverses the recorded one first.
#
# The state comes from one sentinel string per file, the strings
# scripts/benchmark_helper.sh and the managed-launch preflight check, and is
# confirmed by a reverse dry-run: the tree must carry this exact patch. Hunks
# must match exactly (--fuzz=0); a tree in any other state fails loudly rather
# than serving unknown model code.
#
# Usage: scripts/apply_sglang_gemma4_patch.sh [--check | --reverse]
#   (none)     apply; a no-op when the tree already carries this patch
#   --check    report the state; exit 0 only when this patch is applied
#   --reverse  restore the stock files
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$HERE/patches/sglang/v0.5.14/gemma4-spec.patch"

# One entry per patched file: the file, and a string only its patched version
# contains.
TARGETS=(
    "sglang/srt/models/gemma4_causal.py"
    "sglang/srt/models/gemma4_mm.py"
    "sglang/srt/multimodal/processors/gemma4.py"
)
SENTINELS=(
    "aux_hidden_states.append(hidden_states.clone())"
    "and vision_bidirectional"
    "gpu_image_decode = False"
)

case "${1:-}" in
    "" | --check | --reverse) ;;
    *)
        echo "usage: $0 [--check | --reverse]" >&2
        exit 2
        ;;
esac

# find_spec locates the package without importing it (the import pulls in
# torch and takes seconds).
if ! SGL_PARENT="$(python -c 'import importlib.util as u, os; print(os.path.dirname(os.path.dirname(u.find_spec("sglang").origin)))' 2> /dev/null)"; then
    echo "ERROR: python cannot find sglang; activate the environment first" >&2
    exit 1
fi
SGL_VERSION="$(python -c 'import importlib.metadata as m; print(m.version("sglang"))' 2> /dev/null || echo unknown)"
APPLIED_COPY="$SGL_PARENT/sglang/.gemma4_spec_patch.applied"

if [[ "$SGL_VERSION" != 0.5.14* ]]; then
    echo "WARNING: installed sglang is $SGL_VERSION; the patch targets v0.5.14" >&2
fi

# prints: applied | clean | partial | missing
tree_state() {
    local i n_applied=0
    for i in "${!TARGETS[@]}"; do
        if [[ ! -f "$SGL_PARENT/${TARGETS[$i]}" ]]; then
            echo missing
            return
        fi
        if grep -qF "${SENTINELS[$i]}" "$SGL_PARENT/${TARGETS[$i]}"; then
            n_applied=$((n_applied + 1))
        fi
    done
    if [[ $n_applied -eq 0 ]]; then
        echo clean
    elif [[ $n_applied -eq ${#TARGETS[@]} ]]; then
        echo applied
    else
        echo partial
    fi
}

# Whether the tree carries patch $1 exactly: every hunk reverses with no fuzz.
# --force, never --batch: when a hunk does not match, GNU patch with --batch
# prints "Unreversed patch detected! Ignoring -R." and checks it FORWARD,
# which succeeds on an unpatched tree.
carries() {
    patch --reverse --dry-run --force --fuzz=0 -p2 -d "$SGL_PARENT" < "$1" > /dev/null 2>&1
}

reverse() {
    patch --reverse --force --fuzz=0 --no-backup-if-mismatch -p2 -d "$SGL_PARENT" < "$1"
}

state="$(tree_state)"
case "$state" in
    missing)
        echo "ERROR: sglang $SGL_VERSION at $SGL_PARENT has no Gemma 4 support" >&2
        echo "expected ${TARGETS[*]}; the patch targets v0.5.14" >&2
        exit 1
        ;;
    partial)
        # An older revision of this patch that touched fewer files leaves only
        # some of today's sentinels; its recorded copy still reverses exactly.
        if [[ -f "$APPLIED_COPY" ]] && ! cmp -s "$PATCH" "$APPLIED_COPY" && carries "$APPLIED_COPY"; then
            state=older
        else
            echo "ERROR: $SGL_PARENT/sglang carries part of the gemma4-spec patch" >&2
            echo "reinstall sglang (or clear the cached venv) and re-run this script" >&2
            exit 1
        fi
        ;;
esac

if [[ "${1:-}" == "--check" ]]; then
    if [[ "$state" == clean ]]; then
        echo "gemma4-spec patch is NOT applied at $SGL_PARENT/sglang (sglang $SGL_VERSION)" >&2
        exit 1
    fi
    if ! carries "$PATCH"; then
        echo "$SGL_PARENT/sglang does not carry the current gemma4-spec patch exactly" >&2
        echo "re-run $0 without --check to update it" >&2
        exit 1
    fi
    echo "gemma4-spec patch applied at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
    exit 0
fi

if [[ "${1:-}" == "--reverse" ]]; then
    if [[ "$state" == clean ]]; then
        rm -f "$APPLIED_COPY"
        echo "gemma4-spec patch is not applied at $SGL_PARENT/sglang"
        exit 0
    fi
    # Reverse what the tree actually carries: the recorded copy, which may be
    # an older revision than the repo's patch, or else the repo's patch (a
    # tree patched by benchmark_helper.sh has no record).
    applied=""
    for candidate in "$APPLIED_COPY" "$PATCH"; do
        if [[ -f "$candidate" ]] && carries "$candidate"; then
            applied="$candidate"
            break
        fi
    done
    if [[ -z "$applied" ]]; then
        echo "ERROR: $SGL_PARENT/sglang carries an unknown gemma4-spec patch state; nothing reversed" >&2
        echo "reinstall sglang (or clear the cached venv) to restore the stock files" >&2
        exit 1
    fi
    reverse "$applied"
    rm -f "$APPLIED_COPY"
    if [[ "$(tree_state)" != clean ]]; then
        echo "ERROR: the reverse patch reported success but a sentinel remains" >&2
        exit 1
    fi
    echo "gemma4-spec patch --reverse at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
    exit 0
fi

if [[ "$state" == applied || "$state" == older ]]; then
    if carries "$PATCH"; then
        # Also adopts a tree patched by benchmark_helper.sh, which keeps no
        # record.
        cmp -s "$PATCH" "$APPLIED_COPY" 2> /dev/null || cp "$PATCH" "$APPLIED_COPY"
        echo "gemma4-spec patch already applied at $SGL_PARENT/sglang"
        exit 0
    fi
    if [[ -f "$APPLIED_COPY" ]] && carries "$APPLIED_COPY"; then
        echo "gemma4-spec patch changed; reversing the recorded version first"
        reverse "$APPLIED_COPY"
        rm -f "$APPLIED_COPY"
    else
        echo "ERROR: $SGL_PARENT/sglang carries an unknown gemma4-spec patch state" >&2
        echo "reinstall sglang (or clear the cached venv) and re-run this script" >&2
        exit 1
    fi
else
    # A clean tree: a leftover record is stale (sglang was reinstalled, or
    # benchmark_helper.sh --unpatch removed the patch).
    rm -f "$APPLIED_COPY"
fi

# Dry-run first, so a patch that does not apply leaves every file untouched.
if ! dry_run="$(patch -p2 --dry-run --batch -N --fuzz=0 -d "$SGL_PARENT" < "$PATCH" 2>&1)"; then
    echo "$dry_run" >&2
    echo "ERROR: the gemma4-spec patch does not apply to sglang $SGL_VERSION at $SGL_PARENT" >&2
    exit 1
fi
patch -p2 --batch -N --fuzz=0 --no-backup-if-mismatch -d "$SGL_PARENT" < "$PATCH"
if [[ "$(tree_state)" != applied ]]; then
    echo "ERROR: the patch reported success but a sentinel is absent" >&2
    exit 1
fi
cp "$PATCH" "$APPLIED_COPY"
echo "gemma4-spec patch applied at $SGL_PARENT/sglang (sglang $SGL_VERSION)"
