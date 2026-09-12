#!/bin/bash
# Per-anchor accept analysis of DFlash drafts, swept over benchmarks x drafts,
# ending in one machine-readable metrics record per (benchmark, draft) pair.
#
# Everything is set in the CONFIG block below and overridable from the
# environment, so one submission can sweep several drafts over several
# benchmarks, and a later submission can extend the sweep without redoing what
# is already on disk:
#
#   qsub -l select=1:ngpus=1 scripts/analyze_accept_hpc.sh
#   BENCHMARKS="chartqa mmstar" NUM_SAMPLES=50 bash scripts/analyze_accept_hpc.sh
#   DRAFTS_SPEC="zlab=z-lab/Qwen3.5-4B-DFlash" bash scripts/analyze_accept_hpc.sh
#   DRY_RUN=1 bash scripts/analyze_accept_hpc.sh      # print the plan, run nothing
#   FORCE=1   bash scripts/analyze_accept_hpc.sh      # redo pairs already done
#
# RESUMING. Every (benchmark, draft) pair is one unit of work. A pair counts as
# done when its run directory holds a metrics.json AND a run_meta.json whose
# models, benchmark, split, sample count and probe knobs match what is being
# asked for now; those are skipped. A pair whose run_meta disagrees (different
# draft path, different --num-anchors, a run made before the prompts moved into
# the benchmark classes) is NOT silently reused and NOT silently overwritten:
# the stale directory is renamed to <dir>.superseded-<timestamp> and the pair is
# measured again. So an interrupted submission can simply be resubmitted, and
# nothing measured earlier is ever lost.
#
# Outputs are written under the project root (accept_analysis/ beside the code),
# resolved per machine, not to a hard-coded path.
#
# The target's generations depend only on (target, benchmark, split, samples),
# never on the draft, so they are produced once and cached under
# accept_analysis/generations/; every draft then scores the byte-identical token
# sequences, which is what makes the drafts comparable.
#
# Per run directory this leaves:
#   generations.jsonl       the target's answers (copied from the shared cache)
#   per_anchor_accept.jsonl one record per drafted block (the raw measurements)
#   run_meta.json           models, dataset, sample counts, knobs, versions
#   metrics.json            every derived number, with run_meta attached
#   probe.log               the full console output
# and appends the same {run_meta, metrics} line to ${RESULTS_JSONL}, which is
# append-only history: a re-measured pair adds a line rather than replacing one.
#
# Benchmark images are materialised by the benchmark classes themselves, into
# .cache/<benchmark>_specforge/images under the project root, and are therefore
# shared by every run instead of copied per draft.

#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N accept-probe
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=12:00:00

set -uo pipefail

export PYTHONUNBUFFERED=1

# --------------------------- environment -------------------------------------
# qsub starts a shell that inherits none of the caller's conda activation, so
# under PBS this has to activate the environment itself; run interactively with
# an environment already active it does nothing. Without this every pair fails
# identically on `ModuleNotFoundError: No module named 'torch'` -- which is what
# job 601480 spent 73 minutes doing.
CONDA_ENV=${CONDA_ENV:-specforge}
if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "[env] torch is not importable; activating conda env '${CONDA_ENV}'"
    conda_hook=""
    for candidate in \
        "${CONDA_EXE:+$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh}" \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh"
    do
        if [ -n "${candidate}" ] && [ -f "${candidate}" ]; then
            conda_hook="${candidate}"
            break
        fi
    done
    if [ -n "${conda_hook}" ]; then
        # shellcheck disable=SC1090
        . "${conda_hook}"
    elif [ -f "${HOME}/.bashrc" ]; then
        # shellcheck disable=SC1090
        . "${HOME}/.bashrc"
    fi
    conda activate "${CONDA_ENV}" >/dev/null 2>&1 || true
fi
if ! python3 -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: torch is still not importable after trying to activate '${CONDA_ENV}'." >&2
    echo "       python3: $(command -v python3 || echo none)" >&2
    echo "       CONDA_PREFIX: ${CONDA_PREFIX:-<unset>}" >&2
    echo "       Set CONDA_ENV=<name>, or activate the environment before calling." >&2
    exit 1
fi
echo "[env] python3 $(python3 -c 'import sys,torch;print(sys.version.split()[0], "torch", torch.__version__)')"
echo "[env] CONDA_PREFIX=${CONDA_PREFIX:-<unset>}"

# CONDA_PREFIX is only meaningful once the environment is active; setting this
# any earlier under PBS would have produced "/lib:/lib/python3.11/..."
# asked of torch rather than assembled from a hard-coded python version, which
# would silently produce a non-existent path on an env built on another minor
_TORCH_LIB="$(python3 -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))' 2>/dev/null)"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${_TORCH_LIB}:${LD_LIBRARY_PATH:-}"

# ----------------------------- CONFIG ---------------------------------------
PROBE=scripts/analyze_dflash_accept.py

# Project root, resolved without hard-coding a machine, so a checkout on any
# server writes into its own tree.
#
# This script's own location is tried FIRST and PBS_O_WORKDIR only as a
# fallback, because each is wrong in the case the other handles. Under qsub PBS
# spools the script to a private directory, so BASH_SOURCE points nowhere near
# the checkout and only PBS_O_WORKDIR knows where it was submitted from. Run
# interactively the reverse holds: PBS_O_WORKDIR is often still exported from an
# earlier interactive job and points at whatever directory that one was launched
# from. Whether the candidate actually contains the probe is what settles it.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/${PROBE}" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
if [ ! -f "${PROJECT_DIR:-}/${PROBE}" ]; then
    echo "ERROR: cannot locate the SpecForge checkout." >&2
    echo "       looked for ${PROBE} under:" >&2
    echo "         $(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)  (this script's parent)" >&2
    [ -n "${PBS_O_WORKDIR:-}" ] && echo "         ${PBS_O_WORKDIR}  (PBS_O_WORKDIR)" >&2
    echo "       run it from the checkout, or set PBS_O_WORKDIR to it." >&2
    exit 1
fi
cd "${PROJECT_DIR}" || exit 1

# where models are stored -- a data root, kept off the project tree because
# checkpoints do not belong in the repo
ROOT=${ROOT:-/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge}

TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3.5-4B}
# short name used in paths; derived from the model id unless given
TARGET_NAME=${TARGET_NAME:-${TARGET_MODEL##*/}}

# space-separated name=path pairs; the name labels directories and records.
# The two drafts are the ones the throughput table compares head to head:
#   zlab        the published text-only-data baseline
#   llava-ov-1M our DFlash draft on the LLaVA-OneVision-1.5 boxed-prompt regen
#               (the "prompted final" checkpoint of results/)
# Add "sharegpt4v=${ROOT}/draft_models/qwen3.5-4b-mmflash-sharegpt4v-pt-120000"
# to bring the earlier ShareGPT4V-pt draft back into the sweep.
DRAFTS_SPEC=${DRAFTS_SPEC:-"\
zlab=z-lab/Qwen3.5-4B-DFlash \
llava-ov-1M=${ROOT}/draft_models/qwen3.5-4b-dflash-baseline-llava-ov15-1M-prompted-final"}

# space-separated benchmark names, each optionally "name:split" to override the
# split the benchmark itself defaults to. BENCHMARK (singular) still works.
#
# Six columns of the throughput table, so a per-anchor result can be read next
# to the speedup it is meant to explain. They also span the shapes the
# visual-dependency question needs: read-off-the-image answers (ChartQA,
# TextVQA), vision-indispensable questions (MMStar), and figure-fed derivations
# of growing length (DynaMath, MathVista, MathVerse).
#
# SEED-Bench is deliberately absent: under its own answer instruction it
# replies in ~5 tokens, and the probe needs 2*block_size = 32 trainable tokens
# per generation, so only 28 of 200 would be measurable.
#
# "<name>-origin" sends the benchmark's own answer instruction instead of the
# suite's shared boxed one, and works for any benchmark whose class defines
# ORIGINAL_PROMPT_KWARGS.
#
#   BENCHMARKS="chartqa mmstar realworldqa seedbench-image textvqa mmmu mathvision dynamath"
#
# is the older eight-benchmark sweep.
BENCHMARKS=${BENCHMARKS:-${BENCHMARK:-"\
chartqa textvqa mmstar dynamath mathvista mathverse"}}

NUM_SAMPLES=${NUM_SAMPLES:-200}

# Generation budget, one value for every benchmark. Decoding is greedy, so the
# first MAX_NEW_TOKENS tokens of an answer are the same tokens whichever budget
# is set: raising it does not change any block measured below the cap, it only
# adds blocks above it. One budget therefore costs nothing in fidelity and buys
# two things -- every benchmark's anchors are drawn from the same position
# range, which is what makes their quartile tables comparable, and generation
# (the dominant cost of a sweep, 40-250 min per benchmark) stays bounded.
#
# What it gives up: nothing is measured past position 1024. Answers longer than
# the budget, out of 200, on the draft this sweep compares against z-lab
# (results/dflash_baseline_llava_ov_1M_prompted_final_*.jsonl):
#
#   budget   chartqa textvqa | mmstar dynamath mathvista mathverse
#   1024         9       5   |   34      45        44        77
#   3072         4       2   |   16      14        21        32
#
# so on the four long benchmarks this reads the first 1024 tokens of between
# 17% and 39% of the answers rather than all of them. The long-generation
# behaviour those answers would show is a separate question, already measured
# end to end by the throughput runs in results/; this probe is about WHICH
# TOKENS end a block, and 128 anchors per generation is the same sample either
# way. Set MAX_NEW_TOKENS_MAP to bring the per-benchmark budgets back.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
# optional per-benchmark overrides, space-separated name=tokens (e.g.
# "mathverse=3072 mmmu=3072"); empty means every benchmark uses MAX_NEW_TOKENS.
MAX_NEW_TOKENS_MAP=${MAX_NEW_TOKENS_MAP:-""}

# Total sequence budget for the re-encode. Over this a sample is DROPPED, not
# truncated (specforge/data/mm_preprocessing.py:314), and input_ids carry the
# processor's image-token expansion -- SEED-Bench and RealWorldQA send big
# images -- so this has to leave room for prompt + image + generation. At 8192
# the only losses are ~5 SEED-Bench questions whose images alone exceed it.
MAX_LENGTH=${MAX_LENGTH:-8192}

NUM_ANCHORS=${NUM_ANCHORS:-128}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-sdpa}
VISUAL_TOP_N=${VISUAL_TOP_N:-10}
# extra flags handed to analyze_dflash_accept.py verbatim (e.g. "--no-visual-kl")
EXTRA_ARGS=${EXTRA_ARGS:-}

# 1 = re-measure every pair even if it is already done
FORCE=${FORCE:-0}
# 1 = print the plan and exit without loading a model
DRY_RUN=${DRY_RUN:-0}

# saved next to the code that produced it, under the project root
ANALYSIS_ROOT=${ANALYSIS_ROOT:-${PROJECT_DIR}/accept_analysis}
RESULTS_JSONL=${RESULTS_JSONL:-${ANALYSIS_ROOT}/accept_metrics.jsonl}
# -----------------------------------------------------------------------------

# The per-benchmark default splits are read out of the probe's own BENCHMARKS
# table rather than restated here. A second copy would drift -- which is exactly
# how this script's ChartQA prompt ended up two lines away from the benchmark's.
# ast only, so this costs no torch import.
declare -A DEFAULT_SPLIT=()
while IFS=$'\t' read -r _name _split; do
    [ -n "${_name}" ] && DEFAULT_SPLIT["${_name}"]="${_split}"
done < <(python3 - "${PROBE}" <<'PY'
import ast, sys
tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
for node in tree.body:
    target = node.targets[0] if isinstance(node, ast.Assign) else None
    if getattr(target, "id", None) == "BENCHMARKS":
        for name, spec in ast.literal_eval(node.value).items():
            print(f"{name}\t{spec['split']}")
        break
PY
)
if [ "${#DEFAULT_SPLIT[@]}" -eq 0 ]; then
    echo "ERROR: could not read the BENCHMARKS table out of ${PROBE}" >&2
    exit 1
fi

max_new_tokens_for() {  # benchmark
    local entry
    for entry in ${MAX_NEW_TOKENS_MAP}; do
        [ "${entry%%=*}" = "$1" ] && { echo "${entry#*=}"; return; }
    done
    echo "${MAX_NEW_TOKENS}"
}

out_dir_for() {  # benchmark, split, draft-name
    echo "${ANALYSIS_ROOT}/$1-${TARGET_NAME}-$3-$2-${NUM_SAMPLES}"
}

gen_cache_for() {  # benchmark, split
    # -mmb marks generations made from the benchmark classes' prompts; a cache
    # from before that move is a different prompt and must not be reused.
    echo "${ANALYSIS_ROOT}/generations/$1-${TARGET_NAME}-$2-${NUM_SAMPLES}-t$(max_new_tokens_for "$1")-mmb.jsonl"
}

# Is this run directory a finished measurement of exactly what we are asking
# for? Compared field by field against run_meta.json, so a changed draft path,
# sample count, anchor count or prompt source all count as "not done".
pair_is_done() {  # out-dir, draft-path, benchmark, split
    local out="$1"
    [ -s "${out}/metrics.json" ] || return 1
    [ -s "${out}/run_meta.json" ] || return 1
    python3 - "${out}/run_meta.json" "$2" "${TARGET_MODEL}" "$3" "$4" \
        "${NUM_SAMPLES}" "$(max_new_tokens_for "$3")" "${NUM_ANCHORS}" \
        "${ATTENTION_BACKEND}" "${MAX_LENGTH}" <<'PY'
import json, sys

(
    path,
    draft,
    target,
    benchmark,
    split,
    samples,
    new_tokens,
    anchors,
    backend,
    max_length,
) = sys.argv[1:11]
try:
    meta = json.load(open(path, encoding="utf-8"))
except Exception as error:
    print(f"unreadable run_meta.json ({error})")
    raise SystemExit(1)

expected = {
    "draft_model_path": draft,
    "target_model_path": target,
    "benchmark": benchmark,
    "split": split,
    "num_samples_requested": int(samples),
    "max_new_tokens": int(new_tokens),
    "num_anchors": int(anchors),
    "attention_backend": backend,
    "max_length": int(max_length),
    # runs made before the prompts moved into the benchmark classes used a
    # different ChartQA instruction and are not comparable with these
    "prompt_source": "mm_benchmarker",
    # v1 runs measured the "with image" KL without pixel_values (the image
    # slots were embedded as plain tokens), so every KL/entropy-derived number
    # in them is wrong; they are superseded, never deleted
    "visual_kl_version": 2,
}
differences = [
    f"{key}: have {meta.get(key)!r}, want {value!r}"
    for key, value in expected.items()
    if meta.get(key) != value
]
if differences:
    print("; ".join(differences))
    raise SystemExit(1)
raise SystemExit(0)
PY
}

run_probe() {  # draft-name, draft-path, benchmark, split
    local name="$1" draft="$2" bench="$3" split="$4" out cache generations status
    local new_tokens
    new_tokens="$(max_new_tokens_for "${bench}")"
    out="$(out_dir_for "${bench}" "${split}" "${name}")"
    cache="$(gen_cache_for "${bench}" "${split}")"
    mkdir -p "${out}" "$(dirname "${cache}")"

    generations="${out}/generations.jsonl"
    if [ ! -s "${generations}" ] && [ -s "${cache}" ]; then
        cp -f "${cache}" "${generations}"
        echo "[probe] seeded generations from ${cache}"
    fi
    if [ -s "${generations}" ]; then
        local have
        have=$(wc -l < "${generations}")
        echo "[probe] scoring ${have} existing generations"
        if [ "${have}" -ne "${NUM_SAMPLES}" ]; then
            echo "[probe] NOTE: that is not the ${NUM_SAMPLES} requested." >&2
            echo "[probe] --num-samples only governs NEW generations; delete" >&2
            echo "[probe] ${generations} to regenerate. (A benchmark can also" >&2
            echo "[probe] simply hold fewer questions than requested.)" >&2
        fi
    else
        echo "[probe] generating up to ${NUM_SAMPLES} answers of <= ${new_tokens} tokens"
    fi

    python3 "${PROBE}" \
        --draft-model-path "${draft}" \
        --target-model-path "${TARGET_MODEL}" \
        --benchmark "${bench}" \
        --split "${split}" \
        --num-samples "${NUM_SAMPLES}" \
        --max-new-tokens "${new_tokens}" \
        --max-length "${MAX_LENGTH}" \
        --num-anchors "${NUM_ANCHORS}" \
        --attention-backend "${ATTENTION_BACKEND}" \
        --visual-top-n "${VISUAL_TOP_N}" \
        --run-name "${name}" \
        --results-jsonl "${RESULTS_JSONL}" \
        --output-dir "${out}" \
        ${EXTRA_ARGS} 2>&1 | tee "${out}/probe.log"
    status="${PIPESTATUS[0]}"

    # publish this run's generations so every later draft (and later
    # submission) scores the identical sequences
    if [ "${status}" -eq 0 ] && [ ! -s "${cache}" ] && [ -s "${generations}" ]; then
        cp -f "${generations}" "${cache}"
        echo "[probe] cached generations at ${cache}"
    fi
    return "${status}"
}

# ------------------------------ plan -----------------------------------------
BENCH_NAMES=()
BENCH_SPLITS=()
for entry in ${BENCHMARKS}; do
    bench="${entry%%:*}"
    if [ "${entry}" = "${bench}" ]; then
        # "<name>-origin" is the same dataset under the benchmark's own answer
        # instruction, so it inherits the base benchmark's default split
        split="${DEFAULT_SPLIT[${bench}]:-${DEFAULT_SPLIT[${bench%-origin}]:-}}"
        if [ -z "${split}" ]; then
            echo "ERROR: '${bench}' is not a benchmark ${PROBE} knows." >&2
            echo "       known: ${!DEFAULT_SPLIT[*]} (each also as <name>-origin)" >&2
            exit 1
        fi
    else
        split="${entry#*:}"
    fi
    BENCH_NAMES+=("${bench}")
    BENCH_SPLITS+=("${split}")
done

DRAFT_NAMES=()
DRAFT_PATHS=()
for entry in ${DRAFTS_SPEC}; do
    name="${entry%%=*}"
    path="${entry#*=}"
    if [ -z "${name}" ] || [ "${name}" = "${entry}" ]; then
        echo "ERROR: DRAFTS_SPEC entry '${entry}' is not name=path" >&2
        exit 1
    fi
    DRAFT_NAMES+=("${name}")
    DRAFT_PATHS+=("${path}")
done

total=$(( ${#BENCH_NAMES[@]} * ${#DRAFT_NAMES[@]} ))
echo "=================================================================="
echo "  project    : ${PROJECT_DIR}"
echo "  target     : ${TARGET_MODEL}  (as ${TARGET_NAME})"
echo "  benchmarks : ${#BENCH_NAMES[@]}  ->  ${BENCH_NAMES[*]}"
echo "  drafts     : ${#DRAFT_NAMES[@]}  ->  ${DRAFT_NAMES[*]}"
echo "  pairs      : ${total}   (${NUM_SAMPLES} samples, ${NUM_ANCHORS} anchors each)"
echo "  budget     : ${MAX_NEW_TOKENS} new tokens (${MAX_NEW_TOKENS_MAP}), max_length ${MAX_LENGTH}"
echo "  results    : ${RESULTS_JSONL}"
echo "  force      : ${FORCE}      dry run: ${DRY_RUN}"
echo "=================================================================="

# ------------------------------ sweep ----------------------------------------
# Benchmark outer, draft inner: the first draft of a benchmark produces the
# generations that every later draft of that benchmark then reuses.
declare -a DONE_PAIRS=() SKIPPED_PAIRS=() FAILED_PAIRS=()
index=0
for bench_index in "${!BENCH_NAMES[@]}"; do
    bench="${BENCH_NAMES[${bench_index}]}"
    split="${BENCH_SPLITS[${bench_index}]}"
    for draft_index in "${!DRAFT_NAMES[@]}"; do
        name="${DRAFT_NAMES[${draft_index}]}"
        draft="${DRAFT_PATHS[${draft_index}]}"
        index=$((index + 1))
        label="${bench}/${split} x ${name} @$(max_new_tokens_for "${bench}")tok"
        out="$(out_dir_for "${bench}" "${split}" "${name}")"

        echo
        echo "=================================================================="
        echo "  [${index}/${total}] ${label}"
        echo "                 ${draft}"
        echo "                 ${out}"
        echo "=================================================================="

        if [ -d "${out}" ]; then
            if reason="$(pair_is_done "${out}" "${draft}" "${bench}" "${split}")"; then
                if [ "${FORCE}" = "1" ]; then
                    echo "[probe] already done, but FORCE=1 -- measuring again"
                else
                    echo "[probe] already done, skipping (metrics.json is current)"
                    SKIPPED_PAIRS+=("${label}")
                    continue
                fi
            elif [ -s "${out}/metrics.json" ] || [ -s "${out}/run_meta.json" ]; then
                # a finished-looking run of something else: keep it, step aside
                archive="${out}.superseded-$(date +%Y%m%d-%H%M%S)"
                echo "[probe] existing run does not match this request:"
                echo "[probe]   ${reason:-no run_meta.json}"
                if [ "${DRY_RUN}" = "1" ]; then
                    # a dry run must not touch the tree; say what would happen
                    echo "[probe] DRY_RUN=1 -- would preserve it as ${archive}"
                else
                    echo "[probe] preserving it as ${archive}"
                    mv "${out}" "${archive}" || {
                        echo "[probe] ERROR: could not archive ${out}" >&2
                        FAILED_PAIRS+=("${label} (archive failed)")
                        continue
                    }
                fi
            fi
        fi

        if [ "${DRY_RUN}" = "1" ]; then
            echo "[probe] DRY_RUN=1 -- would measure this pair"
            DONE_PAIRS+=("${label} (dry run)")
            continue
        fi

        if run_probe "${name}" "${draft}" "${bench}" "${split}"; then
            DONE_PAIRS+=("${label}")
        else
            echo "[probe] ERROR: ${label} failed; see ${out}/probe.log" >&2
            FAILED_PAIRS+=("${label}")
            # A later failure is usually local to one benchmark -- a dataset
            # that will not download, an image the processor rejects -- and the
            # rest of the sweep is still worth having. A failure with nothing
            # measured yet is a different animal: a missing module, a bad path,
            # a wrong environment, all of which will hit every remaining pair
            # exactly the same way. Stop and say so instead of reproducing one
            # traceback two dozen times.
            if [ "${#DONE_PAIRS[@]}" -eq 0 ] && [ "${#SKIPPED_PAIRS[@]}" -eq 0 ]; then
                echo >&2
                echo "[probe] ABORTING: the first pair failed, so nothing has been" >&2
                echo "[probe] measured yet. This is almost always environment-wide" >&2
                echo "[probe] rather than specific to ${bench}. Last lines:" >&2
                echo >&2
                tail -n 20 "${out}/probe.log" >&2
                echo >&2
                echo "[probe] Fix that, then rerun -- nothing measured is ever redone." >&2
                exit 1
            fi
        fi
    done
done

# ------------------------------ summary --------------------------------------
echo
echo "=================================================================="
echo "  measured ${#DONE_PAIRS[@]} / ${total}   skipped ${#SKIPPED_PAIRS[@]}   failed ${#FAILED_PAIRS[@]}"
echo "=================================================================="
for pair in "${DONE_PAIRS[@]-}";    do [ -n "${pair}" ] && echo "  ok      ${pair}"; done
for pair in "${SKIPPED_PAIRS[@]-}"; do [ -n "${pair}" ] && echo "  skip    ${pair}"; done
for pair in "${FAILED_PAIRS[@]-}";  do [ -n "${pair}" ] && echo "  FAILED  ${pair}"; done
echo
echo "  All runs, one line each: ${RESULTS_JSONL}"
echo "  Re-derive any record without a GPU:"
echo "    python3 ${PROBE} --metrics-only --output-dir <run dir>"
if [ "${#FAILED_PAIRS[@]}" -gt 0 ]; then
    echo "  Resubmit to retry the failures; everything already measured is skipped."
    exit 1
fi
echo "=================================================================="
