#!/bin/bash
# Per-anchor accept length of two DFlash drafts, resolved by token, on the SAME
# target generations.
#
# Runs z-lab/Qwen3.5-4B-DFlash (text-trained) first, then the locally trained
# mmflash checkpoint. The first run produces the target's answers; the second
# scores that exact same file. Regenerating for the second draft would waste the
# slowest stage and, worse, leave a second difference between the two runs --
# the comparison is only about the draft when the token sequence being scored is
# byte-identical.
#
# Submit with `qsub -l select=1:ngpus=1 scripts/analyze_accept_hpc.sh`, or run
# directly on a node that already has a GPU. No SGLang server is involved: the
# target is loaded in-process, which is what gives access to the hidden states
# the draft reads.

#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N accept-probe
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=8:00:00

# cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}" || exit 1
# source ~/.bashrc
# conda activate specforge

set -uo pipefail

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

ROOT=/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge
NUM_SAMPLES=${NUM_SAMPLES:-200}
BENCHMARK=${BENCHMARK:-chartqa}
SPLIT=${SPLIT:-test}

# name -> draft checkpoint. The first entry generates; the rest reuse its file.
DRAFTS=(
    "zlab:z-lab/Qwen3.5-4B-DFlash"
    "mmflash:${ROOT}/draft_models/qwen3.5-4b-mmflash-sharegpt4v-pt-120000"
)

out_dir_for() { echo "${ROOT}/accept_analysis/${BENCHMARK}-$1-${NUM_SAMPLES}"; }

run_probe() {
    local name="$1" draft="$2" out
    out="$(out_dir_for "${name}")"
    mkdir -p "${out}"

    local generations="${out}/generations.jsonl"
    if [ -f "${generations}" ]; then
        local have
        have=$(wc -l < "${generations}")
        echo "[probe] ${name}: scoring ${have} existing generations in ${out}"
        if [ "${have}" -ne "${NUM_SAMPLES}" ]; then
            echo "[probe] WARNING: that is not the ${NUM_SAMPLES} requested." >&2
            echo "[probe] --num-samples only governs NEW generations; delete" >&2
            echo "[probe] ${generations} to regenerate." >&2
        fi
    else
        echo "[probe] ${name}: generating ${NUM_SAMPLES} answers into ${out}"
    fi

    python scripts/analyze_dflash_accept.py \
        --draft-model-path "${draft}" \
        --target-model-path Qwen/Qwen3.5-4B \
        --benchmark "${BENCHMARK}" \
        --split "${SPLIT}" \
        --num-samples "${NUM_SAMPLES}" \
        --max-new-tokens 1024 \
        --num-anchors 128 \
        --attention-backend sdpa \
        --visual-top-n 10 \
        --output-dir "${out}" 2>&1 | tee "${out}/probe.log"
    return "${PIPESTATUS[0]}"
}

first_name="${DRAFTS[0]%%:*}"
shared="$(out_dir_for "${first_name}")/generations.jsonl"

for index in "${!DRAFTS[@]}"; do
    entry="${DRAFTS[$index]}"
    name="${entry%%:*}"
    draft="${entry#*:}"
    out="$(out_dir_for "${name}")"

    echo
    echo "=================================================================="
    echo "  [$((index + 1))/${#DRAFTS[@]}] ${name}   ${draft}"
    echo "=================================================================="

    # Every later draft scores the first one's generations. Copying rather than
    # pointing at them keeps each output directory self-contained, so a single
    # directory is enough to reproduce or re-analyse that run.
    if [ "${index}" -gt 0 ]; then
        if [ ! -s "${shared}" ]; then
            echo "[probe] ERROR: ${shared} is missing or empty;" >&2
            echo "[probe] the first run did not produce generations. Stopping." >&2
            exit 1
        fi
        mkdir -p "${out}"
        cp -f "${shared}" "${out}/generations.jsonl"
    fi

    if ! run_probe "${name}" "${draft}"; then
        echo "[probe] ERROR: ${name} failed; see ${out}/probe.log" >&2
        exit 1
    fi
done

echo
echo "=================================================================="
echo "  done. Compare the TOKEN LEVEL tables in:"
for entry in "${DRAFTS[@]}"; do
    echo "    $(out_dir_for "${entry%%:*}")/probe.log"
done
echo "=================================================================="
