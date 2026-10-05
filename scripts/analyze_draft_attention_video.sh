#!/bin/bash
# Where the 4B MMFlash draft attends in 48-frame video prompts (paper figure).
#
# One GPU, three steps:
#   1. the answers: one full-context MMFlash benchmark run (no window, no
#      sparse; same server flags as the paper's runs) with --save-generations,
#      via scripts/sweep_draft_window.sh -> results/<GEN_NAME>_f16-48-48_winfull_*.
#      ~25 min for 60 questions; skipped when the generation files exist.
#   2. the probe: the HF target re-reads each (frames, prompt, SGLang answer)
#      once for the draft's input features, and the draft's attention is
#      recorded at decode steps over the answer (analyze_draft_attention_video.py);
#      ~11 s a question.
#   3. the figures and numbers (plot_draft_attention_video.py, CPU).
# Same questions, frames and prompts as the video benchmarks (20 each of
# LongVideoBench / MVBench / MovieChat). A re-run resumes every step.
#
#   bash scripts/analyze_draft_attention_video.sh                     # interactive, 1 GPU, ~40 min
#   qsub -l select=1:ngpus=1 -l walltime=02:00:00 scripts/analyze_draft_attention_video.sh
#
# Extra arguments go to the python probe. Probe outputs go to
#   /scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/attention_analysis/video-draft-attn
# and the figures to figures/ in this checkout (video_attn_map_<id>, video_attn_cdf).
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N draft-attn
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=02:00:00

export PYTHONUNBUFFERED=1

CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[env] torch is not importable; activating conda env '${CONDA_ENV}'"
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
fi
if ! python3 -c "import torch" > /dev/null 2>&1; then
    echo "[error] torch is still not importable after activating '${CONDA_ENV}'" >&2
    exit 1
fi

# The benchmarks cache their frames under .cache/ relative to the checkout,
# so run from it.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/scripts/analyze_draft_attention_video.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

BENCHES="${BENCHES:-longvideobench:20 mvbench:20 moviechat:20}"
GEN_NAME="${GEN_NAME:-video_mmflash_qwen35-4B_concurrency1_temp0_4096_gen}"
GEN_PREFIX="results/${GEN_NAME}_f16-48-48_winfull"

# ---- 1. SGLang's answers (full draft context, exactly the benchmark setup) ----
missing=0
for spec in ${BENCHES}; do
    [ -s "${GEN_PREFIX}_${spec%%:*}_generations.jsonl" ] || missing=1
done
if [ "${missing}" = "1" ]; then
    echo "[step 1] generating the benchmark answers with a full-context server"
    WINDOWS=full IMAGE_BENCHES="" VIDEO_BENCHES="${BENCHES}" VIDEO_RUN_NAME="${GEN_NAME}" \
        SAVE_GENERATIONS=1 SUMMARY=0 bash scripts/sweep_draft_window.sh || exit 1
    for spec in ${BENCHES}; do
        if [ ! -s "${GEN_PREFIX}_${spec%%:*}_generations.jsonl" ]; then
            echo "[error] ${GEN_PREFIX}_${spec%%:*}_generations.jsonl was not written; if the results file" >&2
            echo "        ${GEN_PREFIX}_results.jsonl already lists ${spec%%:*}, bench_mm skipped it -- delete that file and re-run" >&2
            exit 1
        fi
    done
else
    echo "[step 1] benchmark answers found under ${GEN_PREFIX}_*"
fi

# ---- 2-3. probe and plot, on the job's first GPU (PBS may name it by UUID) ----
GPU_INDEX="${GPU_INDEX:-0}"
IFS=',' read -ra VISIBLE_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS[${GPU_INDEX}]:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/attention_analysis/video-draft-attn}"
# shellcheck disable=SC2086
python3 scripts/analyze_draft_attention_video.py --output-dir "${OUTPUT_DIR}" --benchmarks ${BENCHES} \
    --bench-generations "${GEN_PREFIX}" "$@" || exit 1
python3 scripts/plot_draft_attention_video.py --input-dir "${OUTPUT_DIR}"
