#!/bin/bash
# Are the 16 draft queries of a DFlash block looking at the same context?
#
# Needs the generations.jsonl that scripts/analyze_accept_hpc.sh produces.
# Submit with `qsub -l select=1:ngpus=1 scripts/analyze_attention_hpc.sh`.

#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N attn-probe
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=4:00:00

# cd "${PBS_O_WORKDIR:-$(dirname "$0")/..}" || exit 1
# source ~/.bashrc
# conda activate specforge

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

ROOT=/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge
DRAFT=${DRAFT:-${ROOT}/draft_models/qwen3.5-4b-mmflash-sharegpt4v-pt-120000}
GEN=${GEN:-${ROOT}/accept_analysis/chartqa-120000/generations.jsonl}
OUT=${OUT:-${ROOT}/attention_analysis/chartqa-120000}

python scripts/analyze_dflash_attention.py \
    --draft-model-path "${DRAFT}" \
    --target-model-path Qwen/Qwen3.5-4B \
    --generations "${GEN}" \
    --num-samples 50 \
    --anchors-per-sample 4 \
    --topk 32 128 512 \
    --output-dir "${OUT}"
