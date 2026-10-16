#!/bin/bash
# Gemma 4 strict overfit + DFLASH serving gate (plan G5) in one 1-GPU PBS job.
# Submit from the repo root:
#
#   qsub scripts/gates/gemma4_overfit_gate_hpc.sh                       # DFlash
#   qsub -v ALGO=mmflash scripts/gates/gemma4_overfit_gate_hpc.sh       # MMFlash
#
# One text row of the Gemma regen is trained to (near) zero loss, then the draft
# is served with SGLang's DFLASH worker on the same prompt; the gate requires the
# served accept length to reach the block size. This is the test that the DFlash
# family's target embedding scale (sqrt(hidden_size) for Gemma) matches serving:
# a wrong scale trains fine and then collapses the accept length.
#
# The capture server and the trainer share the job's one GPU (CAPTURE_GPUS =
# CONSUMER_GPUS = 0, job-relative), so the capture server keeps a small static
# memory fraction. Ports are derived from the job number. Everything lands in
# /scratch/$USER/specforge_gates/gemma4_overfit_gate/<jobid>-<algo>/, with the
# gate's own output in <that dir>.gate.log: the trainer writes a ~6 GB checkpoint,
# which does not belong on the 40 GB NFS home.
#PBS -P CFP04-CF-054
#PBS -j oe
#PBS -k oed
#PBS -N gemma4-overfit
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=03:00:00

set -uo pipefail
export PYTHONUNBUFFERED=1
CONDA_ENV="${CONDA_ENV:-specforge}"
if ! python3 -c "import torch" > /dev/null 2>&1; then
    set +u
    # shellcheck disable=SC1091
    [ -f "${HOME}/.bashrc" ] && source "${HOME}/.bashrc"
    conda activate "${CONDA_ENV}" > /dev/null 2>&1 || true
    set -u
fi
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." 2> /dev/null && pwd)"
if [ ! -f "${PROJECT_DIR:-}/specforge/cli.py" ] && [ -n "${PBS_O_WORKDIR:-}" ]; then
    PROJECT_DIR="${PBS_O_WORKDIR}"
fi
cd "${PROJECT_DIR}" || exit 1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export FLASHINFER_USE_CUDA_NORM=1 SGLANG_NUMA_BIND_V2=0

ALGO="${ALGO:-dflash}"
case "${ALGO}" in
    dflash) GATE_CONFIG=examples/configs/qwen3-8b-dflash-disaggregated.yaml ;;
    mmflash) GATE_CONFIG=examples/configs/qwen3-8b-mmflash-online.yaml ;;
    *) echo "[gate] ERROR: ALGO must be dflash or mmflash, got '${ALGO}'" >&2; exit 1 ;;
esac
JOBNUM="${PBS_JOBID:-0}"; JOBNUM="${JOBNUM%%.*}"
case "${JOBNUM}" in ''|*[!0-9]*) JOBNUM=0 ;; esac
OFFSET=$(( (10#${JOBNUM} % 400) * 10 ))
CACHE_TAG="${JOBNUM}-${ALGO}"
export TRITON_CACHE_DIR="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}/triton-gate-${CACHE_TAG}"
export TORCHINDUCTOR_CACHE_DIR="${SPECFORGE_CACHE_ROOT:-/scratch/${USER}/tmp}/inductor-gate-${CACHE_TAG}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
WORK_DIR="${WORK_DIR:-/scratch/${USER}/specforge_gates/gemma4_overfit_gate/${CACHE_TAG}}"
rm -rf "${WORK_DIR}"
mkdir -p "$(dirname "${WORK_DIR}")"

echo "[gate] $(date '+%F %T') host=$(hostname) job=${PBS_JOBID:-local} algo=${ALGO} work=${WORK_DIR}"
CONFIG="${GATE_CONFIG}" \
TARGET_MODEL_PATH="${TARGET_MODEL_PATH:-/scratch/fengsicheng/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2}" \
DRAFT_CONFIG_PATH="configs/gemma4-e4b-${ALGO}.json" \
SOURCE_DATA_PATH="${SOURCE_DATA_PATH:-/scratch/Projects/CFP-04/CFP04-CF-054/fengsicheng/specforge/regen_data/gemma4-E4B-it_llava-ov15-1M-prompted_regen_first_turn.smoke_every500.jsonl}" \
EXPECTED_PROJECTOR_TYPE="${EXPECTED_PROJECTOR_TYPE:-dflash}" CHAT_TEMPLATE=gemma4 \
EMBEDDING_KEY=model.language_model.embed_tokens.weight \
CAPTURE_ATTENTION_BACKEND=triton \
REASONING_POLICY=forbidden ENABLE_THINKING=false \
CAPTURE_GPUS=0 CONSUMER_GPUS=0 CAPTURE_MEM_FRACTION="${CAPTURE_MEM_FRACTION:-0.30}" \
CAPTURE_PORT=$(( 36000 + OFFSET )) MOONCAKE_RPC_PORT=$(( 46000 + OFFSET )) \
MOONCAKE_HTTP_PORT=$(( 46001 + OFFSET )) MOONCAKE_METRICS_PORT=$(( 46002 + OFFSET )) \
SERVING_PORT=$(( 36001 + OFFSET )) \
WORK_DIR="${WORK_DIR}" RUN_ID="gemma4-${ALGO}-overfit-${JOBNUM}" \
bash scripts/gates/run_disaggregated_overfit_gate.sh 2>&1 | tee "${WORK_DIR}.gate.log"
status=${PIPESTATUS[0]}
echo "[gate] $(date '+%F %T') exit ${status}"
exit "${status}"
